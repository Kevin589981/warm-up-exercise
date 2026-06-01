import os
import re
import json
import math
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# ============================================================
# Positional Encoding
# ============================================================


class SinCosPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class RoPEPositionalEncoding(nn.Module):
    """Rotary Positional Encoding (RoPE).

    修复说明：
    1. inv_freq 按 d_k（每个 head 的维度）而非 d_model 计算，
       在 apply_rope 里动态截断以适配实际 d_k。
    2. apply_rope 中 sin/cos 通过 cat([s,s]) repeat 到完整 d_k 长度，
       避免与 _rotate_half 结果广播错位。
    3. forward 保留 dropout（embed 后调用），不再承担加位置信息的职责；
       位置信息通过 apply_rope 在 attention 内部注入。
    """

    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        # 预计算最大 d_model/2 个频率，apply_rope 里按实际 d_k 截断
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        self.dropout = nn.Dropout(dropout)

    def _rotate_half(self, x):
        """将向量分前后两半，后半取负后拼到前半前面。"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope(self, x, pos):
        """
        x  : (batch, heads, seq, d_k)
        pos: (seq,)  float tensor of absolute positions
        """
        d_k = x.shape[-1]
        # Bug 1 fix: 按实际 d_k 截断，确保 inv_freq 长度 == d_k // 2
        inv_freq = self.inv_freq[: d_k // 2]

        # (seq, d_k/2)
        sin_half = torch.sin(pos.unsqueeze(-1) * inv_freq.unsqueeze(0))
        cos_half = torch.cos(pos.unsqueeze(-1) * inv_freq.unsqueeze(0))

        # Bug 2 fix: repeat 到完整 d_k，与 _rotate_half 的分组方式对齐
        sin = torch.cat([sin_half, sin_half], dim=-1)   # (seq, d_k)
        cos = torch.cat([cos_half, cos_half], dim=-1)   # (seq, d_k)

        # 扩维以匹配 (batch, heads, seq, d_k)
        sin = sin.unsqueeze(0).unsqueeze(0)
        cos = cos.unsqueeze(0).unsqueeze(0)

        return x * cos + self._rotate_half(x) * sin

    def forward(self, x):
        # RoPE 不在 embedding 层加位置，只保留 dropout
        return self.dropout(x)


# ============================================================
# Attention
# ============================================================


class MultiHeadAttention(nn.Module):
    """标准多头注意力（sincos 位置编码用）。"""

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        Q = self.W_q(query).view(B, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.nhead, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)


class MultiHeadAttentionRoPE(nn.Module):
    """支持 RoPE 的多头注意力。
    
    Bug 3 fix: rope_enc 从外部传入并实际使用，q_pos/k_pos 由调用层提供。
    """

    def __init__(self, d_model, nhead, dropout=0.1, rope_enc=None):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.rope_enc = rope_enc

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, q_pos=None, k_pos=None):
        B = query.size(0)
        Q = self.W_q(query).view(B, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.nhead, self.d_k).transpose(1, 2)

        # Bug 3 fix: 只有 rope_enc 非 None 且位置信息存在时才施加 RoPE
        if self.rope_enc is not None:
            if q_pos is None:
                q_pos = torch.arange(Q.size(2), device=Q.device).float()
            if k_pos is None:
                k_pos = torch.arange(K.size(2), device=K.device).float()
            Q = self.rope_enc.apply_rope(Q, q_pos)
            K = self.rope_enc.apply_rope(K, k_pos)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)


# ============================================================
# Feed-Forward
# ============================================================


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ============================================================
# Encoder / Decoder Layers
# ============================================================


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1, rope_enc=None):
        super().__init__()
        # Bug 3 fix: rope_enc 正确传入 attention 层
        if rope_enc is not None:
            self.self_attn = MultiHeadAttentionRoPE(d_model, nhead, dropout, rope_enc)
        else:
            self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1, rope_enc=None):
        super().__init__()
        # 自注意力施加 RoPE；cross-attention 不施加（key 来自 encoder，位置已编码）
        if rope_enc is not None:
            self.self_attn = MultiHeadAttentionRoPE(d_model, nhead, dropout, rope_enc)
        else:
            self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        cross_out = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout(cross_out))
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


# ============================================================
# 完整 Transformer (Encoder-Decoder)
# ============================================================


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        nhead,
        num_layers,
        d_ff,
        max_len=50,
        dropout=0.1,
        pad_idx=0,
        pos_enc_type="sincos",
    ):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc_type = pos_enc_type

        if pos_enc_type == "rope":
            self.rope_enc = RoPEPositionalEncoding(d_model, max_len, dropout)
            self.pos_encoding = self.rope_enc   # forward 里统一调用做 dropout
        else:
            self.rope_enc = None
            self.pos_encoding = SinCosPositionalEncoding(d_model, max_len, dropout)

        # Bug 3 fix: rope_enc 正确传入各 layer
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, d_ff, dropout, rope_enc=self.rope_enc)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, d_ff, dropout, rope_enc=self.rope_enc)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.pad_idx = pad_idx

    def encode(self, src, src_mask):
        x = self.src_embed(src)
        x = self.pos_encoding(x)   # sincos: 加位置 + dropout；rope: 仅 dropout
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, tgt_mask, memory_mask):
        x = self.tgt_embed(tgt)
        x = self.pos_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, tgt_mask, memory_mask)
        return self.fc_out(out)


# ============================================================
# Decoder-only Transformer
# ============================================================


class TransformerDecoderOnly(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_layers,
        d_ff,
        max_len=50,
        dropout=0.1,
        pad_idx=0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = SinCosPositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.pad_idx = pad_idx

    def forward(self, x, mask=None):
        x = self.embed(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, x, mask, mask)
        return self.fc_out(x)


# ============================================================
# Mask 工具函数
# ============================================================


def make_pad_mask(seq, pad_idx=0):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def make_subsequent_mask(size, device):
    return (
        torch.tril(torch.ones(size, size, device=device))
        .unsqueeze(0).unsqueeze(0).bool()
    )


def make_decoder_mask(tgt, pad_idx=0):
    device = tgt.device
    pad_mask = make_pad_mask(tgt, pad_idx)
    sub_mask = make_subsequent_mask(tgt.size(1), device)
    return pad_mask & sub_mask


# ============================================================
# Dataset
# ============================================================


class Seq2SeqDataset(Dataset):
    def __init__(self, src_seqs, tgt_seqs):
        self.src_seqs = src_seqs
        self.tgt_seqs = tgt_seqs

    def __len__(self):
        return len(self.src_seqs)

    def __getitem__(self, idx):
        return self.src_seqs[idx], self.tgt_seqs[idx]


# ============================================================
# 数据生成
# ============================================================


def generate_addition_pairs(num_samples, digits_a, digits_b):
    pairs = []
    for _ in range(num_samples):
        a = random.randint(10 ** (digits_a - 1), 10 ** digits_a - 1)
        b = random.randint(10 ** (digits_b - 1), 10 ** digits_b - 1)
        pairs.append((f"{a}+{b}=", str(a + b)))
    return pairs


def build_vocab(pairs):
    chars = set()
    for src, tgt in pairs:
        chars.update(src)
        chars.update(tgt)
    special = ["<PAD>", "<GO>", "<EOS>"]
    vocab = {c: i + len(special) for i, c in enumerate(sorted(chars))}
    for i, s in enumerate(special):
        vocab[s] = i
    return vocab


def encode_pair(src, tgt, vocab, max_len):
    src_ids = [vocab.get(c, vocab["<PAD>"]) for c in src]
    src_ids = src_ids[:max_len]
    if len(src_ids) < max_len:
        src_ids += [vocab["<PAD>"]] * (max_len - len(src_ids))

    tgt_ids = (
        [vocab["<GO>"]]
        + [vocab.get(c, vocab["<PAD>"]) for c in tgt]
        + [vocab["<EOS>"]]
    )
    tgt_ids = tgt_ids[:max_len]
    if len(tgt_ids) < max_len:
        tgt_ids += [vocab["<PAD>"]] * (max_len - len(tgt_ids))

    return src_ids, tgt_ids


# ============================================================
# 推理函数
# ============================================================


def greedy_decode(
    model, src, src_mask, max_len, vocab, device, pad_idx=0, decoder_only=False
):
    model.eval()
    inv_vocab = {idx: c for c, idx in vocab.items()
                 if c not in ("<PAD>", "<GO>", "<EOS>")}

    with torch.no_grad():
        if decoder_only:
            # decoder-only: 训练时 logits 从 src_len 位开始对应 tgt_output[0]
            # 推理时同样拼接 [src | <GO>]，取最后一位预测下一个 token
            tgt = torch.tensor([[vocab["<GO>"]]], dtype=torch.long, device=device)
            for _ in range(max_len - 1):
                inp = torch.cat([src, tgt], dim=1)
                mask = make_decoder_mask(inp, pad_idx)
                logits = model(inp, mask)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tgt = torch.cat([tgt, next_token], dim=1)
                if next_token.item() == vocab["<EOS>"]:
                    break
            # tgt[0] = [<GO>, t1, t2, ..., <EOS>?]，跳过 <GO>
            token_ids = tgt[0, 1:].tolist()
        else:
            tgt = torch.tensor([[vocab["<GO>"]]], dtype=torch.long, device=device)
            memory = model.encode(src, src_mask)
            for _ in range(max_len - 1):
                tgt_mask = make_decoder_mask(tgt, pad_idx)
                logits = model.decode(tgt, memory, tgt_mask, src_mask)
                logits = model.fc_out(logits)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tgt = torch.cat([tgt, next_token], dim=1)
                if next_token.item() == vocab["<EOS>"]:
                    break
            token_ids = tgt[0, 1:].tolist()

    result = []
    for tid in token_ids:
        if tid == vocab["<EOS>"] or tid == vocab["<PAD>"]:
            break
        if tid in inv_vocab:
            result.append(inv_vocab[tid])
    return "".join(result)


# ============================================================
# Warmup learning rate scheduler
# ============================================================


class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.d_model ** -0.5 * min(
            self.step_num ** -0.5,
            self.step_num * self.warmup_steps ** -1.5,
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


# ============================================================
# 单个实验的运行逻辑
# ============================================================


def run_single_experiment(exp, args, device_str):
    import sys
    from torch.utils.data import DataLoader

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    device = torch.device(device_str)
    set_seed(42)

    print(f"[{exp['name']}] Generating data...", flush=True)
    all_train_pairs = make_train_gen(exp["train_config"])()
    test_pairs = make_test_gen(exp["test_config"])()
    print(
        f"[{exp['name']}] Total train: {len(all_train_pairs)}, Test: {len(test_pairs)}",
        flush=True,
    )

    vocab = build_vocab(all_train_pairs + test_pairs)
    vocab_size = len(vocab)
    max_len = args.max_len

    def to_tensors(pairs):
        src_list, tgt_list = [], []
        for src, tgt in pairs:
            s, t = encode_pair(src, tgt, vocab, max_len)
            src_list.append(s)
            tgt_list.append(t)
        return (
            torch.tensor(src_list, dtype=torch.long),
            torch.tensor(tgt_list, dtype=torch.long),
        )

    all_src_t, all_tgt_t = to_tensors(all_train_pairs)
    test_src_t, _ = to_tensors(test_pairs)

    n_val = max(1, int(len(all_train_pairs) * 0.1))
    indices = torch.randperm(len(all_train_pairs))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_src_t, train_tgt_t = all_src_t[train_idx], all_tgt_t[train_idx]
    val_src_t = all_src_t[val_idx]
    val_pairs = [all_train_pairs[i] for i in val_idx.tolist()]

    train_loader = DataLoader(
        Seq2SeqDataset(train_src_t, train_tgt_t),
        batch_size=args.batch_size,
        shuffle=True,
    )

    d_model = exp.get("d_model", args.d_model)
    num_layers = exp.get("num_layers", args.num_layers)
    pos_enc_type = exp.get("pos_enc_type", "sincos")
    use_warmup = exp.get("use_warmup", False)

    if exp["model_type"] == "decoder_only":
        model = TransformerDecoderOnly(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=args.nhead,
            num_layers=num_layers,
            d_ff=d_model * 2,
            max_len=max_len * 2,
            dropout=args.dropout,
            pad_idx=0,
        )
    else:
        model = Transformer(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=d_model,
            nhead=args.nhead,
            num_layers=num_layers,
            d_ff=d_model * 2,
            max_len=max_len,
            dropout=args.dropout,
            pad_idx=0,
            pos_enc_type=pos_enc_type,
        )

    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = WarmupScheduler(optimizer, d_model, warmup_steps=4000) if use_warmup else None
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    history = {"train_loss": [], "val_acc": []}
    best_acc = -1.0
    best_state = None
    curriculum = exp.get("curriculum", None)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_n = 0

        current_loader = train_loader
        if curriculum is not None:
            stage_pairs = []
            for stage_end, stage_digits in curriculum:
                if epoch < stage_end:
                    stage_pairs = [
                        (s, t) for s, t in all_train_pairs
                        if _pair_digits(s) <= stage_digits
                    ]
                    break
            if stage_pairs:
                src_l, tgt_l = to_tensors(stage_pairs)
                current_loader = DataLoader(
                    Seq2SeqDataset(src_l, tgt_l),
                    batch_size=args.batch_size,
                    shuffle=True,
                )

        for src_batch, tgt_batch in current_loader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]
            src_mask = make_pad_mask(src_batch, 0)

            optimizer.zero_grad()

            if exp["model_type"] == "decoder_only":
                combined = torch.cat([src_batch, tgt_input], dim=1)
                mask = make_decoder_mask(combined, 0)
                logits = model(combined, mask)
                src_len = src_batch.size(1)
                # 取 src_len 之后的位置预测 tgt_output（与推理对齐）
                logits = logits[:, src_len:, :]
            else:
                tgt_mask = make_decoder_mask(tgt_input, 0)
                logits = model(src_batch, tgt_input, src_mask, tgt_mask, src_mask)

            loss = criterion(logits.reshape(-1, vocab_size), tgt_output.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item() * tgt_output.numel()
            total_n += tgt_output.numel()

        train_loss = total_loss / total_n

        # Validation
        correct = 0
        val_sample = min(200, len(val_pairs))
        for i in range(val_sample):
            src_tensor = val_src_t[i: i + 1].to(device)
            src_mask = make_pad_mask(src_tensor, 0)
            pred = greedy_decode(
                model, src_tensor, src_mask, max_len, vocab, device,
                pad_idx=0, decoder_only=(exp["model_type"] == "decoder_only"),
            )
            if pred == val_pairs[i][1]:
                correct += 1

        acc = correct / val_sample
        history["train_loss"].append(train_loss)
        history["val_acc"].append(acc)

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[{exp['name']}] epoch {epoch + 1:02d}/{args.epochs} | "
            f"loss={train_loss:.4f} acc={acc:.4f}",
            flush=True,
        )

    # Test
    model.load_state_dict(best_state)
    model = model.to(device)
    correct = 0
    error_detail = {}
    wrong_samples = []

    for i in range(len(test_pairs)):
        src_tensor = test_src_t[i: i + 1].to(device)
        src_mask = make_pad_mask(src_tensor, 0)
        pred = greedy_decode(
            model, src_tensor, src_mask, max_len, vocab, device,
            pad_idx=0, decoder_only=(exp["model_type"] == "decoder_only"),
        )
        true_ans = test_pairs[i][1]
        is_correct = pred == true_ans
        if is_correct:
            correct += 1
        elif len(wrong_samples) < 20:
            wrong_samples.append({
                "input": test_pairs[i][0],
                "expected": true_ans,
                "predicted": pred,
            })

        src_text = test_pairs[i][0]
        parts = src_text.rstrip("=").split("+")
        length_key = f"{len(parts[0])}+{len(parts[1])}"
        if length_key not in error_detail:
            error_detail[length_key] = {"correct": 0, "total": 0}
        error_detail[length_key]["total"] += 1
        if is_correct:
            error_detail[length_key]["correct"] += 1

    full_acc = correct / len(test_pairs)
    print(
        f"[{exp['name']}] Done | val_acc={best_acc:.4f} "
        f"test_acc={full_acc:.4f} ({len(test_pairs)} samples)",
        flush=True,
    )

    return {
        "config": {
            "name": exp["name"],
            "desc": exp["desc"],
            "model_type": exp["model_type"],
        },
        "best_val_acc": best_acc,
        "test_acc": full_acc,
        "test_size": len(test_pairs),
        "history": history,
        "best_state": best_state,
        "error_detail": error_detail,
        "wrong_samples": wrong_samples,
    }


def _pair_digits(src_text):
    parts = src_text.rstrip("=").split("+")
    return max(len(parts[0]), len(parts[1]))


# ============================================================
# 主程序
# ============================================================


def make_train_gen(config):
    def gen():
        if config == "3plus3":
            return generate_addition_pairs(8000, 3, 3)
        elif config == "mixed":
            return (
                generate_addition_pairs(2000, 2, 2)
                + generate_addition_pairs(3000, 3, 3)
                + generate_addition_pairs(2000, 4, 4)
                + generate_addition_pairs(1000, 5, 5)
            )
        return []
    return gen


def make_test_gen(config):
    def gen():
        if config == "3plus3":
            return generate_addition_pairs(1000, 3, 3)
        elif config == "4plus4":
            return generate_addition_pairs(1000, 4, 4)
        elif config == "5plus5":
            return generate_addition_pairs(1000, 5, 5)
        elif config == "mixed":
            return (
                generate_addition_pairs(500, 2, 2)
                + generate_addition_pairs(500, 3, 3)
                + generate_addition_pairs(500, 4, 4)
                + generate_addition_pairs(500, 5, 5)
            )
        return []
    return gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="outputs_addition")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    device_str = str(device)
    print(f"Using device: {device_str}")

    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        # === Original 5 ===
        {
            "name": "baseline_3plus3",
            "model_type": "enc_dec",
            "train_config": "3plus3",
            "test_config": "3plus3",
            "desc": "3-digit+3-digit identically distributed",
        },
        {
            "name": "extrapolate_4plus4",
            "model_type": "enc_dec",
            "train_config": "3plus3",
            "test_config": "4plus4",
            "desc": "3-digit training -> 4-digit testing (extrapolation)",
        },
        {
            "name": "extrapolate_5plus5",
            "model_type": "enc_dec",
            "train_config": "3plus3",
            "test_config": "5plus5",
            "desc": "3-digit training -> 5-digit testing (extrapolation)",
        },
        {
            "name": "mixed_training",
            "model_type": "enc_dec",
            "train_config": "mixed",
            "test_config": "mixed",
            "desc": "Mixed length training",
        },
        {
            "name": "decoder_only_3plus3",
            "model_type": "decoder_only",
            "train_config": "3plus3",
            "test_config": "3plus3",
            "desc": "Decoder-only variant",
        },
        # === New 7 ===
        {
            "name": "mixed_test_4plus4",
            "model_type": "enc_dec",
            "train_config": "mixed",
            "test_config": "4plus4",
            "desc": "Mixed training -> test on 4-digit only",
        },
        {
            "name": "deeper_4layers",
            "model_type": "enc_dec",
            "train_config": "3plus3",
            "test_config": "3plus3",
            "desc": "Deeper model (4 layers)",
            "num_layers": 4,
        },
        {
            "name": "wider_256d",
            "model_type": "enc_dec",
            "train_config": "3plus3",
            "test_config": "3plus3",
            "desc": "Wider model (d_model=256)",
            "d_model": 256,
        },
        {
            "name": "rope_enc_dec",
            "model_type": "enc_dec",
            "train_config": "3plus3",
            "test_config": "3plus3",
            "desc": "RoPE positional encoding",
            "pos_enc_type": "rope",
        },
        {
            "name": "curriculum_learning",
            "model_type": "enc_dec",
            "train_config": "mixed",
            "test_config": "mixed",
            "desc": "Curriculum: easy->hard staged training",
            "curriculum": [(15, 2), (30, 3), (40, 4), (50, 5)],
        },
        {
            "name": "warmup_scheduler",
            "model_type": "enc_dec",
            "train_config": "3plus3",
            "test_config": "3plus3",
            "desc": "Warmup + inverse sqrt LR schedule",
            "use_warmup": True,
        },
        {
            "name": "error_analysis_4plus4",
            "model_type": "enc_dec",
            "train_config": "mixed",
            "test_config": "4plus4",
            "desc": "Error analysis on 4-digit (from mixed model)",
        },
    ]

    if args.workers > 1:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        tasks = []
        for i, exp in enumerate(experiments):
            dev = f"cuda:{i % num_gpus}" if num_gpus > 0 else "cpu"
            tasks.append((exp, args, dev))

        with mp.Pool(processes=args.workers) as pool:
            results = pool.starmap(run_single_experiment, tasks)
    else:
        results = [run_single_experiment(exp, args, device_str) for exp in experiments]

    best_overall = max(results, key=lambda x: x["test_acc"])
    print(
        f"\nBest overall: {best_overall['config']['name']} | "
        f"val_acc={best_overall['best_val_acc']:.4f} "
        f"test_acc={best_overall['test_acc']:.4f}"
    )

    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        serializable = [
            {k: v for k, v in r.items() if k != "best_state"}
            for r in results
        ]
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    for r in results:
        if r.get("error_detail"):
            print(f"\n--- Error analysis: {r['config']['name']} ---")
            for length_key, stats in sorted(r["error_detail"].items()):
                acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                print(f"  {length_key}: {stats['correct']}/{stats['total']} = {acc:.3f}")
            for ws in (r.get("wrong_samples") or [])[:10]:
                print(
                    f"    {ws['input']} -> expected={ws['expected']}, got={ws['predicted']}"
                )

    torch.save(best_overall["best_state"], out_dir / "best_model.pt")
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()