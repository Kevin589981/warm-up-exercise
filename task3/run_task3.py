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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# 固定随机种子
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# ============================================================
# Transformer 组件
# ============================================================


class MultiHeadAttention(nn.Module):
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
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ V
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
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
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
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


class PositionalEncoding(nn.Module):
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
    ):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.pad_idx = pad_idx

    def encode(self, src, src_mask):
        x = self.src_embed(src)
        x = self.pos_encoding(x)
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
# Decoder-only Transformer (类似 GPT)
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
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)]
        )
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
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def make_subsequent_mask(size, device):
    mask = (
        torch.tril(torch.ones(size, size, device=device))
        .unsqueeze(0)
        .unsqueeze(0)
        .bool()
    )
    return mask


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
        a = random.randint(10 ** (digits_a - 1), 10**digits_a - 1)
        b = random.randint(10 ** (digits_b - 1), 10**digits_b - 1)
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
        [vocab["<GO>"]] + [vocab.get(c, vocab["<PAD>"]) for c in tgt] + [vocab["<EOS>"]]
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
    tgt = torch.tensor([[vocab["<GO>"]]], dtype=torch.long, device=device)

    with torch.no_grad():
        if decoder_only:
            # Decoder-only: 把 src 和 tgt 拼在一起
            for _ in range(max_len - 1):
                combined = torch.cat([src, tgt], dim=1)
                mask = make_decoder_mask(combined, pad_idx)
                logits = model(combined, mask)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tgt = torch.cat([tgt, next_token], dim=1)
                if next_token.item() == vocab["<EOS>"]:
                    break
        else:
            memory = model.encode(src, src_mask)
            for _ in range(max_len - 1):
                tgt_mask = make_decoder_mask(tgt, pad_idx)
                logits = model.decode(tgt, memory, tgt_mask, src_mask)
                logits = model.fc_out(logits)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tgt = torch.cat([tgt, next_token], dim=1)
                if next_token.item() == vocab["<EOS>"]:
                    break

    # 解码输出
    result = []
    for tid in tgt[0, 1:].tolist():  # 跳过 <GO>
        if tid == vocab["<EOS>"] or tid == vocab["<PAD>"]:
            break
        for c, idx in vocab.items():
            if idx == tid and c not in ["<PAD>", "<GO>", "<EOS>"]:
                result.append(c)
                break
    return "".join(result)


# ============================================================
# 单个实验的运行逻辑（供多进程调用）
# ============================================================


def run_single_experiment(exp, args, device_str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    import random
    import numpy as np

    device = torch.device(device_str)
    set_seed(42)

    print(f"[{exp['name']}] Generating data...")
    train_gen = make_train_gen(exp["train_config"])
    test_gen = make_test_gen(exp["test_config"])
    train_pairs = train_gen()
    test_pairs = test_gen()
    print(f"[{exp['name']}] Train: {len(train_pairs)}, Test: {len(test_pairs)}")

    vocab = build_vocab(train_pairs + test_pairs)
    vocab_size = len(vocab)
    max_len = args.max_len

    train_src, train_tgt = [], []
    for src, tgt in train_pairs:
        s, t = encode_pair(src, tgt, vocab, max_len)
        train_src.append(s)
        train_tgt.append(t)

    test_src, test_tgt = [], []
    for src, tgt in test_pairs:
        s, t = encode_pair(src, tgt, vocab, max_len)
        test_src.append(s)
        test_tgt.append(t)

    train_src_t = torch.tensor(train_src, dtype=torch.long)
    train_tgt_t = torch.tensor(train_tgt, dtype=torch.long)
    test_src_t = torch.tensor(test_src, dtype=torch.long)

    train_dataset = Seq2SeqDataset(train_src_t, train_tgt_t)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    if exp["model_type"] == "decoder_only":
        model = TransformerDecoderOnly(
            vocab_size=vocab_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            d_ff=args.d_ff,
            max_len=max_len,
            dropout=args.dropout,
            pad_idx=0,
        )
    else:
        model = Transformer(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            d_ff=args.d_ff,
            max_len=max_len,
            dropout=args.dropout,
            pad_idx=0,
        )

    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    history = {"train_loss": [], "val_acc": []}
    best_acc = 0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_n = 0

        for src_batch, tgt_batch in train_loader:
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
                logits = logits[:, src_len - 1 : -1, :]
            else:
                tgt_mask = make_decoder_mask(tgt_input, 0)
                logits = model(src_batch, tgt_input, src_mask, tgt_mask, src_mask)

            loss = criterion(logits.reshape(-1, vocab_size), tgt_output.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * tgt_output.numel()
            total_n += tgt_output.numel()

        train_loss = total_loss / total_n

        correct = 0
        sample_size = min(200, len(test_pairs))
        for i in range(sample_size):
            src_tensor = test_src_t[i : i + 1].to(device)
            src_mask = make_pad_mask(src_tensor, 0)
            pred = greedy_decode(
                model,
                src_tensor,
                src_mask,
                max_len,
                vocab,
                device,
                pad_idx=0,
                decoder_only=(exp["model_type"] == "decoder_only"),
            )
            if pred == test_pairs[i][1]:
                correct += 1

        acc = correct / sample_size
        history["train_loss"].append(train_loss)
        history["val_acc"].append(acc)

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[{exp['name']}] epoch {epoch + 1:02d}/{args.epochs} | "
            f"loss={train_loss:.4f} acc={acc:.4f}"
        )

    model.load_state_dict(best_state)
    model = model.to(device)
    correct = 0
    for i in range(len(test_pairs)):
        src_tensor = test_src_t[i : i + 1].to(device)
        src_mask = make_pad_mask(src_tensor, 0)
        pred = greedy_decode(
            model,
            src_tensor,
            src_mask,
            max_len,
            vocab,
            device,
            pad_idx=0,
            decoder_only=(exp["model_type"] == "decoder_only"),
        )
        if pred == test_pairs[i][1]:
            correct += 1

    full_acc = correct / len(test_pairs)
    print(
        f"[{exp['name']}] Done | best_acc={best_acc:.4f} "
        f"full_acc={full_acc:.4f} ({len(test_pairs)} samples)"
    )

    return {
        "config": {
            "name": exp["name"],
            "desc": exp["desc"],
            "model_type": exp["model_type"],
        },
        "best_acc": best_acc,
        "full_test_acc": full_acc,
        "test_size": len(test_pairs),
        "history": history,
        "best_state": best_state,
    }


# ============================================================
# 主程序
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="outputs_addition")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="0 = sequential, >0 = multiprocessing with N workers",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "auto" else "cpu"
    )
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    device_str = str(device)
    print(f"Using device: {device_str}")

    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

def make_train_gen(config):
    """Return a callable that generates training pairs for the given config."""
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
    """Return a callable that generates test pairs for the given config."""
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


experiments = [
    {
        "name": "baseline_3plus3",
        "model_type": "enc_dec",
        "train_config": "3plus3",
        "test_config": "3plus3",
        "desc": "3位数+3位数 同分布",
    },
    {
        "name": "extrapolate_4plus4",
        "model_type": "enc_dec",
        "train_config": "3plus3",
        "test_config": "4plus4",
        "desc": "3位训练 → 4位测试 (外推)",
    },
    {
        "name": "extrapolate_5plus5",
        "model_type": "enc_dec",
        "train_config": "3plus3",
        "test_config": "5plus5",
        "desc": "3位训练 → 5位测试 (外推)",
    },
    {
        "name": "mixed_training",
        "model_type": "enc_dec",
        "train_config": "mixed",
        "test_config": "mixed",
        "desc": "混合长度训练",
    },
    {
        "name": "decoder_only_3plus3",
        "model_type": "decoder_only",
        "train_config": "3plus3",
        "test_config": "3plus3",
        "desc": "Decoder-only 变体",
    },
]

    if args.workers > 1:
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)
        with mp.Pool(processes=args.workers) as pool:
            results = pool.starmap(
                run_single_experiment,
                [(exp, args, device_str) for exp in experiments],
            )
    else:
        results = [run_single_experiment(exp, args, device_str) for exp in experiments]

    all_results = results

    best_overall = max(all_results, key=lambda x: x["full_test_acc"])
    print(
        f"\nBest overall: {best_overall['config']['name']} | "
        f"acc={best_overall['full_test_acc']:.4f}"
    )

    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        serializable = []
        for r in all_results:
            entry = {k: v for k, v in r.items() if k != "best_state"}
            serializable.append(entry)
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    plt.figure(figsize=(10, 6))
    for r in all_results:
        plt.plot(
            r["history"]["val_acc"],
            label=f"{r['config']['name']} ({r['config']['desc']})",
        )
    plt.title("Addition Task — Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curves.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    for r in all_results:
        plt.plot(r["history"]["train_loss"], label=r["config"]["name"])
    plt.title("Addition Task — Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "train_loss.png", dpi=180)
    plt.close()

    names = [r["config"]["name"] for r in all_results]
    best_accs = [r["best_acc"] for r in all_results]
    full_accs = [r["full_test_acc"] for r in all_results]

    x_pos = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(
        x_pos - width / 2,
        best_accs,
        width,
        label="Best Acc (200 samples)",
        color="steelblue",
    )
    ax.bar(x_pos + width / 2, full_accs, width, label="Full Test Acc", color="coral")
    ax.set_xticks(x_pos)
    ax.