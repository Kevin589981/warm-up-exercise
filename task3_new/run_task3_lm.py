import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

# 复用子任务1里已经实现的组件
from task3_subtask1 import (
    FeedForward,
    MultiHeadAttention,
    MultiHeadAttentionRoPE,
    RoPEPositionalEncoding,
    SinCosPositionalEncoding,
    WarmupScheduler,
    make_subsequent_mask,
)


# ============================================================
# Utils
# ============================================================


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def init_distributed(ddp: str) -> dict:
    """初始化 DDP（torchrun 启动时生效）。

    返回：
      {
        enabled: bool,
        rank: int,
        local_rank: int,
        world_size: int,
        is_main: bool,
      }
    """
    env_has_rank = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    enabled = (ddp in {"auto", "on"} and env_has_rank) and ddp != "off"

    if not enabled:
        return {"enabled": False, "rank": 0, "local_rank": 0, "world_size": 1, "is_main": True}

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    # 指定 device_ids 可避免 NCCL 提示“devices used by this process are unknown”
    if torch.cuda.is_available():
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()
    return {"enabled": True, "rank": rank, "local_rank": local_rank, "world_size": world_size, "is_main": rank == 0}


def ddp_barrier(ddp_info: dict) -> None:
    if not ddp_info.get("enabled", False):
        return
    local_rank = int(ddp_info.get("local_rank", 0))
    if torch.cuda.is_available():
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def decode_escaped_text(s: str) -> str:
    """把命令行里常见的转义序列（如 \n）转成真实字符。

    例："ROMEO:\\n" -> "ROMEO:\n"
    """
    try:
        return bytes(s, "utf-8").decode("unicode_escape")
    except Exception:
        return s


def maybe_time_to_stop(
    ddp_info: dict,
    start_time: float,
    max_minutes: float,
    step: int,
    check_interval: int,
) -> bool:
    """在 DDP 下同步判断是否达到时间上限。

    为避免每步都做 all_reduce，这里按 check_interval 步检查一次。
    """
    if max_minutes <= 0:
        return False
    if step % max(1, int(check_interval)) != 0:
        return False

    elapsed = time.monotonic() - start_time
    local_stop = 1 if elapsed >= max_minutes * 60.0 else 0

    if ddp_info.get("enabled", False):
        t = torch.tensor([local_stop], dtype=torch.int32, device="cuda" if torch.cuda.is_available() else "cpu")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(int(t.item()) == 1)
    return bool(local_stop == 1)


def download_if_needed(url: str, dst_path: Path, timeout: int = 30) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() and dst_path.stat().st_size > 0:
        return

    try:
        print(f"Downloading dataset -> {dst_path} ...", flush=True)
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read()
        dst_path.write_bytes(data)
        print("Download done.", flush=True)
    except Exception as e:
        raise RuntimeError(
            f"下载失败：{url}\n"
            f"目标路径：{dst_path}\n"
            f"错误：{e}\n\n"
            "你可以手动下载后放到上述路径，再重新运行。"
        )


# ============================================================
# Tokenizer (char-level)
# ============================================================


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> list[int]:
        missing = [c for c in set(s) if c not in self.stoi]
        if missing:
            missing = "".join(sorted(missing))
            raise ValueError(f"start_text 含未知字符：{missing!r}")
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


# ============================================================
# Dataset
# ============================================================


class TextDataset(Dataset):
    """自回归语言模型数据：给定长度 seq_len 的片段，预测下一个字符。"""

    def __init__(self, data: torch.Tensor, seq_len: int):
        assert data.dtype == torch.long
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        # x: [i, i+seq_len)
        # y: [i+1, i+seq_len+1)
        return max(0, self.data.numel() - self.seq_len - 1)

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


# ============================================================
# Model: decoder-only Transformer (GPT-like)
# ============================================================


class DecoderOnlyLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1, rope_enc=None):
        super().__init__()
        self.self_attn = CausalSelfAttentionSDPA(d_model, nhead, dropout, rope_enc=rope_enc)

        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class CausalSelfAttentionSDPA(nn.Module):
    """更快的自注意力：使用 PyTorch scaled_dot_product_attention + causal。

    形状约定：
      x: (B, T, C)
      返回: (B, T, C)
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, rope_enc=None):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model 必须能被 nhead 整除")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_p = float(dropout)
        self.rope_enc = rope_enc

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        if self.rope_enc is not None:
            pos = torch.arange(T, device=x.device).float()
            q = self.rope_enc.apply_rope(q, pos)
            k = self.rope_enc.apply_rope(k, pos)

        attn_dropout = self.dropout_p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=attn_dropout, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        d_ff: int,
        max_len: int,
        dropout: float = 0.1,
        pos_enc_type: str = "sincos",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout = dropout
        self.pos_enc_type = pos_enc_type

        self.embed = nn.Embedding(vocab_size, d_model)

        if pos_enc_type == "rope":
            self.rope_enc = RoPEPositionalEncoding(d_model, max_len=max_len, dropout=dropout)
            self.pos_encoding = self.rope_enc  # RoPE: 仅做 dropout
        elif pos_enc_type == "sincos":
            self.rope_enc = None
            self.pos_encoding = SinCosPositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        else:
            raise ValueError("pos_enc_type 仅支持 sincos/rope")

        self.layers = nn.ModuleList(
            [
                DecoderOnlyLayer(d_model, nhead, d_ff, dropout, rope_enc=self.rope_enc)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T)
        x = self.embed(idx)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer: CharTokenizer,
    start_text: str,
    max_new_tokens: int,
    seq_len: int,
    temperature: float,
    device: torch.device,
) -> str:
    model.eval()
    if temperature <= 0:
        raise ValueError("temperature 必须 > 0")

    ids = tokenizer.encode(start_text)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -seq_len:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    return tokenizer.decode(idx[0].tolist())


# ============================================================
# Train / Eval
# ============================================================


@torch.no_grad()
def eval_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += float(loss.item())
        total_tokens += int(y.numel())

    return total_loss / max(1, total_tokens)


@torch.no_grad()
def estimate_loss_by_sampling(
    model: nn.Module,
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    eval_iters: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    generator: torch.Generator | None,
) -> float:
    model.eval()
    losses: list[float] = []
    ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if (use_amp and device.type == "cuda")
        else nullcontext()
    )

    for _ in range(int(eval_iters)):
        x, y = get_batch(data, batch_size, seq_len, device, generator)
        with ctx:
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="mean")
        losses.append(float(loss.item()))
    return float(sum(losses) / max(1, len(losses)))


@torch.no_grad()
def estimate_loss_by_sampling_distributed(
    ddp_info: dict,
    model: nn.Module,
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    eval_iters: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    generator: torch.Generator | None,
) -> float:
    """DDP 下分摊 eval_iters 到各个 rank，最后 all_reduce 汇总成全局平均。"""
    model.eval()

    total_iters = int(eval_iters)
    if total_iters <= 0:
        return float("nan")

    if ddp_info.get("enabled", False):
        world_size = int(ddp_info.get("world_size", 1))
        rank = int(ddp_info.get("rank", 0))
        base = total_iters // world_size
        extra = total_iters % world_size
        local_iters = base + (1 if rank < extra else 0)
    else:
        local_iters = total_iters

    ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if (use_amp and device.type == "cuda")
        else nullcontext()
    )

    local_sum = 0.0
    for _ in range(int(local_iters)):
        x, y = get_batch(data, batch_size, seq_len, device, generator)
        with ctx:
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="mean")
        local_sum += float(loss.item())

    if ddp_info.get("enabled", False):
        t = torch.tensor(
            [local_sum, float(local_iters)],
            dtype=torch.float32,
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_sum = float(t[0].item())
        total_count = float(t[1].item())
        return total_sum / max(1.0, total_count)

    return local_sum / max(1, int(local_iters))


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从一维 token 序列中随机采样 batch（向量化实现，避免 DataLoader 开销）。"""
    if data.numel() < seq_len + 2:
        raise ValueError("语料太短：请降低 seq_len 或换更长的 input.txt")

    max_start = data.numel() - seq_len - 1
    ix = torch.randint(0, max_start, (batch_size,), device=device, generator=generator)
    offsets = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, T)
    x = data[ix.unsqueeze(1) + offsets]
    y = data[ix.unsqueeze(1) + offsets + 1]
    return x, y


@dataclass
class RunConfig:
    data_url: str
    data_path: str
    out_dir: str
    seed: int
    device: str
    epochs: int
    batch_size: int
    seq_len: int
    d_model: int
    nhead: int
    num_layers: int
    d_ff: int
    dropout: float
    lr: float
    use_warmup: bool
    warmup_steps: int
    pos_enc_type: str
    num_workers: int
    start_text: str
    max_new_tokens: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Task-3 子任务2：字符级语言模型 (decoder-only Transformer)")
    parser.add_argument(
        "--data_url",
        type=str,
        default="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    )
    parser.add_argument("--data_path", type=str, default="data/input.txt")
    parser.add_argument("--out_dir", type=str, default="outputs_lm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help=(
            "每个 epoch 的 optimizer step 数。"
            "设为 0 表示自动按‘训练集跑一遍’估算："
            "steps_per_epoch = ceil(train_tokens / (batch_size*seq_len*grad_accum_steps*world_size))。"
        ),
    )
    parser.add_argument("--eval_iters", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=64)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--use_warmup", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--pos_enc_type", type=str, default="sincos", choices=["sincos", "rope"])

    parser.add_argument("--ddp", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--amp", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="只做评估/生成，不训练（从 --ckpt_path 或 out_dir/best_model.pt 加载权重）。",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="eval_only 模式下要加载的权重路径；为空则使用 out_dir/best_model.pt",
    )
    parser.add_argument(
        "--skip_generate",
        action="store_true",
        help="评估时跳过文本生成（只计算 val/test loss/ppl 并写 metrics）。",
    )
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--compile", action="store_true")

    parser.add_argument(
        "--max_minutes",
        type=float,
        default=0.0,
        help="训练最长运行时间（分钟）。>0 时到点自动停止（DDP 下全 rank 同步停止）。",
    )
    parser.add_argument(
        "--time_check_interval",
        type=int,
        default=20,
        help="每隔多少个 step 检查一次 max_minutes（减少通信开销）。",
    )
    parser.add_argument(
        "--eval_every_epochs",
        type=int,
        default=1,
        help="每隔多少个 epoch 做一次验证；设为 0 表示只在训练结束时验证一次。",
    )

    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集占比")
    parser.add_argument("--test_ratio", type=float, default=0.0, help="测试集占比（0 表示不划分 test）")

    # 兼容旧参数（现在不再依赖 DataLoader 遍历训练集）
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--start_text", type=str, default="ROMEO:\n")
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / args.data_path
    out_dir = base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = RunConfig(
        data_url=args.data_url,
        data_path=str(data_path),
        out_dir=str(out_dir),
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        lr=args.lr,
        use_warmup=bool(args.use_warmup),
        warmup_steps=args.warmup_steps,
        pos_enc_type=args.pos_enc_type,
        num_workers=args.num_workers,
        start_text=decode_escaped_text(args.start_text),
        max_new_tokens=args.max_new_tokens,
    )

    # eval_only 模式：强制关闭 DDP（避免 torchrun 环境变量导致意外 init_process_group）
    ddp_info = init_distributed("off" if bool(args.eval_only) else args.ddp)
    set_seed(cfg.seed + int(ddp_info["rank"]))

    if ddp_info["enabled"]:
        device = torch.device(f"cuda:{ddp_info['local_rank']}" if torch.cuda.is_available() else "cpu")
    else:
        device = resolve_device(cfg.device)

    is_main = bool(ddp_info["is_main"])
    if is_main:
        print(
            f"Using device: {device} | ddp={ddp_info['enabled']} | world_size={ddp_info['world_size']}",
            flush=True,
        )

    # 1) 数据准备（DDP 下只让 rank0 下载，其它 rank 等待）
    if (not ddp_info["enabled"]) or ddp_info["is_main"]:
        download_if_needed(cfg.data_url, data_path)
    ddp_barrier(ddp_info)

    text = data_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    data_ids = torch.tensor([tokenizer.stoi[c] for c in text], dtype=torch.long)

    n = int(data_ids.numel())
    val_ratio = float(args.val_ratio)
    test_ratio = float(args.test_ratio)
    if not (0.0 <= val_ratio < 1.0) or not (0.0 <= test_ratio < 1.0) or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("val_ratio/test_ratio 设置不合法：需满足 0<=ratio 且 val_ratio+test_ratio<1")

    train_end = int((1.0 - val_ratio - test_ratio) * n)
    val_end = int((1.0 - test_ratio) * n)
    train_ids = data_ids[:train_end]
    val_ids = data_ids[train_end:val_end]
    test_ids = data_ids[val_end:] if test_ratio > 0 else None

    # 为了吞吐：把长序列直接放到训练设备上，训练时随机采样 batch（避免 DataLoader 遍历每个字符位置）
    train_data = train_ids.to(device, non_blocking=True)
    val_data = val_ids.to(device, non_blocking=True)
    test_data = test_ids.to(device, non_blocking=True) if test_ids is not None else None

    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed + 1234 + int(ddp_info["rank"]))

    # 2) 模型
    model = LanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        max_len=max(cfg.seq_len, 512),
        dropout=cfg.dropout,
        pos_enc_type=cfg.pos_enc_type,
    ).to(device)

    if args.compile:
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
            if is_main:
                print("torch.compile enabled", flush=True)
        except Exception as e:
            if is_main:
                print(f"[warn] torch.compile 失败，继续运行：{e}", flush=True)

    if ddp_info["enabled"] and device.type == "cuda":
        model = DDP(model, device_ids=[ddp_info["local_rank"]], output_device=ddp_info["local_rank"], broadcast_buffers=False)

    # ========================================================
    # eval_only: 从 checkpoint 直接评估/生成，不训练
    # ========================================================
    if bool(args.eval_only):
        if not is_main:
            # eval_only 已强制关闭 DDP，这里只是防御性处理
            return

        ckpt_path = Path(args.ckpt_path) if str(args.ckpt_path).strip() else (out_dir / "best_model.pt")
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"找不到 checkpoint：{ckpt_path}.\n"
                "请指定 --ckpt_path，或把 best_model.pt 放到 --out_dir 指向的目录下。"
            )

        raw_model = model.module if hasattr(model, "module") else model
        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location="cpu")
        raw_model.load_state_dict(state)
        raw_model.to(device)

        use_amp = (args.amp == "on") or (args.amp == "auto" and device.type == "cuda")
        amp_dtype = torch.float16

        eval_iters = int(args.eval_iters)
        if eval_iters <= 0:
            raise ValueError("eval_only 模式下 --eval_iters 必须 > 0")

        val_loss = estimate_loss_by_sampling(
            model=raw_model,
            data=val_data,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            eval_iters=eval_iters,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            generator=gen,
        )
        val_ppl = float(math.exp(min(20.0, float(val_loss))))
        print(f"Eval-only val_loss={float(val_loss):.4f} | val_ppl={float(val_ppl):.2f}", flush=True)

        test_loss = None
        test_ppl = None
        if test_data is not None:
            test_loss = estimate_loss_by_sampling(
                model=raw_model,
                data=test_data,
                batch_size=cfg.batch_size,
                seq_len=cfg.seq_len,
                eval_iters=eval_iters,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                generator=gen,
            )
            test_ppl = float(math.exp(min(20.0, float(test_loss))))
            print(f"Eval-only test_loss={float(test_loss):.4f} | test_ppl={float(test_ppl):.2f}", flush=True)

        metrics = {
            "mode": "eval_only",
            "ckpt_path": str(ckpt_path),
            "config": {**asdict(cfg), "steps_per_epoch": int(args.steps_per_epoch), "eval_iters": eval_iters},
            "vocab_size": tokenizer.vocab_size,
            "val_loss": float(val_loss),
            "val_ppl": float(val_ppl),
            "test_loss": None if test_loss is None else float(test_loss),
            "test_ppl": None if test_ppl is None else float(test_ppl),
            "ddp": ddp_info,
            "amp": {"enabled": bool(use_amp and device.type == "cuda"), "dtype": str(amp_dtype)},
        }
        (out_dir / "metrics_eval_only.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if not bool(args.skip_generate):
            samples_path = out_dir / "generated_samples_eval_only.txt"
            temps = [0.7, args.temperature, 1.3]
            seen = set()
            temps = [t for t in temps if not (t in seen or seen.add(t))]

            parts = []
            parts.append(f"# start_text: {cfg.start_text!r}\n")
            parts.append(f"# ckpt_path: {str(ckpt_path)}\n")
            parts.append(f"# val_ppl: {float(val_ppl):.2f}\n\n")

            for t in temps:
                txt = generate(
                    model=raw_model,
                    tokenizer=tokenizer,
                    start_text=cfg.start_text,
                    max_new_tokens=cfg.max_new_tokens,
                    seq_len=cfg.seq_len,
                    temperature=float(t),
                    device=device,
                )
                parts.append(f"===== temperature={t} =====\n")
                parts.append(txt)
                parts.append("\n\n")

            samples_path.write_text("".join(parts), encoding="utf-8")
            print(f"\nSaved eval-only outputs to: {out_dir}", flush=True)
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    scheduler = (
        WarmupScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)
        if cfg.use_warmup
        else None
    )

    use_amp = (args.amp == "on") or (args.amp == "auto" and device.type == "cuda")
    amp_dtype = torch.float16
    scaler_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(scaler_device, enabled=bool(use_amp and device.type == "cuda"))

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_ppl": [],
        "epoch_time_sec": [],
    }

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1

    # 3) 训练（固定 steps_per_epoch，随机采样 batch）
    run_start = time.monotonic()
    stopped_early = False
    last_val_loss = float("nan")
    last_val_ppl = float("nan")

    last_epoch = -1
    for epoch in range(cfg.epochs):
        last_epoch = epoch
        t0 = time.time()
        model.train()

        total_loss = 0.0
        total_steps = 0
        tokens_trained = 0

        ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if (use_amp and device.type == "cuda")
            else nullcontext()
        )

        grad_accum_steps = max(1, int(args.grad_accum_steps))

        # steps_per_epoch=0：按“训练集 tokens 跑一遍”为 1 epoch 来自动估算步数。
        # 注意：训练是随机采样（有放回），这里的 epoch 表示“处理的 token 数≈训练集 token 数”。
        if int(args.steps_per_epoch) == 0:
            train_tokens = int(train_ids.numel())
            world_size = int(ddp_info.get("world_size", 1))
            tokens_per_step_global = int(cfg.batch_size) * int(cfg.seq_len) * int(grad_accum_steps) * int(world_size)
            steps_per_epoch = int(math.ceil(train_tokens / max(1, tokens_per_step_global)))
        else:
            steps_per_epoch = int(args.steps_per_epoch)

        for step in range(steps_per_epoch):
            if maybe_time_to_stop(
                ddp_info=ddp_info,
                start_time=run_start,
                max_minutes=float(args.max_minutes),
                step=step,
                check_interval=int(args.time_check_interval),
            ):
                stopped_early = True
                break

            optimizer.zero_grad(set_to_none=True)

            for micro in range(grad_accum_steps):
                x, y = get_batch(train_data, cfg.batch_size, cfg.seq_len, device, gen)
                tokens_trained += int(y.numel())

                sync_ctx = nullcontext()
                if ddp_info["enabled"] and hasattr(model, "no_sync") and micro != grad_accum_steps - 1:
                    sync_ctx = model.no_sync()  # type: ignore[assignment]

                with sync_ctx:
                    with ctx:
                        logits = model(x)
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            y.reshape(-1),
                            reduction="mean",
                        )
                        loss = loss / grad_accum_steps

                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                total_loss += float(loss.item())
                total_steps += 1

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                if float(args.clip_grad_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.clip_grad_norm))
                scaler.step(optimizer)
                scaler.update()
            else:
                if float(args.clip_grad_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.clip_grad_norm))
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if is_main and (step + 1) % max(1, int(args.log_interval)) == 0:
                avg_loss = total_loss / max(1, total_steps)
                elapsed = time.time() - t0
                tok_s = tokens_trained / max(1e-6, elapsed)
                print(
                    f"Epoch {epoch+1:>3}/{cfg.epochs} | step {step+1:>4}/{steps_per_epoch} | "
                    f"loss={avg_loss:.4f} | tokens/s={tok_s:,.0f}",
                    flush=True,
                )

        if stopped_early and is_main:
            print(f"[time] Reached max_minutes={args.max_minutes}, stopping...", flush=True)

        # 到点停止时，可能正好进入一个新的 epoch 并在 step=0 立即触发停止。
        # 这种情况下本 epoch 没有任何训练 step，不应把 train_loss=0/nan 写进 history（会污染曲线）。
        if stopped_early and total_steps == 0:
            if is_main:
                print(
                    f"Epoch {epoch+1:>3}/{cfg.epochs} | 0 steps executed (time limit hit at epoch start). "
                    "Skip recording this epoch.",
                    flush=True,
                )
            last_epoch = epoch - 1
            break

        # 只在主进程做评估/保存（避免 8 卡重复评估拖慢）
        # 注意：如果因为到点停止导致本 epoch 一个 step 都没跑到，
        # total_steps==0 时不应该把 train_loss 记为 0（会污染曲线）。
        if total_steps == 0:
            train_loss = float("nan")
        else:
            train_loss = total_loss / total_steps

        do_eval = int(args.eval_every_epochs) != 0 and ((epoch + 1) % max(1, int(args.eval_every_epochs)) == 0)
        # 如果提前停止，为了留一个结果，这里强制做一次 eval（除非 eval_every_epochs=0 明确只想结束 eval）
        if stopped_early and int(args.eval_every_epochs) != 0:
            do_eval = True

        if do_eval:
            eval_model = model.module if hasattr(model, "module") else model
            val_loss = estimate_loss_by_sampling_distributed(
                ddp_info=ddp_info,
                model=eval_model,
                data=val_data,
                batch_size=cfg.batch_size,
                seq_len=cfg.seq_len,
                eval_iters=int(args.eval_iters),
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                generator=gen,
            )
            if is_main:
                last_val_loss = val_loss
                last_val_ppl = float(math.exp(min(20.0, val_loss)))
        else:
            val_loss = float("nan")

        val_ppl = float(math.exp(min(20.0, val_loss))) if is_main and do_eval else float("nan")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        history["epoch_time_sec"].append(time.time() - t0)

        if is_main:
            if do_eval and (val_loss < best_val_loss):
                best_val_loss = val_loss
                best_epoch = epoch
                raw_model = model.module if hasattr(model, "module") else model
                best_state = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}

            if do_eval:
                msg = (
                    f"Epoch {epoch+1:>3}/{cfg.epochs} | "
                    f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f} | "
                    f"epoch_time={history['epoch_time_sec'][-1]:.1f}s"
                )
            else:
                msg = (
                    f"Epoch {epoch+1:>3}/{cfg.epochs} | "
                    f"train_loss={train_loss:.4f} | (skip val) | "
                    f"epoch_time={history['epoch_time_sec'][-1]:.1f}s"
                )
            print(msg, flush=True)

        if stopped_early:
            break

    # ========================================================
    # DDP 收尾：训练结束后立刻销毁进程组，避免 rank0 后处理太久导致其它 rank 在 collective 里超时。
    # 非主 rank 直接退出；主 rank 继续做评估/保存/生成（单进程、无 NCCL）。
    # ========================================================
    if ddp_info.get("enabled", False):
        ddp_barrier(ddp_info)
        dist.destroy_process_group()
        if not is_main:
            return
        ddp_info = {"enabled": False, "rank": 0, "local_rank": 0, "world_size": 1, "is_main": True}

    # 若 eval_every_epochs=0：训练结束后做一次验证（仅主进程；best_state 取最终权重）
    if int(args.eval_every_epochs) == 0 and int(args.eval_iters) > 0:
        raw_model = model.module if hasattr(model, "module") else model
        # eval_every_epochs=0 时没有中途 best，这里默认保存最终权重
        best_epoch = max(0, last_epoch)
        best_state = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}
        val_loss = estimate_loss_by_sampling(
            model=raw_model,
            data=val_data,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            eval_iters=int(args.eval_iters),
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            generator=gen,
        )
        last_val_loss = val_loss
        last_val_ppl = float(math.exp(min(20.0, val_loss)))
        # 把最终一次验证结果写回 history，避免 val 曲线/图像为空
        if len(history["val_loss"]) > 0:
            history["val_loss"][-1] = float(val_loss)
        else:
            history["val_loss"].append(float(val_loss))
        if len(history["val_ppl"]) > 0:
            history["val_ppl"][-1] = float(last_val_ppl)
        else:
            history["val_ppl"].append(float(last_val_ppl))

        best_val_loss = float(val_loss) if best_val_loss == float("inf") else best_val_loss
        print(f"Final val_loss={val_loss:.4f} | val_ppl={last_val_ppl:.2f}", flush=True)

    # 4) 保存（仅主进程）
    if is_main:
        if best_state is not None:
            torch.save(best_state, out_dir / "best_model.pt")

        test_loss = None
        test_ppl = None
        if test_data is not None and best_state is not None and int(args.eval_iters) > 0:
            raw_model = model.module if hasattr(model, "module") else model
            raw_model.load_state_dict(best_state)
            test_loss = estimate_loss_by_sampling(
                model=raw_model,
                data=test_data,
                batch_size=cfg.batch_size,
                seq_len=cfg.seq_len,
                eval_iters=int(args.eval_iters),
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                generator=gen,
            )
            test_ppl = float(math.exp(min(20.0, float(test_loss))))
            print(f"Test loss={float(test_loss):.4f} | test_ppl={float(test_ppl):.2f}", flush=True)

        metrics = {
            "config": {**asdict(cfg), "steps_per_epoch": int(args.steps_per_epoch), "eval_iters": int(args.eval_iters)},
            "vocab_size": tokenizer.vocab_size,
            "best_epoch": best_epoch + 1,
            "best_val_loss": best_val_loss,
            "best_val_ppl": float(math.exp(min(20.0, best_val_loss))),
            "final_val_loss": None if math.isnan(last_val_loss) else float(last_val_loss),
            "final_val_ppl": None if math.isnan(last_val_ppl) else float(last_val_ppl),
            "test_loss": None if test_loss is None else float(test_loss),
            "test_ppl": None if test_ppl is None else float(test_ppl),
            "history": history,
            "ddp": ddp_info,
            "amp": {"enabled": bool(use_amp and device.type == "cuda"), "dtype": str(amp_dtype)},
        }
        (out_dir / "metrics_summary.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 5) 画图（仅主进程）
    if is_main:
        try:
            import matplotlib.pyplot as plt

            def _valid_xy(values: list[float]):
                xs = []
                ys = []
                for i, v in enumerate(values):
                    if v is None:
                        continue
                    if isinstance(v, float) and math.isnan(v):
                        continue
                    xs.append(i)
                    ys.append(v)
                return xs, ys

            fig, ax = plt.subplots(figsize=(9, 4))
            lines = []
            tx, ty = _valid_xy(history["train_loss"])
            if tx:
                (ln,) = ax.plot(tx, ty, label="train_loss", marker="o")
                lines.append(ln)
            vx, vy = _valid_xy(history["val_loss"])
            if vx:
                (ln,) = ax.plot(vx, vy, label="val_loss", marker="o")
                lines.append(ln)
            ax.set_title("LM Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(alpha=0.3)
            if lines:
                ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "train_loss.png", dpi=180)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(9, 4))
            px, py = _valid_xy(history["val_ppl"])
            if px:
                ax.plot(px, py, label="val_perplexity", marker="o")
            ax.set_title("Validation Perplexity")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Perplexity")
            ax.grid(alpha=0.3)
            if px:
                ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "val_perplexity.png", dpi=180)
            plt.close(fig)
        except Exception as e:
            print(f"[warn] 绘图失败：{e}", flush=True)

    # 6) 文本生成（仅主进程）
    if is_main:
        raw_model = model.module if hasattr(model, "module") else model
        if best_state is not None:
            raw_model.load_state_dict(best_state)

        samples_path = out_dir / "generated_samples.txt"
        temps = [0.7, args.temperature, 1.3]
        # 去重但保持顺序
        seen = set()
        temps = [t for t in temps if not (t in seen or seen.add(t))]

        parts = []
        parts.append(f"# start_text: {cfg.start_text!r}\n")
        parts.append(f"# best_epoch: {best_epoch+1}\n")
        parts.append(f"# best_val_ppl: {float(math.exp(min(20.0, best_val_loss))):.2f}\n\n")

        for t in temps:
            txt = generate(
                model=raw_model,
                tokenizer=tokenizer,
                start_text=cfg.start_text,
                max_new_tokens=cfg.max_new_tokens,
                seq_len=cfg.seq_len,
                temperature=float(t),
                device=device,
            )
            parts.append(f"===== temperature={t} =====\n")
            parts.append(txt)
            parts.append("\n\n")

        samples_path.write_text("".join(parts), encoding="utf-8")

        print(f"\nSaved outputs to: {out_dir}", flush=True)

    if ddp_info["enabled"]:
        ddp_barrier(ddp_info)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
