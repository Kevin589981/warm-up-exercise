import os
import re
import json
import random
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm


# 固定随机种子
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


# ============================================================
# 模型定义
# ============================================================


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_filters,
        kernel_sizes,
        num_classes,
        dropout=0.5,
        pad_idx=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes]
        )
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)  # (B, L, D)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (B, D, L)
        pooled = []
        for conv in self.convs:
            out = F.relu(conv(x))  # (B, F, L-k+1)
            pooled.append(F.max_pool1d(out, out.size(2)).squeeze(2))
        x = torch.cat(pooled, dim=1)
        x = self.dropout(x)
        return self.fc(x)


class TextLSTM(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.5, pad_idx=0
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)  # (B, L, D)
        out, _ = self.lstm(x)  # (B, L, 2H)
        # 拼接前向最后一个和后向第一个
        forward_last = out[:, -1, : out.size(2) // 2]
        backward_first = out[:, 0, out.size(2) // 2 :]
        x = torch.cat([forward_last, backward_first], dim=1)
        x = self.dropout(x)
        return self.fc(x)


class TextTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        num_classes,
        max_len=100,
        dropout=0.1,
        pad_idx=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, : x.size(1), :]
        x = self.dropout(x)
        mask = x.sum(dim=2) == 0  # padding mask
        out = self.transformer(x, src_key_padding_mask=mask)
        # mean pooling (忽略 padding)
        non_pad = (~mask).unsqueeze(-1).float()
        out = (out * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp(min=1)
        return self.fc(out)


# ============================================================
# Dataset
# ============================================================


class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# 主程序
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="../new_train.tsv")
    parser.add_argument("--test_path", type=str, default="../new_test.tsv")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=80)
    parser.add_argument("--vocab_size", type=int, default=20000)
    parser.add_argument("--embed_dim", type=int, default=128)
    args = parser.parse_args()

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 读取数据 ----
    print("Loading data...")
    train_df = pd.read_csv(
        base_dir / args.train_path, sep="\t", header=None, names=["text", "label"]
    )
    test_df = pd.read_csv(
        base_dir / args.test_path, sep="\t", header=None, names=["text", "label"]
    )
    train_df["text"] = train_df["text"].fillna("").astype(str)
    test_df["text"] = test_df["text"].fillna("").astype(str)
    train_df["label"] = train_df["label"].astype(int)
    test_df["label"] = test_df["label"].astype(int)

    texts = train_df["text"].tolist()
    labels = train_df["label"].to_numpy()
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].to_numpy()
    num_classes = int(labels.max()) + 1

    # ---- 2. 构建词表 ----
    print("Building vocabulary...")
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, _ in counter.most_common(args.vocab_size - 2):
        if counter[token] >= 1:
            vocab[token] = len(vocab)

    print(f"Vocab size: {len(vocab)}")

    # ---- 3. 句子转 ID 序列 + padding ----
    def encode(text_list):
        seqs = []
        for t in text_list:
            tokens = tokenize(t)[: args.max_len]
            ids = [vocab.get(tk, 1) for tk in tokens]  # 1 = <UNK>
            if len(ids) < args.max_len:
                ids += [0] * (args.max_len - len(ids))  # 0 = <PAD>
            seqs.append(ids)
        return torch.tensor(seqs, dtype=torch.long)

    print("Encoding sentences...")
    X_all = encode(texts)
    X_test = encode(test_texts)

    # ---- 4. 划分训练集/验证集 ----
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)), test_size=0.2, random_state=args.seed, stratify=labels
    )

    X_train, y_train = X_all[train_idx], labels[train_idx]
    X_val, y_val = X_all[val_idx], labels[val_idx]

    train_dataset = TextDataset(X_train, y_train)
    val_dataset = TextDataset(X_val, y_val)
    test_dataset = TextDataset(X_test, test_labels)

    # ---- 5. 定义实验 ----
    experiments = [
        # 1. CNN baseline
        {
            "name": "cnn_baseline",
            "model": "cnn",
            "lr": 0.001,
            "opt": "adam",
            "num_filters": 64,
            "kernel_sizes": [2, 3, 4],
        },
        # 2. 更多卷积核
        {
            "name": "cnn_more_filters",
            "model": "cnn",
            "lr": 0.001,
            "opt": "adam",
            "num_filters": 128,
            "kernel_sizes": [2, 3, 4],
        },
        # 3. 更大卷积核
        {
            "name": "cnn_wide_kernels",
            "model": "cnn",
            "lr": 0.001,
            "opt": "adam",
            "num_filters": 64,
            "kernel_sizes": [3, 4, 5],
        },
        # 4. 大学习率
        {
            "name": "cnn_lr0.01",
            "model": "cnn",
            "lr": 0.01,
            "opt": "adam",
            "num_filters": 64,
            "kernel_sizes": [2, 3, 4],
        },
        # 5. SGD 优化器
        {
            "name": "cnn_sgd",
            "model": "cnn",
            "lr": 0.01,
            "opt": "sgd",
            "num_filters": 64,
            "kernel_sizes": [2, 3, 4],
        },
        # 6. BiLSTM
        {
            "name": "lstm_baseline",
            "model": "lstm",
            "lr": 0.001,
            "opt": "adam",
            "hidden_dim": 128,
        },
        # 7. Transformer
        {
            "name": "transformer_baseline",
            "model": "transformer",
            "lr": 0.001,
            "opt": "adam",
            "num_heads": 4,
            "num_layers": 2,
        },
    ]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    all_results = []

    for exp in experiments:
        print(f"\n{'=' * 60}")
        print(f"Running: {exp['name']}")
        print(f"{'=' * 60}")

        # 创建模型
        if exp["model"] == "cnn":
            model = TextCNN(
                vocab_size=len(vocab),
                embed_dim=args.embed_dim,
                num_filters=exp["num_filters"],
                kernel_sizes=exp["kernel_sizes"],
                num_classes=num_classes,
                dropout=0.5,
                pad_idx=0,
            )
        elif exp["model"] == "lstm":
            model = TextLSTM(
                vocab_size=len(vocab),
                embed_dim=args.embed_dim,
                hidden_dim=exp["hidden_dim"],
                num_classes=num_classes,
                dropout=0.5,
                pad_idx=0,
            )
        elif exp["model"] == "transformer":
            model = TextTransformer(
                vocab_size=len(vocab),
                embed_dim=args.embed_dim,
                num_heads=exp["num_heads"],
                num_layers=exp["num_layers"],
                num_classes=num_classes,
                max_len=args.max_len,
                dropout=0.1,
                pad_idx=0,
            )

        model = model.to(device)

        # 优化器和损失函数
        if exp["opt"] == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=exp["lr"])
        elif exp["opt"] == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=exp["lr"], momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        # 训练
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_acc = 0
        best_state = None

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            total_correct = 0
            total_n = 0

            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(by)
                total_correct += (logits.argmax(dim=1) == by).sum().item()
                total_n += len(by)

            train_loss = total_loss / total_n
            train_acc = total_correct / total_n

            # 验证
            model.eval()
            val_loss = 0
            val_correct = 0
            val_n = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    logits = model(bx)
                    loss = criterion(logits, by)
                    val_loss += loss.item() * len(by)
                    val_correct += (logits.argmax(dim=1) == by).sum().item()
                    val_n += len(by)

            val_loss /= val_n
            val_acc = val_correct / val_n
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            print(
                f"  epoch {epoch + 1:02d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        # 测试集评估
        model.load_state_dict(best_state)
        model = model.to(device)
        model.eval()
        test_correct = 0
        test_n = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                test_correct += (logits.argmax(dim=1) == by).sum().item()
                test_n += len(by)
        test_acc = test_correct / test_n

        print(f"  => Best val_acc: {best_val_acc:.4f}, Test acc: {test_acc:.4f}")

        all_results.append(
            {
                "config": exp,
                "best_val_acc": best_val_acc,
                "test_acc": test_acc,
                "history": history,
            }
        )

    # ---- 6. 选最佳模型 ----
    best = max(all_results, key=lambda x: x["best_val_acc"])
    print(
        f"\nBest experiment: {best['config']['name']} | "
        f"val_acc={best['best_val_acc']:.4f} | test_acc={best['test_acc']:.4f}"
    )

    # ---- 7. 保存结果 ----
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # ---- 8. 画图 ----
    # 学习曲线
    plt.figure(figsize=(10, 6))
    for r in all_results:
        plt.plot(r["history"]["val_acc"], label=r["config"]["name"])
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "val_accuracy.png", dpi=180)
    plt.close()

    # loss 曲线
    plt.figure(figsize=(10, 6))
    for r in all_results:
        plt.plot(r["history"]["train_loss"], label=f"{r['config']['name']}-train")
        plt.plot(
            r["history"]["val_loss"], linestyle="--", label=f"{r['config']['name']}-val"
        )
    plt.title("Train/Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "train_val_loss.png", dpi=180)
    plt.close()

    # 柱状对比图
    names = [r["config"]["name"] for r in all_results]
    val_accs = [r["best_val_acc"] for r in all_results]
    test_accs = [r["test_acc"] for r in all_results]

    x_pos = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x_pos - width / 2, val_accs, width, label="Val Acc", color="steelblue")
    ax.bar(x_pos + width / 2, test_accs, width, label="Test Acc", color="coral")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Best Validation & Test Accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "bar_comparison.png", dpi=180)
    plt.close()

    # 保存最佳模型
    torch.save(best_state, out_dir / "best_model.pt")

    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
