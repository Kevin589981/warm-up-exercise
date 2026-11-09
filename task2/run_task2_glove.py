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

set_seed = lambda s=0: (
    random.seed(s),
    np.random.seed(s),
    torch.manual_seed(s),
    torch.cuda.manual_seed_all(s) if torch.cuda.is_available() else None,
)
set_seed(42)

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


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
        x = self.embedding(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            out = F.relu(conv(x))
            pooled.append(F.max_pool1d(out, out.size(2)).squeeze(2))
        x = torch.cat(pooled, dim=1)
        x = self.dropout(x)
        return self.fc(x)


class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_glove(glove_path, vocab, embed_dim):
    print(f"  Loading GloVe from {glove_path}...")
    glove = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            glove[parts[0]] = torch.tensor(
                [float(x) for x in parts[1:]], dtype=torch.float32
            )

    matrix = torch.randn(len(vocab), embed_dim) * 0.1
    matched = 0
    for token, idx in vocab.items():
        if token in glove:
            matrix[idx] = glove[token]
            matched += 1
    print(f"  Matched {matched}/{len(vocab)} words ({matched / len(vocab):.1%})")
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="../new_train.tsv")
    parser.add_argument("--test_path", type=str, default="../new_test.tsv")
    parser.add_argument("--glove_dir", type=str, default="glove.6B")
    parser.add_argument("--out_dir", type=str, default="outputs_glove")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=80)
    parser.add_argument("--vocab_size", type=int, default=20000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 读取数据 ----
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

    # ---- 构建词表 ----
    print("Building vocabulary...")
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, _ in counter.most_common(args.vocab_size - 2):
        if counter[token] >= 1:
            vocab[token] = len(vocab)
    print(f"Vocab size: {len(vocab)}")

    # ---- 编码 ----
    def encode(text_list):
        seqs = []
        for t in text_list:
            tokens = tokenize(t)[: args.max_len]
            ids = [vocab.get(tk, 1) for tk in tokens]
            if len(ids) < args.max_len:
                ids += [0] * (args.max_len - len(ids))
            seqs.append(ids)
        return torch.tensor(seqs, dtype=torch.long)

    print("Encoding sentences...")
    X_all = encode(texts)
    X_test = encode(test_texts)

    train_idx, val_idx = train_test_split(
        np.arange(len(labels)), test_size=0.2, random_state=42, stratify=labels
    )
    X_train, y_train = X_all[train_idx], labels[train_idx]
    X_val, y_val = X_all[val_idx], labels[val_idx]

    train_loader = DataLoader(
        TextDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TextDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TextDataset(X_test, test_labels), batch_size=args.batch_size, shuffle=False
    )

    # ---- 定义实验 ----
    experiments = [
        {"name": "random_init", "glove": None, "embed_dim": 100},
        {"name": "glove_50d", "glove": "glove.6B.50d.txt", "embed_dim": 50},
        {"name": "glove_100d", "glove": "glove.6B.100d.txt", "embed_dim": 100},
        {"name": "glove_200d", "glove": "glove.6B.200d.txt", "embed_dim": 200},
        {"name": "glove_300d", "glove": "glove.6B.300d.txt", "embed_dim": 300},
        {
            "name": "glove_100d_frozen",
            "glove": "glove.6B.100d.txt",
            "embed_dim": 100,
            "frozen": True,
        },
    ]

    all_results = []

    for exp in experiments:
        print(f"\n{'=' * 60}")
        print(f"Running: {exp['name']} (embed_dim={exp['embed_dim']})")
        print(f"{'=' * 60}")

        model = TextCNN(
            vocab_size=len(vocab),
            embed_dim=exp["embed_dim"],
            num_filters=64,
            kernel_sizes=[2, 3, 4],
            num_classes=num_classes,
            dropout=0.5,
            pad_idx=0,
        )

        # 加载 GloVe
        if exp["glove"]:
            glove_path = base_dir / args.glove_dir / exp["glove"]
            glove_matrix = load_glove(glove_path, vocab, exp["embed_dim"])
            model.embedding.weight.data.copy_(glove_matrix)

            # 是否冻结
            if exp.get("frozen", False):
                model.embedding.weight.requires_grad = False
                print("  Embedding frozen (requires_grad=False)")

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

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

    # ---- 保存结果 ----
    best = max(all_results, key=lambda x: x["best_val_acc"])
    print(
        f"\nBest: {best['config']['name']} | val_acc={best['best_val_acc']:.4f} | test_acc={best['test_acc']:.4f}"
    )

    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # ---- 画图 ----
    plt.figure(figsize=(10, 6))
    for r in all_results:
        plt.plot(r["history"]["val_acc"], label=r["config"]["name"])
    plt.title("Validation Accuracy (GloVe Comparison)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "val_accuracy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    for r in all_results:
        plt.plot(r["history"]["train_loss"], label=f"{r['config']['name']}-train")
        plt.plot(
            r["history"]["val_loss"], linestyle="--", label=f"{r['config']['name']}-val"
        )
    plt.title("Train/Validation Loss (GloVe Comparison)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "train_val_loss.png", dpi=180)
    plt.close()

    # 柱状图
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
    ax.set_title("GloVe Embedding Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "bar_comparison.png", dpi=180)
    plt.close()

    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
