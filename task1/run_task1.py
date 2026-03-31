import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy.sparse import csr_matrix

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


@dataclass
class ExperimentConfig:
    name: str
    ngram_min: int
    ngram_max: int
    loss_type: str
    lr: float
    epochs: int
    batch_size: int
    num_features: int
    weight_decay: float
    binary: bool


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["text", "label"])
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df[["text", "label"]]


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_ngram_tokens(tokens: List[str], ngram_range: Tuple[int, int]) -> List[str]:
    min_n, max_n = ngram_range
    out: List[str] = []
    for n in range(min_n, max_n + 1):
        if len(tokens) < n:
            continue
        if n == 1:
            out.extend(tokens)
        else:
            for i in range(len(tokens) - n + 1):
                out.append("_".join(tokens[i : i + n]))
    return out


def stable_hash(token: str, num_features: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % num_features


def hash_features_for_text(
    text: str,
    num_features: int,
    ngram_range: Tuple[int, int],
    binary: bool,
) -> Dict[int, float]:
    tokens = tokenize(text)
    grams = build_ngram_tokens(tokens, ngram_range)
    feat: Dict[int, float] = {}
    if not grams:
        return feat

    for g in grams:
        idx = stable_hash(g, num_features)
        if binary:
            feat[idx] = 1.0
        else:
            feat[idx] = feat.get(idx, 0.0) + 1.0
    return feat


def vectorize_corpus_sparse(
    texts: List[str],
    num_features: int,
    ngram_range: Tuple[int, int],
    binary: bool,
    desc: str,
) -> List[Dict[int, float]]:
    sparse_list: List[Dict[int, float]] = []
    for text in tqdm(texts, desc=desc):
        sparse_list.append(
            hash_features_for_text(
                text=text,
                num_features=num_features,
                ngram_range=ngram_range,
                binary=binary,
            )
        )
    return sparse_list


def vectorize_corpus_to_csr(
    texts: List[str],
    num_features: int,
    ngram_range: Tuple[int, int],
    binary: bool,
    desc: str,
) -> csr_matrix:
    data, indices, indptr = [], [], [0]

    for text in tqdm(texts, desc=desc):
        # 复用你原来的分词和哈希逻辑
        feat_dict = hash_features_for_text(text, num_features, ngram_range, binary)

        # 将字典转为稀疏格式需要的三个数组
        for col in sorted(feat_dict.keys()):
            indices.append(col)
            data.append(feat_dict[col])
        indptr.append(len(indices))

    # 构建 CSR 矩阵：这在内存中是连续存放的
    return csr_matrix(
        (data, indices, indptr), shape=(len(texts), num_features), dtype=np.float32
    )


def build_dense_batch(
    indices: np.ndarray,
    sparse_matrix: csr_matrix,
    num_features: int,
    device: torch.device,
) -> torch.Tensor:
    # 1. 利用 scipy 高效切片并转为密集 NumPy 数组
    # 这是 C 级别的操作，比 Python 循环快几百倍
    batch_data = sparse_matrix[indices].toarray()

    # 2. 一次性转为 Tensor 并移动到 GPU
    return torch.from_numpy(batch_data).to(device)


def linear_forward(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return x @ w + b


def cross_entropy_manual(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_z = torch.logsumexp(logits, dim=1)
    chosen = logits[torch.arange(logits.shape[0], device=logits.device), targets]
    return (log_z - chosen).mean()


def mse_on_softmax_manual(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    shifted = logits - logits.max(dim=1, keepdim=True).values
    probs = torch.exp(shifted)
    probs = probs / probs.sum(dim=1, keepdim=True)
    one_hot = torch.zeros(
        (logits.shape[0], num_classes), dtype=logits.dtype, device=logits.device
    )
    one_hot[torch.arange(logits.shape[0], device=logits.device), targets] = 1.0
    return ((probs - one_hot) ** 2).mean()


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def evaluate_model(
    indices: np.ndarray,
    labels: np.ndarray,
    sparse_features: csr_matrix,
    num_features: int,
    batch_size: int,
    device: torch.device,
    w: torch.Tensor,
    b: torch.Tensor,
    loss_type: str,
    num_classes: int,
) -> Tuple[float, float]:
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            x = build_dense_batch(batch_idx, sparse_features, num_features, device)
            y = torch.tensor(labels[batch_idx], dtype=torch.long, device=device)

            logits = linear_forward(x, w, b)
            if loss_type == "ce":
                loss = cross_entropy_manual(logits, y)
            elif loss_type == "mse":
                loss = mse_on_softmax_manual(logits, y, num_classes)
            else:
                raise ValueError(f"Unsupported loss_type: {loss_type}")

            acc = compute_accuracy(logits, y)
            bs = len(batch_idx)
            total_loss += loss.item() * bs
            total_acc += acc * bs
            total_n += bs

    return total_loss / total_n, total_acc / total_n


def predict_labels(
    indices: np.ndarray,
    sparse_features: csr_matrix,
    num_features: int,
    batch_size: int,
    device: torch.device,
    w: torch.Tensor,
    b: torch.Tensor,
) -> np.ndarray:
    all_preds: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            x = build_dense_batch(batch_idx, sparse_features, num_features, device)
            logits = linear_forward(x, w, b)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
    return np.concatenate(all_preds, axis=0)


def save_predictions(path: Path, preds: np.ndarray) -> None:
    out = pd.DataFrame({"id": np.arange(len(preds)), "label": preds.astype(int)})
    out.to_csv(path, index=False)


def plot_histories(results: List[dict], out_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    for r in results:
        plt.plot(r["history"]["val_acc"], label=r["config"]["name"])
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "val_accuracy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    for r in results:
        plt.plot(r["history"]["train_loss"], label=f"{r['config']['name']}-train")
        plt.plot(
            r["history"]["val_loss"], linestyle="--", label=f"{r['config']['name']}-val"
        )
    plt.title("Train/Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "train_val_loss.png", dpi=180)
    plt.close()


def train_one_experiment(
    cfg: ExperimentConfig,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    labels: np.ndarray,
    sparse_features: List[Dict[int, float]],
    num_classes: int,
    device: torch.device,
) -> dict:
    w = (
        0.01 * torch.randn(cfg.num_features, num_classes, device=device)
    ).requires_grad_(True)
    b = torch.zeros(num_classes, dtype=torch.float32, device=device, requires_grad=True)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None

    for epoch in range(cfg.epochs):
        shuffled = np.random.permutation(train_indices)
        epoch_loss = 0.0
        total_n = 0

        for start in range(0, len(shuffled), cfg.batch_size):
            batch_idx = shuffled[start : start + cfg.batch_size]
            x = build_dense_batch(batch_idx, sparse_features, cfg.num_features, device)
            y = torch.tensor(labels[batch_idx], dtype=torch.long, device=device)

            logits = linear_forward(x, w, b)
            if cfg.loss_type == "ce":
                loss = cross_entropy_manual(logits, y)
            elif cfg.loss_type == "mse":
                loss = mse_on_softmax_manual(logits, y, num_classes)
            else:
                raise ValueError(f"Unsupported loss_type: {cfg.loss_type}")

            loss.backward()
            with torch.no_grad():
                w -= cfg.lr * (w.grad + cfg.weight_decay * w)
                b -= cfg.lr * b.grad
                w.grad.zero_()
                b.grad.zero_()

            bs = len(batch_idx)
            epoch_loss += loss.item() * bs
            total_n += bs

        train_loss = epoch_loss / total_n
        val_loss, val_acc = evaluate_model(
            indices=val_indices,
            labels=labels,
            sparse_features=sparse_features,
            num_features=cfg.num_features,
            batch_size=cfg.batch_size,
            device=device,
            w=w,
            b=b,
            loss_type=cfg.loss_type,
            num_classes=num_classes,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[{cfg.name}] epoch {epoch + 1:02d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "w": w.detach().clone(),
                "b": b.detach().clone(),
                "epoch": epoch + 1,
            }

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model state.")

    return {
        "config": asdict(cfg),
        "history": history,
        "best_val_acc": best_val_acc,
        "best_epoch": best_state["epoch"],
        "best_w": best_state["w"],
        "best_b": best_state["b"],
    }


def build_default_experiments(
    epochs: int,
    batch_size: int,
    num_features: int,
) -> List[ExperimentConfig]:
    """8 组最小可行实验矩阵（控制变量法）"""
    return [
        # 1. Baseline: BoW + CE + lr0.10 + count + wd1e-4
        ExperimentConfig(
            name="baseline_bow_ce_lr0.10",
            ngram_min=1,
            ngram_max=1,
            loss_type="ce",
            lr=0.10,
            epochs=epochs,
            batch_size=batch_size,
            num_features=num_features,
            weight_decay=1e-4,
            binary=False,
        ),
        # 2. +bigram: N-gram(1,2) 效果
        ExperimentConfig(
            name="bigram_ce_lr0.10",
            ngram_min=1,
            ngram_max=2,
            loss_type="ce",
            lr=0.10,
            epochs=epochs,
            batch_size=batch_size,
            num_features=num_features,
            weight_decay=1e-4,
            binary=False,
        ),
        # 3. +trigram: N-gram(1,3) 效果
        ExperimentConfig(
            name="trigram_ce_lr0.10",
            ngram_min=1,
            ngram_max=3,
            loss_type="ce",
            lr=0.10,
            epochs=epochs,
            batch_size=batch_size,
            num_features=num_features,
            weight_decay=1e-4,
            binary=False,
        ),
        # 4. +mse: 损失函数对比
        ExperimentConfig(
            name="bow_mse_lr0.10",
            ngram_min=1,
            ngram_max=1,
            loss_type="mse",
            lr=0.10,
            epochs=epochs,
            batch_size=batch_size,
            num_features=num_features,
            weight_decay=1e-4,
            binary=False,
        ),
        # 5. +lr_small: 小学习率
        ExperimentConfig(
            name="bow_ce_lr0.01",
            ngram_min=1,
            ngram_max=1,
            loss_type="ce",
            lr=0.01,
            epochs=epochs,
            batch_size=batch_size,
            num_features=num_features,
            weight_decay=1e-4,
            binary=False,
        ),
        # 6. +lr_large: 大学习率
        ExperimentConfig(
            name="bow_ce_lr0.30",
            ngram_min=1,
            ngram_max=1,
            loss_type="ce",
            lr=0.30,
            epochs=epochs,
            batch_size=batch_size,
            num_features=num_features,
            weight_decay=1e-4,
            binary=False,
        ),
        # 7. +binary: 二值特征
        ExperimentConfig(
            name="bow_ce_lr0.10_binary",
            ngram_min=1,
            ngram_max=1,
            loss_type="ce",
            lr=0.10,
            epochs=epochs,
            batch_size=batch_size,
            num_features=num_features,
            weight_decay=1e-4,
            binary=True,
        ),
        # 8. +no_reg: 无正则化
        ExperimentConfig(
            name="bow_ce_lr0.10_no_reg",
            ngram_min=1,
            ngram_max=1,
            loss_type="ce",
            lr=0.10,
            epochs=epochs,
            batch_size=batch_size,
            num_features=num_features,
            weight_decay=0,
            binary=False,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task1: text classification without torch.nn"
    )
    parser.add_argument("--train_path", type=str, default="../new_train.tsv")
    parser.add_argument("--test_path", type=str, default="../new_test.tsv")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--num_features", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    base_dir = Path(__file__).resolve().parent
    train_path = (base_dir / args.train_path).resolve()
    test_path = (base_dir / args.test_path).resolve()
    out_dir = (base_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    train_df = load_tsv(train_path)
    test_df = load_tsv(test_path)

    texts = train_df["text"].tolist()
    labels = train_df["label"].to_numpy(dtype=np.int64)
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].to_numpy(dtype=np.int64)

    all_indices = np.arange(len(texts))
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=labels,
    )

    num_classes = int(labels.max()) + 1
    experiments = build_default_experiments(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_features=args.num_features,
    )

    feature_cache: Dict[
        Tuple[int, int, int, bool],
        Tuple[csr_matrix, csr_matrix],
    ] = {}
    all_results: List[dict] = []

    for cfg in experiments:
        cache_key = (cfg.ngram_min, cfg.ngram_max, cfg.num_features, cfg.binary)
        if cache_key not in feature_cache:
            ngram_range = (cfg.ngram_min, cfg.ngram_max)
            sparse_train = vectorize_corpus_to_csr(
                texts=texts,
                num_features=cfg.num_features,
                ngram_range=ngram_range,
                binary=cfg.binary,
                desc=f"Vectorizing train ngram={ngram_range}",
            )
            sparse_test = vectorize_corpus_to_csr(
                texts=test_texts,
                num_features=cfg.num_features,
                ngram_range=ngram_range,
                binary=cfg.binary,
                desc=f"Vectorizing test ngram={ngram_range}",
            )
            feature_cache[cache_key] = (sparse_train, sparse_test)

        sparse_train, sparse_test = feature_cache[cache_key]

        result = train_one_experiment(
            cfg=cfg,
            train_indices=train_indices,
            val_indices=val_indices,
            labels=labels,
            sparse_features=sparse_train,
            num_classes=num_classes,
            device=device,
        )

        # Evaluate best checkpoint on provided test split for reference.
        test_indices = np.arange(len(test_texts))
        test_logits_acc = evaluate_model(
            indices=test_indices,
            labels=test_labels,
            sparse_features=sparse_test,
            num_features=cfg.num_features,
            batch_size=cfg.batch_size,
            device=device,
            w=result["best_w"],
            b=result["best_b"],
            loss_type=cfg.loss_type,
            num_classes=num_classes,
        )[1]
        result["test_acc_reference"] = float(test_logits_acc)

        all_results.append(result)

    best = max(all_results, key=lambda x: x["best_val_acc"])
    print(
        f"Best experiment: {best['config']['name']} | val_acc={best['best_val_acc']:.4f}"
    )
    best_cfg = best["config"]
    best_cache_key = (
        best_cfg["ngram_min"],
        best_cfg["ngram_max"],
        best_cfg["num_features"],
        best_cfg["binary"],
    )
    _, best_sparse_test = feature_cache[best_cache_key]
    test_indices = np.arange(len(test_texts))

    preds = predict_labels(
        indices=test_indices,
        sparse_features=best_sparse_test,
        num_features=best_cfg["num_features"],
        batch_size=best_cfg["batch_size"],
        device=device,
        w=best["best_w"],
        b=best["best_b"],
    )
    save_predictions(out_dir / "predictions_best.csv", preds)

    _, final_test_acc = evaluate_model(
        indices=test_indices,
        labels=test_labels,
        sparse_features=best_sparse_test,
        num_features=best_cfg["num_features"],
        batch_size=best_cfg["batch_size"],
        device=device,
        w=best["best_w"],
        b=best["best_b"],
        loss_type=best_cfg["loss_type"],
        num_classes=num_classes,
    )
    print(f"Final test accuracy (best experiment): {final_test_acc:.4f}")

    # save_predictions(out_dir / "predictions_best.csv", preds)

    serializable = []
    for r in all_results:
        serializable.append(
            {
                "config": r["config"],
                "best_val_acc": r["best_val_acc"],
                "best_epoch": r["best_epoch"],
                "test_acc_reference": r["test_acc_reference"],
                "history": r["history"],
            }
        )

    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    plot_histories(all_results, out_dir)

    torch.save(
        {
            "config": best_cfg,
            "w": best["best_w"].cpu(),
            "b": best["best_b"].cpu(),
        },
        out_dir / "best_model.pt",
    )

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
