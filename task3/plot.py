"""
plot_results.py
从 metrics_summary.json 读取训练结果并生成三张图。

用法：
    python plot_results.py                          # 默认读取 outputs_addition/
    python plot_results.py --out_dir my_outputs     # 指定目录
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# 每条线的样式：线型 + marker，保证曲线重叠时仍可区分
LINE_STYLES = [
    {"linestyle": "-",  "marker": "o", "markevery": 5, "markersize": 4},
    {"linestyle": "--", "marker": "s", "markevery": 5, "markersize": 4},
    {"linestyle": "-.", "marker": "^", "markevery": 5, "markersize": 4},
    {"linestyle": ":",  "marker": "D", "markevery": 5, "markersize": 4},
    {"linestyle": "-",  "marker": "x", "markevery": 5, "markersize": 5},
]


def load_results(out_dir: Path) -> list[dict]:
    json_path = out_dir / "metrics_summary.json"
    if not json_path.exists():
        raise FileNotFoundError(f"找不到 {json_path}，请先运行训练脚本生成结果。")
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def plot_val_acc_curves(results: list[dict], out_dir: Path) -> None:
    """每个 epoch 的 validation accuracy 曲线。
    注意：若多个实验共用同一训练配置（train split 相同），曲线会天然重叠；
    此处通过不同线型 + marker 使重叠可见。
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, r in enumerate(results):
        style = LINE_STYLES[i % len(LINE_STYLES)]
        label = f"{r['config']['name']}  ({r['config']['desc']})"
        ax.plot(r["history"]["val_acc"], label=label, **style)

    ax.set_title("Addition Task — Validation Accuracy per Epoch\n"
                 "(curves sharing the same train split will overlap)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_path = out_dir / "val_acc_curves.png"
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_train_loss_curves(results: list[dict], out_dir: Path) -> None:
    """每个 epoch 的 training loss 曲线。"""
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, r in enumerate(results):
        style = LINE_STYLES[i % len(LINE_STYLES)]
        ax.plot(r["history"]["train_loss"], label=r["config"]["name"], **style)

    ax.set_title("Addition Task — Training Loss per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_path = out_dir / "train_loss.png"
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {save_path}")


def _annotate_bar(ax, bar, value: float) -> None:
    """在柱子顶部（或底部，当高度为 0 时）标注数值。"""
    h = bar.get_height()
    x = bar.get_x() + bar.get_width() / 2
    if value == 0.0:
        # 高度为 0 时标注在轴线稍上方，否则文字贴地不可见
        ax.text(x, 0.012, "0.000", ha="center", va="bottom",
                fontsize=6.5, color="gray", rotation=0)
    else:
        ax.text(x, h + 0.008, f"{value:.3f}", ha="center", va="bottom",
                fontsize=6.5)


def plot_bar_comparison(results: list[dict], out_dir: Path) -> None:
    """Best val acc vs. full test acc 的柱状对比图。
    修复：test_acc / val_acc 为 0 时柱高为 0，原版标注偏移导致不可见，
    现改为在轴线附近显示 '0.000'。
    """
    names         = [r["config"]["name"] for r in results]
    best_val_accs = [r["best_val_acc"]   for r in results]
    test_accs     = [r["test_acc"]       for r in results]

    x_pos = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    bars_val  = ax.bar(x_pos - width / 2, best_val_accs, width,
                       label="Best Val Acc (200 samples)", color="steelblue")
    bars_test = ax.bar(x_pos + width / 2, test_accs,     width,
                       label="Full Test Acc",              color="coral")

    for bar, val in zip(bars_val,  best_val_accs):
        _annotate_bar(ax, bar, val)
    for bar, val in zip(bars_test, test_accs):
        _annotate_bar(ax, bar, val)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_title("Addition Task — Accuracy Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_path = out_dir / "bar_comparison.png"
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {save_path}")


def print_summary(results: list[dict]) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"{'Name':<28} {'Best Val':>10} {'Test Acc':>10} {'Test N':>8}")
    print("-" * 62)
    for r in results:
        print(
            f"{r['config']['name']:<28} "
            f"{r['best_val_acc']:>10.4f} "
            f"{r['test_acc']:>10.4f} "
            f"{r['test_size']:>8}"
        )
    best = max(results, key=lambda x: x["test_acc"])
    print("-" * 62)
    print(f"Best overall: {best['config']['name']}  test_acc={best['test_acc']:.4f}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Plot results from metrics_summary.json")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs_addition",
        help="包含 metrics_summary.json 的目录（同时也是图片输出目录）",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    out_dir  = base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(out_dir)
    print(f"Loaded {len(results)} experiment(s) from {out_dir / 'metrics_summary.json'}")

    plot_val_acc_curves(results, out_dir)
    plot_train_loss_curves(results, out_dir)
    plot_bar_comparison(results, out_dir)
    print_summary(results)


if __name__ == "__main__":
    main()