"""
plot_metrics.py — Generate training graphs from checkpoints/training_metrics.json

Usage:
    python plot_metrics.py                          # saves plots to checkpoints/
    python plot_metrics.py --metrics path/to/training_metrics.json
    python plot_metrics.py --show                   # also opens an interactive window
"""

import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; swap to "TkAgg" if you want --show
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

COLORS = {
    "train_loss": "#534AB7",   # purple
    "val_loss":   "#1D9E75",   # teal
    "f1":         "#E8593C",   # coral
    "precision":  "#BA7517",   # amber
    "recall":     "#378ADD",   # blue
    "lr":         "#888780",   # gray
}


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ax_style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="500", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", linewidth=0.4, color="#d3d1c7")
    ax.grid(axis="x", linewidth=0.2, color="#d3d1c7", linestyle="--")


def plot_step_loss(metrics: dict, out_dir: Path) -> None:
    """Step-level training loss — one point every LOG_EVERY steps."""
    steps = metrics.get("step_losses", [])
    if not steps:
        print("  No step-level data found, skipping.")
        return

    xs = [s["global_step"] for s in steps]
    ys = [s["train_loss"]  for s in steps]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, ys, color=COLORS["train_loss"], linewidth=1.2, label="Train loss (step avg)")
    ax.fill_between(xs, ys, alpha=0.08, color=COLORS["train_loss"])
    _ax_style(ax, "Training loss (per step)", "Global step", "Loss (token_mean)")
    ax.legend(fontsize=10)
    fig.tight_layout()
    out = out_dir / "step_loss.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_epoch_losses(metrics: dict, out_dir: Path) -> None:
    """Epoch-level train vs val loss on one chart."""
    epochs_data = metrics.get("epoch_stats", [])
    if not epochs_data:
        print("  No epoch data found, skipping.")
        return

    epochs     = [e["epoch"]          for e in epochs_data]
    train_loss = [e["avg_train_loss"] for e in epochs_data]
    val_loss   = [e["avg_val_loss"]   for e in epochs_data]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_loss, color=COLORS["train_loss"], linewidth=2,
            marker="o", markersize=5, label="Train loss")
    ax.plot(epochs, val_loss,   color=COLORS["val_loss"],   linewidth=2,
            marker="s", markersize=5, label="Val loss",   linestyle="--")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    _ax_style(ax, "Train vs validation loss (per epoch)", "Epoch", "Loss (token_mean)")
    ax.legend(fontsize=10)
    fig.tight_layout()
    out = out_dir / "epoch_losses.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_f1(metrics: dict, out_dir: Path) -> None:
    """Epoch-level val F1, precision, recall."""
    epochs_data = metrics.get("epoch_stats", [])
    if not epochs_data:
        return

    epochs    = [e["epoch"]         for e in epochs_data]
    f1        = [e["val_f1"]        for e in epochs_data]
    precision = [e["val_precision"] for e in epochs_data]
    recall    = [e["val_recall"]    for e in epochs_data]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, f1,        color=COLORS["f1"],        linewidth=2.5,
            marker="o", markersize=5, label="F1")
    ax.plot(epochs, precision, color=COLORS["precision"], linewidth=1.5,
            marker="^", markersize=4, label="Precision", linestyle="--")
    ax.plot(epochs, recall,    color=COLORS["recall"],    linewidth=1.5,
            marker="v", markersize=4, label="Recall",    linestyle=":")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    _ax_style(ax, "Validation span-level F1 (per epoch)", "Epoch", "Score")
    ax.legend(fontsize=10)
    fig.tight_layout()
    out = out_dir / "val_f1.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_lr(metrics: dict, out_dir: Path) -> None:
    """Learning rate schedule over global steps."""
    steps = metrics.get("step_losses", [])
    if not steps:
        return

    xs = [s["global_step"] for s in steps]
    ys = [s["lr"]          for s in steps]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(xs, ys, color=COLORS["lr"], linewidth=1.5)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.5f"))
    _ax_style(ax, "Learning rate schedule", "Global step", "LR")
    fig.tight_layout()
    out = out_dir / "lr_schedule.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_summary(metrics: dict, out_dir: Path) -> None:
    """2×2 dashboard: step loss · epoch losses · F1 · LR."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Training summary", fontsize=14, fontweight="500", y=1.01)

    steps       = metrics.get("step_losses", [])
    epochs_data = metrics.get("epoch_stats", [])

    # ── top-left: step loss ──────────────────────────────────────────────────
    ax = axes[0, 0]
    if steps:
        xs = [s["global_step"] for s in steps]
        ys = [s["train_loss"]  for s in steps]
        ax.plot(xs, ys, color=COLORS["train_loss"], linewidth=1)
        ax.fill_between(xs, ys, alpha=0.08, color=COLORS["train_loss"])
    _ax_style(ax, "Train loss (step)", "Step", "Loss")

    # ── top-right: epoch losses ───────────────────────────────────────────────
    ax = axes[0, 1]
    if epochs_data:
        ep = [e["epoch"] for e in epochs_data]
        ax.plot(ep, [e["avg_train_loss"] for e in epochs_data],
                color=COLORS["train_loss"], linewidth=2, marker="o", markersize=4, label="Train")
        ax.plot(ep, [e["avg_val_loss"]   for e in epochs_data],
                color=COLORS["val_loss"],   linewidth=2, marker="s", markersize=4, label="Val", linestyle="--")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=9)
    _ax_style(ax, "Train / val loss (epoch)", "Epoch", "Loss")

    # ── bottom-left: F1 ───────────────────────────────────────────────────────
    ax = axes[1, 0]
    if epochs_data:
        ep = [e["epoch"] for e in epochs_data]
        ax.plot(ep, [e["val_f1"]        for e in epochs_data],
                color=COLORS["f1"],        linewidth=2, marker="o", markersize=4, label="F1")
        ax.plot(ep, [e["val_precision"] for e in epochs_data],
                color=COLORS["precision"], linewidth=1.2, marker="^", markersize=3, label="P", linestyle="--")
        ax.plot(ep, [e["val_recall"]    for e in epochs_data],
                color=COLORS["recall"],    linewidth=1.2, marker="v", markersize=3, label="R", linestyle=":")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=9)
    _ax_style(ax, "Val F1 (epoch)", "Epoch", "Score")

    # ── bottom-right: LR ──────────────────────────────────────────────────────
    ax = axes[1, 1]
    if steps:
        xs = [s["global_step"] for s in steps]
        ys = [s["lr"]          for s in steps]
        ax.plot(xs, ys, color=COLORS["lr"], linewidth=1.5)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.5f"))
    _ax_style(ax, "LR schedule", "Step", "LR")

    fig.tight_layout()
    out = out_dir / "training_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument("--metrics", default="checkpoints/training_metrics.json",
                        help="Path to training_metrics.json")
    parser.add_argument("--out", default=None,
                        help="Output directory for PNG files (default: same dir as metrics file)")
    parser.add_argument("--show", action="store_true",
                        help="Open plots in an interactive window after saving")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise SystemExit(f"Metrics file not found: {metrics_path}")

    out_dir = Path(args.out) if args.out else metrics_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load(str(metrics_path))
    print(f"Loaded metrics from {metrics_path}")
    print(f"  step records : {len(metrics.get('step_losses', []))}")
    print(f"  epoch records: {len(metrics.get('epoch_stats', []))}")
    print(f"Saving plots to {out_dir}/")

    plot_step_loss(metrics, out_dir)
    plot_epoch_losses(metrics, out_dir)
    plot_f1(metrics, out_dir)
    plot_lr(metrics, out_dir)
    plot_summary(metrics, out_dir)

    if args.show:
        matplotlib.use("TkAgg")
        plt.show()


if __name__ == "__main__":
    main()