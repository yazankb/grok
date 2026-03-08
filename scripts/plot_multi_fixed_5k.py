#!/usr/bin/env python
"""
Plot specialist/merged/baseline curves for the multi_fixed 5k experiment.
"""

import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _load_latest_metrics(run_dir: str) -> pd.DataFrame | None:
    matches = sorted(
        glob.glob(os.path.join(run_dir, "lightning_logs", "version_*", "metrics.csv"))
    )
    if not matches:
        return None
    df = pd.read_csv(matches[-1]).dropna(how="all")
    if "step" in df.columns:
        df = df.sort_values("step").reset_index(drop=True)
    return df


def _series(df: pd.DataFrame | None, col: str) -> tuple[pd.Series, pd.Series]:
    if df is None or col not in df.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    sub = df[["step", col]].dropna()
    return sub["step"], sub[col]


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    exp_dir = repo / "logs" / "multi_fixed"
    out_path = exp_dir / "grokking_curves_5k.png"

    specs = {}
    for i in range(4):
        specs[f"specialist_{i}"] = _load_latest_metrics(str(exp_dir / f"specialist_{i}"))

    merged = _load_latest_metrics(str(exp_dir / "merged_5k"))
    baseline = _load_latest_metrics(str(exp_dir / "baseline_5k"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Multi-Model Grokking (multi_fixed, 5k phase budget)", fontsize=14)

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    # Panel 1: specialists train/val
    ax = axes[0]
    ax.set_title("Specialists: full-train and val accuracy")
    for idx, (name, df) in enumerate(specs.items()):
        xs, ys = _series(df, "full_train_acc")
        if len(xs):
            ax.plot(xs, ys, lw=1.8, color=colors[idx], label=f"{name} train")
        xv, yv = _series(df, "val_accuracy")
        if len(xv):
            ax.plot(xv, yv, lw=1.0, ls="--", alpha=0.7, color=colors[idx], label=f"{name} val")
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, ncol=2)

    # Panel 2: merged vs baseline val
    ax = axes[1]
    ax.set_title("Merged vs baseline validation accuracy")
    for label, df, color in [
        ("merged_5k", merged, "#DD8452"),
        ("baseline_5k", baseline, "#222222"),
    ]:
        xv, yv = _series(df, "val_accuracy")
        if len(xv):
            ax.plot(xv, yv, lw=2.2, color=color, label=label)
    ax.set_xlabel("Step")
    ax.set_ylabel("Val accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)

    # Panel 3: merged vs baseline train accuracy
    ax = axes[2]
    ax.set_title("Merged vs baseline full-train accuracy")
    for label, df, color in [
        ("merged_5k", merged, "#DD8452"),
        ("baseline_5k", baseline, "#222222"),
    ]:
        xt, yt = _series(df, "full_train_acc")
        if len(xt):
            ax.plot(xt, yt, lw=2.2, color=color, label=label)
    ax.set_xlabel("Step")
    ax.set_ylabel("Full-train accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
