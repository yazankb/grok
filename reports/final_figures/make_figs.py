"""Generate figures for the final consistency-grokking report.

Reads pilot_v2/, pilot_v3/, pilot_v4/ sweep outputs from the repo root and
writes one PNG per figure into reports/final_figures/.
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "reports" / "final_figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_eval(cell_dir: Path) -> Dict[str, np.ndarray]:
    path = cell_dir / "consistency_eval.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0].keys()}
    return cols


def load_summary(sweep: str) -> Dict:
    with open(ROOT / sweep / "sweep_summary.json") as f:
        return json.load(f)


def cell_dir(sweep: str, name: str) -> Path:
    return ROOT / sweep / name


# ------------------------------------------------------------------
# Figure 1: single-model vs multi+consistency headline (v2 cells)
# ------------------------------------------------------------------
def fig_v2_headline():
    names = [
        ("single_model_50pct", "single model\n(50% data)"),
        ("multi_lam0", "multi, $\\lambda=0$\n(no consistency)"),
        ("multi_kl_lam01_warm5k", "multi, KL, $\\lambda=0.1$\nfull grid"),
        ("multi_kl_lam1_warm5k", "multi, KL, $\\lambda=1.0$\nfull grid"),
        ("multi_kl_lam01_warm5k_trainonly", "multi, KL, $\\lambda=0.1$\ntrain inputs only"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for name, label in names:
        d = load_eval(cell_dir("pilot_v2", name))
        ax.plot(d["step"], d["val_acc_ensemble"], label=label, linewidth=1.4)
    ax.set_xlabel("training step")
    ax.set_ylabel("ensemble val acc (%)")
    ax.set_title("Pilot v2: single-model baseline vs multi-specialist + consistency (25k steps)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_ylim(-1, 50)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_v2_headline.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 2: v3 seed robustness
# ------------------------------------------------------------------
def fig_v3_seeds():
    seeds = [42, 43, 44, 45]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(seeds)))
    for c, s in zip(colors, seeds):
        d = load_eval(cell_dir("pilot_v3", f"seed{s}_trainonly_lam01"))
        ax.plot(d["step"], d["val_acc_ensemble"], color=c, label=f"seed {s}", linewidth=1.4)
    ax.axhline(12.33, linestyle="--", color="grey", linewidth=1,
               label="single-model ceiling (12.3%)")
    ax.set_xlabel("training step")
    ax.set_ylabel("ensemble val acc (%)")
    ax.set_title("Pilot v3: seed robustness (train-inputs-only KL, $\\lambda=0.1$, 25k steps)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(-1, 55)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_v3_seeds.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 3: v3 long runs (single vs multi at 100k)
# ------------------------------------------------------------------
def fig_v3_long():
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    d_single = load_eval(cell_dir("pilot_v3", "long_single_50pct_100k"))
    d_multi = load_eval(cell_dir("pilot_v3", "long_trainonly_lam01_100k"))
    ax.plot(d_single["step"], d_single["val_acc_spec_0"],
            color="#d95f02", label="single model (spec_0)", linewidth=1.4)
    ax.plot(d_multi["step"], d_multi["val_acc_ensemble"],
            color="#1b9e77", label="multi + consistency (ensemble)", linewidth=1.4)
    ax.set_xlabel("training step")
    ax.set_ylabel("val acc (%)")
    ax.set_title("Pilot v3: single-model vs multi+consistency at 100k steps (seed 42, $\\lambda=0.1$)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(-1, 50)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_v3_long.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 4: v3 lambda ablation (at seed 42, 25k)
# ------------------------------------------------------------------
def fig_v3_lambda():
    cells = [
        ("trainonly_lam003", 0.03),
        ("seed42_trainonly_lam01", 0.1),
        ("trainonly_lam03", 0.3),
        ("trainonly_lam10", 1.0),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.4))
    colors = plt.cm.plasma(np.linspace(0.15, 0.8, len(cells)))
    for c, (name, lam) in zip(colors, cells):
        d = load_eval(cell_dir("pilot_v3", name))
        ax1.plot(d["step"], d["val_acc_ensemble"],
                 color=c, label=f"$\\lambda={lam}$", linewidth=1.4)
    ax1.set_xlabel("training step")
    ax1.set_ylabel("ensemble val acc (%)")
    ax1.set_title("Pilot v3: lambda ablation (seed 42, 25k)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_ylim(-1, 35)
    # Right panel: best val vs lambda
    lams = [lam for _, lam in cells]
    bests = []
    for name, _ in cells:
        d = load_eval(cell_dir("pilot_v3", name))
        bests.append(float(np.max(d["val_acc_ensemble"])))
    ax2.plot(lams, bests, "o-", color="#1b9e77", linewidth=1.6)
    for lam, best in zip(lams, bests):
        ax2.annotate(f"{best:.1f}%", xy=(lam, best), xytext=(0, 6),
                     textcoords="offset points", ha="center", fontsize=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("$\\lambda$ (log scale)")
    ax2.set_ylabel("best ensemble val acc (%)")
    ax2.set_title("best val acc vs $\\lambda$")
    ax2.set_ylim(10, 32)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_v3_lambda.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 5: v4 three cells at 100k
# ------------------------------------------------------------------
def fig_v4_main():
    cells = [
        ("seed44_lam1_100k",  "seed 44, $\\lambda=1.0$",  "#1b9e77"),
        ("seed43_lam1_100k",  "seed 43, $\\lambda=1.0$",  "#7570b3"),
        ("seed44_lam01_100k", "seed 44, $\\lambda=0.1$",  "#d95f02"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    for name, label, color in cells:
        d = load_eval(cell_dir("pilot_v4", name))
        ax.plot(d["step"], d["val_acc_ensemble"], color=color, label=label, linewidth=1.4)
    ax.axhline(12.33, linestyle="--", color="grey", linewidth=1,
               label="single-model ceiling (12.3%)")
    ax.axhline(100, linestyle=":", color="red", linewidth=0.8,
               label="grokking threshold (100%)")
    ax.set_xlabel("training step")
    ax.set_ylabel("ensemble val acc (%)")
    ax.set_title("Pilot v4: final three cells at 100k steps (train-inputs-only KL)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(-1, 105)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_v4_main.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 6: v4 per-specialist breakdown
# ------------------------------------------------------------------
def fig_v4_perspec():
    cells = [
        ("seed44_lam1_100k",  "seed 44, $\\lambda=1.0$"),
        ("seed43_lam1_100k",  "seed 43, $\\lambda=1.0$"),
        ("seed44_lam01_100k", "seed 44, $\\lambda=0.1$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, (name, title) in zip(axes, cells):
        d = load_eval(cell_dir("pilot_v4", name))
        for i in range(4):
            ax.plot(d["step"], d[f"val_acc_spec_{i}"],
                    linewidth=1.0, alpha=0.8, label=f"spec {i}")
        ax.plot(d["step"], d["val_acc_ensemble"],
                linewidth=1.8, color="black", label="ensemble")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        ax.set_ylim(-1, 60)
    axes[0].set_ylabel("val acc (%)")
    axes[-1].legend(loc="upper left", fontsize=7)
    fig.suptitle("Pilot v4: per-specialist val acc vs ensemble (100k steps)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_v4_perspec.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 7: mechanism (entropy + pairwise KL) for v4
# ------------------------------------------------------------------
def fig_v4_mechanism():
    cells = [
        ("seed44_lam1_100k",  "seed 44, $\\lambda=1.0$",  "#1b9e77"),
        ("seed43_lam1_100k",  "seed 43, $\\lambda=1.0$",  "#7570b3"),
        ("seed44_lam01_100k", "seed 44, $\\lambda=0.1$",  "#d95f02"),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.3))
    for name, label, color in cells:
        d = load_eval(cell_dir("pilot_v4", name))
        ax1.plot(d["step"], d["unsup_entropy_mean"], color=color, label=label, linewidth=1.4)
        ax2.plot(d["step"], d["pairwise_kl_val_mean"], color=color, label=label, linewidth=1.4)
    # Reference: log(97) ≈ 4.575 for V=97 tokens
    import math
    ax1.axhline(math.log(97), linestyle="--", color="grey",
                linewidth=0.8, label=f"$\\log V = {math.log(97):.2f}$")
    ax1.set_xlabel("step")
    ax1.set_ylabel("mean output entropy (nats)")
    ax1.set_title("Unsup output entropy")
    ax1.legend(loc="upper right", fontsize=7)
    ax2.set_xlabel("step")
    ax2.set_ylabel("pairwise KL on val (nats)")
    ax2.set_title("Pairwise specialist KL on val grid")
    ax2.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_v4_mechanism.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 8: summary bar — best ensemble across all sweeps
# ------------------------------------------------------------------
def fig_summary_bar():
    rows: List[Tuple[str, float, str]] = [
        ("single model 50%\n(v2, 25k)",                     12.33, "single"),
        ("single model 50%\n(v3, 100k)",                    12.33, "single"),
        ("multi $\\lambda=0$\n(v2, 25k)",                   1.38,  "multi_nocons"),
        ("full-grid KL $\\lambda=0.1$\n(v2, 25k)",          7.61,  "fullgrid"),
        ("full-grid KL $\\lambda=1.0$\n(v2, 25k)",          11.75, "fullgrid"),
        ("train-only $\\lambda=0.1$\n(v2 seed42, 25k)",     21.15, "trainonly"),
        ("train-only $\\lambda=0.1$\n(v3 seed44, 25k)",     45.33, "trainonly"),
        ("train-only $\\lambda=0.1$\n(v3 seed42, 100k)",    38.04, "trainonly"),
        ("train-only $\\lambda=1.0$\n(v4 seed44, 100k)",    49.31, "v4_best"),
        ("train-only $\\lambda=1.0$\n(v4 seed43, 100k)",    47.97, "v4_best"),
        ("train-only $\\lambda=0.1$\n(v4 seed44, 100k)",    48.01, "v4_best"),
    ]
    color_map = {
        "single":       "#636363",
        "multi_nocons": "#e31a1c",
        "fullgrid":     "#fdbf6f",
        "trainonly":    "#a6cee3",
        "v4_best":      "#1b9e77",
    }
    fig, ax = plt.subplots(figsize=(11, 4.2))
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [color_map[r[2]] for r in rows]
    x = np.arange(len(rows))
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.4)
    for xi, v in zip(x, vals):
        ax.annotate(f"{v:.1f}", xy=(xi, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.axhline(100, linestyle=":", color="red", linewidth=0.8)
    ax.text(len(rows) - 0.5, 101, "grokking threshold", color="red", fontsize=8,
            ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("best ensemble val acc (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Summary: best ensemble val acc across all pilots")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_summary_bar.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_v2_headline()
    fig_v3_seeds()
    fig_v3_long()
    fig_v3_lambda()
    fig_v4_main()
    fig_v4_perspec()
    fig_v4_mechanism()
    fig_summary_bar()
    print("Wrote figures to", FIGDIR)
