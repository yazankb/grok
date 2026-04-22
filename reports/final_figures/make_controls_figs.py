"""Plots for the lr_control and loss_control_kl isolation experiments.

Reads:
    lr_control/v4_winner_lr1e3/            (M=4, kl_softmax, lr=1e-3, everything else = v4 winner)
    loss_control_kl/kl_infonce_hparams_100k/   (M=2, kl_softmax, InfoNCE pilot hparams)
    pilot_v4/seed44_lam1_100k/             (M=4, kl_softmax, lr=5e-4 -- v4 winner for comparison)
    infonce_pilot_v3/infonce_long_100k/    (M=2, infonce, 92% grok for comparison)

Writes:
    fig_lr_control.png, fig_loss_control.png, fig_controls_mechanism.png,
    fig_summary_bar_final.png
"""

import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "reports" / "final_figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(sweep: str, cell: str):
    path = ROOT / sweep / cell / "consistency_eval.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0].keys()}


# --- Fig A: lr_control vs v4 winner (only lr differs) ------------------
def fig_lr_control():
    v4 = load("pilot_v4", "seed44_lam1_100k")
    lrc = load("lr_control", "v4_winner_lr1e3")

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.plot(v4["step"], v4["val_acc_ensemble"], color="#d95f02",
            linewidth=1.5,
            label="v4 winner (M=4, $\\lambda{=}1$, seed 44, lr=5e-4) -- 49%")
    ax.plot(lrc["step"], lrc["val_acc_ensemble"], color="#1b9e77",
            linewidth=1.9,
            label="lr control: same config, lr=1e-3 -- 100%")
    ax.axhline(100, linestyle=":", color="red", linewidth=0.8)
    ax.text(100_000, 101.5, "grokking (100%)", color="red", fontsize=8,
            ha="right", va="bottom")
    ax.axhline(12.3, linestyle="--", color="grey", linewidth=0.6)
    ax.text(1_000, 13.5, "single-model ceiling (12.3%)", color="grey", fontsize=8)
    ax.set_xlabel("training step")
    ax.set_ylabel("ensemble val acc (%)")
    ax.set_title("LR control: raising lr from 5e-4 to 1e-3 breaks the v4 ceiling")
    ax.legend(loc="center left", fontsize=9)
    ax.set_ylim(-1, 108)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_lr_control.png", bbox_inches="tight")
    plt.close(fig)


# --- Fig B: loss_control_kl vs infonce_long (only loss differs) ---------
def fig_loss_control():
    infonce = load("infonce_pilot_v3", "infonce_long_100k")
    kl = load("loss_control_kl", "kl_infonce_hparams_100k")

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.plot(infonce["step"], infonce["val_acc_ensemble"], color="#7570b3",
            linewidth=1.5,
            label="InfoNCE (M=2, lr=1e-3, $\\lambda{=}5$, T=0.03) -- 92% @ 99k")
    ax.plot(kl["step"], kl["val_acc_ensemble"], color="#1b9e77",
            linewidth=1.9,
            label="loss control: same hparams, KL-softmax -- 100% @ 22k")
    ax.axhline(100, linestyle=":", color="red", linewidth=0.8)
    ax.text(100_000, 101.5, "grokking (100%)", color="red", fontsize=8,
            ha="right", va="bottom")
    ax.set_xlabel("training step")
    ax.set_ylabel("ensemble val acc (%)")
    ax.set_title("Loss control: swapping InfoNCE -> KL-softmax (same hparams) groks faster and higher")
    ax.legend(loc="center left", fontsize=9)
    ax.set_ylim(-1, 108)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_loss_control.png", bbox_inches="tight")
    plt.close(fig)


# --- Fig C: mechanism diagnostics on both controls ---------------------
def fig_controls_mechanism():
    lrc = load("lr_control", "v4_winner_lr1e3")
    kl = load("loss_control_kl", "kl_infonce_hparams_100k")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    ax1.plot(lrc["step"], lrc["val_acc_spec_0"], color="#1b9e77", linewidth=1.0, label="spec 0")
    ax1.plot(lrc["step"], lrc["val_acc_spec_1"], color="#7570b3", linewidth=1.0, label="spec 1")
    ax1.plot(lrc["step"], lrc["val_acc_spec_2"], color="#d95f02", linewidth=1.0, label="spec 2")
    ax1.plot(lrc["step"], lrc["val_acc_spec_3"], color="#e7298a", linewidth=1.0, label="spec 3")
    ax1.plot(lrc["step"], lrc["val_acc_ensemble"], color="black", linewidth=1.8, label="ensemble")
    ax1.set_xlabel("step"); ax1.set_ylabel("val acc (%)")
    ax1.set_title("lr_control (M=4): per-specialist vs ensemble")
    ax1.legend(loc="center left", fontsize=8); ax1.set_ylim(-1, 108)

    ax2.plot(kl["step"], kl["val_acc_spec_0"], color="#1b9e77", linewidth=1.0, label="spec 0")
    ax2.plot(kl["step"], kl["val_acc_spec_1"], color="#7570b3", linewidth=1.0, label="spec 1")
    ax2.plot(kl["step"], kl["val_acc_ensemble"], color="black", linewidth=1.8, label="ensemble")
    ax2.set_xlabel("step"); ax2.set_ylabel("val acc (%)")
    ax2.set_title("loss_control_kl (M=2): per-specialist vs ensemble")
    ax2.legend(loc="center right", fontsize=8); ax2.set_ylim(-1, 108)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_controls_perspec.png", bbox_inches="tight")
    plt.close(fig)


# --- Fig D: final summary bar (now including the grokking controls) ----
def fig_summary_bar_final():
    # (label, ensemble best val acc %, family)
    rows = [
        ("single-model\n(50% data, 100k)",          12.3,  "single"),
        ("v1 multi + MSE\n(full grid, 25k)",         1.0,  "v1"),
        ("v2 KL full grid\n($\\lambda{=}0.1$, 25k)",  7.6,  "fullgrid"),
        ("v2 KL full grid\n($\\lambda{=}1$, 25k)",   11.8,  "fullgrid"),
        ("v2 KL train-only\n($\\lambda{=}0.1$, 25k)",21.2,  "v2trainonly"),
        ("v3 KL train-only\nseed 44, 25k",           45.3,  "v3"),
        ("v4 seed 44 $\\lambda{=}1$\nlr=5e-4, 100k", 49.3,  "v4"),
        ("InfoNCE long 100k\nlr=1e-3, M=2, T=0.03",  92.2,  "infonce"),
        ("loss control (KL)\nlr=1e-3, M=2, 100k",   100.0,  "grok"),
        ("lr control (KL)\nlr=1e-3, M=4, 100k",     100.0,  "grok"),
    ]
    colors = {
        "single":       "#636363",
        "v1":           "#b22222",
        "fullgrid":     "#e39d42",
        "v2trainonly":  "#3182bd",
        "v3":           "#2171b5",
        "v4":           "#08519c",
        "infonce":      "#7570b3",
        "grok":         "#1b9e77",
    }
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cs = [colors[r[2]] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 4.2))
    xs = np.arange(len(rows))
    bars = ax.bar(xs, vals, color=cs, width=0.72, edgecolor="black", linewidth=0.3)
    for x, v in zip(xs, vals):
        ax.text(x, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.axhline(100, linestyle=":", color="red", linewidth=0.7)
    ax.axhline(12.3, linestyle="--", color="grey", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("best ensemble val acc (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Best ensemble val acc across every milestone in the narrative")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_summary_bar_final.png", bbox_inches="tight")
    plt.close(fig)


# --- Fig E: project overview (single plot, every key trajectory) -------
def fig_project_overview():
    """One-panel trajectory view of the whole investigation, for §1."""
    single = load("pilot_v3", "long_single_50pct_100k")
    v2 = load("pilot_v2", "multi_kl_lam01_warm5k_trainonly")
    v4 = load("pilot_v4", "seed44_lam1_100k")
    infonce = load("infonce_pilot_v3", "infonce_long_100k")
    lrc = load("lr_control", "v4_winner_lr1e3")
    kl = load("loss_control_kl", "kl_infonce_hparams_100k")

    fig, ax = plt.subplots(figsize=(10, 4.4))

    ax.plot(single["step"], single["val_acc_ensemble"], color="#636363",
            linewidth=1.2, linestyle="--",
            label="single-model 50%, 100k (v3 budget probe) -- 12.3%")
    ax.plot(v2["step"], v2["val_acc_ensemble"], color="#3182bd", linewidth=1.2,
            label="v2 KL train-only, 25k ($\\lambda{=}0.1$) -- 21.2%")
    ax.plot(v4["step"], v4["val_acc_ensemble"], color="#08519c", linewidth=1.6,
            label="v4 winner (M=4, $\\lambda{=}1$, lr=5e-4) -- 49.3%")
    ax.plot(infonce["step"], infonce["val_acc_ensemble"], color="#7570b3", linewidth=1.6,
            label="v5 InfoNCE long (M=2, lr=1e-3) -- 92.2%")
    ax.plot(kl["step"], kl["val_acc_ensemble"], color="#1b9e77", linewidth=2.0,
            label="loss control (KL, InfoNCE hparams) -- 100%")
    ax.plot(lrc["step"], lrc["val_acc_ensemble"], color="#006d2c", linewidth=2.0,
            label="lr control (v4 config, lr=1e-3) -- 100%")

    ax.axhline(100, linestyle=":", color="red", linewidth=0.7)
    ax.text(100_000, 101.5, "grokking (100%)", color="red", fontsize=8, ha="right", va="bottom")
    ax.axhline(12.3, linestyle="--", color="grey", linewidth=0.5)
    ax.text(1_000, 13.3, "single-model ceiling (12.3%)", color="grey", fontsize=8)
    ax.set_xlabel("training step")
    ax.set_ylabel("ensemble val acc (%)")
    ax.set_title("Project overview: every key trajectory in one plot")
    ax.legend(loc="center right", fontsize=8, framealpha=0.9)
    ax.set_ylim(-1, 108)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_project_overview.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_lr_control()
    fig_loss_control()
    fig_controls_mechanism()
    fig_summary_bar_final()
    fig_project_overview()
    print("wrote figures to", FIGDIR)
