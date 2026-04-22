"""Plots for the InfoNCE pilot (infonce_pilot_v3/)."""

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

def load(cell):
    path = ROOT / "infonce_pilot_v3" / cell / "consistency_eval.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0].keys()}


# --- Fig 1: main result — infonce 100k vs KL-softmax 100k vs single 100k ---
def fig_infonce_vs_kl():
    infonce = load("infonce_long_100k")
    kl100 = ROOT / "pilot_v3" / "long_trainonly_lam01_100k" / "consistency_eval.csv"
    single = ROOT / "pilot_v3" / "long_single_50pct_100k" / "consistency_eval.csv"
    with open(kl100) as f: kl = list(csv.DictReader(f))
    with open(single) as f: sg = list(csv.DictReader(f))
    kl_step = np.array([float(r["step"]) for r in kl])
    kl_ens = np.array([float(r["val_acc_ensemble"]) for r in kl])
    sg_step = np.array([float(r["step"]) for r in sg])
    sg_ens = np.array([float(r["val_acc_spec_0"]) for r in sg])

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.plot(sg_step, sg_ens, color="#636363", linewidth=1.3,
            label="single model (v3 long, M=1)")
    ax.plot(kl_step, kl_ens, color="#d95f02", linewidth=1.4,
            label="KL-softmax (v3 long, M=4, $\\lambda{=}0.1$)")
    ax.plot(infonce["step"], infonce["val_acc_ensemble"],
            color="#1b9e77", linewidth=1.8,
            label="InfoNCE (M=2, $\\lambda{=}5$, $T{=}0.03$)")
    ax.axhline(100, linestyle=":", color="red", linewidth=0.7)
    ax.text(100_000, 101, "grokking (100%)", color="red", fontsize=8,
            ha="right", va="bottom")
    ax.set_xlabel("training step")
    ax.set_ylabel("ensemble val acc (%)")
    ax.set_title("InfoNCE long vs KL-softmax long vs single-model at 100k steps")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(-1, 108)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_infonce_vs_kl.png", bbox_inches="tight")
    plt.close(fig)


# --- Fig 2: per-specialist vs ensemble on the 100k infonce cell ---
def fig_infonce_perspec():
    d = load("infonce_long_100k")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ax1.plot(d["step"], d["val_acc_spec_0"], color="#1b9e77", linewidth=1.2, label="spec 0")
    ax1.plot(d["step"], d["val_acc_spec_1"], color="#7570b3", linewidth=1.2, label="spec 1")
    ax1.plot(d["step"], d["val_acc_ensemble"], color="black", linewidth=1.8, label="ensemble")
    ax1.set_xlabel("step"); ax1.set_ylabel("val acc (%)")
    ax1.set_title("InfoNCE 100k: per-specialist val acc")
    ax1.legend(loc="upper left", fontsize=9); ax1.set_ylim(-1, 100)

    ax2.plot(d["step"], d["unsup_entropy_mean"], color="#1b9e77", label="entropy (nats)")
    ax2.set_ylabel("entropy (nats)", color="#1b9e77")
    ax2.tick_params(axis="y", labelcolor="#1b9e77")
    ax2b = ax2.twinx()
    ax2b.plot(d["step"], d["pairwise_kl_val_mean"], color="#d95f02",
              label="pairwise KL on val (nats)")
    ax2b.set_ylabel("pairwise KL (nats)", color="#d95f02")
    ax2b.tick_params(axis="y", labelcolor="#d95f02")
    ax2b.grid(False)
    ax2.set_xlabel("step")
    ax2.set_title("InfoNCE 100k: mechanism diagnostics")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_infonce_perspec.png", bbox_inches="tight")
    plt.close(fig)


# --- Fig 3: infonce 25k vs 100k — the phase transition ---
def fig_infonce_phase():
    d25 = load("infonce_seed42")
    d100 = load("infonce_long_100k")
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.plot(d25["step"], d25["val_acc_ensemble"], color="#d95f02",
            linewidth=1.3, label="25k run (seed 42, T=0.03)")
    ax.plot(d100["step"], d100["val_acc_ensemble"], color="#1b9e77",
            linewidth=1.7, label="100k run (seed 42, T=0.03)")
    ax.axvline(25_000, color="grey", linestyle="--", linewidth=0.8)
    ax.text(25_500, 90, "25k cutoff", color="grey", fontsize=8)
    ax.set_xlabel("training step")
    ax.set_ylabel("ensemble val acc (%)")
    ax.set_title("InfoNCE: 25k and 100k runs share the same trajectory until ~50k; grokking starts near 75k")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(-1, 100)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_infonce_phase.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_infonce_vs_kl()
    fig_infonce_perspec()
    fig_infonce_phase()
    print("wrote to", FIGDIR)
