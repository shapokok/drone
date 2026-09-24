"""Builds the paper's 4 tables + 4 figures from outputs/results.csv and
outputs/predictions/. Each table/figure pair is deliberately non-
redundant -- see README section 5 for which axis each one owns.

Color usage follows the project's validated categorical palette (fixed
hue order, never cycled): EKF=blue, FusionTransformer=orange,
FusionLSTM=aqua, ablation arms=yellow/magenta/violet in a fixed order.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CATEGORICAL = {
    "ekf": "#2a78d6",                 # slot 1 blue -- classical baseline
    "fusion_transformer": "#eb6834",  # slot 2 orange -- proposed method
    "fusion_lstm": "#1baf7a",         # slot 3 aqua -- backbone ablation
    "no_cross_attn": "#eda100",       # slot 4 yellow -- ablation arm
    "imu_only": "#e87ba4",            # slot 5 magenta
    "gps_only": "#4a3aa7",            # slot 7 violet
}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRIDLINE, "axes.labelcolor": INK_PRIMARY,
    "text.color": INK_PRIMARY, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "grid.color": GRIDLINE, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False,
})


def label_for(model, ablation):
    if model == "ekf":
        return "EKF (no bias)" if ablation == "no_bias" else "EKF"
    base = "FusionNav" if model == "fusion_transformer" else "FusionNav-LSTM"
    return base if ablation == "full" else f"FusionNav ({ablation})"


def color_for(model, ablation):
    if ablation != "full" and ablation in CATEGORICAL:
        return CATEGORICAL[ablation]
    return CATEGORICAL.get(model, INK_MUTED)


# --------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------- #

def table_main_accuracy(df):
    """Table 1: headline accuracy, full GPS availability, full models only."""
    sub = df[(df.ablation == "full") & (df.eval_outage_rate.fillna(0) == 0)]
    cols = ["ATE", "RPE", "RMSE_E", "RMSE_N", "RMSE_U"]
    return (sub.groupby("model")[cols].agg(["mean", "std"]).round(4)
            .sort_values(("ATE", "mean")))


def table_gps_denied(df):
    """Table 2: robustness across outage rates, full models, trained clean."""
    sub = df[(df.ablation == "full") & (df.train_outage_rate.fillna(0) == 0)
             & df.eval_outage_rate.notna()]
    return (sub.pivot_table(index="model", columns="eval_outage_rate", values="ATE",
                             aggfunc="mean").round(3))


def table_ablation(df):
    """Table 3: ablation arms at full GPS availability."""
    sub = df[(df.eval_outage_rate.fillna(0) == 0) & (df.train_outage_rate.fillna(0) == 0)]
    cols = ["ATE", "RPE", "RMSE_E", "RMSE_N", "RMSE_U"]
    sub = sub.assign(variant=sub.model + "/" + sub.ablation)
    return sub.groupby("variant")[cols].mean().round(4).sort_values("ATE")


def table_seed_spread(df):
    """Table 4: mean+-std of Table 1's headline ATE/RPE over seeds."""
    sub = df[(df.ablation == "full") & (df.eval_outage_rate.fillna(0) == 0)]
    g = sub.groupby("model").agg(
        ATE_mean=("ATE", "mean"), ATE_std=("ATE", "std"),
        RPE_mean=("RPE", "mean"), RPE_std=("RPE", "std"),
        n_seeds=("seed", "nunique"),
    ).round(4)
    return g


# --------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------- #

def figure_trajectory_overlay(pred_dir, out_dir, tag_gt, tags_pred):
    """Fig 1: qualitative top-down + altitude overlay."""
    gt = np.load(pred_dir / f"{tag_gt}_gt.npy")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    axes[0].plot(gt[:, 0], gt[:, 1], color=INK_PRIMARY, lw=1.5, label="Ground truth")
    axes[1].plot(np.arange(len(gt)), gt[:, 2], color=INK_PRIMARY, lw=1.5, label="Ground truth")

    for label, tag, color in tags_pred:
        pred = np.load(pred_dir / f"{tag}_pred.npy")
        n = min(len(pred), len(gt))
        axes[0].plot(pred[:n, 0], pred[:n, 1], color=color, lw=1.2, alpha=0.9, label=label)
        axes[1].plot(np.arange(n), pred[:n, 2], color=color, lw=1.2, alpha=0.9, label=label)

    axes[0].set_xlabel("East (m)"); axes[0].set_ylabel("North (m)")
    axes[0].set_title("Top-down trajectory")
    axes[1].set_xlabel("Timestep"); axes[1].set_ylabel("Up (m)")
    axes[1].set_title("Altitude profile")
    for ax in axes:
        ax.grid(True, lw=0.5)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_trajectory_overlay.png", dpi=200)
    plt.close(fig)


def figure_outage_error_growth(pred_dir, out_dir, tags):
    """Fig 2: error growth WITHIN a single sustained outage (time-since-loss
    axis) -- distinct from Table 2's across-outage-rate axis.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, tag, color in tags:
        pred = np.load(pred_dir / f"{tag}_pred.npy")
        gt = np.load(pred_dir / f"{tag}_gt.npy")
        n = min(len(pred), len(gt))
        err = np.linalg.norm(pred[:n] - gt[:n], axis=1)
        ax.plot(np.arange(n), err, color=color, lw=1.5, label=label)
    ax.set_xlabel("Timestep since outage start")
    ax.set_ylabel("Position error (m)")
    ax.set_title("Error growth during a sustained GPS outage")
    ax.grid(True, lw=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_outage_error_growth.png", dpi=200)
    plt.close(fig)


def figure_attention_shift(xai_dir, out_dir):
    """Fig 3: attention weight on concurrent GPS vs its availability -- the
    XAI angle, distinct from both accuracy tables.
    """
    weight = np.load(xai_dir / "attention_weight.npy")
    avail = np.load(xai_dir / "gps_availability.npy")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    t = np.arange(len(weight))
    ax.plot(t, weight, color=CATEGORICAL["fusion_transformer"], lw=1.5,
            label="Attention on concurrent GPS")
    ax.fill_between(t, 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                     where=~avail, color=INK_MUTED, alpha=0.15, label="GPS unavailable")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Attention weight")
    ax.set_title("Cross-attention reliance shifts to IMU when GPS drops")
    ax.grid(True, lw=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_attention_shift.png", dpi=200)
    plt.close(fig)


def figure_ate_distribution(df, out_dir):
    """Fig 4: ATE spread across seeds/segments -- complements Table 4's
    single mean+-std with the full distribution.
    """
    sub = df[(df.ablation == "full") & (df.eval_outage_rate.fillna(0) == 0)]
    models = sorted(sub.model.unique(), key=lambda m: {"ekf": 0}.get(m, 1))
    data = [sub[sub.model == m].ATE.dropna().values for m in models]
    colors = [CATEGORICAL.get(m, INK_MUTED) for m in models]

    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(data, labels=[label_for(m, "full") for m in models],
                     patch_artist=True, widths=0.5, medianprops={"color": INK_PRIMARY})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor(color)
    ax.set_ylabel("ATE (m)")
    ax.set_title("ATE distribution across seeds")
    ax.grid(True, lw=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_ate_distribution.png", dpi=200)
    plt.close(fig)


def build_all(results_csv, out_dir, pred_dir):
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)

    tables = {
        "table1_main_accuracy": table_main_accuracy(df),
        "table2_gps_denied": table_gps_denied(df),
        "table3_ablation": table_ablation(df),
        "table4_seed_spread": table_seed_spread(df),
    }
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv")
        print(f"\n=== {name} ===\n{t}")

    build_figures(df, Path(pred_dir), out_dir, fig_dir)
    return tables


def build_figures(df, pred_dir, out_dir, fig_dir):
    """Best-effort: each figure is skipped (not failed) with a printed
    reason if the predictions/xai artifacts it needs aren't present yet
    (e.g. a partial sweep) -- doesn't block the tables from being built.
    """
    seed0_full_evaloutage0 = "s0_trainoutage0.0_evaloutage0.0"
    try:
        tags = [
            (label_for("ekf", "full"), f"ekf_full_{seed0_full_evaloutage0}", CATEGORICAL["ekf"]),
            (label_for("fusion_transformer", "full"),
             f"fusion_transformer_full_{seed0_full_evaloutage0}", CATEGORICAL["fusion_transformer"]),
        ]
        figure_trajectory_overlay(pred_dir, fig_dir, tags[0][1], tags)
        print("fig1 saved")
    except FileNotFoundError as e:
        print(f"fig1 skipped: {e}")

    try:
        seed0_evaloutage3 = "s0_trainoutage0.0_evaloutage0.3"
        tags = [
            (label_for("ekf", "full"), f"ekf_full_{seed0_evaloutage3}", CATEGORICAL["ekf"]),
            (label_for("fusion_transformer", "full"),
             f"fusion_transformer_full_{seed0_evaloutage3}", CATEGORICAL["fusion_transformer"]),
        ]
        figure_outage_error_growth(pred_dir, fig_dir, tags)
        print("fig2 saved")
    except FileNotFoundError as e:
        print(f"fig2 skipped: {e}")

    try:
        figure_attention_shift(out_dir / "xai", fig_dir)
        print("fig3 saved")
    except FileNotFoundError as e:
        print(f"fig3 skipped: {e}")

    try:
        figure_ate_distribution(df, fig_dir)
        print("fig4 saved")
    except (ValueError, IndexError) as e:
        print(f"fig4 skipped: {e}")

    print(f"\ntables + figures saved under {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", default="outputs/results.csv")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--pred_dir", default="outputs/predictions")
    args = ap.parse_args()
    build_all(args.results_csv, args.out_dir, args.pred_dir)
