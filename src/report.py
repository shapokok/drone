"""Builds the paper's 4 tables + 4 figures from outputs/results.csv and
outputs/predictions/. Each table/figure pair is deliberately non-
redundant -- see README section 5 for which axis each one owns.

Color usage follows the project's validated categorical palette (fixed
hue order, never cycled): EKF=blue, FusionTransformer=orange,
FusionLSTM=aqua, ablation arms=yellow/magenta/violet in a fixed order.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import apply_block_outage, outage_rng  # noqa: E402

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


def find_longest_outage_segment(avail, margin=15):
    """Longest contiguous False-run in a GPS-availability mask, padded
    with `margin` samples of context on each side. Returns (slice,
    onset_index_within_slice).
    """
    padded = np.concatenate(([True], avail, [True])).astype(int)
    edges = np.diff(padded)
    starts = np.where(edges == -1)[0]
    ends = np.where(edges == 1)[0]
    i = np.argmax(ends - starts)
    start, end = int(starts[i]), int(ends[i])
    lo, hi = max(0, start - margin), min(len(avail), end + margin)
    return slice(lo, hi), start - lo


def figure_outage_error_growth(pred_dir, out_dir, tags, outage_rate=0.3, dt=1.0, margin=15):
    """Fig 2: error growth within ONE representative long GPS-outage
    episode, time axis in seconds since loss -- distinct from Table 2's
    across-outage-rate axis (which aggregates over the whole test set
    and every outage episode in it, not a single one).

    The outage mask isn't saved alongside predictions -- it's
    regenerated here with the same (n, outage_rate) -> outage_rng seed
    evaluate.py used, which is deterministic by construction (see
    data/dataset.py's outage_rng), so this reproduces the exact mask
    without needing extra saved files.
    """
    first_pred = np.load(pred_dir / f"{tags[0][1]}_pred.npy")
    n = len(first_pred)
    avail, _ = apply_block_outage(n, np.zeros((n, 3)), target_rate=outage_rate,
                                   rng=outage_rng(outage_rate))
    sl, onset = find_longest_outage_segment(avail, margin=margin)

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, tag, color in tags:
        pred = np.load(pred_dir / f"{tag}_pred.npy")
        gt = np.load(pred_dir / f"{tag}_gt.npy")
        err = np.linalg.norm(pred[sl] - gt[sl], axis=1)
        t_seconds = (np.arange(len(err)) - onset) * dt
        ax.plot(t_seconds, err, color=color, lw=1.5, label=label)
    ax.axvline(0, color=INK_MUTED, lw=1.0, linestyle="--")
    ax.set_xlabel("Seconds since GPS loss")
    ax.set_ylabel("Position error (m)")
    ax.set_title("Error growth during one representative GPS outage episode")
    ax.grid(True, lw=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_outage_error_growth.png", dpi=200)
    plt.close(fig)


def figure_branch_contribution(xai_dir, out_dir):
    """Fig 3: CAUSAL contribution of each branch -- excess position error
    when that branch's input is blinded at inference on the same trained
    checkpoint (occlusion/ablation probing), not raw attention weight.
    Raw attention on the concurrent GPS token was tried first and gave an
    uninterpretable result (it rose during outage, contradicting Table 2's
    evidence that the model's actual robustness comes from the IMU branch)
    -- attention weight is correlational, not causal. This is the XAI
    angle, distinct from both accuracy tables.
    """
    imu_contrib = np.load(xai_dir / "imu_contribution.npy")
    gps_contrib = np.load(xai_dir / "gps_contribution.npy")
    avail = np.load(xai_dir / "gps_availability.npy")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    t = np.arange(len(imu_contrib))
    ax.plot(t, imu_contrib, color=CATEGORICAL["imu_only"], lw=1.5, label="Losing the IMU branch")
    ax.plot(t, gps_contrib, color=CATEGORICAL["gps_only"], lw=1.5, label="Losing the GPS branch")
    ax.axhline(0, color=INK_MUTED, lw=0.8)
    ymax = max(float(imu_contrib.max()), float(gps_contrib.max()), 0.1)
    ax.fill_between(t, 0, ymax, where=~avail, color=INK_MUTED, alpha=0.15, label="GPS unavailable")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Excess position error when blinded (m)")
    ax.set_title("Causal contribution of each branch (inference-time ablation)")
    ax.grid(True, lw=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_branch_contribution.png", dpi=200)
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


def build_all(results_csv, out_dir, pred_dir, processed_dir="data/processed"):
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

    build_figures(df, Path(pred_dir), out_dir, fig_dir, processed_dir=processed_dir)
    return tables


def build_figures(df, pred_dir, out_dir, fig_dir, processed_dir="data/processed"):
    """Best-effort: each figure is skipped (not failed) with a printed
    reason if the predictions/xai artifacts it needs aren't present yet
    (e.g. a partial sweep) -- doesn't block the tables from being built.
    """
    dt = 1.0
    meta_path = Path(processed_dir) / "meta.json"
    if meta_path.exists():
        import json
        dt = 1.0 / json.loads(meta_path.read_text())["imu_rate_hz"]
    else:
        print(f"fig2: {meta_path} not found, x-axis will be in raw timesteps not seconds")

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
        figure_outage_error_growth(pred_dir, fig_dir, tags, dt=dt)
        print("fig2 saved")
    except FileNotFoundError as e:
        print(f"fig2 skipped: {e}")

    try:
        figure_branch_contribution(out_dir / "xai", fig_dir)
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
    ap.add_argument("--processed_dir", default="data/processed")
    args = ap.parse_args()
    build_all(args.results_csv, args.out_dir, args.pred_dir, args.processed_dir)
