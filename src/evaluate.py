"""GPS-outage sweep: evaluates FusionTransformer / FusionLSTM checkpoints
and the EKF baseline on the same test split, under the same synthetic
outage masks (one shared RNG seed per outage_rate, independent of which
model is being scored) so the comparison in Table 2 is apples-to-apples.

Appends rows to results.csv and saves per-run trajectories to
outputs/predictions/ for report.py's Figure 1/2.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines.ekf import run_ekf  # noqa: E402
from data.dataset import apply_block_outage, make_windows, outage_rng, temporal_split  # noqa: E402
from metrics import trajectory_metrics  # noqa: E402
from models.fusion_transformer import FusionTransformer  # noqa: E402
from train import append_result, build_model, load_processed, MODELS  # noqa: E402


def load_tuned_ekf_params(out_dir):
    """tune_ekf.py's output, if it's been run -- see baselines/tune_ekf.py.
    Falls back to run_ekf's untuned defaults (with a warning) otherwise,
    so evaluate.py still works standalone before tuning has happened.
    """
    path = Path(out_dir) / "ekf_tuned_params.json"
    if not path.exists():
        print(f"WARNING: {path} not found -- run baselines/tune_ekf.py first for a fair "
              f"comparison (README section 5: baselines get the same tuning budget as the "
              f"proposed model). Falling back to untuned defaults.")
        return {}
    params = json.loads(path.read_text())
    return {k: params[k] for k in ("q_accel", "q_gyro", "r_gps")}


def eval_ekf(imu, gps_held, gt_pos, te, outage_rate, dt, estimate_bias=True, ekf_params=None):
    """imu is (N, 6) = [accel_xyz, gyro_xyz] per prepare.py's load_imu."""
    n = te.stop - te.start
    avail, gps_out = apply_block_outage(n, gps_held[te], target_rate=outage_rate, rng=outage_rng(outage_rate))
    accel, gyro = imu[te, :3], imu[te, 3:]
    pred = run_ekf(accel, gyro, gps_out, avail, dt, gps_pos0=gt_pos[te][0], estimate_bias=estimate_bias,
                    **(ekf_params or {}))
    return pred, gt_pos[te]


def eval_neural(model, imu, gps_held, gt_pos, te, outage_rate, window, device):
    n = te.stop - te.start
    avail, gps_out = apply_block_outage(n, gps_held[te], target_rate=outage_rate, rng=outage_rng(outage_rate))
    windows = make_windows(imu[te], gps_out, avail, gt_pos[te], window=window, stride=window)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(windows["imu"])):
            b_imu = torch.from_numpy(windows["imu"][i]).unsqueeze(0).to(device)
            b_gps = torch.from_numpy(windows["gps"][i]).unsqueeze(0).to(device)
            b_mask = torch.from_numpy(windows["gps_mask"][i]).unsqueeze(0).to(device)
            pred = model(b_imu, b_gps, b_mask)
            preds.append(pred.squeeze(0).cpu().numpy())
    pred_all = np.concatenate(preds, axis=0)
    gt_all = windows["gt_pos"].reshape(-1, 3)
    return pred_all, gt_all


def already_done(results_csv, model_name, ablation, seed, train_outage_rate, eval_outage_rate):
    if not Path(results_csv).exists():
        return False
    df = pd.read_csv(results_csv)
    if df.empty or "eval_outage_rate" not in df.columns:
        return False
    match = (
        (df.model == model_name) & (df.seed == seed)
        & (df.get("ablation", "full") == ablation)
        & np.isclose(df.get("train_outage_rate", -1), train_outage_rate)
        & np.isclose(df.eval_outage_rate, eval_outage_rate)
    )
    return match.any()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    imu, gps_held, gt_pos = load_processed(args.processed_dir)
    tr, va, te = temporal_split(len(imu))
    dt = 1.0 / 50.0  # overwritten below once meta.json's real imu_rate_hz is known

    meta_path = Path(args.processed_dir) / "meta.json"
    if meta_path.exists():
        dt = 1.0 / json.loads(meta_path.read_text())["imu_rate_hz"]

    if args.model != "ekf":
        ckpt_path = (Path(args.out_dir) / "checkpoints"
                     / f"{args.model}_{args.ablation}_s{args.seed}_outage{args.train_outage_rate}.pt")
        model = build_model(args.model, args.ablation).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        ekf_params = load_tuned_ekf_params(args.out_dir)

    pred_dir = Path(args.out_dir) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for rate in args.outage_rates:
        if not args.force and already_done(args.results_csv, args.model, args.ablation, args.seed,
                                            args.train_outage_rate, rate):
            print(f"skip: {args.model}/{args.ablation} seed={args.seed} eval_outage_rate={rate} already logged")
            continue

        if args.model == "ekf":
            pred, gt = eval_ekf(imu, gps_held, gt_pos, te, rate, dt,
                                 estimate_bias=(args.ablation != "no_bias"), ekf_params=ekf_params)
        else:
            pred, gt = eval_neural(model, imu, gps_held, gt_pos, te, rate, args.window, device)

        metrics = trajectory_metrics(pred, gt, rpe_delta=10)
        row = {
            "model": args.model, "ablation": args.ablation, "seed": args.seed,
            "train_outage_rate": args.train_outage_rate, "eval_outage_rate": rate,
            **metrics,
        }
        append_result(args.results_csv, row)

        tag = f"{args.model}_{args.ablation}_s{args.seed}_trainoutage{args.train_outage_rate}_evaloutage{rate}"
        np.save(pred_dir / f"{tag}_pred.npy", pred)
        np.save(pred_dir / f"{tag}_gt.npy", gt)
        print(f"eval done: {row}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS) + ["ekf"], required=True)
    ap.add_argument("--ablation", default="full",
                     choices=list(FusionTransformer.ABLATIONS) + ["no_bias"],
                     help="fusion_transformer: no_cross_attn/imu_only/gps_only. "
                          "ekf: no_bias disables online bias estimation. Table 3.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_outage_rate", type=float, default=0.0,
                     help="which trained checkpoint to load (ignored for --model ekf)")
    ap.add_argument("--outage_rates", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.5, 0.7])
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--results_csv", default="outputs/results.csv")
    ap.add_argument("--window", type=int, default=192)
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args())
