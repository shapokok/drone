"""Branch-ablation probing for Figure 3: the CAUSAL contribution of each
input branch, not raw attention weight.

Raw cross-attention weight on the concurrent GPS token was tried first
and gave a counter-intuitive, uninterpretable result: attention on GPS
INCREASED during outage, even though Table 2 shows the model's actual
output correction under outage clearly comes from the IMU branch (that
is where FusionNav's robustness over the EKF baseline comes from).
Attention weight is correlational, not causal -- a well-documented gap
in the interpretability literature (raw attention scores don't reliably
indicate which input actually drove the output).

This probes causally instead: on the same TRAINED "full" checkpoint, at
INFERENCE time (no retraining), blind one branch's input (zero it) and
measure how much WORSE the prediction gets relative to both inputs
present. This is a standard input-ablation / occlusion probe -- excess
error when a branch is blinded is direct evidence of what that branch
was actually contributing, not just what the model looked at.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import apply_block_outage  # noqa: E402
from models.fusion_transformer import FusionTransformer  # noqa: E402
from train import load_processed  # noqa: E402


def branch_ablation_errors(model, imu, gps, gps_mask, gt, device):
    """One window. Returns per-timestep position error (T,) for: the
    normal (both-branch) prediction, IMU-blinded, and GPS-blinded.
    """
    model.eval()
    b_imu = torch.from_numpy(imu).unsqueeze(0).to(device)
    b_gps = torch.from_numpy(gps).unsqueeze(0).to(device)
    b_mask = torch.from_numpy(gps_mask[:, None].astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_full = model(b_imu, b_gps, b_mask)
        pred_imu_blind = model(torch.zeros_like(b_imu), b_gps, b_mask)
        # zeroing gps here also zeroes the residual base (pos_pred = gps +
        # delta), so this is a genuine no-GPS-anywhere probe, not just a
        # blinded embedding -- same fix as the imu_only ablation's bug.
        pred_gps_blind = model(b_imu, torch.zeros_like(b_gps), torch.zeros_like(b_mask))

    def err(pred):
        return np.linalg.norm(pred.squeeze(0).cpu().numpy() - gt, axis=1)

    return err(pred_full), err(pred_imu_blind), err(pred_gps_blind)


def extract(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    imu, gps_held, gt_pos = load_processed(args.processed_dir)

    n = args.window
    start = args.start if args.start is not None else len(imu) - n - 1
    sl = slice(start, start + n)

    rng = np.random.default_rng(0)
    avail, gps_out = apply_block_outage(n, gps_held[sl], target_rate=args.outage_rate, rng=rng)

    model = FusionTransformer().to(device)
    ckpt_path = (Path(args.out_dir) / "checkpoints"
                 / f"fusion_transformer_full_s{args.seed}_outage{args.train_outage_rate}.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    err_full, err_imu_blind, err_gps_blind = branch_ablation_errors(
        model, imu[sl], gps_out, avail, gt_pos[sl], device)

    # excess error caused by losing that branch -- the causal contribution
    imu_contribution = err_imu_blind - err_full
    gps_contribution = err_gps_blind - err_full

    out_dir = Path(args.out_dir) / "xai"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "imu_contribution.npy", imu_contribution)
    np.save(out_dir / "gps_contribution.npy", gps_contribution)
    np.save(out_dir / "gps_availability.npy", avail)
    np.save(out_dir / "err_full.npy", err_full)

    unavail_imu = imu_contribution[~avail].mean() if (~avail).any() else float("nan")
    unavail_gps = gps_contribution[~avail].mean() if (~avail).any() else float("nan")
    print(f"xai ok | window={n} outage_rate={args.outage_rate} | "
          f"IMU contribution avail={imu_contribution[avail].mean():.4f} "
          f"unavail={unavail_imu:.4f} | "
          f"GPS contribution avail={gps_contribution[avail].mean():.4f} "
          f"unavail={unavail_gps:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_outage_rate", type=float, default=0.0)
    ap.add_argument("--outage_rate", type=float, default=0.3,
                     help="synthetic outage injected into the probe window, so the "
                          "figure actually shows a GPS-loss interval")
    ap.add_argument("--window", type=int, default=192)
    ap.add_argument("--start", type=int, default=None)
    extract(ap.parse_args())
