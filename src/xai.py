"""Attention-weight extraction for Figure 3 (branch reliance vs GPS
availability). Only meaningful for FusionTransformer -- FusionLSTM's
gated fusion has no attention map to extract, so it isn't used here.

For each IMU query timestep t, reports the attention mass placed on the
concurrent GPS key at t (averaged over heads and cross-attention layers).
This is a standard "self-position attention" probe: high weight means the
model is leaning on the (possibly stale/frozen) GPS input at that instant;
low weight is the proxy for "falling back on IMU reasoning."
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import apply_block_outage, resample_gps_to_grid  # noqa: E402
from models.fusion_transformer import FusionTransformer  # noqa: E402
from train import load_processed  # noqa: E402


def concurrent_attention_weight(attn_maps):
    """attn_maps: list of (B, T, T) tensors, one per cross-attention layer
    (already head-averaged by nn.MultiheadAttention's average_attn_weights).
    Returns (B, T): mean over layers of the diagonal (query t -> key t).
    """
    diag_per_layer = [a.diagonal(dim1=-2, dim2=-1) for a in attn_maps]  # (B, T) each
    return torch.stack(diag_per_layer, dim=0).mean(dim=0)


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
    model.eval()

    b_imu = torch.from_numpy(imu[sl]).unsqueeze(0).to(device)
    b_gps = torch.from_numpy(gps_out).unsqueeze(0).to(device)
    b_mask = torch.from_numpy(avail[:, None].astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        _, attn_maps = model(b_imu, b_gps, b_mask, return_attn=True)
    weight = concurrent_attention_weight(attn_maps).squeeze(0).cpu().numpy()

    out_dir = Path(args.out_dir) / "xai"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "attention_weight.npy", weight)
    np.save(out_dir / "gps_availability.npy", avail)
    print(f"xai ok | window={n} outage_rate={args.outage_rate} "
          f"mean_attn(available)={weight[avail].mean():.4f} "
          f"mean_attn(unavailable)={weight[~avail].mean() if (~avail).any() else float('nan'):.4f}")


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
