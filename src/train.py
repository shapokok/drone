"""Train FusionTransformer / FusionLSTM on the processed Zurich MAV arrays.

Every finished (model, seed, train_outage_rate) triple appends one row to
results.csv; re-running the same command skips what's already there
(--force overrides) so a Kaggle 12h timeout costs one run, not the batch.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import (  # noqa: E402
    apply_block_outage, make_windows, resample_gps_to_grid, temporal_split,
    TrajectoryWindowDataset,
)
from metrics import trajectory_metrics  # noqa: E402
from models.fusion_lstm import FusionLSTM  # noqa: E402
from models.fusion_transformer import FusionTransformer  # noqa: E402

MODELS = {"fusion_transformer": FusionTransformer, "fusion_lstm": FusionLSTM}


def build_model(name, ablation="full"):
    if name == "fusion_transformer":
        return FusionTransformer(ablation=ablation)
    if ablation != "full":
        raise ValueError(
            f"{name} has no ablation variants of its own -- it IS the "
            f"LSTM-backbone ablation arm of fusion_transformer"
        )
    return FusionLSTM()


def load_processed(processed_dir):
    processed_dir = Path(processed_dir)
    t_imu = np.load(processed_dir / "t_imu.npy")
    imu = np.load(processed_dir / "imu.npy")
    t_gps = np.load(processed_dir / "t_gps.npy")
    gps_pos = np.load(processed_dir / "gps_pos.npy")
    t_gt = np.load(processed_dir / "t_gt.npy")
    gt_pos = np.load(processed_dir / "gt_pos.npy")

    gps_held, _native_mask = resample_gps_to_grid(t_imu, t_gps, gps_pos)
    # Ground truth (~1 Hz on the real dataset) is SPARSER than the IMU
    # grid (~10 Hz), not denser -- nearest-hold would train the model
    # against a staircase target. The true trajectory is smooth between
    # fixes, so linear interpolation is the honest choice here.
    gt_on_grid = np.stack(
        [np.interp(t_imu, t_gt, gt_pos[:, i]) for i in range(gt_pos.shape[1])], axis=1
    )
    return imu.astype(np.float32), gps_held.astype(np.float32), gt_on_grid.astype(np.float32)


def build_loaders(imu, gps_held, gt_pos, window, train_stride, batch_size, train_outage_rate, seed):
    """Train windows overlap (denser augmentation); val/test windows tile
    the split back-to-back (stride=window) so concatenating them for
    ATE/RPE reconstructs one temporally contiguous trajectory per split.
    """
    rng = np.random.default_rng(seed)
    tr, va, te = temporal_split(len(imu))
    loaders = {}
    for name, sl in (("train", tr), ("val", va), ("test", te)):
        n = sl.stop - sl.start
        rate = train_outage_rate if name == "train" else 0.0
        avail, gps_out = apply_block_outage(n, gps_held[sl], target_rate=rate, rng=rng)
        stride = train_stride if name == "train" else window
        windows = make_windows(imu[sl], gps_out, avail, gt_pos[sl], window=window, stride=stride)
        ds = TrajectoryWindowDataset(windows)
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"),
                                    drop_last=(name == "train"))
    return loaders


def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    huber = torch.nn.HuberLoss()
    total, n = 0.0, 0
    for batch in loader:
        imu = batch["imu"].to(device)
        gps = batch["gps"].to(device)
        mask = batch["gps_mask"].to(device)
        gt = batch["gt_pos"].to(device)
        with torch.set_grad_enabled(train):
            pred = model(imu, gps, mask)
            loss = huber(pred, gt)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += loss.item() * len(imu)
        n += len(imu)
    return total / max(n, 1)


@torch.no_grad()
def evaluate_trajectory(model, loader, device):
    model.eval()
    preds, gts = [], []
    for batch in loader:
        pred = model(batch["imu"].to(device), batch["gps"].to(device), batch["gps_mask"].to(device))
        preds.append(pred.cpu().numpy())
        gts.append(batch["gt_pos"].numpy())
    pred_all = np.concatenate(preds, axis=0).reshape(-1, 3)
    gt_all = np.concatenate(gts, axis=0).reshape(-1, 3)
    return trajectory_metrics(pred_all, gt_all, rpe_delta=10)


def already_done(results_csv, model_name, seed, train_outage_rate, ablation):
    if not Path(results_csv).exists():
        return False
    df = pd.read_csv(results_csv)
    if df.empty:
        return False
    match = (
        (df.model == model_name) & (df.seed == seed)
        & np.isclose(df.train_outage_rate, train_outage_rate)
        & (df.get("ablation", "full") == ablation)
    )
    return match.any()


def append_result(results_csv, row):
    """train.py and evaluate.py log different column sets to the same
    file (epochs_run/train_time_s vs eval_outage_rate). A blind
    mode="a" append writes by column POSITION, not name, so mixing the
    two silently shifts values into the wrong columns the moment the
    schemas diverge -- reconcile by column name instead, even though
    that costs a full rewrite per call (results.csv stays small).
    """
    new_row = pd.DataFrame([row])
    path = Path(results_csv)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_row], ignore_index=True, sort=False)
    else:
        combined = new_row
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)


def train_one(args, seed):
    if not args.force and already_done(args.results_csv, args.model, seed,
                                        args.train_outage_rate, args.ablation):
        print(f"skip: {args.model}/{args.ablation} seed={seed} "
              f"train_outage_rate={args.train_outage_rate} already in {args.results_csv}")
        return

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    imu, gps_held, gt_pos = load_processed(args.processed_dir)
    loaders = build_loaders(imu, gps_held, gt_pos, args.window, args.stride,
                             args.batch_size, args.train_outage_rate, seed)

    model = build_model(args.model, args.ablation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * max(len(loaders["train"]), 1))

    best_val = float("inf")
    patience_left = args.patience
    ckpt_dir = Path(args.out_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{args.model}_{args.ablation}_s{seed}_outage{args.train_outage_rate}.pt"

    t0 = time.time()
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, loaders["train"], device, optimizer)
        for _ in range(len(loaders["train"])):
            scheduler.step()
        val_loss = run_epoch(model, loaders["val"], device, optimizer=None)
        print(f"[{args.model} s{seed}] epoch {epoch+1}/{args.epochs} "
              f"train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_left = args.patience
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    metrics = evaluate_trajectory(model, loaders["test"], device)

    row = {
        "model": args.model, "ablation": args.ablation, "seed": seed,
        "train_outage_rate": args.train_outage_rate,
        "epochs_run": epoch + 1, "train_time_s": round(time.time() - t0, 1),
        **metrics,
    }
    append_result(args.results_csv, row)
    print(f"done: {row}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--ablation", choices=FusionTransformer.ABLATIONS, default="full",
                     help="fusion_transformer only -- Table 3's no_cross_attn/imu_only/gps_only arms")
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--results_csv", default="outputs/results.csv")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--window", type=int, default=192)
    ap.add_argument("--stride", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--train_outage_rate", type=float, default=0.0,
                     help="synthetic GPS-outage rate injected during TRAINING "
                          "(robustness training); evaluation outage sweep is evaluate.py's job")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seeds = args.seeds or ([args.seed] if args.seed is not None else [0])
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for s in seeds:
        train_one(args, s)
