"""Inference-only runner for a pinned R7 export; invoked manually on Kaggle.

Does not import train.py/evaluate.py, append R7 results, tune, or train.
Only writes a NEW outputs/diagnostic_clean_outages directory.
"""
import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from diagnostics.clean_outages import (
    availability_from_blocks, clean_gnss, common_validation_indices,
    constant_velocity_gnss, make_scenarios, raw_errors, raw_metrics,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def verify_r7(root, config):
    """Check exact file bytes BEFORE loading any checkpoint."""
    root = Path(root)
    verified = {}
    for relative, expected in config["r7"]["artifacts"].items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing R7 artifact: {path}")
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"R7 identity mismatch: {path}; expected {expected['sha256']}")
        verified[relative] = {**expected, "resolved_path": str(path.resolve())}
    return verified


def discover_r7(input_root, config):
    roots = sorted({p.parents[3] for p in Path(input_root).rglob("drone/data/processed/t_imu.npy")})
    matches, rejected = [], []
    for root in roots:
        try:
            verified = verify_r7(root, config)
            matches.append((root, verified))
        except (ValueError, FileNotFoundError) as error:
            rejected.append(str(error))
    if not matches:
        detail = "\n".join(rejected) or "No export with drone/data/processed/t_imu.npy was found."
        raise FileNotFoundError(
            "Attach the saved output of shapok/drone-nav-full-run version 7 "
            "(output ID 352644733), retaining its directory structure.\n" + detail
            + "\nRequired paths:\n" + "\n".join(config["r7"]["artifacts"]))
    if len(matches) != 1:
        raise ValueError("Multiple matching R7 mounts; pass --r7-root explicitly: "
                         + ", ".join(str(x[0]) for x in matches))
    return matches[0]


def prepare_validation(root, config):
    """Keep R7 frames, channel order, float32 features and GPS searchsorted.

    GPS order is intentionally NOT silently repaired. Record reversals; clean
    GNSS construction refuses future or reversed selected sources. GT is
    interpolated only inside original support, on the common validation grid.
    """
    data = Path(root) / "drone/data/processed"
    arrays = {name: np.load(data / (name + ".npy"), allow_pickle=False)
              for name in ("t_imu", "imu", "t_gps", "gps_pos", "t_gt", "gt_pos")}
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite R7 input: {name}")
    t, tg, t_gt = arrays["t_imu"], arrays["t_gps"], arrays["t_gt"]
    if arrays["imu"].shape != (len(t), 6) or arrays["gps_pos"].shape != (len(tg), 3):
        raise ValueError("Unexpected R7 input shapes")
    if arrays["gt_pos"].shape != (len(t_gt), 3):
        raise ValueError("Unexpected GT shape")
    split = config["split"]
    if split["name"] != "validation" or len(t) != split["expected_total_imu"]:
        raise ValueError("This locked diagnostic supports only the R7 validation split")
    indices, support = common_validation_indices(
        t, t_gt, split["train_fraction"], split["val_fraction"], config["window"])
    if (support["original_split_indices_half_open"] != split["expected_original_indices"]
            or support["common_indices_half_open"] != split["expected_common_indices"]):
        raise ValueError("Unexpected split/support; do not silently change locked protocol")
    common_t = t[indices].astype(np.float64)
    # EXACT existing R7 numeric resampling, including its unsorted-source risk.
    source = np.clip(np.searchsorted(tg, common_t, side="right") - 1, 0, len(tg) - 1)
    gps = arrays["gps_pos"][source].astype(np.float32)
    gt = np.stack([np.interp(common_t, t_gt, arrays["gt_pos"][:, axis])
                   for axis in range(3)], axis=1).astype(np.float32)
    support.update(gps_timestamp_reversal_count=int(np.sum(np.diff(tg) < 0)),
                   selected_future_source_count=int(np.sum(tg[source] > common_t)),
                   selected_source_time_reversal_count=int(np.sum(np.diff(tg[source]) < 0)),
                   imu_dt_min_s=float(np.diff(common_t).min()),
                   imu_dt_median_s=float(np.median(np.diff(common_t))),
                   imu_dt_max_s=float(np.diff(common_t).max()),
                   source_identity="processed GPS row, not independently verified physical receiver fix")
    return {"t": common_t, "imu": arrays["imu"][indices].astype(np.float32), "gps": gps,
            "gt": gt, "source_indices": source, "source_timestamps": tg[source],
            "original_indices": indices, "support": support}


def infer_windows(model, imu, gps, available, window, batch_windows, device, zero_imu=False):
    """Reset state per nonoverlapping window, no ground-truth input or reset."""
    import torch
    n = len(imu)
    if n % window or batch_windows < 1:
        raise ValueError("all methods require the same complete-window prefix")
    if imu.shape != (n, 6) or gps.shape != (n, 3) or available.shape != (n,):
        raise ValueError("invalid inference input shapes")
    imu_windows = np.asarray(imu, dtype=np.float32).reshape(-1, window, 6)
    gps_windows = np.asarray(gps, dtype=np.float32).reshape(-1, window, 3)
    mask_windows = np.asarray(available, dtype=np.float32).reshape(-1, window, 1)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(imu_windows), batch_windows):
            stop = start + batch_windows
            i = torch.from_numpy(imu_windows[start:stop]).to(device)
            if zero_imu:
                i = torch.zeros_like(i)
            g = torch.from_numpy(gps_windows[start:stop]).to(device)
            m = torch.from_numpy(mask_windows[start:stop]).to(device)
            out = model(i, g, m)
            predictions.append(out.cpu().numpy().reshape(-1, 3))
    result = np.concatenate(predictions)
    if result.shape != (n, 3) or not np.isfinite(result).all():
        raise ValueError("invalid inference result")
    return result


def unique_rows(rows):
    keys = [(r["scenario_id"], r["method"], r["seed"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate evaluation rows")
    return rows


def write_csv(path, rows):
    if not rows:
        raise ValueError("no rows")
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reserve_output(path):
    """No append, resume, force flag, or overwrite of existing diagnostics."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_prediction(directory, pred, data, clean, scenario, metadata):
    """One self-contained bundle per method/scenario; no pickle needed to read."""
    directory.mkdir()
    correction = pred.astype(np.float64) - clean["held_gnss"].astype(np.float64)
    errors = raw_errors(pred, data["gt"])
    np.savez_compressed(
        directory / "prediction.npz", pred=pred, gt=data["gt"], timestamps_s=data["t"],
        original_imu_indices=data["original_indices"],
        scenario_id=np.array(scenario["scenario_id"]), metadata_json=np.array(json.dumps(metadata)),
        correction_xyz_m=correction, correction_norm_m=np.linalg.norm(correction, axis=1),
        **clean, **errors)
    write_json(directory / "metadata.json", metadata)
    selected = ~clean["availability_mask"]
    if not selected.any():
        selected = np.ones(len(pred), dtype=bool)
    norms = np.linalg.norm(correction[selected], axis=1)
    return {"correction_rms_norm_m": float(np.sqrt(np.mean(norms ** 2))),
            "correction_mean_norm_m": float(norms.mean()),
            "correction_max_norm_m": float(norms.max())}


def run(config_path, input_root="/kaggle/input", r7_root=None,
        out_dir="outputs/diagnostic_clean_outages"):
    import torch
    from models.fusion_lstm import FusionLSTM
    from models.fusion_transformer import FusionTransformer

    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    project = Path(__file__).resolve().parents[2]
    for relative, expected in config["r7"]["model_source_sha256"].items():
        if sha256_file(project / relative) != expected:
            raise ValueError(f"R7 architecture source changed: {relative}")
    if r7_root is None:
        root, verified = discover_r7(input_root, config)
    else:
        root = Path(r7_root)
        verified = verify_r7(root, config)
    out_path = Path(out_dir).resolve()
    if root.resolve() == out_path or root.resolve() in out_path.parents:
        raise ValueError("diagnostic output must be outside the R7 export")
    data = prepare_validation(root, config)
    scenarios = make_scenarios(data["t"], config["protocol"])
    locked_path = config_path.with_name("diagnostic_clean_outages_manifest.json")
    locked = json.loads(locked_path.read_text())
    t_hash = hashlib.sha256(data["t"].astype("<f8").tobytes()).hexdigest()
    if scenarios != locked["scenarios"] or t_hash != locked["common_timestamps_float64_le_sha256"]:
        raise ValueError("scenario manifest differs from the timestamp-only preflight")
    if locked["config_sha256"] != sha256_file(config_path):
        raise ValueError("configuration differs from the locked preflight")
    if len({m["id"] for m in config["methods"]}) != len(config["methods"]):
        raise ValueError("duplicate configured methods")

    torch.manual_seed(config["inference"]["seed"])
    np.random.seed(config["inference"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["inference"]["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    for method in config["methods"]:
        checkpoint = method.get("checkpoint")
        if checkpoint and checkpoint not in models:
            model = (FusionTransformer(ablation=method["ablation"])
                     if method["architecture"] == "fusion_transformer" else FusionLSTM())
            state = torch.load(root / checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            models[checkpoint] = model.to(device).eval()
    out = reserve_output(out_path)
    write_json(out / "status.json", {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()})
    try:
        shutil.copyfile(config_path, out / "config.json")
        shutil.copyfile(locked_path, out / "locked_protocol_manifest.json")
        snapshot = out / "code"
        for relative in ("src/diagnostics/__init__.py", "src/diagnostics/clean_outages.py",
                         "src/diagnostics/run_clean_outages.py", *config["r7"]["model_source_sha256"]):
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / relative, destination)
        provenance = {
            "protocol_id": config["protocol_id"], "r7": config["r7"], "verified_artifacts": verified,
            "config_sha256": sha256_file(config_path), "locked_manifest_sha256": sha256_file(locked_path),
            "executed_source_sha256": {str(p.relative_to(snapshot)): sha256_file(p)
                                       for p in sorted(snapshot.rglob("*.py"))},
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "torch": torch.__version__, "cuda": torch.version.cuda, "device": device,
                            "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor()},
            "validation_support": data["support"],
            "claim_limits": ["validation diagnostic, one seed and one flight",
                             "Transformer is noncausal within each 192-sample window",
                             "original coordinate agreement and IMU synchronization unresolved",
                             "source provenance identifies exported GPS rows, not physical receiver packets",
                             "no bitwise guarantee across Torch/CUDA versions"],
        }
        write_json(out / "provenance.json", provenance)
        write_json(out / "protocol_manifest.json", {**locked, "runtime_support": data["support"]})
        rows, pairs = [], []
        for scenario in scenarios:
            sid = scenario["scenario_id"]
            scenario_dir = out / sid
            scenario_dir.mkdir()
            available = availability_from_blocks(len(data["t"]), scenario["unavailable_index_runs"])
            clean = clean_gnss(data["t"], data["gps"], data["source_indices"],
                               data["source_timestamps"], available, scenario["source_blocked_intervals_s"])
            if sid == "val_control" and not np.array_equal(clean["held_gnss"], data["gps"]):
                raise ValueError("control altered original R7 GPS values")
            cv, cv_manifest = constant_velocity_gnss(data["t"], clean, **{
                k: config["constant_velocity"][k] for k in ("history_s", "min_span_s")})
            write_json(scenario_dir / "scenario.json", {**scenario, "constant_velocity_fit": cv_manifest})
            np.savez_compressed(scenario_dir / "inputs.npz", timestamps_s=data["t"], imu=data["imu"],
                                gt=data["gt"], original_imu_indices=data["original_indices"], **clean)
            scenario_rows = {}
            for method in config["methods"]:
                mid = method["id"]
                if mid == "held_gnss":
                    pred = clean["held_gnss"].copy()
                elif mid == "constant_velocity_gnss":
                    pred = cv
                else:
                    pred = infer_windows(models[method["checkpoint"]], data["imu"], clean["held_gnss"],
                                         available, config["window"], config["inference"]["batch_windows"],
                                         device, zero_imu=method.get("zero_imu", False))
                checkpoint = method.get("checkpoint")
                metadata = {"scenario": scenario, "method": method,
                            "checkpoint_sha256": verified[checkpoint]["sha256"] if checkpoint else None,
                            "protocol_id": config["protocol_id"], "r7_output_id": config["r7"]["output_id"],
                            "config_sha256": provenance["config_sha256"], "metric_definition": config["metrics"],
                            "model_semantics": config["inference"], "claim_limits": provenance["claim_limits"]}
                correction = save_prediction(scenario_dir / mid, pred, data, clean, scenario, metadata)
                row = {"protocol_id": config["protocol_id"], "scenario_id": sid, "split": "validation",
                       "episode_id": scenario["episode_id"], "requested_duration_s": scenario["requested_duration_s"],
                       "method": mid, "kind": method["kind"], "seed": method.get("seed"),
                       "checkpoint_sha256": metadata["checkpoint_sha256"],
                       "mask_uint8_sha256": scenario["mask_uint8_sha256"], "n_common": len(data["t"]),
                       **raw_metrics(pred, data["gt"], available), **correction,
                       "prediction_file": f"{sid}/{mid}/prediction.npz"}
                rows.append(row)
                scenario_rows[mid] = row
                print(f"{sid} / {mid}: raw {row['metric_scope']} RMSE3D={row['rmse_3d_m']:.6f} m")
            full = scenario_rows["fusion_full"]
            pair = {"scenario_id": sid, "episode_id": scenario["episode_id"],
                    "requested_duration_s": scenario["requested_duration_s"], "split": "validation",
                    "metric_scope": full["metric_scope"], "n_evaluated": full["n_evaluated"],
                    "seed": 0, "delta_definition": "full minus comparator; negative means smaller full RMSE"}
            for other in ("fusion_gps_only", "held_gnss", "constant_velocity_gnss", "fusion_lstm"):
                pair[f"full_minus_{other}_rmse_3d_m"] = full["rmse_3d_m"] - scenario_rows[other]["rmse_3d_m"]
            pair["sensitivity_zero_imu_minus_full_rmse_3d_m"] = (
                scenario_rows["fusion_full_zero_imu"]["rmse_3d_m"] - full["rmse_3d_m"])
            pairs.append(pair)
        unique_rows(rows)
        if len(rows) != len(scenarios) * len(config["methods"]):
            raise ValueError("incomplete evaluation matrix")
        write_csv(out / "summary.csv", rows)
        write_csv(out / "paired_differences.csv", pairs)
        write_json(out / "status.json", {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
                                         "evaluation_rows": len(rows), "scenarios": len(scenarios),
                                         "training_performed": False, "test_evaluated": False})
        catalog = {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                   for p in sorted(out.rglob("*")) if p.is_file()}
        write_json(out / "artifact_manifest.json", catalog)
        return out
    except Exception as error:
        write_json(out / "status.json", {"status": "failed", "error": str(error),
                                         "training_performed": False, "test_evaluated": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--r7-root")
    parser.add_argument("--out-dir", default="outputs/diagnostic_clean_outages")
    args = parser.parse_args()
    run(args.config, args.input_root, args.r7_root, args.out_dir)


if __name__ == "__main__":
    main()
