"""Small synthetic tests only: no R7 inference, training or Kaggle jobs.

Run: python3 -B -m unittest discover -s tests -p 'test_clean_outages.py' -v
"""
import hashlib
import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from diagnostics.clean_outages import (
    availability_from_blocks, clean_gnss, common_validation_indices,
    constant_velocity_gnss, false_runs, make_scenarios, mask_sha256,
    raw_errors, raw_metrics,
)
from diagnostics.run_clean_outages import (
    discover_r7, infer_windows, prepare_validation, reserve_output,
    run, save_prediction, unique_rows, verify_r7, write_csv,
)


class CleanGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(20, dtype=float)
        self.gps = np.column_stack((self.t, self.t * 2, -self.t)).astype(np.float32)
        self.ids = np.arange(20)

    def clean(self, blocks, gps=None):
        return clean_gnss(self.t, self.gps if gps is None else gps, self.ids,
                          self.t - 0.1, availability_from_blocks(20, blocks))

    def assert_constant_runs(self, clean):
        for a, b in false_runs(clean["availability_mask"]):
            self.assertFalse(clean["gnss_update_mask"][a:b].any())
            np.testing.assert_array_equal(clean["held_gnss"][a:b],
                                          np.repeat(self.gps[a - 1:a], b - a, axis=0))
            self.assertTrue((clean["held_source_index"][a:b] == a - 1).all())
            self.assertTrue((clean["held_source_timestamp_s"][a:b] < self.t[a]).all())
            self.assertTrue((clean["accepted_at_grid_index"][a:b] < a).all())

    def test_single_outage_strict_previous_fix(self):
        result = self.clean([(5, 9)])
        self.assert_constant_runs(result)
        np.testing.assert_array_equal(result["held_gnss"][5], self.gps[4])
        self.assertFalse(np.array_equal(result["held_gnss"][5], self.gps[5]))

    def test_overlaps_both_orders_same_final_mask_and_values(self):
        left = self.clean([(5, 10), (8, 14)])
        right = self.clean([(8, 14), (5, 10)])
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
        self.assertEqual(false_runs(left["availability_mask"]), [(5, 14)])
        self.assert_constant_runs(left)

    def test_nested_blocks(self):
        left = self.clean([(4, 16), (7, 10)])
        right = self.clean([(7, 10), (4, 16)])
        self.assert_constant_runs(left)
        np.testing.assert_array_equal(left["held_gnss"], right["held_gnss"])

    def test_outage_to_record_end(self):
        self.assert_constant_runs(self.clean([(6, 20)]))

    def test_adjacent_blocks_are_one_outage(self):
        result = self.clean([(4, 8), (8, 12)])
        self.assertEqual(false_runs(result["availability_mask"]), [(4, 12)])
        self.assert_constant_runs(result)

    def test_masked_values_cannot_affect_inputs(self):
        blocks = [(5, 9), (12, 20)]
        changed = self.gps.copy()
        changed[~availability_from_blocks(20, blocks)] = 999999
        left, right = self.clean(blocks), self.clean(blocks, changed)
        np.testing.assert_array_equal(left["held_gnss"], right["held_gnss"])

    def test_control_preserves_input(self):
        np.testing.assert_array_equal(self.clean([])["held_gnss"], self.gps)

    def test_cannot_fill_initial_missing_with_gt_or_first_masked_fix(self):
        with self.assertRaisesRegex(ValueError, "no legal pre-outage fix"):
            self.clean([(0, 4)])

    def test_repeated_source_is_not_a_new_measurement(self):
        ids = self.ids // 2
        result = clean_gnss(self.t, self.gps[ids], ids, ids * 2 - 0.1, np.ones(20, bool))
        np.testing.assert_array_equal(np.flatnonzero(result["gnss_update_mask"]), np.arange(0, 20, 2))
        self.assertTrue(result["availability_mask"].all())

    def test_buffered_outage_fix_rejected_at_recovery(self):
        t = np.array([0., 1., 2., 3., 4.])
        st = np.array([0., 0.9, 1.9, 2.4, 3.9])
        gps = np.repeat(np.arange(5.)[:, None], 3, axis=1)
        result = clean_gnss(t, gps, np.arange(5), st, np.array([1, 1, 0, 1, 1], bool), [(2., 2.5)])
        self.assertEqual(result["held_source_index"].tolist(), [0, 1, 1, 1, 4])
        self.assertTrue(result["rejected_buffered_source_mask"][3])
        self.assertTrue(result["availability_mask"][3])
        self.assertFalse(result["gnss_update_mask"][3])

    def test_future_selected_source_fails(self):
        with self.assertRaisesRegex(ValueError, "future GPS"):
            clean_gnss(self.t, self.gps, self.ids, self.t + 0.01, np.ones(20, bool))

    def test_reversed_selected_source_fails(self):
        ids = self.ids.copy()
        ids[9] = 7
        with self.assertRaisesRegex(ValueError, "order reversed"):
            clean_gnss(self.t, self.gps[ids], ids, self.t[ids] - 0.1, np.ones(20, bool))

    def test_source_intervals_must_match_final_mask(self):
        with self.assertRaisesRegex(ValueError, "match the final grid mask"):
            clean_gnss(self.t, self.gps, self.ids, self.t - 0.1,
                        availability_from_blocks(20, [(3, 8)]), [(3, 7)])


class ProtocolTests(unittest.TestCase):
    def test_gt_support_and_tail_crop_shared(self):
        indices, info = common_validation_indices(np.arange(100.), np.array([52., 66.]), .5, .2, 4)
        np.testing.assert_array_equal(indices, np.arange(52, 64))
        self.assertEqual(info["excluded_outside_gt_support_n"], 5)
        self.assertEqual(info["discarded_incomplete_window_n"], 3)
        self.assertEqual(info["common_indices_half_open"], [52, 64])

    def test_irregular_time_duration_and_common_deterministic_starts(self):
        t = 1000 + np.cumsum(np.tile([.11, .21, .33, .17], 1200))
        config = {"durations_s": [10, 30, 60], "start_fractions": [.25, .5, .75],
                  "warmup_s": 30, "recovery_s": 20, "generator_seed": 0}
        scenarios = make_scenarios(t, config)
        self.assertEqual(scenarios, make_scenarios(t.copy(), config))
        self.assertEqual(len(scenarios), 10)
        for scenario in scenarios[1:]:
            a, b = scenario["unavailable_index_runs"][0]
            d = scenario["requested_duration_s"]
            self.assertLess(t[b - 1] - t[a], d)
            self.assertGreaterEqual(t[b] - t[a], d)
            self.assertGreaterEqual(t[a] - t[0], 30)
            self.assertGreaterEqual(t[-1] - t[b], 20)
            self.assertNotEqual(b - a, round(d / .1))
        for episode in (1, 2, 3):
            self.assertEqual(len({s["onset_s"] for s in scenarios[1:] if s["episode_id"] == episode}), 1)

    def test_too_short_protocol_fails_instead_of_changing_starts(self):
        config = {"durations_s": [60], "start_fractions": [.25, .5, .75],
                  "warmup_s": 30, "recovery_s": 20, "generator_seed": 0}
        with self.assertRaisesRegex(ValueError, "too short"):
            make_scenarios(np.arange(100.), config)

    def test_locked_manifest_identity_masks_and_common_counts(self):
        config_path = PROJECT / "configs/diagnostic_clean_outages.json"
        manifest = json.loads((PROJECT / "configs/diagnostic_clean_outages_manifest.json").read_text())
        self.assertEqual(manifest["config_sha256"], hashlib.sha256(config_path.read_bytes()).hexdigest())
        self.assertEqual(len(manifest["scenarios"]), 10)
        for scenario in manifest["scenarios"]:
            mask = availability_from_blocks(scenario["n_common"], scenario["unavailable_index_runs"])
            self.assertEqual(scenario["n_common"], 2688)
            self.assertEqual(mask_sha256(mask), scenario["mask_uint8_sha256"])
            self.assertEqual(int((~mask).sum()), scenario["n_unavailable"])

    def test_prepare_validation_does_not_extrapolate_gt(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "drone/data/processed"
            data.mkdir(parents=True)
            arrays = {"t_imu": np.arange(100.), "imu": np.zeros((100, 6)),
                      "t_gps": np.arange(100.) - .1, "gps_pos": np.zeros((100, 3)),
                      "t_gt": np.array([52., 66.]), "gt_pos": np.array([[0, 0, 0], [14, 28, 42]])}
            for name, array in arrays.items():
                np.save(data / (name + ".npy"), array)
            config = {"window": 4, "split": {"name": "validation", "train_fraction": .5,
                      "val_fraction": .2, "expected_total_imu": 100,
                      "expected_original_indices": [50, 70], "expected_common_indices": [52, 64]}}
            result = prepare_validation(temporary, config)
            np.testing.assert_array_equal(result["t"], np.arange(52., 64.))
            np.testing.assert_allclose(result["gt"][-1], [11, 22, 33])


class VelocityAndMetricsTests(unittest.TestCase):
    def test_cv_velocity_anchor_source_time_and_future_independence(self):
        t = np.arange(25.)
        st = t - .25
        gps = np.column_stack((2 * st, -st, .5 * st)).astype(np.float32)
        mask = availability_from_blocks(len(t), [(10, 18)])
        clean = clean_gnss(t, gps, np.arange(len(t)), st, mask)
        pred, records = constant_velocity_gnss(t, clean)
        np.testing.assert_allclose(records[0]["velocity_m_s"], [2, -1, .5])
        np.testing.assert_allclose(pred[10:18], np.column_stack((2 * t, -t, .5 * t))[10:18])
        changed = gps.copy()
        changed[10:] = 1e6
        pred_changed, records_changed = constant_velocity_gnss(t, clean_gnss(t, changed, np.arange(len(t)), st, mask))
        np.testing.assert_array_equal(pred[10:18], pred_changed[10:18])
        self.assertEqual(records, records_changed)
        self.assertTrue(all(x < 10 for x in records[0]["history_grid_indices"]))

    def test_cv_fallback_with_insufficient_unique_history(self):
        t = np.arange(20.)
        clean = clean_gnss(t, np.ones((20, 3)), np.zeros(20, int), np.zeros(20),
                           availability_from_blocks(20, [(4, 10)]))
        pred, records = constant_velocity_gnss(t, clean)
        self.assertTrue(records[0]["zero_velocity_fallback"])
        np.testing.assert_array_equal(pred, np.ones((20, 3)))

    def test_cv_deduplicates_source_timestamps(self):
        t = np.arange(12.)
        st = np.floor(t / 2) * 2
        pos = np.repeat(st[:, None], 3, axis=1)
        clean = clean_gnss(t, pos, np.arange(12), st, availability_from_blocks(12, [(8, 12)]))
        _, records = constant_velocity_gnss(t, clean)
        self.assertEqual(records[0]["history_source_timestamps_s"], [4., 6.])
        np.testing.assert_allclose(records[0]["velocity_m_s"], [1, 1, 1])

    def test_raw_metrics_outage_only_and_last_unavailable(self):
        pred = np.array([[999, 999, 999], [3, 4, 12], [0, 0, 5], [999, 999, 999.]])
        row = raw_metrics(pred, np.zeros_like(pred), np.array([1, 0, 0, 1], bool))
        self.assertEqual(row["n_evaluated"], 2)
        self.assertAlmostEqual(row["rmse_3d_m"], np.sqrt(97))
        self.assertAlmostEqual(row["rmse_horizontal_m"], np.sqrt(12.5))
        self.assertAlmostEqual(row["rmse_vertical_m"], np.sqrt(84.5))
        self.assertEqual(row["final_unavailable_error_3d_m"], 5)
        self.assertEqual(row["final_unavailable_error_horizontal_m"], 0)
        self.assertEqual(row["final_unavailable_abs_vertical_m"], 5)

    def test_control_raw_no_alignment_and_no_fake_outage_endpoint(self):
        row = raw_metrics(np.ones((3, 3)), np.zeros((3, 3)), np.ones(3, bool))
        self.assertAlmostEqual(row["rmse_3d_m"], np.sqrt(3))
        self.assertEqual(row["metric_scope"], "whole_common_trace")
        self.assertIsNone(row["final_unavailable_error_3d_m"])

    def test_nonfinite_metrics_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            raw_metrics(np.array([[np.nan, 0, 0]]), np.zeros((1, 3)), np.ones(1, bool))


class ArtifactAndSmokeTests(unittest.TestCase):
    def test_end_to_end_runner_on_tiny_synthetic_fixture_only(self):
        """Exercise manifests, 60 unique rows and bundles, never a real checkpoint."""
        import torch
        from models.fusion_transformer import FusionTransformer
        from models.fusion_lstm import FusionLSTM
        torch.manual_seed(0)
        torch.set_num_threads(1)
        full = FusionTransformer(d_model=8, n_heads=2, n_layers=1, dropout=0)
        gps_only = FusionTransformer(d_model=8, n_heads=2, n_layers=1, dropout=0, ablation="gps_only")
        lstm = FusionLSTM(hidden=8, n_layers=1, dropout=0)
        def synthetic_state(path, **kwargs):
            name = str(path)
            return (lstm if "fusion_lstm" in name else gps_only if "gps_only" in name else full).state_dict()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "r7_fixture"
            data_dir = root / "drone/data/processed"
            data_dir.mkdir(parents=True)
            t = np.arange(100.)
            arrays = {"t_imu": t, "imu": np.zeros((100, 6), np.float32),
                      "t_gps": t - .1, "gps_pos": np.repeat((t - .1)[:, None], 3, axis=1).astype(np.float32),
                      "t_gt": np.array([0., 99.]), "gt_pos": np.array([[0., 0., 0.], [99., 99., 99.]])}
            for name, array in arrays.items():
                np.save(data_dir / (name + ".npy"), array)
            (data_dir / "meta.json").write_text('{}')
            config = json.loads((PROJECT / "configs/diagnostic_clean_outages.json").read_text())
            config.update(protocol_id="synthetic_unit_test_only", window=8)
            config["split"].update(train_fraction=.5, val_fraction=.3, expected_total_imu=100,
                                   expected_original_indices=[50, 80], expected_common_indices=[50, 74])
            config["protocol"].update(durations_s=[1, 2, 3], warmup_s=1, recovery_s=1)
            for relative in config["r7"]["artifacts"]:
                path = root / relative
                if relative.endswith('.pt'):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"synthetic placeholder; loading is mocked; not weights")
                raw = path.read_bytes()
                config["r7"]["artifacts"][relative] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            config_path = base / "config.json"
            config_path.write_text(json.dumps(config))
            manifest = {"scenarios": make_scenarios(t[50:74], config["protocol"]),
                        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                        "common_timestamps_float64_le_sha256": hashlib.sha256(t[50:74].astype('<f8').tobytes()).hexdigest()}
            (base / "diagnostic_clean_outages_manifest.json").write_text(json.dumps(manifest))
            out = base / "diagnostic_output"
            with patch("models.fusion_transformer.FusionTransformer", side_effect=lambda ablation: gps_only if ablation == "gps_only" else full), \
                    patch("models.fusion_lstm.FusionLSTM", return_value=lstm), \
                    patch("torch.load", side_effect=synthetic_state), \
                    patch("torch.cuda.is_available", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                run(config_path, r7_root=root, out_dir=out)
            with (out / "summary.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 60)
            unique_rows(rows)
            self.assertEqual(json.loads((out / "status.json").read_text())["status"], "complete")
            for row in rows:
                with np.load(out / row["prediction_file"], allow_pickle=False) as bundle:
                    self.assertEqual(len(bundle["timestamps_s"]), 24)
                    expected = raw_metrics(bundle["pred"], bundle["gt"], bundle["availability_mask"])
                    self.assertAlmostEqual(float(row["rmse_3d_m"]), expected["rmse_3d_m"], places=12)
                    np.testing.assert_array_equal(bundle["timestamps_s"], t[50:74])
                    self.assertEqual(mask_sha256(bundle["availability_mask"]), row["mask_uint8_sha256"])
            catalog = json.loads((out / "artifact_manifest.json").read_text())
            for relative, info in catalog.items():
                self.assertEqual(hashlib.sha256((out / relative).read_bytes()).hexdigest(), info["sha256"])
            with (out / "paired_differences.csv").open() as stream:
                pairs = list(csv.DictReader(stream))
            self.assertEqual(len(pairs), 10)
            for pair in pairs:
                subset = {r["method"]: r for r in rows if r["scenario_id"] == pair["scenario_id"]}
                delta = float(subset["fusion_full"]["rmse_3d_m"]) - float(subset["fusion_gps_only"]["rmse_3d_m"])
                self.assertAlmostEqual(float(pair["full_minus_fusion_gps_only_rmse_3d_m"]), delta)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = reserve_output(Path(temporary) / "diagnostic")
            (path / "sentinel").write_text("unchanged")
            with self.assertRaises(FileExistsError):
                reserve_output(path)
            self.assertEqual((path / "sentinel").read_text(), "unchanged")

    def test_duplicate_evaluation_rows_rejected(self):
        row = {"scenario_id": "a", "method": "b", "seed": 0}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            unique_rows([row, row.copy()])

    def test_artifact_identity_missing_and_tampered(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mounted"
            file = root / "drone/data/processed/t_imu.npy"
            file.parent.mkdir(parents=True)
            file.write_bytes(b"synthetic")
            config = {"r7": {"artifacts": {"drone/data/processed/t_imu.npy": {
                "bytes": 9, "sha256": hashlib.sha256(b"synthetic").hexdigest()}}}}
            self.assertEqual(discover_r7(temporary, config)[0], root)
            file.write_bytes(b"tampered!")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                verify_r7(root, config)
            file.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Missing R7"):
                verify_r7(root, config)

    def test_synthetic_models_and_self_contained_bundle(self):
        import torch
        from models.fusion_transformer import FusionTransformer
        from models.fusion_lstm import FusionLSTM
        torch.manual_seed(0)
        torch.set_num_threads(1)
        n, window = 16, 8
        t = np.arange(n, dtype=float)
        imu = np.ones((n, 6), np.float32)
        gps = np.repeat(t[:, None], 3, axis=1).astype(np.float32)
        mask = availability_from_blocks(n, [(5, 11)])
        clean = clean_gnss(t, gps, np.arange(n), t, mask)
        tiny_models = [FusionTransformer(d_model=8, n_heads=2, n_layers=1, dropout=0),
                       FusionTransformer(d_model=8, n_heads=2, n_layers=1, dropout=0, ablation="gps_only"),
                       FusionLSTM(hidden=8, n_layers=1, dropout=0)]
        for model in tiny_models:
            for zero in (False, True):
                pred = infer_windows(model, imu, clean["held_gnss"], mask, window, 1, "cpu", zero_imu=zero)
                self.assertEqual(pred.shape, (n, 3))
                self.assertTrue(np.isfinite(pred).all())
                with torch.inference_mode():
                    ref = model(torch.from_numpy(imu[:8] * (0 if zero else 1))[None],
                                torch.from_numpy(clean["held_gnss"][:8])[None],
                                torch.from_numpy(mask[:8].astype(np.float32))[None, :, None]).numpy()[0]
                np.testing.assert_allclose(pred[:8], ref, rtol=1e-5, atol=1e-6)
        with tempfile.TemporaryDirectory() as temporary:
            data = {"gt": gps, "t": t, "original_indices": np.arange(n)}
            scenario = {"scenario_id": "synthetic_test_only"}
            metadata = {"checkpoint": None, "synthetic_random_model": True}
            save_prediction(Path(temporary) / "bundle", pred, data, clean, scenario, metadata)
            with np.load(Path(temporary) / "bundle/prediction.npz", allow_pickle=False) as bundle:
                for key in ("pred", "gt", "timestamps_s", "availability_mask", "held_source_index",
                            "error_3d_m", "correction_xyz_m", "scenario_id", "metadata_json"):
                    self.assertIn(key, bundle.files)
                np.testing.assert_array_equal(bundle["availability_mask"], mask)
                np.testing.assert_allclose(bundle["error_3d_m"], raw_errors(pred, gps)["error_3d_m"])
                self.assertEqual(json.loads(str(bundle["metadata_json"])), metadata)
            csv_path = Path(temporary) / "summary.csv"
            write_csv(csv_path, [{"scenario_id": "synthetic", **raw_metrics(pred, gps, mask)}])
            self.assertEqual(len(csv_path.read_text().splitlines()), 2)
            with self.assertRaises(FileExistsError):
                write_csv(csv_path, [{"scenario_id": "duplicate"}])


if __name__ == "__main__":
    unittest.main()
