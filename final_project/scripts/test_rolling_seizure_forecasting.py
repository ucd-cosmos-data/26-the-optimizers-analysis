"""Regression tests for the configurable rolling EEG forecasting module."""

from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

try:
    from . import rolling_seizure_forecasting as rsf
except ImportError:
    import rolling_seizure_forecasting as rsf


class DeterministicHazardModel:
    """Small predict-only model for probability and interpretation tests."""

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        logit = -3.0 + 0.6 * values[:, -2] + 0.2 * values[:, 0]
        positive = 1.0 / (1.0 + np.exp(-logit))
        return np.column_stack([1.0 - positive, positive])


def feature_frame(rows: int) -> pd.DataFrame:
    values = np.zeros((rows, len(rsf.MODEL_FEATURE_COLUMNS)), dtype=float)
    values[:, 0] = np.linspace(-1.0, 1.0, rows)
    return pd.DataFrame(values, columns=rsf.MODEL_FEATURE_COLUMNS)


class RollingForecastTests(unittest.TestCase):
    def test_feature_set_is_time_domain_only(self) -> None:
        forbidden = ("delta", "theta", "alpha", "beta", "gamma", "power")
        joined = " ".join(rsf.MICRO_FEATURE_NAMES).lower()
        self.assertFalse(any(token in joined for token in forbidden))
        self.assertEqual(len(rsf.MICRO_FEATURE_NAMES), 23)
        self.assertIn("log_rms", rsf.MICRO_FEATURE_NAMES)
        self.assertIn("log_rms_channel_iqr", rsf.MICRO_FEATURE_NAMES)
        self.assertIn("log_rms_channel_p90", rsf.MICRO_FEATURE_NAMES)
        self.assertIn(
            "median_absolute_channel_correlation", rsf.MICRO_FEATURE_NAMES
        )
        self.assertIn("usable_channel_fraction", rsf.MICRO_FEATURE_NAMES)

    def test_config_is_dynamic_and_rejects_invalid_values(self) -> None:
        config = rsf.ForecastConfig(
            bin_seconds=10,
            horizon_seconds=120,
            context_seconds=120,
            pre_onset_seconds=240,
        )
        config.validate()
        self.assertEqual(config.n_bins, 12)
        self.assertEqual(config.context_bins, 12)
        self.assertEqual(config.pre_onset_steps, 24)
        self.assertEqual(config.reading_count, 25)

        with self.assertRaisesRegex(
            ValueError, "horizon_seconds must be positive"
        ):
            rsf.ForecastConfig(horizon_seconds=0).validate()
        with self.assertRaisesRegex(ValueError, "divisible"):
            rsf.ForecastConfig(bin_seconds=7).validate()
        with self.assertRaisesRegex(ValueError, "exactly 120"):
            rsf.ForecastConfig(context_seconds=60).validate()
        one_channel = rsf.ForecastConfig(
            included_eeg_channels=("Fp1",)
        )
        one_channel.validate()
        self.assertEqual(one_channel.selected_eeg_channels, ("FP1",))
        self.assertEqual(one_channel.effective_min_eeg_channels, 1)
        with self.assertRaisesRegex(ValueError, "Unknown included"):
            rsf.ForecastConfig(
                included_eeg_channels=("NOT_A_CHANNEL",)
            ).validate()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            rsf.ForecastConfig(
                included_eeg_channels=("Cz", "CZ")
            ).validate()

    def test_channel_selection_is_canonical_ordered_and_strict(self) -> None:
        labels = ["EEG Fp1", "ECG", "EEG CZ", "EEG P9"]
        indices, selected = rsf._select_eeg_channel_indices(
            labels, ("cz", "FP1")
        )
        self.assertEqual(indices, [2, 0])
        self.assertEqual(selected, ["CZ", "FP1"])
        with self.assertRaisesRegex(ValueError, "missing"):
            rsf._select_eeg_channel_indices(labels, ("P10",))

    def test_single_selected_channel_produces_finite_features(self) -> None:
        config = rsf.ForecastConfig(
            included_eeg_channels=("FP1",),
            target_sample_rate=64,
        )
        samples = np.sin(
            np.linspace(0, 20 * np.pi, config.bin_seconds * 64)
        )[None, :]
        features = rsf.segment_micro_features(samples, 64.0, config)
        self.assertEqual(features.shape, (1, len(rsf.MICRO_FEATURE_NAMES)))
        self.assertTrue(np.isfinite(features).all())

    def test_dynamic_horizon_probabilities_sum_to_one(self) -> None:
        config = rsf.ForecastConfig(
            bin_seconds=10,
            horizon_seconds=120,
            context_seconds=120,
            pre_onset_seconds=240,
        )
        hazards, pmf, no_event = rsf.predict_horizon_distribution(
            DeterministicHazardModel(), feature_frame(7), config
        )
        self.assertEqual(hazards.shape, (7, 12))
        self.assertEqual(pmf.shape, (7, 12))
        self.assertEqual(no_event.shape, (7,))
        self.assertTrue(
            all(rsf.probability_invariants(hazards, pmf, no_event).values())
        )

    def test_two_stage_model_preserves_probability_mass(self) -> None:
        config = rsf.ForecastConfig()
        frame = feature_frame(6)
        frame["episode_id"] = ["a"] * 3 + ["b"] * 3
        frame["landmark_step"] = [0, 1, 2, 0, 1, 2]
        risk_column = rsf.MODEL_FEATURE_COLUMNS[0]
        risk_model = DummyClassifier(strategy="prior").fit(
            frame[[risk_column]], np.array([0, 0, 0, 1, 1, 1])
        )
        model = rsf.TwoStageForecastModel(
            timing_model=DeterministicHazardModel(),
            risk_model=risk_model,
            risk_feature_columns=(risk_column,),
            smoothing_alpha=0.1,
        )
        hazards, pmf, no_event = rsf.predict_horizon_distribution(
            model, frame, config
        )
        self.assertEqual(pmf.shape, (6, config.n_bins))
        self.assertTrue(
            all(rsf.probability_invariants(hazards, pmf, no_event).values())
        )

    def test_requested_allocation_is_exact_and_event_disjoint(self) -> None:
        rows: list[dict[str, object]] = []
        patients = sorted(
            set().union(
                *(
                    counts.keys()
                    for counts in rsf.REQUESTED_SEIZURE_SPLITS.values()
                )
            )
        )
        for patient_id in patients:
            count = sum(
                split.get(patient_id, 0)
                for split in rsf.REQUESTED_SEIZURE_SPLITS.values()
            )
            for index in range(count):
                rows.append(
                    {
                        "patient_id": patient_id,
                        "episode_id": (
                            f"{patient_id}_S{index + 1:02d}_preictal"
                        ),
                        "source_event_id": f"{patient_id}_S{index + 1:02d}",
                        "episode_type": "preictal",
                        "recording": f"{patient_id}-1.edf",
                    }
                )
                rows.append(
                    {
                        "patient_id": patient_id,
                        "episode_id": (
                            f"{patient_id}_interictal_{index + 1:02d}"
                        ),
                        "source_event_id": "",
                        "episode_type": "interictal",
                        "recording": f"{patient_id}-1.edf",
                    }
                )
        assigned, allocation = rsf.assign_requested_splits(pd.DataFrame(rows))
        self.assertEqual(
            allocation.groupby("dataset_split").size().to_dict(),
            {"test": 8, "train": 39},
        )
        self.assertEqual(allocation["source_event_id"].nunique(), 47)
        self.assertFalse(allocation["source_event_id"].duplicated().any())
        self.assertFalse(assigned["dataset_split"].eq("").any())
        development, test, split = rsf.split_development_test(
            pd.DataFrame(
                {
                    **assigned.to_dict(orient="list"),
                    "has_event_in_horizon": assigned["episode_type"]
                    .eq("preictal")
                    .astype(int),
                }
            )
        )
        self.assertEqual(
            development.loc[
                development["episode_type"].eq("preictal"), "episode_id"
            ].nunique(),
            39,
        )
        self.assertEqual(
            test.loc[test["episode_type"].eq("preictal"), "episode_id"
            ].nunique(),
            8,
        )
        self.assertIn("out-of-fold", split["design"])

    def test_split_only_cache_change_reuses_features_and_relabels_rows(self) -> None:
        config = rsf.ForecastConfig()
        manifest = pd.DataFrame(
            [
                {
                    "episode_id": "PN00_S01_preictal",
                    "patient_id": "PN00",
                    "source_event_id": "PN00_S01",
                    "episode_type": "preictal",
                    "recording": "PN00-1.edf",
                    "edf_path": "/tmp/PN00-1.edf",
                    "anchor_seconds": 100.0,
                    "training_episode_end_seconds": 400.0,
                    "event_onset_seconds": 400.0,
                    "dataset_split": "train",
                }
            ]
        )
        changed = manifest.copy()
        changed["dataset_split"] = "validation"
        self.assertTrue(
            rsf._feature_signature_matches(
                rsf._cache_signature(manifest, config),
                rsf._cache_signature(changed, config),
            )
        )
        landmarks = pd.DataFrame(
            {
                "episode_id": ["PN00_S01_preictal"],
                "dataset_split": ["train"],
            }
        )
        relabeled = rsf._attach_current_dataset_splits(landmarks, changed)
        self.assertEqual(relabeled["dataset_split"].tolist(), ["validation"])

    def test_alarm_needs_two_highs_and_releases_after_65_seconds(self) -> None:
        high_risk = np.array([0.7, 0.8], dtype=float)
        lows = np.full(13, 0.1, dtype=float)
        raw, alarm = rsf.alarm_state_from_risk(
            np.r_[high_risk, lows], threshold=0.5
        )
        self.assertTrue(raw[:2].all())
        self.assertFalse(alarm[0])
        self.assertTrue(alarm[1])
        self.assertTrue(alarm[-2])
        self.assertFalse(alarm[-1])

        _, alarm_before_release = rsf.alarm_state_from_risk(
            np.r_[high_risk, np.full(12, 0.1)], threshold=0.5
        )
        self.assertTrue(alarm_before_release[-1])

    def test_refractory_period_suppresses_immediate_rearming(self) -> None:
        risk = np.array(
            [0.8, 0.8, 0.1, 0.1, 0.8, 0.8, 0.8, 0.8],
            dtype=float,
        )
        _, without_refractory = rsf.alarm_state_from_risk(
            risk,
            threshold=0.5,
            alarm_on_consecutive=2,
            alarm_off_consecutive=2,
            refractory_consecutive=0,
        )
        _, with_refractory = rsf.alarm_state_from_risk(
            risk,
            threshold=0.5,
            alarm_on_consecutive=2,
            alarm_off_consecutive=2,
            refractory_consecutive=4,
        )
        self.assertTrue(without_refractory[-1])
        self.assertFalse(with_refractory[-1])

    def test_patient_normalization_preserves_public_feature_frame(self) -> None:
        frame = feature_frame(6)
        frame["patient_id"] = ["a"] * 3 + ["b"] * 3
        frame["episode_id"] = ["a0"] * 3 + ["b0"] * 3
        frame["episode_type"] = "interictal"
        original = frame.copy(deep=True)
        medians, scales, patient_medians, patient_scales = (
            rsf._risk_normalization_statistics(frame)
        )
        design = rsf._normalized_risk_design(
            frame,
            frame[rsf.MODEL_FEATURE_COLUMNS].to_numpy(dtype=float),
            medians,
            scales,
            patient_medians,
            patient_scales,
            "combined",
        )
        self.assertEqual(
            design.shape,
            (len(frame), 2 * len(rsf.MODEL_FEATURE_COLUMNS) + 1),
        )
        pd.testing.assert_frame_equal(frame, original)

    def test_cache_signature_changes_with_annotation_time(self) -> None:
        config = rsf.ForecastConfig()
        manifest = pd.DataFrame(
            [
                {
                    "episode_id": "PN00_S01_preictal",
                    "patient_id": "PN00",
                    "source_event_id": "PN00_S01",
                    "episode_type": "preictal",
                    "recording": "PN00-1.edf",
                    "edf_path": "PN00/PN00-1.edf",
                    "anchor_seconds": 843.0,
                    "training_episode_end_seconds": 1143.0,
                    "event_onset_seconds": 1143.0,
                    "dataset_split": "train",
                },
                {
                    "episode_id": "PN00_interictal_01",
                    "patient_id": "PN00",
                    "source_event_id": "",
                    "episode_type": "interictal",
                    "recording": "PN00-1.edf",
                    "edf_path": "PN00/PN00-1.edf",
                    "anchor_seconds": 120.0,
                    "training_episode_end_seconds": 420.0,
                    "event_onset_seconds": np.nan,
                    "dataset_split": "train",
                },
            ]
        )
        original = rsf._cache_signature(manifest, config)
        csv_round_trip = manifest.copy()
        csv_round_trip.loc[
            csv_round_trip["episode_type"].eq("interictal"),
            "source_event_id",
        ] = np.nan
        self.assertEqual(
            original, rsf._cache_signature(csv_round_trip, config)
        )
        changed_manifest = manifest.copy()
        changed_manifest.loc[0, "event_onset_seconds"] = 1144.0
        changed = rsf._cache_signature(changed_manifest, config)
        self.assertNotEqual(original, changed)
        channel_changed = rsf._cache_signature(
            manifest,
            rsf.ForecastConfig(included_eeg_channels=("FP1", "CZ")),
        )
        self.assertNotEqual(original, channel_changed)
        json.dumps(original, allow_nan=False)

    def test_cached_manifest_paths_rebase_without_changing_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            patient_dir = raw / "PN00"
            patient_dir.mkdir()
            local_edf = patient_dir / "PN00-1.edf"
            local_edf.touch()
            manifest = pd.DataFrame(
                {
                    "episode_id": ["PN00_S01_preictal"],
                    "patient_id": ["PN00"],
                    "recording": ["PN00-1.edf"],
                    "edf_path": [r"C:\old-machine\PN00\PN00-1.edf"],
                }
            )
            rebased = rsf.rebase_manifest_edf_paths(manifest, raw)
            self.assertEqual(
                rebased.loc[0, "edf_path"], str(local_edf.resolve())
            )
            self.assertEqual(
                rebased.loc[0, "episode_id"], manifest.loc[0, "episode_id"]
            )

    def test_model_bundle_uses_portable_plain_config(self) -> None:
        model = DummyClassifier(strategy="prior").fit(
            np.zeros((4, len(rsf.HAZARD_FEATURE_COLUMNS))),
            np.array([0, 0, 0, 1]),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.joblib"
            config = rsf.ForecastConfig()
            rsf.save_model_bundle(
                path,
                model,
                config,
                threshold=0.4,
                split={"design": "test"},
                metrics=pd.DataFrame(
                    {"metric": ["nll"], "value": [1.0]}
                ),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                raw = joblib.load(path)
            self.assertIsInstance(raw["config"], dict)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                loaded = rsf.load_model_bundle(path)
            self.assertEqual(loaded["config"], config)

    def test_rolling_engine_rejects_backward_time(self) -> None:
        engine = object.__new__(rsf.RollingForecastEngine)
        engine.episode_anchor_seconds = 100.0
        engine.config = rsf.ForecastConfig()
        engine.max_step = 60
        engine.last_step = 2
        engine.last_result = {"episode_step": 2}
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            engine.advance_to(105.0)


if __name__ == "__main__":
    unittest.main()
