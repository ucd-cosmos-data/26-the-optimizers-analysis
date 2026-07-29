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
            {"test": 8, "train": 35, "validation": 4},
        )
        self.assertEqual(allocation["source_event_id"].nunique(), 47)
        self.assertFalse(allocation["source_event_id"].duplicated().any())
        self.assertFalse(assigned["dataset_split"].eq("").any())

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
        json.dumps(original, allow_nan=False)

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
