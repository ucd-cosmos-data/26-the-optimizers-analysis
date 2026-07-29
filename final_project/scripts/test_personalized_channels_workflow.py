import numpy as np
import pandas as pd

import rolling_seizure_forecasting as rsf
import personalized_channels_workflow as pc


def _synthetic_landmarks(n_events: int = 3) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rng = np.random.default_rng(7)
    rows = []
    channel_map = {"A": ["a1", "a2"], "B": ["b1", "b2"]}
    n_bins = rsf.ForecastConfig().n_bins
    for event in range(n_events):
        for episode_type in ("preictal", "interictal"):
            positive = episode_type == "preictal"
            for step in range(n_bins):
                rows.append(
                    {
                        "patient_id": "PX",
                        "source_event_id": f"E{event}",
                        "episode_id": f"E{event}_{episode_type}",
                        "episode_type": episode_type,
                        "recording": f"R{event}",
                        "landmark_step": step,
                        "landmark_seconds": float(step * rsf.BIN_SECONDS),
                        "time_to_event_seconds": (
                            float(rsf.HORIZON_SECONDS - step * rsf.BIN_SECONDS)
                            if positive
                            else np.nan
                        ),
                        "event_bin": n_bins - step - 1 if positive else -1,
                        "has_event_in_5m": int(positive),
                        "a1": float(positive) * 3 + rng.normal(scale=0.2),
                        "a2": rng.normal(scale=0.2),
                        "b1": rng.normal(),
                        "b2": rng.normal(),
                    }
                )
    return pd.DataFrame(rows), channel_map


def test_chronological_split_keeps_matched_pairs_together():
    manifest = pd.DataFrame(
        [
            {
                "patient_id": "PX",
                "source_event_id": event,
                "episode_id": f"{event}_{kind}",
                "episode_type": kind,
                "recording": f"R{event}",
                "event_onset_seconds": 100.0,
            }
            for event in ("E1", "E2", "E3")
            for kind in ("preictal", "interictal")
        ]
    )
    train, test, details = pc.chronological_event_split(manifest)
    assert details["n_train_seizures"] == 2
    assert details["n_test_seizures"] == 1
    assert set(train["source_event_id"]) == {"E1", "E2"}
    assert set(test["source_event_id"]) == {"E3"}


def test_chronological_split_keeps_multiple_controls_per_seizure():
    rows = []
    for event_number, event in enumerate(("E1", "E2", "E3"), start=1):
        rows.append(
            {
                "patient_id": "PX",
                "source_event_id": event,
                "episode_id": f"{event}_preictal",
                "episode_type": "preictal",
                "recording": f"R{event_number}",
                "event_onset_seconds": 100.0,
            }
        )
        rows.extend(
            {
                "patient_id": "PX",
                "source_event_id": event,
                "episode_id": f"{event}_interictal_{control}",
                "episode_type": "interictal",
                "recording": f"R{event_number}",
                "event_onset_seconds": np.nan,
            }
            for control in range(4)
        )
    manifest = pd.DataFrame(rows)
    train, test, _ = pc.chronological_event_split(manifest)

    assert set(train["source_event_id"]) == {"E1", "E2"}
    assert set(test["source_event_id"]) == {"E3"}
    assert test["episode_type"].value_counts().to_dict() == {
        "interictal": 4,
        "preictal": 1,
    }


def test_fixed_k_selection_and_hazard_evaluation():
    frame, channel_map = _synthetic_landmarks()
    train = frame.loc[frame["source_event_id"].isin(["E0", "E1"])].copy()
    test = frame.loc[frame["source_event_id"].eq("E2")].copy()
    config = pc.PersonalizedConfig(k=1, swap_refinement=False)
    selected, trace, method = pc.select_fixed_k_channels(
        train, channel_map, config
    )
    assert selected == ["A"]
    assert len(trace) == 1
    assert method == "expanding_chronological_validation"

    forecast_config = rsf.ForecastConfig(max_iter=10, min_samples_leaf=5)
    model = pc.fit_hazard_model(train, channel_map["A"], forecast_config)
    metrics, predictions = pc.evaluate_model(
        model, train, test, channel_map["A"], forecast_config
    )
    assert 0 <= metrics["auprc"] <= 1
    assert metrics["threshold_method"] == "chronological_oof"
    assert len(predictions) == len(test)

    binary_metrics, binary_predictions = pc.fit_and_evaluate_binary(
        train,
        test,
        channel_map,
        ["A"],
        config,
        forecast_config,
    )
    assert 0 <= binary_metrics["auprc"] <= 1
    assert binary_metrics["threshold_method"] == "chronological_oof"
    assert len(binary_predictions) == len(test)


def test_positive_platt_calibration_is_monotone_and_nonsaturated():
    frame = pd.DataFrame(
        {
            "episode_id": np.repeat(["n", "p"], 4),
            "has_event_in_5m": np.repeat([0, 1], 4),
        }
    )
    raw = np.array([0.90, 0.92, 0.94, 0.96, 0.95, 0.97, 0.98, 0.99])
    calibrator = pc.fit_risk_calibrator(frame, raw)
    calibrated = calibrator.transform(raw)
    order = np.argsort(raw)
    assert np.all(np.diff(calibrated[order]) >= 0)
    assert calibrated.min() > 0
    assert calibrated.max() < 1
