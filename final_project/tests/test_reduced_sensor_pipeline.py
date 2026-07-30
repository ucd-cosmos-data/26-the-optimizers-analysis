from __future__ import annotations

import numpy as np
import pandas as pd

from src.reduced_sensor_pipeline import (
    COMPACT_FEATURES,
    PositivePlattCalibrator,
    chronological_patient_split,
    greedy_channel_ranking,
    reassign_overlapping_controls,
)


def synthetic_patient(n_seizures: int) -> pd.DataFrame:
    rows = []
    for event in range(1, n_seizures + 1):
        event_id = f"PNXX_S{event:02d}"
        for episode_type, count in (("preictal", 1), ("interictal", 4)):
            for episode in range(count):
                for step in range(3):
                    rows.append(
                        {
                            "episode_id": (
                                f"{event_id}_{episode_type}_{episode}"
                            ),
                            "patient_id": "PNXX",
                            "source_event_id": event_id,
                            "episode_type": episode_type,
                            "recording": f"PNXX-{event}.edf",
                            "landmark_step": step,
                            "landmark_seconds": event * 100 + step * 5,
                            "time_to_event_seconds": (
                                15 - step * 5
                                if episode_type == "preictal"
                                else np.nan
                            ),
                            "event_bin": (
                                2 - step if episode_type == "preictal" else -1
                            ),
                            "has_event_in_5m": int(
                                episode_type == "preictal"
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def test_split_holds_out_at_least_two_complete_seizure_groups():
    train, test, details = chronological_patient_split(synthetic_patient(5))
    assert details["n_train_seizures"] == 3
    assert details["n_test_seizures"] == 2
    assert set(train.source_event_id).isdisjoint(test.source_event_id)
    assert test.episode_id.nunique() == 10


def test_twenty_percent_rule_still_holds_for_ten_seizures():
    _, _, details = chronological_patient_split(synthetic_patient(10))
    assert details["n_train_seizures"] == 8
    assert details["n_test_seizures"] == 2


def test_split_reserves_a_shared_test_recording_and_excludes_earlier_events():
    frame = synthetic_patient(10)
    shared = frame["source_event_id"].isin(
        ["PNXX_S07", "PNXX_S08", "PNXX_S09"]
    )
    frame.loc[shared, "recording"] = "PNXX-7.8.9.edf"
    train, test, details = chronological_patient_split(frame)
    assert details["n_nominal_train_seizures"] == 8
    assert details["n_train_seizures"] == 6
    assert details["train_events_excluded_for_recording_isolation"] == [
        "PNXX_S07",
        "PNXX_S08",
    ]
    assert set(train["recording"]).isdisjoint(test["recording"])
    assert not set(train["source_event_id"]) & set(test["source_event_id"])


def test_greedy_ranking_prefers_best_probability_column():
    truth = np.asarray([0, 0, 1, 1])
    probabilities = np.column_stack(
        [
            [0.1, 0.2, 0.8, 0.9],
            [0.4, 0.4, 0.6, 0.6],
            [0.9, 0.8, 0.2, 0.1],
        ]
    )
    ranking, trace = greedy_channel_ranking(
        truth, probabilities, ["GOOD", "WEAK", "BAD"]
    )
    assert ranking[0] == "GOOD"
    assert trace.iloc[0]["rank"] == 1
    assert len(ranking) == 3


def test_positive_platt_transform_stays_in_probability_range():
    calibrator = PositivePlattCalibrator(slope=2.0, intercept=-0.5)
    transformed = calibrator.transform([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.all((transformed > 0) & (transformed < 1))
    assert np.all(np.diff(transformed) > 0)


def test_overlapping_controls_are_kept_in_one_event_group():
    frame = synthetic_patient(3)
    # Make two controls from different historical groups represent overlapping
    # raw time in the same recording.
    first = frame["episode_id"].eq("PNXX_S01_interictal_0")
    second = frame["episode_id"].eq("PNXX_S02_interictal_0")
    frame.loc[first | second, "recording"] = "shared.edf"
    frame.loc[first, "landmark_seconds"] = [500, 505, 510]
    frame.loc[second, "landmark_seconds"] = [560, 565, 570]
    for channel in ("C3",):
        for feature in COMPACT_FEATURES:
            for aggregation in ("mean", "last", "slope"):
                frame[f"ch::{channel}::{feature}::{aggregation}"] = np.arange(
                    len(frame), dtype=float
                )
    cleaned, audit = reassign_overlapping_controls(frame, ["C3"])
    groups = (
        cleaned.loc[
            cleaned["episode_id"].isin(
                ["PNXX_S01_interictal_0", "PNXX_S02_interictal_0"]
            )
        ]
        .groupby("episode_id")["source_event_id"]
        .first()
    )
    assert groups.nunique() == 1
    assert audit["raw_interval_cross_group_overlaps_after"] == 0
