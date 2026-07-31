from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
TABLES = PROJECT / "results" / "final" / "tables"
FIGURES = PROJECT / "results" / "final" / "figures"


def test_k_summary_matches_the_first_observed_passing_count():
    summary = json.loads((TABLES / "k_finder_summary.json").read_text())
    curve = pd.read_csv(TABLES / "k_finder_loss_curve.csv")
    passing = curve.loc[
        curve["meets_0.03_worst_patient_limit"].astype(bool), "k"
    ]
    assert int(passing.min()) == summary["selected_k"]
    assert (
        summary["selected_k_worst_patient_auprc_loss"]
        <= summary["allowed_worst_patient_auprc_loss"]
    )
    assert len(pd.read_csv(TABLES / "k_finder_patient_scores.csv")) == 14


def test_p_and_g_use_the_same_five_patient_test_rows():
    comparison = pd.read_csv(TABLES / "patient_model_comparison.csv")
    assert len(comparison) == 5
    assert comparison["n_test_seizures"].eq(2).all()
    assert comparison["k"].nunique() == 1

    predictions = pd.read_csv(TABLES / "held_out_predictions.csv")
    keys = [
        "patient_id",
        "source_event_id",
        "episode_id",
        "recording",
        "landmark_step",
    ]
    p_keys = predictions.loc[predictions["model"].eq("P"), keys]
    g_keys = predictions.loc[predictions["model"].eq("G"), keys]
    assert not p_keys.duplicated().any()
    assert not g_keys.duplicated().any()
    assert set(map(tuple, p_keys.to_numpy())) == set(
        map(tuple, g_keys.to_numpy())
    )
    for column in ("selected_k_probability", "full_29_probability"):
        values = predictions[column].to_numpy(dtype=float)
        assert np.isfinite(values).all()
        assert np.all((values > 0) & (values < 1))


def test_recording_isolation_is_exposed_in_the_final_table():
    comparison = pd.read_csv(TABLES / "patient_model_comparison.csv")
    for row in comparison.itertuples(index=False):
        train = {
            value.strip()
            for value in str(row.train_recordings_P).split(",")
            if value.strip()
        }
        test = {
            value.strip()
            for value in str(row.test_recordings).split(",")
            if value.strip()
        }
        assert train.isdisjoint(test)
    pn10 = comparison.set_index("patient_id").loc["PN10"]
    assert pn10["n_train_seizures_P"] == 6
    assert pn10["n_nominal_train_seizures_P"] == 8
    assert pn10["train_events_excluded_for_recording_isolation"] == (
        "PN10_S07, PN10_S08"
    )


def test_seizure_allocation_figure_matches_the_final_splits():
    comparison = pd.read_csv(TABLES / "patient_model_comparison.csv")
    expected = {
        "PN00": (3, 0, 2),
        "PN06": (3, 0, 2),
        "PN10": (6, 2, 2),
        "PN12": (2, 0, 2),
        "PN14": (2, 0, 2),
    }

    observed: dict[str, tuple[int, int, int]] = {}
    for row in comparison.itertuples(index=False):
        excluded = (
            int(row.n_nominal_train_seizures_P)
            - int(row.n_train_seizures_P)
        )
        allocation = (
            int(row.n_train_seizures_P)
            + excluded
            + int(row.n_test_seizures)
        )
        assert allocation == int(row.n_seizures)
        assert int(row.n_test_seizures) >= 2
        observed[str(row.patient_id)] = (
            int(row.n_train_seizures_P),
            excluded,
            int(row.n_test_seizures),
        )

    assert observed == expected
    figure = FIGURES / "05_seizure_allocation.png"
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert figure.stat().st_size > 10_000
