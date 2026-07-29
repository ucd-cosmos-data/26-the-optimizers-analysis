import sys
import types

import numpy as np
import pandas as pd

# Structural unit tests do not read EDF files.  The production notebook checks
# for the real optional dependency before feature extraction.
sys.modules.setdefault("pyedflib", types.SimpleNamespace(EdfReader=object))

import generalized_personalized_comparison as gp
import personalized_channels_workflow as pc


def _feature_data(patient_id: str, channels: list[str]) -> pc.PatientFeatureData:
    columns = gp.compact_feature_columns(channels)
    rows = []
    for event in range(2):
        for episode_type in ("preictal", "interictal"):
            for step in range(3):
                row = {
                    "episode_id": f"{patient_id}_{event}_{episode_type}",
                    "patient_id": patient_id,
                    "source_event_id": f"{patient_id}_S{event}",
                    "episode_type": episode_type,
                    "recording": f"{patient_id}.edf",
                    "landmark_step": step,
                    "landmark_seconds": float(step * 5),
                    "time_to_event_seconds": (
                        float(15 - step * 5)
                        if episode_type == "preictal"
                        else np.nan
                    ),
                    "event_bin": 2 - step if episode_type == "preictal" else -1,
                    "has_event_in_5m": int(episode_type == "preictal"),
                }
                row.update({column: float(step + event) for column in columns})
                rows.append(row)
    channel_map = {
        channel: [
            column
            for column in columns
            if column.startswith(f"ch::{channel}::")
        ]
        for channel in channels
    }
    return pc.PatientFeatureData(
        patient_id=patient_id,
        frame=pd.DataFrame(rows),
        channel_names=channels,
        channel_feature_columns=channel_map,
    )


def test_audit_confirms_eight_plus_six_partition():
    rows = []
    all_patients = gp.COMPARISON_PATIENTS + gp.EXCLUDED_LOW_EVENT_PATIENTS
    for patient_id in all_patients:
        for event in range(3 if patient_id in gp.COMPARISON_PATIENTS else 1):
            rows.append(
                {
                    "patient_id": patient_id,
                    "source_event_id": f"{patient_id}_S{event}",
                    "episode_type": "preictal",
                }
            )
    audit = gp.audit_patient_eligibility(pd.DataFrame(rows))
    assert (audit["study_role"] == "comparison").sum() == 8
    assert (audit["study_role"] == "excluded_low_event_count").sum() == 6


def test_alignment_has_seven_training_patients_and_one_target():
    data = {
        patient_id: _feature_data(
            patient_id,
            ["A", "B"] if patient_id != "PN05" else ["A"],
        )
        for patient_id in gp.COMPARISON_PATIENTS
    }
    train, test, columns, coverage = gp.align_patient_feature_frames(
        data,
        target_patient="PN00",
        selected_channels=["A", "B"],
    )
    assert set(train["patient_id"]) == set(gp.COMPARISON_PATIENTS) - {"PN00"}
    assert set(test["patient_id"]) == {"PN00"}
    assert len(columns) == 2 * len(pc.COMPACT_FEATURE_NAMES) * len(
        pc.COMPACT_AGGREGATIONS
    )
    assert len(coverage) == 8 * 2
    assert train.loc[train["patient_id"].eq("PN05"), columns[24:]].isna().all().all()


def test_handoff_reports_pending_patients(tmp_path):
    summary = pd.DataFrame(
        {
            "patient_id": ["PN00"],
            "status": ["included"],
            "selected_channels": ["A, B"],
            "n_seizures": [5],
            "n_train_seizures": [4],
            "n_test_seizures": [1],
        }
    )
    summary_path = tmp_path / "personalized.csv"
    summary.to_csv(summary_path, index=False)
    completed, pending = gp.load_personalized_handoff(summary_path)
    assert completed.iloc[0]["selected_channel_list"] == ["A", "B"]
    assert len(pending) == 7


def test_comparison_is_g_minus_p_without_optimization():
    p = pd.DataFrame(
        {
            "patient_id": ["PN00"],
            "n_train_seizures": [4],
            "n_test_seizures": [1],
            "selected_auprc": [0.7],
            "selected_auroc": [0.8],
            "selected_binary_brier": [0.2],
            "selected_seizure_sensitivity": [1.0],
            "selected_time_in_warning": [0.25],
            "selected_false_alarms_per_hour": [0.1],
            "selected_median_warning_lead_seconds": [120.0],
        }
    )
    g = pd.DataFrame(
        {
            "patient_id": ["PN00"],
            "model": ["G_i(i)"],
            "n_train_seizures": [32],
            "n_test_seizures": [5],
            "auprc": [0.6],
            "auroc": [0.75],
            "binary_brier": [0.25],
            "seizure_sensitivity": [0.8],
            "time_in_warning": [0.2],
            "false_alarms_per_hour": [0.2],
            "median_warning_lead_seconds": [100.0],
        }
    )
    long, differences = gp.compare_personalized_generalized(p, g)
    assert set(long["model"]) == {"P_i(i)", "G_i(i)"}
    auprc = differences.loc[differences["metric"].eq("auprc")].iloc[0]
    assert np.isclose(auprc["G_minus_P"], -0.1)
    assert auprc["P_test_seizures"] == 1
    assert auprc["G_test_seizures"] == 5
