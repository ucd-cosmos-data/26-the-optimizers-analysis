"""Leave-one-patient-out generalized versus personalized seizure forecasts.

This module is the reusable implementation behind
``generalized_vs_personalized.ipynb``.  It deliberately reuses the discrete-time
hazard architecture and per-channel rolling features from
``personalized_channels_workflow.py`` so that, for patient i, P_i and G_i differ
in their fitting population rather than their model class.

This is exploratory research code, not a clinical warning system.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

import personalized_channels_workflow as pc
import rolling_seizure_forecasting as rsf


COMPARISON_PATIENTS = (
    "PN00",
    "PN05",
    "PN06",
    "PN09",
    "PN10",
    "PN12",
    "PN13",
    "PN14",
)
EXCLUDED_LOW_EVENT_PATIENTS = (
    "PN01",
    "PN03",
    "PN07",
    "PN11",
    "PN16",
    "PN17",
)

LANDMARK_COLUMNS = (
    "episode_id",
    "patient_id",
    "source_event_id",
    "episode_type",
    "recording",
    "landmark_step",
    "landmark_seconds",
    "time_to_event_seconds",
    "event_bin",
    "has_event_in_5m",
)

COMPARISON_METRICS = (
    "auprc",
    "auroc",
    "binary_brier",
    "seizure_sensitivity",
    "time_in_warning",
    "false_alarms_per_hour",
    "median_warning_lead_seconds",
)


@dataclass
class GeneralizedResult:
    """Outputs for G_i evaluated on patient i."""

    patient_id: str
    selected_channels: list[str]
    training_patients: list[str]
    training_seizures: int
    test_seizures: int
    metrics: dict[str, Any]
    predictions: pd.DataFrame


def audit_patient_eligibility(
    manifest: pd.DataFrame,
    *,
    included: Sequence[str] = COMPARISON_PATIENTS,
    excluded: Sequence[str] = EXCLUDED_LOW_EVENT_PATIENTS,
) -> pd.DataFrame:
    """Count usable seizure/control pairs and validate the 8+6 study partition."""

    required = {"patient_id", "source_event_id", "episode_type"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")

    event_rows = manifest.loc[manifest["episode_type"].eq("preictal")].copy()
    counts = (
        event_rows.groupby("patient_id")["source_event_id"]
        .nunique()
        .rename("n_seizures")
        .reset_index()
    )
    counts["study_role"] = np.select(
        [
            counts["patient_id"].isin(included),
            counts["patient_id"].isin(excluded),
        ],
        ["comparison", "excluded_low_event_count"],
        default="unexpected",
    )
    observed = set(counts["patient_id"])
    expected = set(included) | set(excluded)
    if observed != expected:
        raise ValueError(
            f"Patient partition mismatch; missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    if len(set(included)) != 8 or len(set(excluded)) != 6:
        raise ValueError("The study partition must contain 8 included and 6 excluded patients.")
    return counts.sort_values(["study_role", "patient_id"]).reset_index(drop=True)


def load_personalized_handoff(
    summary_csv: str | Path,
    *,
    patient_ids: Sequence[str] = COMPARISON_PATIENTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load completed P_i summaries and return completed/pending patient tables."""

    summary_path = Path(summary_csv)
    if not summary_path.exists():
        completed = pd.DataFrame(columns=["patient_id", "selected_channels"])
    else:
        completed = pd.read_csv(summary_path)
        required = {
            "patient_id",
            "status",
            "selected_channels",
            "n_seizures",
            "n_train_seizures",
            "n_test_seizures",
        }
        missing = sorted(required - set(completed.columns))
        if missing:
            raise ValueError(f"Personalized summary is missing columns: {missing}")
        completed = completed.loc[
            completed["patient_id"].isin(patient_ids)
            & completed["status"].eq("included")
        ].copy()

    if completed["patient_id"].duplicated().any():
        duplicates = completed.loc[
            completed["patient_id"].duplicated(), "patient_id"
        ].tolist()
        raise ValueError(f"Duplicate personalized summaries: {duplicates}")

    completed["selected_channel_list"] = completed.get(
        "selected_channels", pd.Series(dtype=str)
    ).fillna("").map(
        lambda value: [part.strip() for part in str(value).split(",") if part.strip()]
    )
    observed = set(completed["patient_id"])
    pending = pd.DataFrame(
        {
            "patient_id": [patient for patient in patient_ids if patient not in observed],
            "status": "awaiting_personalized_model",
            "required_artifact": str(summary_path),
        }
    )
    return completed.reset_index(drop=True), pending


def selected_channels_for_patient(
    personalized_summary: pd.DataFrame,
    patient_id: str,
) -> list[str]:
    """Return the P_i channel schema that both P_i and G_i must use."""

    rows = personalized_summary.loc[
        personalized_summary["patient_id"].eq(patient_id)
    ]
    if len(rows) != 1:
        raise KeyError(
            f"{patient_id} needs exactly one completed personalized summary; "
            f"found {len(rows)}"
        )
    channels = list(rows.iloc[0]["selected_channel_list"])
    if not channels:
        raise ValueError(f"{patient_id} has no selected channels in the handoff.")
    return channels


def compact_feature_columns(channels: Iterable[str]) -> list[str]:
    """Return the exact rolling feature schema for an ordered channel list."""

    return [
        f"ch::{channel}::{feature}::{aggregation}"
        for channel in channels
        for feature in pc.COMPACT_FEATURE_NAMES
        for aggregation in pc.COMPACT_AGGREGATIONS
    ]


def align_patient_feature_frames(
    feature_data: dict[str, pc.PatientFeatureData],
    *,
    target_patient: str,
    selected_channels: Sequence[str],
    comparison_patients: Sequence[str] = COMPARISON_PATIENTS,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    """Create G_i train and all-event patient-i test frames with one schema.

    Channels unavailable in a training patient are represented by NaN.  The
    HistGradientBoosting model handles missing values natively, but every model
    feature must be observed in at least one of the seven training patients.
    """

    required_patients = set(comparison_patients)
    missing_patients = sorted(required_patients - set(feature_data))
    if missing_patients:
        raise ValueError(f"Missing feature data for patients: {missing_patients}")
    if target_patient not in required_patients:
        raise ValueError(f"{target_patient} is not one of the eight comparison patients.")

    feature_columns = compact_feature_columns(selected_channels)
    aligned: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    for patient_id in comparison_patients:
        patient = feature_data[patient_id]
        frame = patient.frame.copy()
        metadata = frame.reindex(columns=LANDMARK_COLUMNS)
        features = frame.reindex(columns=feature_columns)
        aligned_frame = pd.concat([metadata, features], axis=1)
        aligned.append(aligned_frame)
        available = set(patient.channel_names)
        coverage_rows.extend(
            {
                "target_patient": target_patient,
                "training_or_test_patient": patient_id,
                "channel": channel,
                "available": channel in available,
            }
            for channel in selected_channels
        )

    combined = pd.concat(aligned, ignore_index=True)
    train = combined.loc[~combined["patient_id"].eq(target_patient)].copy()
    test = combined.loc[combined["patient_id"].eq(target_patient)].copy()
    if train.empty or test.empty:
        raise ValueError("Generalized train and target-patient test frames must be nonempty.")
    if set(train["patient_id"]) != required_patients - {target_patient}:
        raise AssertionError("G_i training rows do not contain exactly the other seven patients.")
    if set(test["patient_id"]) != {target_patient}:
        raise AssertionError("G_i test rows contain a non-target patient.")
    if train[feature_columns].notna().sum().eq(0).any():
        unavailable = (
            train[feature_columns].notna().sum().loc[lambda values: values.eq(0)].index
        )
        raise ValueError(
            "No generalized-training patient supplies these features: "
            + ", ".join(unavailable[:10])
        )

    coverage = pd.DataFrame(coverage_rows)
    return train, test, feature_columns, coverage


def _leave_one_training_patient_out_risk(
    train: pd.DataFrame,
    feature_columns: Sequence[str],
    forecast_config: rsf.ForecastConfig,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Patient-disjoint OOF risk used only for calibration and thresholding."""

    validation_frames: list[pd.DataFrame] = []
    raw_risks: list[np.ndarray] = []
    training_patients = sorted(train["patient_id"].unique())
    if len(training_patients) < 2:
        raise ValueError("Generalized OOF calibration needs at least two patients.")
    for validation_patient in training_patients:
        fit = train.loc[~train["patient_id"].eq(validation_patient)]
        validation = train.loc[train["patient_id"].eq(validation_patient)]
        model = pc.fit_hazard_model(fit, feature_columns, forecast_config)
        _, _, no_event = pc.predict_horizon_distribution(
            model, validation, feature_columns
        )
        validation_frames.append(validation.copy())
        raw_risks.append(1.0 - no_event)
    return pd.concat(validation_frames, ignore_index=True), np.concatenate(raw_risks)


def _distribution_metrics(
    test: pd.DataFrame,
    calibrated_pmf: np.ndarray,
    calibrated_no_event: np.ndarray,
) -> dict[str, float]:
    """Categorical forecast metrics shared with the personalized workflow."""

    has_event = test["has_event_in_5m"].to_numpy(dtype=bool)
    event_bin = test["event_bin"].to_numpy(dtype=int)
    rows = np.arange(len(test))
    observed_probability = np.where(
        has_event,
        calibrated_pmf[rows, event_bin],
        calibrated_no_event,
    )
    nll = -np.log(np.clip(observed_probability, 1e-12, None))
    squared = np.square(calibrated_pmf).sum(axis=1) + np.square(
        calibrated_no_event
    )
    categorical_brier = squared.copy()
    categorical_brier[has_event] += (
        1.0 - 2.0 * calibrated_pmf[rows[has_event], event_bin[has_event]]
    )
    categorical_brier[~has_event] += (
        1.0 - 2.0 * calibrated_no_event[~has_event]
    )
    return {
        "categorical_brier": float(categorical_brier.mean()),
        "negative_log_likelihood": float(nll.mean()),
    }


def fit_evaluate_generalized(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Sequence[str],
    forecast_config: rsf.ForecastConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit G_i on seven patients and evaluate it on every episode of patient i."""

    overlap = set(train["patient_id"]) & set(test["patient_id"])
    if overlap:
        raise AssertionError(f"Patient leakage between G_i train and test: {sorted(overlap)}")
    if train["has_event_in_5m"].nunique() != 2:
        raise ValueError("Generalized training rows must contain both outcome classes.")
    if test["has_event_in_5m"].nunique() != 2:
        raise ValueError("Target-patient rows must contain seizure and interictal landmarks.")

    threshold_frame, raw_threshold_risk = _leave_one_training_patient_out_risk(
        train, feature_columns, forecast_config
    )
    calibrator = pc.fit_risk_calibrator(threshold_frame, raw_threshold_risk)
    threshold_risk = calibrator.transform(raw_threshold_risk)
    threshold, _ = rsf.select_warning_threshold(
        threshold_frame,
        threshold_risk,
        forecast_config.warning_time_target,
    )

    model = pc.fit_hazard_model(train, feature_columns, forecast_config)
    _, raw_pmf, raw_no_event = pc.predict_horizon_distribution(
        model, test, feature_columns
    )
    pmf, no_event, risk = pc._calibrate_distribution(
        raw_pmf, raw_no_event, calibrator
    )
    risk_metrics, predictions = pc._evaluate_risk(test, risk, threshold)
    predictions["raw_event_risk_5m"] = 1.0 - raw_no_event
    metrics: dict[str, Any] = {
        **risk_metrics,
        **_distribution_metrics(test, pmf, no_event),
        "calibration_method": calibrator.method,
        "threshold_method": "leave_one_training_patient_out_oof",
        "n_training_patients": int(train["patient_id"].nunique()),
    }
    return metrics, predictions


def run_generalized_patient(
    feature_data: dict[str, pc.PatientFeatureData],
    personalized_summary: pd.DataFrame,
    target_patient: str,
    forecast_config: rsf.ForecastConfig,
) -> tuple[GeneralizedResult, pd.DataFrame]:
    """Fit one G_i using P_i's channel schema, then test on all of patient i."""

    channels = selected_channels_for_patient(personalized_summary, target_patient)
    train, test, columns, coverage = align_patient_feature_frames(
        feature_data,
        target_patient=target_patient,
        selected_channels=channels,
    )
    metrics, predictions = fit_evaluate_generalized(
        train, test, columns, forecast_config
    )
    predictions["model"] = "generalized_leave_one_patient_out"
    event_rows = lambda frame: frame.loc[frame["episode_type"].eq("preictal")]
    result = GeneralizedResult(
        patient_id=target_patient,
        selected_channels=channels,
        training_patients=sorted(train["patient_id"].unique().tolist()),
        training_seizures=int(event_rows(train)["source_event_id"].nunique()),
        test_seizures=int(event_rows(test)["source_event_id"].nunique()),
        metrics=metrics,
        predictions=predictions,
    )
    return result, coverage


def generalized_summary(results: Sequence[GeneralizedResult]) -> pd.DataFrame:
    """Convert G_i results to one comparison-ready row per target patient."""

    rows = []
    for result in results:
        rows.append(
            {
                "patient_id": result.patient_id,
                "model": "G_i(i)",
                "selected_channels": ", ".join(result.selected_channels),
                "n_training_patients": len(result.training_patients),
                "n_train_seizures": result.training_seizures,
                "n_test_seizures": result.test_seizures,
                **result.metrics,
            }
        )
    return pd.DataFrame(rows)


def compare_personalized_generalized(
    personalized_summary: pd.DataFrame,
    generalized: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return side-by-side P_i(i)/G_i(i) values and transparent raw differences.

    Differences are always ``G_i(i) - P_i(i)``.  They are descriptive only:
    G_i is tested on all target-patient seizures, whereas P_i follows the
    chronological approximately 80/20 split requested for the study.
    """

    personalized_columns = {
        "selected_auprc": "auprc",
        "selected_auroc": "auroc",
        "selected_binary_brier": "binary_brier",
        "selected_seizure_sensitivity": "seizure_sensitivity",
        "selected_time_in_warning": "time_in_warning",
        "selected_false_alarms_per_hour": "false_alarms_per_hour",
        "selected_median_warning_lead_seconds": "median_warning_lead_seconds",
    }
    required = {"patient_id", "n_train_seizures", "n_test_seizures"} | set(
        personalized_columns
    )
    missing = sorted(required - set(personalized_summary.columns))
    if missing:
        raise ValueError(f"Personalized handoff is missing comparison fields: {missing}")

    p = personalized_summary[
        ["patient_id", "n_train_seizures", "n_test_seizures", *personalized_columns]
    ].rename(columns=personalized_columns)
    p["model"] = "P_i(i)"
    keep = [
        "patient_id",
        "model",
        "n_train_seizures",
        "n_test_seizures",
        *COMPARISON_METRICS,
    ]
    g = generalized.reindex(columns=keep)
    long = pd.concat([p.reindex(columns=keep), g], ignore_index=True).sort_values(
        ["patient_id", "model"]
    )

    joined = p.merge(g, on="patient_id", suffixes=("_P", "_G"))
    difference_rows = []
    for row in joined.itertuples(index=False):
        for metric in COMPARISON_METRICS:
            difference_rows.append(
                {
                    "patient_id": row.patient_id,
                    "metric": metric,
                    "P_i(i)": getattr(row, f"{metric}_P"),
                    "G_i(i)": getattr(row, f"{metric}_G"),
                    "G_minus_P": getattr(row, f"{metric}_G")
                    - getattr(row, f"{metric}_P"),
                    "preferred_direction": (
                        "lower"
                        if metric
                        in {
                            "binary_brier",
                            "time_in_warning",
                            "false_alarms_per_hour",
                        }
                        else "higher"
                    ),
                    "P_test_seizures": row.n_test_seizures_P,
                    "G_test_seizures": row.n_test_seizures_G,
                }
            )
    return long.reset_index(drop=True), pd.DataFrame(difference_rows)

