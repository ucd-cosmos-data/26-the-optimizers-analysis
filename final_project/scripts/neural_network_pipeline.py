"""Neural-network seizure-risk scoring for the Siena EEG project.

This module trains on the filter-aware window table produced from the raw EDF
files by ``analyze_final60s_bandpower.py``.  At inference time it reads a
6-second window directly from an EDF and returns:

1. an experimental probability score for seizure onset within 60 seconds; and
2. an estimated number of seconds to onset, only when the score crosses the
   validation-selected warning threshold.

The model is exploratory and is not a medical device.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analyze_final60s_bandpower import (
    BAND_ORDER,
    read_eeg_window,
    window_bandpowers,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_FEATURE_TABLE = (
    PROJECT_DIR / "data" / "processed" / "preictal_60s_zscore" / "all_window_features.csv"
)
PREDICTION_HORIZON_SECONDS = 60
WINDOW_SECONDS = 6
MIN_CHANNELS = 10
POWER_FEATURES = [f"{band}_relative_power" for band in BAND_ORDER]
QUALITY_FEATURES = [
    "artifact_channel_fraction",
    "artifact_usable_channel_fraction",
    "sample_rate_hz",
]
FEATURE_COLUMNS = POWER_FEATURES + QUALITY_FEATURES


@dataclass(frozen=True)
class ModelConfig:
    hidden_layers: tuple[int, ...] = (32, 16)
    learning_rate: float = 1e-3
    max_epochs: int = 300
    validation_fraction: float = 0.2
    patience: int = 30
    random_seed: int = 42


def _window_key_columns(frame: pd.DataFrame) -> list[str]:
    """Columns that uniquely identify a physical EEG window."""
    return [
        "patient_id",
        "event_id",
        "interictal_block_id",
        "condition",
        "recording",
        "window_start_seconds",
        "time_to_onset_midpoint_seconds",
    ]


def load_training_windows(path: str | Path = DEFAULT_FEATURE_TABLE) -> pd.DataFrame:
    """Convert the long filter-aware band table into one row per EEG window."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Training table not found: {path}. Run "
            "analyze_final60s_bandpower.py to build it from the raw EDF files."
        )
    long = pd.read_csv(path)
    required = set(_window_key_columns(long)) | {
        "band",
        "relative_power",
        "artifact_channel_fraction",
        "artifact_usable_channel_count",
        "eeg_channel_count",
        "sample_rate_hz",
    }
    missing = sorted(required - set(long.columns))
    if missing:
        raise ValueError(f"Training table is missing columns: {missing}")

    keys = _window_key_columns(long)
    # ``pivot_table(dropna=False)`` builds a Cartesian product of all key
    # levels in recent pandas versions and can request hundreds of GB. Group
    # first and unstack only combinations that are actually present.
    powers = (
        long.groupby(keys + ["band"], dropna=False)["relative_power"]
        .first()
        .unstack("band")
        .reset_index()
    )
    powers.columns.name = None
    powers = powers.rename(columns={band: f"{band}_relative_power" for band in BAND_ORDER})

    quality = (
        long.groupby(keys, dropna=False, as_index=False)
        .agg(
            artifact_channel_fraction=("artifact_channel_fraction", "first"),
            artifact_usable_channel_count=("artifact_usable_channel_count", "first"),
            eeg_channel_count=("eeg_channel_count", "first"),
            sample_rate_hz=("sample_rate_hz", "first"),
        )
    )
    windows = powers.merge(quality, on=keys, how="inner", validate="one_to_one")
    windows["artifact_usable_channel_fraction"] = (
        windows["artifact_usable_channel_count"] / windows["eeg_channel_count"].clip(lower=1)
    )
    windows["seizure_within_60s"] = (windows["condition"] == "preictal").astype(int)
    windows["seconds_to_seizure"] = np.where(
        windows["seizure_within_60s"].eq(1),
        -pd.to_numeric(windows["time_to_onset_midpoint_seconds"], errors="coerce"),
        np.nan,
    )
    if windows["seizure_within_60s"].nunique() != 2:
        raise ValueError("Training data must contain both preictal and interictal windows.")
    return windows.sort_values(keys, na_position="last").reset_index(drop=True)


def _make_classifier(config: ModelConfig) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "network",
                MLPClassifier(
                    hidden_layer_sizes=config.hidden_layers,
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
                    learning_rate_init=config.learning_rate,
                    max_iter=config.max_epochs,
                    early_stopping=True,
                    validation_fraction=config.validation_fraction,
                    n_iter_no_change=config.patience,
                    random_state=config.random_seed,
                ),
            ),
        ]
    )


def _make_regressor(config: ModelConfig) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "network",
                MLPRegressor(
                    hidden_layer_sizes=config.hidden_layers,
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
                    learning_rate_init=config.learning_rate,
                    max_iter=config.max_epochs,
                    early_stopping=True,
                    validation_fraction=config.validation_fraction,
                    n_iter_no_change=config.patience,
                    random_state=config.random_seed,
                ),
            ),
        ]
    )


def _balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Both target classes are required.")
    return len(labels) / (2.0 * counts[labels])


def _choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Select a warning threshold using validation balanced accuracy."""
    candidates = np.unique(np.r_[0.05, np.arange(0.1, 0.91, 0.02), 0.95])
    scores = np.array(
        [balanced_accuracy_score(labels, probabilities >= value) for value in candidates]
    )
    best = candidates[np.isclose(scores, scores.max())]
    return float(best[np.argmin(np.abs(best - 0.5))])


def train_models(
    windows: pd.DataFrame,
    config: ModelConfig = ModelConfig(),
) -> dict[str, Any]:
    """Train and patient-holdout validate the classifier and timing network."""
    missing = sorted(set(FEATURE_COLUMNS) - set(windows.columns))
    if missing:
        raise ValueError(f"Window table is missing model features: {missing}")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=config.random_seed
    )
    train_index, test_index = next(
        splitter.split(windows, windows["seizure_within_60s"], groups=windows["patient_id"])
    )
    train = windows.iloc[train_index].copy()
    test = windows.iloc[test_index].copy()
    if train["seizure_within_60s"].nunique() != 2 or test["seizure_within_60s"].nunique() != 2:
        raise ValueError("Patient holdout split did not preserve both classes.")

    classifier = _make_classifier(config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        classifier.fit(
            train[FEATURE_COLUMNS],
            train["seizure_within_60s"],
            network__sample_weight=_balanced_sample_weights(
                train["seizure_within_60s"].to_numpy()
            ),
        )
    holdout_probability = classifier.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    warning_threshold = _choose_threshold(
        test["seizure_within_60s"].to_numpy(), holdout_probability
    )

    positive_train = train[train["seizure_within_60s"].eq(1)]
    positive_test = test[test["seizure_within_60s"].eq(1)]
    regressor = _make_regressor(config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        regressor.fit(positive_train[FEATURE_COLUMNS], positive_train["seconds_to_seizure"])
    holdout_seconds = np.clip(
        regressor.predict(positive_test[FEATURE_COLUMNS]),
        WINDOW_SECONDS / 2,
        PREDICTION_HORIZON_SECONDS - WINDOW_SECONDS / 2,
    )
    holdout_auc = float(
        roc_auc_score(test["seizure_within_60s"], holdout_probability)
    )
    holdout_balanced_accuracy = float(
        balanced_accuracy_score(
            test["seizure_within_60s"],
            holdout_probability >= warning_threshold,
        )
    )
    timing_mae = float(
        mean_absolute_error(positive_test["seconds_to_seizure"], holdout_seconds)
    )
    baseline_seconds = float(positive_train["seconds_to_seizure"].median())
    timing_baseline_mae = float(
        mean_absolute_error(
            positive_test["seconds_to_seizure"],
            np.full(len(positive_test), baseline_seconds),
        )
    )
    classifier_validated = holdout_auc >= 0.60 and holdout_balanced_accuracy >= 0.60
    timing_validated = timing_mae < timing_baseline_mae

    metrics = {
        "holdout_patients": sorted(test["patient_id"].unique().tolist()),
        "training_windows": int(len(train)),
        "holdout_windows": int(len(test)),
        "holdout_roc_auc": holdout_auc,
        "holdout_balanced_accuracy": holdout_balanced_accuracy,
        "holdout_timing_mae_seconds": timing_mae,
        "holdout_timing_baseline_mae_seconds": timing_baseline_mae,
        "warning_threshold": warning_threshold,
        "classifier_validated": classifier_validated,
        "timing_validated": timing_validated,
    }

    # Refit deployable models on all available patients after honest holdout metrics.
    final_classifier = _make_classifier(config)
    final_regressor = _make_regressor(config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        final_classifier.fit(
            windows[FEATURE_COLUMNS],
            windows["seizure_within_60s"],
            network__sample_weight=_balanced_sample_weights(
                windows["seizure_within_60s"].to_numpy()
            ),
        )
        positives = windows[windows["seizure_within_60s"].eq(1)]
        final_regressor.fit(positives[FEATURE_COLUMNS], positives["seconds_to_seizure"])

    return {
        "classifier": final_classifier,
        "regressor": final_regressor,
        "feature_columns": FEATURE_COLUMNS.copy(),
        "warning_threshold": warning_threshold,
        "classifier_validated": classifier_validated,
        "timing_validated": timing_validated,
        "metrics": metrics,
        "config": config,
    }


def extract_edf_features(
    edf_path: str | Path,
    window_start_seconds: int,
    window_seconds: int = WINDOW_SECONDS,
) -> pd.DataFrame:
    """Read one raw EDF window and calculate the exact model input features."""
    if window_seconds != WINDOW_SECONDS:
        raise ValueError(f"This model expects exactly {WINDOW_SECONDS}-second windows.")
    edf_path = Path(edf_path)
    data, sample_rate, eeg_indices, header = read_eeg_window(
        edf_path, int(window_start_seconds), window_seconds
    )
    powers, _, quality = window_bandpowers(
        data, sample_rate, eeg_indices, header, MIN_CHANNELS
    )
    usable_fraction = quality["artifact_usable_channel_count"] / max(
        quality["eeg_channel_count"], 1
    )
    row = {
        **{f"{band}_relative_power": powers[band] for band in BAND_ORDER},
        "artifact_channel_fraction": quality["artifact_channel_fraction"],
        "artifact_usable_channel_fraction": usable_fraction,
        "sample_rate_hz": quality["sample_rate_hz"],
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_edf_window(
    fitted: dict[str, Any],
    edf_path: str | Path,
    window_start_seconds: int,
) -> dict[str, Any]:
    """Score a raw EDF window for onset in the following 60 seconds."""
    features = extract_edf_features(edf_path, window_start_seconds)
    probability = float(fitted["classifier"].predict_proba(features)[0, 1])
    predicted_seconds = float(
        np.clip(
            fitted["regressor"].predict(features)[0],
            WINDOW_SECONDS / 2,
            PREDICTION_HORIZON_SECONDS - WINDOW_SECONDS / 2,
        )
    )
    threshold = float(fitted["warning_threshold"])
    classifier_validated = bool(fitted.get("classifier_validated", False))
    timing_validated = bool(fitted.get("timing_validated", False))
    warning = classifier_validated and probability >= threshold
    report_timing = warning and timing_validated
    return {
        "seizure_likelihood": probability,
        "prediction_horizon_seconds": PREDICTION_HORIZON_SECONDS,
        "warning_threshold": threshold,
        "warning": bool(warning),
        "classifier_validated": classifier_validated,
        "timing_validated": timing_validated,
        "predicted_seconds_to_seizure": predicted_seconds if report_timing else None,
        "edf": str(Path(edf_path)),
        "window_start_seconds": int(window_start_seconds),
        "window_end_seconds": int(window_start_seconds + WINDOW_SECONDS),
        "features": features.iloc[0].to_dict(),
        "interpretation": (
            f"Experimental score {probability:.3f}: estimated onset in "
            f"{predicted_seconds:.1f} seconds."
            if report_timing
            else (
                f"Experimental score {probability:.3f}; no countdown is reported "
                "because patient-holdout validation did not support deployment."
                if not (classifier_validated and timing_validated)
                else f"Experimental score {probability:.3f}: below the warning threshold."
            )
        ),
    }
