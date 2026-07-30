"""Discrete-time seizure forecasting from EDF scalp EEG recordings.

This module supports the companion ``rolling_seizure_forecasting.ipynb``.
It implements a configurable rolling forecast that updates at fixed landmarks.
The model keeps a fixed two-minute causal EEG context and predicts ``r`` minutes
into the future using bins of length ``i``. A discrete hazard model produces an
internally consistent distribution over the future bins plus a separate
``no seizure in the configured horizon`` outcome.

The current feature path is deliberately time-domain only. It does not compute
delta/theta/alpha/beta/gamma band powers or any FFT-based spectral features.

The implementation is an exploratory research model, not a medical device.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from concurrent.futures import ThreadPoolExecutor
import json
import math
import re
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import pyedflib
from scipy import signal
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODULE_VERSION = "8.2.0"
FEATURE_CACHE_VERSION = "7.1.0"
BIN_SECONDS = 5
HORIZON_SECONDS = 300
CONTEXT_SECONDS = 120
PRE_ONSET_SECONDS = 300
INTERICTAL_BUFFER_SECONDS = 300
TARGET_SAMPLE_RATE = 64
RANDOM_SEED = 42
MIN_EEG_CHANNELS = 8
CANONICAL_EEG_CHANNELS = (
    "FP1",
    "F3",
    "C3",
    "P3",
    "O1",
    "F7",
    "T3",
    "T5",
    "FC1",
    "FC5",
    "CP1",
    "CP5",
    "F9",
    "FZ",
    "CZ",
    "PZ",
    "F4",
    "C4",
    "P4",
    "O2",
    "F8",
    "T4",
    "T6",
    "FC2",
    "FC6",
    "CP2",
    "CP6",
    "F10",
    "FP2",
    "P9",
    "P10",
)

CHANNEL_BASE_FEATURE_NAMES = (
    "log_rms",
    "log_line_length",
    "log_mad",
    "log_robust_range",
    "zero_crossing_rate",
    "hjorth_mobility",
    "hjorth_complexity",
)
CHANNEL_DISTRIBUTION_NAMES = (
    "median",
    "channel_iqr",
    "channel_p90",
)
MICRO_FEATURE_NAMES = tuple(
    base if statistic == "median" else f"{base}_{statistic}"
    for statistic in CHANNEL_DISTRIBUTION_NAMES
    for base in CHANNEL_BASE_FEATURE_NAMES
) + (
    "median_absolute_channel_correlation",
    "usable_channel_fraction",
)

AGGREGATIONS = ("mean", "std", "last", "slope")
MODEL_FEATURE_COLUMNS = [
    f"{feature}_{aggregation}"
    for feature in MICRO_FEATURE_NAMES
    for aggregation in AGGREGATIONS
]
HAZARD_TIME_COLUMNS = ("lead_bin", "lead_fraction", "log1p_lead_seconds")
HAZARD_FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS + list(HAZARD_TIME_COLUMNS)

REQUESTED_SEIZURE_SPLITS: dict[str, dict[str, int]] = {
    "train": {
        "PN00": 4,
        "PN01": 2,
        "PN03": 2,
        "PN05": 2,
        "PN06": 4,
        "PN07": 1,
        "PN09": 2,
        "PN10": 9,
        "PN11": 1,
        "PN12": 3,
        "PN13": 2,
        "PN14": 4,
        "PN16": 2,
        "PN17": 1,
    },
    # Kept as an explicit empty key for backward-compatible allocation files.
    # Model and alarm selection use grouped out-of-fold train predictions.
    "validation": {},
    "test": {
        "PN00": 1,
        "PN05": 1,
        "PN06": 1,
        "PN09": 1,
        "PN10": 1,
        "PN12": 1,
        "PN13": 1,
        "PN17": 1,
    },
}

FIXED_TEST_SOURCE_EVENT_IDS = (
    "PN00_S01",
    "PN05_S03",
    "PN06_S04",
    "PN09_S03",
    "PN10_S08",
    "PN12_S01",
    "PN13_S01",
    "PN17_S02",
)


@dataclass(frozen=True)
class ForecastConfig:
    """Configuration used to build features and fit the hazard model."""

    bin_seconds: int = BIN_SECONDS
    horizon_seconds: int = HORIZON_SECONDS
    context_seconds: int = CONTEXT_SECONDS
    pre_onset_seconds: int = PRE_ONSET_SECONDS
    interictal_buffer_seconds: int = INTERICTAL_BUFFER_SECONDS
    target_sample_rate: int = TARGET_SAMPLE_RATE
    min_eeg_channels: int = MIN_EEG_CHANNELS
    included_eeg_channels: tuple[str, ...] | None = None
    random_seed: int = RANDOM_SEED
    test_fraction: float = 0.25
    warning_time_target: float = 0.25
    minimum_development_sensitivity: float | None = None
    false_alarm_target_per_hour: float = 5.0
    interictal_controls_per_seizure: int = 4
    alarm_on_consecutive: int = 2
    # At five-second landmarks, 13 low-risk readings are 65 seconds: strictly
    # more than one minute, as requested.
    alarm_off_consecutive: int = 13
    max_iter: int = 120
    learning_rate: float = 0.07
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 80
    l2_regularization: float = 1e-4

    @property
    def n_bins(self) -> int:
        return self.horizon_seconds // self.bin_seconds

    @property
    def context_bins(self) -> int:
        return self.context_seconds // self.bin_seconds

    @property
    def pre_onset_steps(self) -> int:
        """Number of trainable readings from -m through -i seconds."""

        return self.pre_onset_seconds // self.bin_seconds

    @property
    def reading_count(self) -> int:
        """Number of requested displays from -m through onset, inclusive."""

        return self.pre_onset_steps + 1

    @property
    def selected_eeg_channels(self) -> tuple[str, ...] | None:
        """Canonical selected channels, or None for legacy all-available mode."""

        if self.included_eeg_channels is None:
            return None
        return tuple(
            canonical_eeg_channel_name(channel)
            for channel in self.included_eeg_channels
        )

    @property
    def effective_min_eeg_channels(self) -> int:
        """Quality threshold capped by the number explicitly selected."""

        selected = self.selected_eeg_channels
        return (
            self.min_eeg_channels
            if selected is None
            else min(self.min_eeg_channels, len(selected))
        )

    def validate(self) -> None:
        if self.bin_seconds <= 0:
            raise ValueError("bin_seconds must be positive.")
        if self.horizon_seconds <= 0:
            raise ValueError("horizon_seconds must be positive.")
        if self.context_seconds <= 0:
            raise ValueError("context_seconds must be positive.")
        if self.horizon_seconds % self.bin_seconds:
            raise ValueError("horizon_seconds must be divisible by bin_seconds.")
        if self.context_seconds % self.bin_seconds:
            raise ValueError("context_seconds must be divisible by bin_seconds.")
        if self.pre_onset_seconds <= 0:
            raise ValueError("pre_onset_seconds must be positive.")
        if self.pre_onset_seconds % self.bin_seconds:
            raise ValueError("pre_onset_seconds must be divisible by bin_seconds.")
        if self.context_seconds != 120:
            raise ValueError(
                "This experiment requires exactly 120 seconds of past EEG context."
            )
        if self.context_bins < 2:
            raise ValueError(
                "The two-minute context must contain at least two intervals."
            )
        if self.target_sample_rate < 32:
            raise ValueError(
                "target_sample_rate must be at least 32 Hz for time-domain features."
            )
        if self.min_eeg_channels <= 0:
            raise ValueError("min_eeg_channels must be positive.")
        selected = self.selected_eeg_channels
        if selected is not None:
            if not selected:
                raise ValueError(
                    "included_eeg_channels must select at least one channel."
                )
            if len(set(selected)) != len(selected):
                raise ValueError(
                    "included_eeg_channels contains duplicate canonical names."
                )
            unknown = sorted(set(selected) - set(CANONICAL_EEG_CHANNELS))
            if unknown:
                raise ValueError(
                    "Unknown included EEG channels: " + ", ".join(unknown)
                )
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be in (0, 1).")
        if not 0 <= self.warning_time_target <= 1:
            raise ValueError("warning_time_target must be in [0, 1].")
        if (
            self.minimum_development_sensitivity is not None
            and not 0 <= self.minimum_development_sensitivity <= 1
        ):
            raise ValueError(
                "minimum_development_sensitivity must be in [0, 1]."
            )
        if self.false_alarm_target_per_hour < 0:
            raise ValueError(
                "false_alarm_target_per_hour must be nonnegative."
            )
        if self.interictal_controls_per_seizure <= 0:
            raise ValueError("interictal_controls_per_seizure must be positive.")
        if self.alarm_on_consecutive <= 0:
            raise ValueError("alarm_on_consecutive must be positive.")
        if self.alarm_off_consecutive <= 0:
            raise ValueError("alarm_off_consecutive must be positive.")


@dataclass
class TwoStageForecastModel:
    """Nonlinear horizon-risk model plus a discrete-hazard timing model."""

    timing_model: Any
    risk_model: Any
    risk_feature_columns: tuple[str, ...]
    smoothing_alpha: float = 1.0
    feature_medians: np.ndarray | None = None
    feature_scales: np.ndarray | None = None
    patient_feature_medians: dict[str, np.ndarray] | None = None
    patient_feature_scales: dict[str, np.ndarray] | None = None
    normalization_mode: str = "legacy"
    risk_model_name: str = "Extra Trees"
    risk_cv_auroc: float = float("nan")
    risk_oof_predictions: np.ndarray | None = None
    warning_policy: dict[str, float | int] | None = None


@dataclass
class CrossFittedRiskComponent:
    """One fold model and the training-only normalization it requires."""

    estimator: Any
    feature_medians: np.ndarray
    feature_scales: np.ndarray
    patient_feature_medians: dict[str, np.ndarray]
    patient_feature_scales: dict[str, np.ndarray]


@dataclass
class CrossFittedRiskEnsemble:
    """Average risk from models trained on complementary development folds."""

    components: tuple[CrossFittedRiskComponent, ...]
    normalization_mode: str

    def predict_feature_proba(
        self,
        frame: pd.DataFrame | None,
        base: np.ndarray,
    ) -> np.ndarray:
        probabilities = []
        for component in self.components:
            design = _normalized_risk_design(
                frame,
                base,
                component.feature_medians,
                component.feature_scales,
                component.patient_feature_medians,
                component.patient_feature_scales,
                self.normalization_mode,
            )
            probabilities.append(
                component.estimator.predict_proba(design)[:, 1]
            )
        mean_positive = np.mean(np.vstack(probabilities), axis=0)
        return np.column_stack([1.0 - mean_positive, mean_positive])


def project_paths(start: str | Path | None = None) -> dict[str, Path]:
    """Resolve project paths whether called from the repo root or notebook."""

    here = Path(start or Path.cwd()).resolve()
    candidates = [here, *here.parents]
    project_dir: Path | None = None
    for candidate in candidates:
        if (candidate / "final_project" / "data" / "raw").exists():
            project_dir = candidate / "final_project"
            break
        if candidate.name == "final_project" and (candidate / "data" / "raw").exists():
            project_dir = candidate
            break
    if project_dir is None:
        raise FileNotFoundError(
            "Could not locate final_project/data/raw from the current directory."
        )

    paths = {
        "project": project_dir,
        "raw": project_dir / "data" / "raw",
        "event_inventory": project_dir / "results" / "event_inventory.csv",
        "processed": project_dir / "data" / "processed" / "rolling_forecast",
        "results": project_dir / "results" / "rolling_forecast",
        "scripts": project_dir / "scripts",
    }
    return paths


def _clean_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_event_inventory(path: str | Path) -> pd.DataFrame:
    """Load the project's audited seizure annotations."""

    path = Path(path)
    events = pd.read_csv(path)
    required = {
        "patient_id",
        "event_id",
        "recording",
        "onset_seconds",
        "seizure_duration_seconds",
        "eligible_preictal",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Event inventory is missing columns: {missing}")
    events["eligible_preictal"] = _clean_bool(events["eligible_preictal"])
    events["onset_seconds"] = pd.to_numeric(events["onset_seconds"], errors="raise")
    events["seizure_duration_seconds"] = pd.to_numeric(
        events["seizure_duration_seconds"], errors="coerce"
    ).fillna(0)
    return events.sort_values(["patient_id", "recording", "onset_seconds"]).reset_index(
        drop=True
    )


def _resolve_edf(raw_dir: Path, patient_id: str, recording: str) -> Path:
    direct = raw_dir / patient_id / recording
    if direct.exists():
        return direct
    normalized = recording.upper().replace("O", "0").replace(" ", "")
    candidates = list((raw_dir / patient_id).glob("*.edf"))
    matches = [
        path
        for path in candidates
        if path.name.upper().replace("O", "0").replace(" ", "") == normalized
    ]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Could not uniquely resolve {recording!r} for patient {patient_id}."
    )


def rebase_manifest_edf_paths(
    manifest: pd.DataFrame,
    raw_dir: str | Path,
) -> pd.DataFrame:
    """Resolve cached machine-specific EDF paths without changing episodes."""

    required = {"patient_id", "recording", "edf_path"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing EDF path columns: {missing}")
    result = manifest.copy()
    resolved: dict[tuple[str, str], str] = {}
    for index, row in result.iterrows():
        current = Path(str(row["edf_path"]))
        if current.exists():
            resolved_path = current.resolve()
        else:
            key = (str(row["patient_id"]), str(row["recording"]))
            if key not in resolved:
                resolved[key] = str(
                    _resolve_edf(Path(raw_dir), key[0], key[1]).resolve()
                )
            resolved_path = Path(resolved[key])
        result.at[index, "edf_path"] = str(resolved_path)
    return result


def edf_metadata(path: str | Path) -> dict[str, Any]:
    """Read EDF metadata without loading the complete recording."""

    path = Path(path)
    reader = pyedflib.EdfReader(str(path))
    try:
        labels = [str(label).strip() for label in reader.getSignalLabels()]
        sample_rates = np.asarray(
            [reader.getSampleFrequency(i) for i in range(reader.signals_in_file)],
            dtype=float,
        )
        return {
            "path": path,
            "duration_seconds": float(reader.getFileDuration()),
            "labels": labels,
            "sample_rates": sample_rates,
            "signal_count": int(reader.signals_in_file),
        }
    finally:
        reader.close()


def eeg_channel_availability(
    edf_paths: Iterable[str | Path],
) -> pd.DataFrame:
    """Summarize canonical EEG channel presence across unique EDF files."""

    unique_paths = sorted({str(Path(path).resolve()) for path in edf_paths})
    counts = {channel: 0 for channel in CANONICAL_EEG_CHANNELS}
    for path in unique_paths:
        labels = edf_metadata(path)["labels"]
        present = {
            canonical_eeg_channel_name(label)
            for label in labels
            if _is_eeg_label(label)
        }
        for channel in counts:
            counts[channel] += int(channel in present)
    return pd.DataFrame(
        {
            "channel": list(CANONICAL_EEG_CHANNELS),
            "files_available": [
                counts[channel] for channel in CANONICAL_EEG_CHANNELS
            ],
            "total_files": len(unique_paths),
        }
    )


def _is_eeg_label(label: str) -> bool:
    upper = label.upper().replace("-", " ").strip()
    excluded = ("EKG", "ECG", "SPO2", "STATUS", "EVENT", "EMG", "RESP")
    return upper.startswith("EEG") and not any(token in upper for token in excluded)


def canonical_eeg_channel_name(label: str) -> str:
    """Normalize EDF labels such as ``EEG Fp2`` and ``EEG CZ`` to one name."""

    text = str(label).strip().upper()
    text = re.sub(r"^EEG[\s:_-]*", "", text)
    text = re.sub(r"[\s:_-]*(REF|LE|RE|AVG)$", "", text)
    canonical = re.sub(r"[^A-Z0-9]", "", text)
    if not canonical:
        raise ValueError(f"Cannot canonicalize EEG channel label {label!r}.")
    return canonical


def _select_eeg_channel_indices(
    labels: list[str],
    included_channels: Iterable[str] | None,
) -> tuple[list[int], list[str]]:
    """Return EDF indices in requested canonical order with strict validation."""

    recognized: dict[str, int] = {}
    ordered_available: list[str] = []
    for index, label in enumerate(labels):
        if not _is_eeg_label(label):
            continue
        canonical = canonical_eeg_channel_name(label)
        if canonical in recognized:
            raise ValueError(
                f"EDF contains duplicate canonical EEG channel {canonical}."
            )
        recognized[canonical] = index
        ordered_available.append(canonical)
    if included_channels is None:
        selected = ordered_available
    else:
        selected = [
            canonical_eeg_channel_name(channel)
            for channel in included_channels
        ]
        if len(set(selected)) != len(selected):
            raise ValueError("Selected EEG channels contain duplicates.")
        unknown = sorted(set(selected) - set(CANONICAL_EEG_CHANNELS))
        if unknown:
            raise ValueError(
                "Unknown selected EEG channels: " + ", ".join(unknown)
            )
        missing = [channel for channel in selected if channel not in recognized]
        if missing:
            raise ValueError(
                "Selected EEG channels are missing from this EDF: "
                + ", ".join(missing)
            )
    return [recognized[channel] for channel in selected], selected


def read_edf_eeg_segment(
    path: str | Path,
    start_seconds: float,
    duration_seconds: float,
    min_channels: int = MIN_EEG_CHANNELS,
    included_channels: Iterable[str] | None = None,
) -> tuple[np.ndarray, float, list[str]]:
    """Random-access a scaled multichannel EEG interval from an EDF file."""

    path = Path(path)
    if start_seconds < 0 or duration_seconds <= 0:
        raise ValueError("Requested EDF segment has invalid bounds.")
    reader = pyedflib.EdfReader(str(path))
    try:
        labels = [str(label).strip() for label in reader.getSignalLabels()]
        eeg_indices, selected_labels = _select_eeg_channel_indices(
            labels, included_channels
        )
        if len(eeg_indices) < min_channels:
            raise ValueError(
                f"{path.name} has only {len(eeg_indices)} selected EEG channels; "
                f"at least {min_channels} are required."
            )
        rates = np.asarray(
            [reader.getSampleFrequency(i) for i in eeg_indices], dtype=float
        )
        rounded_rates = np.round(rates, 6)
        values, counts = np.unique(rounded_rates, return_counts=True)
        sample_rate = float(values[np.argmax(counts)])
        rate_pairs = list(
            zip(eeg_indices, selected_labels, rounded_rates, strict=True)
        )
        eeg_indices = [
            index
            for index, _, rate in rate_pairs
            if math.isclose(float(rate), sample_rate, rel_tol=0, abs_tol=1e-6)
        ]
        selected_labels = [
            channel
            for _, channel, rate in rate_pairs
            if math.isclose(float(rate), sample_rate, rel_tol=0, abs_tol=1e-6)
        ]
        if included_channels is not None and len(eeg_indices) != len(rate_pairs):
            off_rate = [
                channel
                for _, channel, rate in rate_pairs
                if not math.isclose(
                    float(rate), sample_rate, rel_tol=0, abs_tol=1e-6
                )
            ]
            raise ValueError(
                "Selected EEG channels do not share the modal sample rate: "
                + ", ".join(off_rate)
            )
        if len(eeg_indices) < min_channels:
            raise ValueError(
                f"{path.name} has fewer than {min_channels} EEG channels at one rate."
            )

        start_sample = int(round(start_seconds * sample_rate))
        sample_count = int(round(duration_seconds * sample_rate))
        if start_sample + sample_count > int(reader.getNSamples()[eeg_indices[0]]):
            raise ValueError(
                f"Requested [{start_seconds}, {start_seconds + duration_seconds}) s "
                f"extends beyond {path.name}."
            )
        arrays = [
            np.asarray(
                reader.readSignal(index, start=start_sample, n=sample_count),
                dtype=np.float64,
            )
            for index in eeg_indices
        ]
        if any(len(values_) != sample_count for values_ in arrays):
            raise ValueError(f"Incomplete EDF read from {path.name}.")
        return np.vstack(arrays), sample_rate, selected_labels
    finally:
        reader.close()


def _quality_mask(window: np.ndarray) -> np.ndarray:
    finite = np.isfinite(window).all(axis=1)
    centered = window - np.nanmedian(window, axis=1, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(centered), axis=1)
    peak = np.nanmax(np.abs(centered), axis=1)
    flat = mad < 0.05
    extreme_scale = mad > 250.0
    extreme_peak = peak > 1500.0
    return finite & ~flat & ~extreme_scale & ~extreme_peak


def _micro_window_features(
    window: np.ndarray,
    sample_rate: float,
    total_channel_count: int,
    min_channels: int,
) -> np.ndarray:
    """Calculate fast robust time-domain features for one EEG interval."""

    # Median centering is substantially faster and less outlier-sensitive than
    # fitting a linear trend independently in every channel and interval.
    window = window - np.nanmedian(window, axis=1, keepdims=True)
    if window.shape[0] > 1:
        window = window - np.nanmedian(window, axis=0, keepdims=True)
    keep = _quality_mask(window)
    if int(keep.sum()) < min_channels:
        raise ValueError(
            f"Only {int(keep.sum())} usable EEG channels in the configured window."
    )
    window = window[keep]

    rms = np.sqrt(np.mean(np.square(window), axis=1))
    line_length = np.mean(np.abs(np.diff(window, axis=1)), axis=1)
    mad = 1.4826 * np.median(np.abs(window), axis=1)
    robust_range = np.percentile(window, 95, axis=1) - np.percentile(
        window, 5, axis=1
    )
    zero_crossings = np.mean(
        np.diff(np.signbit(window), axis=1) != 0, axis=1
    )
    signal_std = np.std(window, axis=1)
    diff_std = np.std(np.diff(window, axis=1), axis=1)
    mobility = diff_std / np.clip(signal_std, 1e-12, None)
    second_diff_std = np.std(np.diff(window, n=2, axis=1), axis=1)
    complexity = (
        second_diff_std / np.clip(diff_std, 1e-12, None)
    ) / np.clip(mobility, 1e-12, None)
    channel_values = np.column_stack(
        [
            np.log1p(rms),
            np.log1p(line_length),
            np.log1p(mad),
            np.log1p(robust_range),
            zero_crossings,
            mobility,
            complexity,
        ]
    )
    distributions: list[float] = []
    for feature_index in range(channel_values.shape[1]):
        values = channel_values[:, feature_index]
        distributions.extend(
            [
                float(np.median(values)),
                float(np.percentile(values, 75) - np.percentile(values, 25)),
                float(np.percentile(values, 90)),
            ]
        )
    if window.shape[0] > 1:
        correlation = np.corrcoef(window)
        upper = correlation[np.triu_indices(window.shape[0], k=1)]
        median_absolute_correlation = float(np.nanmedian(np.abs(upper)))
    else:
        median_absolute_correlation = 0.0

    return np.asarray(
        distributions
        + [
            median_absolute_correlation,
            float(keep.sum() / total_channel_count),
        ],
        dtype=float,
    )


def segment_micro_features(
    data: np.ndarray,
    source_sample_rate: float,
    config: ForecastConfig = ForecastConfig(),
) -> np.ndarray:
    """Convert a contiguous EEG segment to one feature row per five seconds."""

    config.validate()
    if data.ndim != 2:
        raise ValueError("EEG data must have shape (channels, samples).")
    if source_sample_rate < config.target_sample_rate:
        raise ValueError(
            "Source sample rate is below the configured target sample rate."
        )

    gcd = math.gcd(int(round(source_sample_rate)), config.target_sample_rate)
    up = config.target_sample_rate // gcd
    down = int(round(source_sample_rate)) // gcd
    if up != down:
        data = signal.resample_poly(data, up=up, down=down, axis=1)
    sample_rate = float(config.target_sample_rate)

    samples_per_bin = int(round(config.bin_seconds * sample_rate))
    n_windows = data.shape[1] // samples_per_bin
    if n_windows < 1:
        raise ValueError("EEG segment is shorter than one prediction bin.")
    data = data[:, : n_windows * samples_per_bin]
    features = []
    for index in range(n_windows):
        start = index * samples_per_bin
        stop = start + samples_per_bin
        features.append(
            _micro_window_features(
                data[:, start:stop],
                sample_rate,
                total_channel_count=data.shape[0],
                min_channels=config.effective_min_eeg_channels,
            )
        )
    result = np.vstack(features)
    if result.shape[1] != len(MICRO_FEATURE_NAMES):
        raise AssertionError("Unexpected micro-feature width.")
    return result


def aggregate_context(
    micro_features: np.ndarray, bin_seconds: int = BIN_SECONDS
) -> np.ndarray:
    """Aggregate a complete two-minute context into one feature vector."""

    if micro_features.ndim != 2 or micro_features.shape[1] != len(
        MICRO_FEATURE_NAMES
    ):
        raise ValueError(
            "Context micro-features have an unexpected shape: "
            f"{micro_features.shape}."
        )
    context_bins = micro_features.shape[0]
    if context_bins < 2:
        raise ValueError("At least two micro-windows are required for a context.")
    means = np.mean(micro_features, axis=0)
    stds = np.std(micro_features, axis=0)
    last = micro_features[-1]
    times = np.arange(context_bins, dtype=float) * bin_seconds
    centered_time = times - times.mean()
    slopes = (
        centered_time[:, None] * (micro_features - means[None, :])
    ).sum(axis=0) / np.square(centered_time).sum()

    values: list[float] = []
    for feature_index in range(len(MICRO_FEATURE_NAMES)):
        values.extend(
            [
                float(means[feature_index]),
                float(stds[feature_index]),
                float(last[feature_index]),
                float(slopes[feature_index]),
            ]
        )
    result = np.asarray(values, dtype=float)
    if len(result) != len(MODEL_FEATURE_COLUMNS):
        raise AssertionError("Unexpected aggregated feature width.")
    return result


def _safe_from_seizures(
    segment_start: float,
    segment_end: float,
    intervals: Iterable[tuple[float, float]],
    buffer_seconds: int,
) -> bool:
    for onset, end in intervals:
        protected_start = onset - buffer_seconds
        protected_end = end + buffer_seconds
        if segment_start < protected_end and segment_end > protected_start:
            return False
    return True


def build_episode_manifest(
    events: pd.DataFrame,
    raw_dir: str | Path,
    config: ForecastConfig = ForecastConfig(),
) -> pd.DataFrame:
    """Create one seizure episode and one matched interictal episode per event."""

    config.validate()
    raw_dir = Path(raw_dir)
    eligible = events.loc[events["eligible_preictal"]].copy()
    eligible["edf_path"] = [
        _resolve_edf(raw_dir, row.patient_id, row.recording)
        for row in eligible.itertuples()
    ]

    metadata_cache: dict[Path, dict[str, Any]] = {}

    def metadata(path: Path) -> dict[str, Any]:
        if path not in metadata_cache:
            metadata_cache[path] = edf_metadata(path)
        return metadata_cache[path]

    positive_rows: list[dict[str, Any]] = []
    for row in eligible.itertuples():
        path = Path(row.edf_path)
        duration = metadata(path)["duration_seconds"]
        anchor = float(row.onset_seconds) - config.pre_onset_seconds
        if anchor < config.context_seconds or row.onset_seconds > duration:
            continue
        positive_rows.append(
            {
                "episode_id": f"{row.event_id}_preictal",
                "patient_id": row.patient_id,
                "source_event_id": row.event_id,
                "episode_type": "preictal",
                "recording": row.recording,
                "edf_path": str(path),
                "anchor_seconds": anchor,
                "training_episode_end_seconds": (
                    anchor + config.pre_onset_seconds
                ),
                "event_onset_seconds": float(row.onset_seconds),
            }
        )
    positives = pd.DataFrame(positive_rows)
    if positives.empty:
        raise ValueError("No seizure has enough context for the requested episode.")

    interval_lookup: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for (patient_id, recording), group in events.groupby(
        ["patient_id", "recording"], sort=False
    ):
        interval_lookup[(patient_id, recording)] = [
            (float(row.onset_seconds), float(row.onset_seconds + row.seizure_duration_seconds))
            for row in group.itertuples()
        ]

    negative_rows: list[dict[str, Any]] = []
    for patient_id, patient_positives in positives.groupby("patient_id", sort=True):
        patient_events = events.loc[events["patient_id"].eq(patient_id)]
        recordings = sorted(patient_events["recording"].unique())
        candidates: list[dict[str, Any]] = []
        for recording in recordings:
            path = _resolve_edf(raw_dir, patient_id, recording)
            duration = metadata(path)["duration_seconds"]
            intervals = interval_lookup.get((patient_id, recording), [])
            latest_anchor = int(
                math.floor(
                    duration
                    - config.pre_onset_seconds
                    - config.horizon_seconds
                )
            )
            for anchor in range(
                config.context_seconds, latest_anchor + 1, 60
            ):
                # Context covers 120 s before the anchor. Labels for the final
                # trainable landmark require a full forecast horizon afterward.
                segment_start = anchor - config.context_seconds
                label_followup_end = (
                    anchor
                    + config.pre_onset_seconds
                    + config.horizon_seconds
                )
                if _safe_from_seizures(
                    segment_start,
                    label_followup_end,
                    intervals,
                    config.interictal_buffer_seconds,
                ):
                    candidates.append(
                        {
                            "recording": recording,
                            "edf_path": str(path),
                            "anchor_seconds": float(anchor),
                        }
                    )
        if not candidates:
            raise ValueError(
                f"No safe interictal episode found for {patient_id}. "
                "Reduce the buffer only if scientifically justified."
            )

        needed = len(patient_positives) * config.interictal_controls_per_seizure
        candidate_frame = pd.DataFrame(candidates).drop_duplicates(
            ["edf_path", "anchor_seconds"]
        )
        candidate_frame = candidate_frame.sort_values(
            ["recording", "anchor_seconds"]
        ).reset_index(drop=True)
        chosen_indices = np.unique(
            np.linspace(0, len(candidate_frame) - 1, num=min(needed, len(candidate_frame)))
            .round()
            .astype(int)
        )
        chosen = candidate_frame.iloc[chosen_indices].copy()
        if len(chosen) < needed:
            raise ValueError(
                f"{patient_id} has {len(chosen)} safe controls for {needed} seizures."
            )
        for sequence, row in enumerate(chosen.itertuples(), start=1):
            negative_rows.append(
                {
                    "episode_id": f"{patient_id}_interictal_{sequence:02d}",
                    "patient_id": patient_id,
                    "source_event_id": "",
                    "episode_type": "interictal",
                    "recording": row.recording,
                    "edf_path": row.edf_path,
                    "anchor_seconds": float(row.anchor_seconds),
                    "training_episode_end_seconds": float(
                        row.anchor_seconds + config.pre_onset_seconds
                    ),
                    "event_onset_seconds": np.nan,
                }
            )

    manifest = pd.concat(
        [positives, pd.DataFrame(negative_rows)], ignore_index=True
    ).sort_values(["patient_id", "episode_type", "episode_id"])
    counts = manifest.groupby(["patient_id", "episode_type"]).size().unstack(fill_value=0)
    expected_controls = counts["preictal"] * config.interictal_controls_per_seizure
    if not (counts["interictal"] == expected_controls).all():
        raise AssertionError("Interictal episode counts do not match the configured ratio.")
    return manifest.reset_index(drop=True)


def assign_requested_splits(
    manifest: pd.DataFrame,
    split_quotas: dict[str, dict[str, int]] = REQUESTED_SEIZURE_SPLITS,
    random_seed: int = RANDOM_SEED,
    interictal_controls_per_seizure: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly assign disjoint seizures and controls to exact splits.

    The requested design deliberately allows the same patient to occur in
    multiple splits. Individual seizure episodes and interictal episodes remain
    disjoint. A fixed seed makes the random assignment exactly reproducible.
    """

    required = {"patient_id", "episode_id", "episode_type", "source_event_id"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing split columns: {missing}")
    if set(split_quotas) != {"train", "validation", "test"}:
        raise ValueError("Split quotas must define train, validation, and test.")

    result = manifest.copy()
    result["dataset_split"] = ""
    result["split_selection_rank"] = -1
    rng = np.random.default_rng(random_seed)
    allocation_rows: list[dict[str, Any]] = []

    patients = sorted(
        set().union(*(patient_counts.keys() for patient_counts in split_quotas.values()))
    )
    for patient_id in patients:
        patient_positive = result.index[
            result["patient_id"].eq(patient_id)
            & result["episode_type"].eq("preictal")
        ].to_numpy()
        patient_negative = result.index[
            result["patient_id"].eq(patient_id)
            & result["episode_type"].eq("interictal")
        ].to_numpy()
        required_count = sum(
            split_quotas[split_name].get(patient_id, 0)
            for split_name in ("train", "validation", "test")
        )
        if len(patient_positive) != required_count:
            raise ValueError(
                f"{patient_id} has {len(patient_positive)} eligible seizures, "
                f"but the requested quotas require {required_count}."
            )
        required_negative = required_count * interictal_controls_per_seizure
        if len(patient_negative) != required_negative:
            raise ValueError(
                f"{patient_id} has {len(patient_negative)} controls for "
                f"{required_count} requested seizures at a "
                f"{interictal_controls_per_seizure}:1 control ratio."
            )

        positive_order = rng.permutation(patient_positive)
        negative_order = rng.permutation(patient_negative)
        positive_cursor = 0
        negative_cursor = 0
        for split_name in ("train", "validation", "test"):
            count = split_quotas[split_name].get(patient_id, 0)
            negative_count = count * interictal_controls_per_seizure
            selected_positive = positive_order[
                positive_cursor : positive_cursor + count
            ]
            selected_negative = negative_order[
                negative_cursor : negative_cursor + negative_count
            ]
            for rank, row_index in enumerate(selected_positive, start=1):
                result.loc[row_index, ["dataset_split", "split_selection_rank"]] = [
                    split_name,
                    rank,
                ]
                allocation_rows.append(
                    {
                        "dataset_split": split_name,
                        "patient_id": patient_id,
                        "selection_rank": rank,
                        "source_event_id": result.at[row_index, "source_event_id"],
                        "episode_id": result.at[row_index, "episode_id"],
                        "recording": result.at[row_index, "recording"],
                    }
                )
            for rank, row_index in enumerate(selected_negative, start=1):
                result.loc[row_index, ["dataset_split", "split_selection_rank"]] = [
                    split_name,
                    rank,
                ]
            positive_cursor += count
            negative_cursor += negative_count

    unused = result.loc[result["dataset_split"].eq("")]
    if not unused.empty:
        raise ValueError(
            "Every episode must be assigned exactly once; unassigned episodes: "
            + ", ".join(unused["episode_id"].astype(str).head(10))
        )
    allocation = pd.DataFrame(allocation_rows).sort_values(
        ["dataset_split", "patient_id", "selection_rank"]
    )
    duplicated_events = allocation["source_event_id"].duplicated().any()
    if duplicated_events:
        raise AssertionError("A seizure was assigned to more than one split.")
    return result.reset_index(drop=True), allocation.reset_index(drop=True)


def _episode_landmarks(
    episode: pd.Series,
    config: ForecastConfig,
) -> pd.DataFrame:
    """Extract trainable readings from ``m`` minutes before onset to ``-i``."""

    read_start = float(episode["anchor_seconds"]) - config.context_seconds
    read_duration = config.context_seconds + config.pre_onset_seconds
    data, sample_rate, labels = read_edf_eeg_segment(
        episode["edf_path"],
        read_start,
        read_duration,
        min_channels=config.effective_min_eeg_channels,
        included_channels=config.selected_eeg_channels,
    )
    micro = segment_micro_features(data, sample_rate, config)
    expected_micro_rows = (
        config.context_seconds + config.pre_onset_seconds
    ) // config.bin_seconds
    if len(micro) != expected_micro_rows:
        raise ValueError(
            f"Expected {expected_micro_rows} micro-windows, obtained {len(micro)}."
        )

    rows: list[dict[str, Any]] = []
    is_event = episode["episode_type"] == "preictal"
    for step in range(config.pre_onset_steps):
        context = micro[step : step + config.context_bins]
        aggregated = aggregate_context(context, config.bin_seconds)
        landmark = float(episode["anchor_seconds"]) + step * config.bin_seconds
        if is_event:
            time_to_event = float(episode["event_onset_seconds"]) - landmark
            has_event = 0 < time_to_event <= config.horizon_seconds
            event_bin = (
                int(math.ceil(time_to_event / config.bin_seconds) - 1)
                if has_event
                else -1
            )
            if has_event and not 0 <= event_bin < config.n_bins:
                raise AssertionError("Preictal landmark target is outside the horizon.")
        else:
            time_to_event = np.nan
            event_bin = -1
            has_event = False
        row = {
            "episode_id": episode["episode_id"],
            "patient_id": episode["patient_id"],
            "source_event_id": episode["source_event_id"],
            "episode_type": episode["episode_type"],
            "dataset_split": episode.get("dataset_split", ""),
            "recording": episode["recording"],
            "edf_path": episode["edf_path"],
            "anchor_seconds": float(episode["anchor_seconds"]),
            "training_episode_end_seconds": float(
                episode["training_episode_end_seconds"]
            ),
            "landmark_step": step,
            "landmark_seconds": landmark,
            "time_to_event_seconds": time_to_event,
            "event_bin": event_bin,
            "has_event_in_horizon": int(has_event),
            "source_sample_rate": sample_rate,
            "eeg_channel_count": len(labels),
        }
        row.update(dict(zip(MODEL_FEATURE_COLUMNS, aggregated, strict=True)))
        rows.append(row)
    return pd.DataFrame(rows)


def _cache_signature(
    manifest: pd.DataFrame, config: ForecastConfig
) -> dict[str, Any]:
    signature_columns = [
        "episode_id",
        "patient_id",
        "source_event_id",
        "episode_type",
        "recording",
        "edf_path",
        "anchor_seconds",
        "training_episode_end_seconds",
        "event_onset_seconds",
    ]
    if "dataset_split" in manifest:
        signature_columns.append("dataset_split")
    signature_manifest = (
        manifest[signature_columns]
        .copy()
        .sort_values("episode_id")
    )
    text_columns = [
        "episode_id",
        "patient_id",
        "source_event_id",
        "episode_type",
        "recording",
        "edf_path",
    ]
    if "dataset_split" in signature_manifest:
        text_columns.append("dataset_split")
    for column in text_columns:
        # CSV round-tripping converts empty control event IDs to NaN. Treat
        # blank and missing identifiers identically for cache validity.
        signature_manifest[column] = (
            signature_manifest[column].fillna("").astype(str)
        )
    signature_manifest = signature_manifest.astype(object)
    signature_manifest = signature_manifest.where(
        pd.notna(signature_manifest), None
    )
    signature_config = asdict(config)
    signature_config["included_eeg_channels"] = (
        list(config.selected_eeg_channels)
        if config.selected_eeg_channels is not None
        else None
    )
    return {
        "module_version": FEATURE_CACHE_VERSION,
        "config": signature_config,
        "manifest": signature_manifest.to_dict(orient="records"),
    }


def _feature_signature_matches(
    cached_signature: dict[str, Any], current_signature: dict[str, Any]
) -> bool:
    """Compare cache signatures while ignoring alarm/model-only settings.

    Changing an operational alarm threshold or its consecutive-reading rule
    must not force EDF rereads: neither changes the signal features or labels.
    """

    data_fields = {
        "bin_seconds",
        "horizon_seconds",
        "context_seconds",
        "pre_onset_seconds",
        "target_sample_rate",
        "min_eeg_channels",
        "included_eeg_channels",
    }

    def canonical(signature: dict[str, Any]) -> dict[str, Any]:
        result = dict(signature)
        result["config"] = {
            key: value
            for key, value in signature.get("config", {}).items()
            if key in data_fields
        }
        # The signal features and targets are properties of an episode, not of
        # the split to which it is assigned. Ignoring this metadata lets a new
        # train/validation allocation reuse the exact same extracted features.
        result["manifest"] = [
            {
                key: value
                for key, value in record.items()
                if key != "dataset_split"
            }
            for record in signature.get("manifest", [])
        ]
        return result

    return canonical(cached_signature) == canonical(current_signature)


def _attach_current_dataset_splits(
    landmarks: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    """Replace cached split labels with the current manifest assignment."""

    split_by_episode = (
        manifest[["episode_id", "dataset_split"]]
        .drop_duplicates("episode_id")
        .set_index("episode_id")["dataset_split"]
    )
    result = landmarks.copy()
    result["dataset_split"] = result["episode_id"].map(split_by_episode)
    if result["dataset_split"].isna().any():
        missing = sorted(
            result.loc[result["dataset_split"].isna(), "episode_id"]
            .astype(str)
            .unique()
        )
        raise ValueError(
            "Cached landmark episodes are missing from the current manifest: "
            f"{missing[:5]}"
        )
    return result


def build_landmark_dataset(
    manifest: pd.DataFrame,
    cache_csv: str | Path | None = None,
    cache_metadata_json: str | Path | None = None,
    config: ForecastConfig = ForecastConfig(),
    force: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build or reuse the 5-second landmark feature table."""

    config.validate()
    signature = _cache_signature(manifest, config)
    cache_csv = Path(cache_csv) if cache_csv is not None else None
    cache_metadata_json = (
        Path(cache_metadata_json) if cache_metadata_json is not None else None
    )
    if (
        not force
        and cache_csv is not None
        and cache_metadata_json is not None
        and cache_csv.exists()
        and cache_metadata_json.exists()
    ):
        try:
            cached_signature = json.loads(cache_metadata_json.read_text("utf-8"))
            cached = pd.read_csv(cache_csv)
            if (
                _feature_signature_matches(cached_signature, signature)
                and set(MODEL_FEATURE_COLUMNS).issubset(cached.columns)
                and len(cached) == len(manifest) * config.pre_onset_steps
            ):
                cached = _attach_current_dataset_splits(cached, manifest)
                if verbose:
                    print(f"Loaded {len(cached):,} cached landmark rows.")
                return cached
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    # Keep a small, signature-checked cache per episode while the combined
    # table is being built. EDF extraction is the expensive operation; this
    # makes recovery from one unusable recording resumable instead of forcing
    # a complete reread of every prior recording.
    episode_cache_dir: Path | None = None
    if cache_csv is not None:
        episode_cache_dir = cache_csv.parent / f"{cache_csv.stem}_episodes"
        episode_cache_dir.mkdir(parents=True, exist_ok=True)

    episodes = [episode.copy() for _, episode in manifest.iterrows()]

    def load_or_extract(episode: pd.Series) -> pd.DataFrame:
        episode_frame: pd.DataFrame | None = None
        episode_signature = _cache_signature(
            pd.DataFrame([episode.to_dict()]), config
        )
        if episode_cache_dir is not None:
            safe_id = str(episode["episode_id"]).replace("/", "_").replace("\\", "_")
            episode_cache = episode_cache_dir / f"{safe_id}.pkl"
            episode_metadata = episode_cache_dir / f"{safe_id}.json"
            if episode_cache.exists() and episode_metadata.exists():
                try:
                    if _feature_signature_matches(
                        json.loads(episode_metadata.read_text("utf-8")),
                        episode_signature,
                    ):
                        candidate = pd.read_pickle(episode_cache)
                        if (
                            len(candidate) == config.pre_onset_steps
                            and set(MODEL_FEATURE_COLUMNS).issubset(candidate.columns)
                        ):
                            episode_frame = candidate
                except (OSError, ValueError, json.JSONDecodeError):
                    episode_frame = None
        if episode_frame is None:
            try:
                episode_frame = _episode_landmarks(episode, config)
            except ValueError as error:
                raise ValueError(
                    f"Episode {episode['episode_id']} ({episode['recording']}) "
                    f"cannot be used: {error}"
                ) from error
            if episode_cache_dir is not None:
                episode_frame.to_pickle(episode_cache)
                episode_metadata.write_text(
                    json.dumps(episode_signature, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
        return episode_frame

    # Different EDF files can be read concurrently, but pyEDFlib is not safe
    # when separate threads open the same recording. Group episodes by file so
    # each recording is handled serially within one worker.
    if verbose:
        print(
            f"Building {len(episodes)} episode feature tables with up to 4 EDF workers.",
            flush=True,
        )
    grouped_episodes: dict[str, list[pd.Series]] = {}
    for episode in episodes:
        grouped_episodes.setdefault(str(episode["edf_path"]), []).append(episode)

    def load_recording_group(group: list[pd.Series]) -> list[pd.DataFrame]:
        return [load_or_extract(episode) for episode in group]

    with ThreadPoolExecutor(max_workers=min(4, len(episodes))) as executor:
        grouped_frames = list(
            executor.map(load_recording_group, grouped_episodes.values())
        )
    frame_by_episode = {
        str(frame.iloc[0]["episode_id"]): frame
        for group in grouped_frames
        for frame in group
    }
    frames = [frame_by_episode[str(episode["episode_id"])] for episode in episodes]
    landmarks = pd.concat(frames, ignore_index=True)
    landmarks = _attach_current_dataset_splits(landmarks, manifest)
    expected_rows = len(manifest) * config.pre_onset_steps
    if len(landmarks) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, found {len(landmarks)}.")
    if not np.isfinite(landmarks[MODEL_FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("Non-finite values were found in model features.")

    if cache_csv is not None and cache_metadata_json is not None:
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        cache_metadata_json.parent.mkdir(parents=True, exist_ok=True)
        landmarks.to_csv(cache_csv, index=False)
        cache_metadata_json.write_text(
            json.dumps(signature, indent=2, sort_keys=True), encoding="utf-8"
        )
        if verbose:
            print(f"Saved landmark cache to {cache_csv}.")
    return landmarks


def split_by_patient(
    landmarks: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Create a patient-disjoint development/holdout split."""

    patients = landmarks[["patient_id"]].drop_duplicates().reset_index(drop=True)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config.test_fraction,
        random_state=config.random_seed,
    )
    train_index, test_index = next(
        splitter.split(patients, groups=patients["patient_id"])
    )
    train_patients = sorted(patients.iloc[train_index]["patient_id"].tolist())
    test_patients = sorted(patients.iloc[test_index]["patient_id"].tolist())
    train = landmarks.loc[landmarks["patient_id"].isin(train_patients)].copy()
    test = landmarks.loc[landmarks["patient_id"].isin(test_patients)].copy()
    for name, frame in {"development": train, "holdout": test}.items():
        outcome_column = (
            "has_event_in_horizon"
            if "has_event_in_horizon" in frame
            else "has_event_in_5m"
        )
        if frame[outcome_column].nunique() != 2:
            raise ValueError(f"{name} split does not contain both outcome classes.")
    return train, test, {
        "development_patients": train_patients,
        "holdout_patients": test_patients,
    }


def split_by_assignment(
    landmarks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return the exact event-disjoint train/validation/test assignment."""

    if "dataset_split" not in landmarks:
        raise ValueError("Landmarks do not contain dataset_split assignments.")
    frames = {
        name: landmarks.loc[landmarks["dataset_split"].eq(name)].copy()
        for name in ("train", "validation", "test")
    }
    for name, frame in frames.items():
        if name == "validation" and frame.empty:
            continue
        if frame.empty:
            raise ValueError(f"The {name} split is empty.")
        if frame["has_event_in_horizon"].nunique() != 2:
            raise ValueError(
                f"The {name} split lacks event/no-event horizon labels."
            )
    event_sets = {
        name: set(
            frame.loc[
                frame["episode_type"].eq("preictal"), "source_event_id"
            ].dropna()
        )
        for name, frame in frames.items()
    }
    if (
        event_sets["train"] & event_sets["validation"]
        or event_sets["train"] & event_sets["test"]
        or event_sets["validation"] & event_sets["test"]
    ):
        raise AssertionError("Seizure events overlap across dataset splits.")
    split_info = {
        f"{name}_patients": sorted(frame["patient_id"].unique().tolist())
        for name, frame in frames.items()
    }
    split_info["design"] = (
        "event-disjoint; patient overlap intentionally follows requested quotas"
    )
    return frames["train"], frames["validation"], frames["test"], split_info


def split_development_test(
    landmarks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return all non-test episodes for development and the fixed test set."""

    train, validation, test, split_info = split_by_assignment(landmarks)
    development = pd.concat([train, validation], ignore_index=True)
    if development["has_event_in_horizon"].nunique() != 2:
        raise ValueError("The development split lacks event/no-event labels.")
    development_events = set(
        development.loc[
            development["episode_type"].eq("preictal"), "source_event_id"
        ].dropna()
    )
    test_events = set(
        test.loc[test["episode_type"].eq("preictal"), "source_event_id"].dropna()
    )
    if development_events & test_events:
        raise AssertionError("Development and test seizures overlap.")
    split_info = {
        "development_patients": sorted(
            development["patient_id"].unique().tolist()
        ),
        "test_patients": sorted(test["patient_id"].unique().tolist()),
        "design": (
            "39-seizure development set with grouped out-of-fold tuning; "
            "fixed event-disjoint 8-seizure test set"
        ),
    }
    return development, test, split_info


def make_person_period(
    landmarks: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand landmarks into at-risk bin rows for discrete-hazard likelihood."""

    base = landmarks[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)
    event_bins = landmarks["event_bin"].to_numpy(dtype=int)
    has_event = landmarks["has_event_in_horizon"].to_numpy(dtype=bool)
    counts = np.where(has_event, event_bins + 1, config.n_bins).astype(int)
    total_rows = int(counts.sum())
    x = np.empty((total_rows, len(HAZARD_FEATURE_COLUMNS)), dtype=np.float32)
    y = np.zeros(total_rows, dtype=np.uint8)
    weights = np.empty(total_rows, dtype=np.float32)

    episode_period_totals = (
        pd.Series(counts, index=landmarks.index)
        .groupby(landmarks["episode_id"])
        .transform("sum")
        .to_numpy(dtype=float)
    )
    cursor = 0
    for row_index, count in enumerate(counts):
        stop = cursor + count
        x[cursor:stop, : len(MODEL_FEATURE_COLUMNS)] = base[row_index]
        lead_bins = np.arange(count, dtype=float)
        x[cursor:stop, -3] = lead_bins
        x[cursor:stop, -2] = (lead_bins + 0.5) / config.n_bins
        x[cursor:stop, -1] = np.log1p(
            (lead_bins + 0.5) * config.bin_seconds
        )
        if has_event[row_index]:
            y[stop - 1] = 1
        # Equal total influence per episode despite overlapping landmarks.
        weights[cursor:stop] = 1.0 / episode_period_totals[row_index]
        cursor = stop
    if cursor != total_rows:
        raise AssertionError("Person-period expansion length mismatch.")
    weights *= len(weights) / weights.sum()
    return x, y, weights


def _robust_location_scale(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite median/IQR statistics for the unchanged feature frame."""

    values = frame[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)
    medians = np.nanmedian(values, axis=0)
    q25, q75 = np.nanpercentile(values, [25, 75], axis=0)
    scales = q75 - q25
    standard_deviation = np.nanstd(values, axis=0)
    scales = np.where(scales > 1e-8, scales, standard_deviation)
    scales = np.where(scales > 1e-8, scales, 1.0)
    if not np.isfinite(medians).all() or not np.isfinite(scales).all():
        raise ValueError("Risk normalization statistics must be finite.")
    return medians, scales


def _risk_normalization_statistics(
    development: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Learn global and patient baselines from development interictal EEG only."""

    interictal = development.loc[
        development["episode_type"].eq("interictal")
    ]
    reference = interictal if not interictal.empty else development
    medians, scales = _robust_location_scale(reference)
    patient_medians: dict[str, np.ndarray] = {}
    patient_scales: dict[str, np.ndarray] = {}
    for patient_id, patient_frame in reference.groupby("patient_id"):
        patient_medians[str(patient_id)], patient_scales[str(patient_id)] = (
            _robust_location_scale(patient_frame)
        )
    return medians, scales, patient_medians, patient_scales


def _normalized_risk_design(
    frame: pd.DataFrame | None,
    base: np.ndarray,
    medians: np.ndarray,
    scales: np.ndarray,
    patient_medians: dict[str, np.ndarray],
    patient_scales: dict[str, np.ndarray],
    mode: str,
) -> np.ndarray:
    """Build an internal risk matrix without changing public input columns."""

    global_z = np.clip((base - medians) / scales, -8.0, 8.0)
    if mode == "global":
        return global_z.astype(np.float32)

    patient_z = global_z.copy()
    baseline_available = np.zeros((len(base), 1), dtype=np.float32)
    if frame is not None and "patient_id" in frame:
        patient_ids = frame["patient_id"].astype(str).to_numpy()
        for patient_id in np.unique(patient_ids):
            if patient_id not in patient_medians:
                continue
            rows = patient_ids == patient_id
            patient_z[rows] = np.clip(
                (base[rows] - patient_medians[patient_id])
                / patient_scales[patient_id],
                -8.0,
                8.0,
            )
            baseline_available[rows] = 1.0
    if mode == "patient":
        return patient_z.astype(np.float32)
    if mode == "combined":
        return np.column_stack(
            [global_z, patient_z, baseline_available]
        ).astype(np.float32)
    raise ValueError(f"Unknown risk normalization mode: {mode}")


def _episode_balanced_sample_weight(frame: pd.DataFrame) -> np.ndarray:
    """Give every seizure/control episode equal total fitting influence."""

    episode_sizes = frame.groupby("episode_id")["episode_id"].transform("size")
    weights = 1.0 / episode_sizes.to_numpy(dtype=float)
    return weights * (len(weights) / weights.sum())


def _fit_selected_risk_model(
    development: pd.DataFrame,
    medians: np.ndarray,
    scales: np.ndarray,
    patient_medians: dict[str, np.ndarray],
    patient_scales: dict[str, np.ndarray],
    config: ForecastConfig,
) -> tuple[Any, str, str, float, np.ndarray]:
    """Select risk model and produce leakage-resistant out-of-fold scores."""

    base = development[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)
    labels = development["has_event_in_horizon"].to_numpy(dtype=int)
    groups = development["episode_id"].astype(str).to_numpy()
    weights = _episode_balanced_sample_weight(development)
    candidates = (
        ("global", 5, 0.5),
        ("global", 15, 1.0),
        ("patient", 5, 0.5),
        ("patient", 15, 1.0),
        ("combined", 10, "sqrt"),
        ("combined", 20, 0.5),
    )
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        raise ValueError("At least two training episodes are required.")
    folds = list(
        StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=config.random_seed,
        ).split(base, labels, groups)
    )
    scored: list[
        tuple[float, str, int, float | str, np.ndarray]
    ] = []
    for candidate_number, (mode, min_leaf, max_features) in enumerate(
        candidates
    ):
        out_of_fold_risk = np.full(len(development), np.nan, dtype=float)
        for fold_number, (fit_rows, score_rows) in enumerate(folds):
            if (
                np.unique(labels[fit_rows]).size < 2
                or np.unique(labels[score_rows]).size < 2
            ):
                continue
            fit_frame = development.iloc[fit_rows]
            score_frame = development.iloc[score_rows]
            (
                fold_medians,
                fold_scales,
                fold_patient_medians,
                fold_patient_scales,
            ) = _risk_normalization_statistics(fit_frame)
            fit_design = _normalized_risk_design(
                fit_frame,
                base[fit_rows],
                fold_medians,
                fold_scales,
                fold_patient_medians,
                fold_patient_scales,
                mode,
            )
            score_design = _normalized_risk_design(
                score_frame,
                base[score_rows],
                fold_medians,
                fold_scales,
                fold_patient_medians,
                fold_patient_scales,
                mode,
            )
            estimator = ExtraTreesClassifier(
                n_estimators=160,
                min_samples_leaf=min_leaf,
                max_features=max_features,
                class_weight="balanced",
                n_jobs=-1,
                random_state=(
                    config.random_seed
                    + candidate_number * 100
                    + fold_number
                ),
            )
            estimator.fit(
                fit_design,
                labels[fit_rows],
                sample_weight=weights[fit_rows],
            )
            out_of_fold_risk[score_rows] = estimator.predict_proba(
                score_design
            )[:, 1]
        if not np.isfinite(out_of_fold_risk).all():
            mean_score = float("-inf")
        else:
            mean_score = float(
                roc_auc_score(labels, out_of_fold_risk)
            )
        scored.append(
            (
                mean_score,
                mode,
                min_leaf,
                max_features,
                out_of_fold_risk,
            )
        )

    (
        best_score,
        best_mode,
        best_min_leaf,
        best_max_features,
        best_oof_risk,
    ) = max(
        scored, key=lambda item: (item[0], -item[2])
    )
    # Retain the fold models for deployment. Their averaged predictions have
    # the same out-of-sample character as the scores used to tune the alarm,
    # unlike a refit model whose probability scale can shift substantially.
    components: list[CrossFittedRiskComponent] = []
    best_oof_risk = np.full(len(development), np.nan, dtype=float)
    for fold_number, (fit_rows, score_rows) in enumerate(folds):
        fit_frame = development.iloc[fit_rows]
        score_frame = development.iloc[score_rows]
        (
            fold_medians,
            fold_scales,
            fold_patient_medians,
            fold_patient_scales,
        ) = _risk_normalization_statistics(fit_frame)
        fit_design = _normalized_risk_design(
            fit_frame,
            base[fit_rows],
            fold_medians,
            fold_scales,
            fold_patient_medians,
            fold_patient_scales,
            best_mode,
        )
        score_design = _normalized_risk_design(
            score_frame,
            base[score_rows],
            fold_medians,
            fold_scales,
            fold_patient_medians,
            fold_patient_scales,
            best_mode,
        )
        estimator = ExtraTreesClassifier(
            n_estimators=240,
            min_samples_leaf=best_min_leaf,
            max_features=best_max_features,
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.random_seed + 10_000 + fold_number,
        )
        estimator.fit(
            fit_design,
            labels[fit_rows],
            sample_weight=weights[fit_rows],
        )
        best_oof_risk[score_rows] = estimator.predict_proba(
            score_design
        )[:, 1]
        components.append(
            CrossFittedRiskComponent(
                estimator=estimator,
                feature_medians=fold_medians,
                feature_scales=fold_scales,
                patient_feature_medians=fold_patient_medians,
                patient_feature_scales=fold_patient_scales,
            )
        )
    if not np.isfinite(best_oof_risk).all():
        raise AssertionError("Out-of-fold risk predictions are incomplete.")
    best_score = float(roc_auc_score(labels, best_oof_risk))
    risk_model = CrossFittedRiskEnsemble(
        components=tuple(components),
        normalization_mode=best_mode,
    )
    model_name = (
        f"{len(components)}-fold Extra Trees ensemble "
        f"({best_mode} robust baseline, leaf={best_min_leaf}, "
        f"features={best_max_features})"
    )
    return (
        risk_model,
        best_mode,
        model_name,
        best_score,
        best_oof_risk,
    )


def fit_hazard_model(
    development: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
) -> TwoStageForecastModel:
    """Fit nonlinear horizon risk plus regularized discrete-hazard timing.

    The Extra Trees stage learns whether an onset occurs in the configured
    horizon from robust high-channel-quantile variability. The hazard stage
    retains the complete future-bin timing distribution. Combining them keeps
    the original output contract while avoiding the weak linear risk ranking.
    """

    x, y, weights = make_person_period(development, config)
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=config.l2_regularization,
        learning_rate="optimal",
        max_iter=config.max_iter,
        tol=1e-4,
        average=True,
        n_jobs=-1,
        random_state=config.random_seed,
    )
    timing_model = Pipeline(
        [
            ("standardize", StandardScaler()),
            ("classifier", classifier),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        timing_model.fit(x, y, classifier__sample_weight=weights)

    (
        feature_medians,
        feature_scales,
        patient_feature_medians,
        patient_feature_scales,
    ) = _risk_normalization_statistics(development)
    (
        risk_model,
        normalization_mode,
        risk_model_name,
        risk_cv_auroc,
        risk_oof_predictions,
    ) = _fit_selected_risk_model(
        development,
        feature_medians,
        feature_scales,
        patient_feature_medians,
        patient_feature_scales,
        config,
    )
    return TwoStageForecastModel(
        timing_model=timing_model,
        risk_model=risk_model,
        risk_feature_columns=tuple(MODEL_FEATURE_COLUMNS),
        smoothing_alpha=1.0,
        feature_medians=feature_medians,
        feature_scales=feature_scales,
        patient_feature_medians=patient_feature_medians,
        patient_feature_scales=patient_feature_scales,
        normalization_mode=normalization_mode,
        risk_model_name=risk_model_name,
        risk_cv_auroc=risk_cv_auroc,
        risk_oof_predictions=risk_oof_predictions,
    )


def model_iteration_count(model: Any) -> int | None:
    """Return fitted iterations for supported estimators."""

    if isinstance(model, TwoStageForecastModel):
        if isinstance(model.risk_model, CrossFittedRiskEnsemble):
            return sum(
                len(getattr(component.estimator, "estimators_", []))
                for component in model.risk_model.components
            )
        return len(getattr(model.risk_model, "estimators_", [])) or None
    if isinstance(model, Pipeline):
        classifier = model.named_steps.get("classifier")
        value = getattr(classifier, "n_iter_", None)
    else:
        value = getattr(model, "n_iter_", None)
    if value is None:
        return None
    array = np.asarray(value).reshape(-1)
    return int(array.max())


def predict_horizon_distribution(
    model: Any,
    landmark_features: pd.DataFrame | np.ndarray,
    config: ForecastConfig = ForecastConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return hazards, r/i-bin probability mass, and no-event mass."""

    feature_frame = (
        landmark_features.reset_index(drop=True)
        if isinstance(landmark_features, pd.DataFrame)
        else None
    )
    if feature_frame is not None:
        base = feature_frame[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)
    else:
        base = np.asarray(landmark_features, dtype=float)
    if base.ndim != 2 or base.shape[1] != len(MODEL_FEATURE_COLUMNS):
        raise ValueError("Landmark feature matrix has an unexpected shape.")

    n_samples = len(base)
    repeated = np.repeat(base, config.n_bins, axis=0).astype(np.float32)
    lead_bins = np.tile(np.arange(config.n_bins, dtype=float), n_samples)
    x = np.empty(
        (n_samples * config.n_bins, len(HAZARD_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    x[:, : len(MODEL_FEATURE_COLUMNS)] = repeated
    x[:, -3] = lead_bins
    x[:, -2] = (lead_bins + 0.5) / config.n_bins
    x[:, -1] = np.log1p((lead_bins + 0.5) * config.bin_seconds)

    timing_model = (
        model.timing_model
        if isinstance(model, TwoStageForecastModel)
        else model
    )
    hazards = timing_model.predict_proba(x)[:, 1].reshape(
        n_samples, config.n_bins
    )
    hazards = np.clip(hazards, 1e-7, 1 - 1e-7)
    survival_before = np.concatenate(
        [
            np.ones((n_samples, 1), dtype=float),
            np.cumprod(1.0 - hazards[:, :-1], axis=1),
        ],
        axis=1,
    )
    pmf = hazards * survival_before
    no_event = np.prod(1.0 - hazards, axis=1)

    if isinstance(model, TwoStageForecastModel):
        if isinstance(model.risk_model, CrossFittedRiskEnsemble):
            risk_values = model.risk_model.predict_feature_proba(
                feature_frame, base
            )[:, 1]
        elif (
            model.feature_medians is not None
            and model.feature_scales is not None
            and model.patient_feature_medians is not None
            and model.patient_feature_scales is not None
            and model.normalization_mode != "legacy"
        ):
            risk_input = _normalized_risk_design(
                feature_frame,
                base,
                model.feature_medians,
                model.feature_scales,
                model.patient_feature_medians,
                model.patient_feature_scales,
                model.normalization_mode,
            )
            risk_values = model.risk_model.predict_proba(risk_input)[:, 1]
        else:
            column_indices = [
                MODEL_FEATURE_COLUMNS.index(column)
                for column in model.risk_feature_columns
            ]
            risk_input = pd.DataFrame(
                base[:, column_indices],
                columns=list(model.risk_feature_columns),
            )
            risk_values = model.risk_model.predict_proba(risk_input)[:, 1]
        if (
            model.normalization_mode == "legacy"
            and model.smoothing_alpha < 1.0
            and feature_frame is not None
            and {"episode_id", "landmark_step"}.issubset(feature_frame.columns)
        ):
            smoothed = np.empty_like(risk_values)
            for _, group in feature_frame.groupby("episode_id", sort=False):
                ordered = group.sort_values("landmark_step").index.to_numpy()
                smoothed[ordered] = (
                    pd.Series(risk_values[ordered])
                    .ewm(alpha=model.smoothing_alpha, adjust=False)
                    .mean()
                    .to_numpy()
                )
            risk_values = smoothed
        risk_values = np.clip(risk_values, 1e-7, 1.0 - 1e-7)
        base_risk = 1.0 - no_event
        conditional = pmf / np.clip(base_risk[:, None], 1e-12, None)
        zero_rows = base_risk <= 1e-12
        if zero_rows.any():
            conditional[zero_rows] = 1.0 / config.n_bins
        conditional /= np.clip(
            conditional.sum(axis=1, keepdims=True), 1e-12, None
        )
        pmf = conditional * risk_values[:, None]
        no_event = 1.0 - risk_values
        survival_before = 1.0 - np.concatenate(
            [
                np.zeros((n_samples, 1), dtype=float),
                np.cumsum(pmf[:, :-1], axis=1),
            ],
            axis=1,
        )
        hazards = np.clip(
            pmf / np.clip(survival_before, 1e-12, None),
            1e-7,
            1.0 - 1e-7,
        )
    if not np.allclose(pmf.sum(axis=1) + no_event, 1.0, atol=1e-8):
        raise AssertionError("Forecast probabilities do not sum to one.")
    return hazards, pmf, no_event


def alarm_state_from_risk(
    risk: np.ndarray,
    threshold: float,
    alarm_on_consecutive: int = 2,
    alarm_off_consecutive: int = 13,
    refractory_consecutive: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a hysteretic alarm state machine to one ordered episode.

    A warning turns on at the second consecutive high-risk landmark. Once on,
    it remains on until the thirteenth consecutive low-risk landmark. With
    five-second landmarks, that release condition is 65 seconds of low risk.
    """

    values = np.asarray(risk, dtype=float)
    high = values >= threshold
    active = np.zeros(len(values), dtype=bool)
    is_active = False
    high_run = 0
    low_run = 0
    refractory_remaining = 0
    for index, is_high in enumerate(high):
        if not is_active:
            if refractory_remaining > 0:
                refractory_remaining -= 1
                high_run = 0
                active[index] = False
                continue
            high_run = high_run + 1 if is_high else 0
            if high_run >= alarm_on_consecutive:
                is_active = True
                low_run = 0
        else:
            low_run = 0 if is_high else low_run + 1
            if low_run >= alarm_off_consecutive:
                is_active = False
                high_run = 0
                refractory_remaining = refractory_consecutive
        active[index] = is_active
    return high, active


def _alarm_columns(
    frame: pd.DataFrame,
    risk: np.ndarray,
    threshold: float,
    alarm_on_consecutive: int,
    alarm_off_consecutive: int,
    refractory_consecutive: int = 0,
) -> pd.DataFrame:
    """Return raw threshold flags and episode-local persistent alarm states."""

    if len(frame) != len(risk):
        raise ValueError("Risk vector length must match the scored frame.")
    scored = frame[["episode_id", "landmark_step"]].copy().reset_index(drop=True)
    scored["risk"] = np.asarray(risk, dtype=float)
    scored["raw_high_probability"] = False
    scored["alarm_active"] = False
    for _, indices in scored.sort_values(
        ["episode_id", "landmark_step"]
    ).groupby("episode_id").groups.items():
        ordered = np.asarray(list(indices), dtype=int)
        high, alarm = alarm_state_from_risk(
            scored.loc[ordered, "risk"].to_numpy(dtype=float),
            threshold,
            alarm_on_consecutive,
            alarm_off_consecutive,
            refractory_consecutive,
        )
        scored.loc[ordered, "raw_high_probability"] = high
        scored.loc[ordered, "alarm_active"] = alarm
    return scored[["raw_high_probability", "alarm_active"]]


def _warning_summary(
    frame: pd.DataFrame,
    risk: np.ndarray,
    threshold: float,
    bin_seconds: int = BIN_SECONDS,
    alarm_on_consecutive: int = 2,
    alarm_off_consecutive: int = 13,
    refractory_consecutive: int = 0,
) -> dict[str, float]:
    values = np.asarray(risk, dtype=float)
    scored = frame.reset_index(drop=True)
    if len(scored) != len(values):
        raise ValueError("Risk vector length must match the scored frame.")
    sequences: list[tuple[np.ndarray, str, np.ndarray]] = []
    for _, indices in scored.sort_values(
        ["episode_id", "landmark_step"]
    ).groupby("episode_id").groups.items():
        ordered = np.asarray(list(indices), dtype=int)
        sequences.append(
            (
                values[ordered],
                str(scored.loc[ordered[0], "episode_type"]),
                scored.loc[
                    ordered, "time_to_event_seconds"
                ].to_numpy(dtype=float),
            )
        )
    return _warning_summary_from_sequences(
        sequences,
        threshold,
        len(scored),
        bin_seconds,
        alarm_on_consecutive,
        alarm_off_consecutive,
        refractory_consecutive,
    )


def _warning_summary_from_sequences(
    sequences: list[tuple[np.ndarray, str, np.ndarray]],
    threshold: float,
    total_count: int,
    bin_seconds: int,
    alarm_on_consecutive: int,
    alarm_off_consecutive: int,
    refractory_consecutive: int,
) -> dict[str, float]:
    """Fast warning summary for episode arrays prepared once per search."""

    warning_count = 0
    false_alarms = 0
    negative_count = 0
    captured: list[bool] = []
    lead_times: list[float] = []
    for episode_risk, episode_type, time_to_event in sequences:
        _, warning = alarm_state_from_risk(
            episode_risk,
            threshold,
            alarm_on_consecutive,
            alarm_off_consecutive,
            refractory_consecutive,
        )
        warning_count += int(warning.sum())
        if episode_type == "preictal":
            was_captured = bool(warning.any())
            captured.append(was_captured)
            if was_captured:
                first_warning = int(np.flatnonzero(warning)[0])
                lead_times.append(float(time_to_event[first_warning]))
        else:
            negative_count += len(episode_risk)
            false_alarms += int(
                np.sum(warning & ~np.r_[False, warning[:-1]])
            )
    sensitivity = float(np.mean(captured)) if captured else float("nan")
    time_in_warning = float(warning_count / total_count)
    negative_hours = negative_count * bin_seconds / 3600.0
    false_alarms_per_hour = (
        float(false_alarms / negative_hours) if negative_hours else float("nan")
    )
    return {
        "sensitivity": sensitivity,
        "time_in_warning": time_in_warning,
        "false_alarms_per_hour": false_alarms_per_hour,
        "captured_seizures": float(sum(captured)),
        "total_seizures": float(len(captured)),
        "median_warning_lead_seconds": (
            float(np.median(lead_times)) if lead_times else float("nan")
        ),
    }


def select_warning_threshold(
    development: pd.DataFrame,
    risk: np.ndarray,
    target_time_in_warning: float = 0.25,
    bin_seconds: int = BIN_SECONDS,
    alarm_on_consecutive: int = 2,
    alarm_off_consecutive: int = 13,
    minimum_sensitivity_target: float | None = None,
    refractory_consecutive: int = 0,
    quantile_count: int = 201,
) -> tuple[float, pd.DataFrame]:
    """Choose a development-only operating point.

    When a minimum sensitivity is requested, choose the qualifying point with
    the fewest false alarms and then the least warning time. Without a hard
    target, maximize a clinically balanced utility: seizure capture is given
    double weight, while false-warning rate and time in warning are penalized.
    """

    if quantile_count < 2:
        raise ValueError("quantile_count must be at least two.")
    quantiles = np.linspace(0, 1, quantile_count)
    candidates = np.unique(np.r_[0.0, np.quantile(risk, quantiles), 1.0])
    values = np.asarray(risk, dtype=float)
    scored = development.reset_index(drop=True)
    sequences: list[tuple[np.ndarray, str, np.ndarray]] = []
    for _, indices in scored.sort_values(
        ["episode_id", "landmark_step"]
    ).groupby("episode_id").groups.items():
        ordered = np.asarray(list(indices), dtype=int)
        sequences.append(
            (
                values[ordered],
                str(scored.loc[ordered[0], "episode_type"]),
                scored.loc[
                    ordered, "time_to_event_seconds"
                ].to_numpy(dtype=float),
            )
        )
    rows: list[dict[str, float]] = []
    for threshold in candidates:
        summary = _warning_summary_from_sequences(
            sequences,
            float(threshold),
            len(scored),
            bin_seconds,
            alarm_on_consecutive,
            alarm_off_consecutive,
            refractory_consecutive,
        )
        rows.append({"threshold": float(threshold), **summary})
    curve = pd.DataFrame(rows)
    if minimum_sensitivity_target is not None:
        sensitivity_eligible = curve.loc[
            curve["sensitivity"] >= minimum_sensitivity_target
        ].copy()
        if not sensitivity_eligible.empty:
            selected = sensitivity_eligible.sort_values(
                ["false_alarms_per_hour", "time_in_warning", "threshold"],
                ascending=[True, True, False],
            ).iloc[0]
            return float(selected["threshold"]), curve
    if minimum_sensitivity_target is None:
        curve["operating_utility"] = (
            2.0 * curve["sensitivity"]
            - 0.10 * curve["false_alarms_per_hour"]
            - 0.25 * curve["time_in_warning"]
        )
        selected = curve.sort_values(
            ["operating_utility", "sensitivity", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
        return float(selected["threshold"]), curve
    eligible = curve.loc[curve["time_in_warning"] <= target_time_in_warning].copy()
    if eligible.empty or eligible["sensitivity"].max() <= 0:
        curve["objective"] = curve["sensitivity"] - curve["time_in_warning"]
        selected = curve.sort_values(
            ["objective", "sensitivity", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
    else:
        selected = eligible.sort_values(
            ["sensitivity", "time_in_warning", "threshold"],
            ascending=[False, True, False],
        ).iloc[0]
    return float(selected["threshold"]), curve


def _causal_smooth_risk(
    frame: pd.DataFrame,
    risk: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Exponentially smooth risk within ordered episodes using past values only."""

    if not 0 < alpha <= 1:
        raise ValueError("Risk smoothing alpha must be in (0, 1].")
    values = np.asarray(risk, dtype=float)
    if len(values) != len(frame):
        raise ValueError("Risk vector length must match the scored frame.")
    if alpha == 1:
        return values.copy()
    smoothed = np.empty_like(values)
    ordered_frame = frame.reset_index(drop=True)
    for _, indices in ordered_frame.sort_values(
        ["episode_id", "landmark_step"]
    ).groupby("episode_id").groups.items():
        ordered = np.asarray(list(indices), dtype=int)
        smoothed[ordered] = (
            pd.Series(values[ordered])
            .ewm(alpha=alpha, adjust=False)
            .mean()
            .to_numpy()
        )
    return smoothed


def _rescale_event_distribution(
    pmf: np.ndarray,
    no_event: np.ndarray,
    event_risk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve conditional timing while replacing total event probability."""

    raw_risk = 1.0 - np.asarray(no_event, dtype=float)
    conditional = np.asarray(pmf, dtype=float) / np.clip(
        raw_risk[:, None], 1e-12, None
    )
    zero_rows = raw_risk <= 1e-12
    if zero_rows.any():
        conditional[zero_rows] = 1.0 / pmf.shape[1]
    conditional /= np.clip(
        conditional.sum(axis=1, keepdims=True), 1e-12, None
    )
    adjusted_risk = np.clip(
        np.asarray(event_risk, dtype=float), 1e-7, 1.0 - 1e-7
    )
    return conditional * adjusted_risk[:, None], 1.0 - adjusted_risk


def select_warning_policy(
    development: pd.DataFrame,
    risk: np.ndarray,
    config: ForecastConfig = ForecastConfig(),
) -> tuple[float, pd.DataFrame, np.ndarray, dict[str, float | int]]:
    """Tune a causal alarm policy on development data without changing frames."""

    labels = development["has_event_in_horizon"].to_numpy(dtype=bool)
    candidate_rows: list[dict[str, float | int]] = []
    smoothed_by_alpha: dict[float, np.ndarray] = {}
    for alpha in (0.05, 0.08, 0.15, 0.30, 0.60, 1.0):
        smoothed = _causal_smooth_risk(development, risk, alpha)
        smoothed_by_alpha[alpha] = smoothed
        auroc = (
            float(roc_auc_score(labels, smoothed))
            if np.unique(labels).size == 2
            else float("nan")
        )
        for alarm_on in (2, 3, 4, 6, 12, 18):
            for alarm_off in (6, 13, 25):
                for refractory in (0, 12, 60):
                    threshold, curve = select_warning_threshold(
                        development,
                        smoothed,
                        config.warning_time_target,
                        config.bin_seconds,
                        alarm_on,
                        alarm_off,
                        config.minimum_development_sensitivity,
                        refractory,
                        51,
                    )
                    selected = curve.iloc[
                        (curve["threshold"] - threshold).abs().argmin()
                    ]
                    candidate_rows.append(
                        {
                            "alpha": alpha,
                            "alarm_on_consecutive": alarm_on,
                            "alarm_off_consecutive": alarm_off,
                            "refractory_consecutive": refractory,
                            "threshold": threshold,
                            "sensitivity": float(selected["sensitivity"]),
                            "false_alarms_per_hour": float(
                                selected["false_alarms_per_hour"]
                            ),
                            "time_in_warning": float(
                                selected["time_in_warning"]
                            ),
                            "auroc": auroc,
                            "operating_utility": float(
                                selected.get("operating_utility", np.nan)
                            ),
                        }
                    )

    candidates = pd.DataFrame(candidate_rows)
    if config.minimum_development_sensitivity is None:
        candidates["combined_utility"] = (
            candidates["operating_utility"] + 0.20 * candidates["auroc"]
        )
        selected_policy = candidates.sort_values(
            [
                "combined_utility",
                "sensitivity",
                "false_alarms_per_hour",
                "time_in_warning",
                "threshold",
            ],
            ascending=[False, False, True, True, False],
        ).iloc[0]
    else:
        sensitivity_eligible = candidates.loc[
            candidates["sensitivity"]
            >= config.minimum_development_sensitivity
        ].copy()
        pool = (
            sensitivity_eligible
            if not sensitivity_eligible.empty
            else candidates.copy()
        )
        false_alarm_eligible = pool.loc[
            pool["false_alarms_per_hour"]
            <= config.false_alarm_target_per_hour
        ].copy()
        if not false_alarm_eligible.empty:
            selected_policy = false_alarm_eligible.sort_values(
                [
                    "auroc",
                    "false_alarms_per_hour",
                    "time_in_warning",
                    "threshold",
                ],
                ascending=[False, True, True, False],
            ).iloc[0]
        else:
            selected_policy = pool.sort_values(
                [
                    "sensitivity",
                    "false_alarms_per_hour",
                    "auroc",
                    "time_in_warning",
                    "threshold",
                ],
                ascending=[False, True, False, True, False],
            ).iloc[0]

    policy = {
        "smoothing_alpha": float(selected_policy["alpha"]),
        "alarm_on_consecutive": int(
            selected_policy["alarm_on_consecutive"]
        ),
        "alarm_off_consecutive": int(
            selected_policy["alarm_off_consecutive"]
        ),
        "refractory_consecutive": int(
            selected_policy["refractory_consecutive"]
        ),
    }
    selected_risk = smoothed_by_alpha[policy["smoothing_alpha"]]
    threshold, curve = select_warning_threshold(
        development,
        selected_risk,
        config.warning_time_target,
        config.bin_seconds,
        policy["alarm_on_consecutive"],
        policy["alarm_off_consecutive"],
        config.minimum_development_sensitivity,
        policy["refractory_consecutive"],
        201,
    )
    return threshold, curve, selected_risk, policy


def _class_baseline(
    development: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
) -> np.ndarray:
    targets = np.where(
        development["has_event_in_horizon"].to_numpy(dtype=bool),
        development["event_bin"].to_numpy(dtype=int),
        config.n_bins,
    )
    counts = np.bincount(targets, minlength=config.n_bins + 1).astype(float)
    return counts / counts.sum()


def evaluate_forecasts(
    model: Any,
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    warning_time_target: float = 0.25,
    config: ForecastConfig = ForecastConfig(),
    development_risk: np.ndarray | None = None,
    holdout_risk: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, pd.DataFrame]:
    """Evaluate forecasts with optional out-of-fold alarm-risk vectors."""

    if development_risk is None:
        _, _, development_no_event = predict_horizon_distribution(
            model, development, config
        )
        raw_development_risk = 1.0 - development_no_event
    else:
        raw_development_risk = np.asarray(development_risk, dtype=float)
        if len(raw_development_risk) != len(development):
            raise ValueError(
                "development_risk length must match the development frame."
            )
        if not np.isfinite(raw_development_risk).all():
            raise ValueError("development_risk must contain finite values.")
    raw_development_risk = np.clip(
        raw_development_risk, 1e-7, 1.0 - 1e-7
    )
    (
        threshold,
        threshold_curve,
        development_risk,
        warning_policy,
    ) = select_warning_policy(
        development,
        raw_development_risk,
        config,
    )
    if isinstance(model, TwoStageForecastModel):
        model.warning_policy = {
            **warning_policy,
            "threshold": float(threshold),
        }

    hazards, pmf, no_event = predict_horizon_distribution(
        model, holdout, config
    )
    raw_risk = (
        1.0 - no_event
        if holdout_risk is None
        else np.asarray(holdout_risk, dtype=float)
    )
    if len(raw_risk) != len(holdout):
        raise ValueError("holdout_risk length must match the holdout frame.")
    if not np.isfinite(raw_risk).all():
        raise ValueError("holdout_risk must contain finite values.")
    raw_risk = np.clip(raw_risk, 1e-7, 1.0 - 1e-7)
    risk = _causal_smooth_risk(
        holdout,
        raw_risk,
        float(warning_policy["smoothing_alpha"]),
    )
    risk = np.clip(risk, 1e-7, 1.0 - 1e-7)
    pmf, no_event = _rescale_event_distribution(pmf, no_event, risk)
    has_event = holdout["has_event_in_horizon"].to_numpy(dtype=bool)
    event_bin = holdout["event_bin"].to_numpy(dtype=int)
    rows = np.arange(len(holdout))
    target_probability = np.where(has_event, pmf[rows, event_bin], no_event)
    nll = -np.log(np.clip(target_probability, 1e-12, None))

    squared = np.square(pmf).sum(axis=1) + np.square(no_event)
    categorical_brier = squared.copy()
    categorical_brier[has_event] += (
        1.0 - 2.0 * pmf[rows[has_event], event_bin[has_event]]
    )
    categorical_brier[~has_event] += 1.0 - 2.0 * no_event[~has_event]
    binary_brier = np.square(risk - has_event.astype(float))

    predicted_survival = 1.0 - np.cumsum(pmf, axis=1)
    truth_survival = np.ones_like(predicted_survival)
    if has_event.any():
        truth_survival[has_event] = (
            np.arange(config.n_bins)[None, :] < event_bin[has_event, None]
        )
    integrated_brier = np.mean(
        np.square(predicted_survival - truth_survival), axis=1
    )

    conditional_pmf = pmf / np.clip(risk[:, None], 1e-12, None)
    bin_midpoints = (
        np.arange(config.n_bins) + 0.5
    ) * config.bin_seconds
    expected_seconds = conditional_pmf @ bin_midpoints
    timing_error = np.full(len(holdout), np.nan)
    timing_error[has_event] = np.abs(
        expected_seconds[has_event]
        - holdout.loc[has_event, "time_to_event_seconds"].to_numpy(dtype=float)
    )

    baseline = _class_baseline(development, config)
    reference_brier = np.square(baseline).sum() + 1.0 - 2.0 * np.where(
        has_event, baseline[event_bin], baseline[config.n_bins]
    )
    brier_skill = 1.0 - categorical_brier.mean() / reference_brier.mean()

    prediction_columns = holdout[
        [
            "episode_id",
            "patient_id",
            "source_event_id",
            "episode_type",
            "recording",
            "landmark_step",
            "landmark_seconds",
            "time_to_event_seconds",
            "event_bin",
            "has_event_in_horizon",
        ]
    ].reset_index(drop=True)
    prediction_columns["event_risk"] = risk
    prediction_columns["no_event_probability"] = no_event
    prediction_columns["negative_log_likelihood"] = nll
    prediction_columns["categorical_brier"] = categorical_brier
    prediction_columns["binary_brier"] = binary_brier
    prediction_columns["integrated_brier"] = integrated_brier
    prediction_columns["expected_seconds_given_event"] = expected_seconds
    prediction_columns["absolute_timing_error_seconds"] = timing_error
    alarm = _alarm_columns(
        prediction_columns,
        risk,
        threshold,
        int(warning_policy["alarm_on_consecutive"]),
        int(warning_policy["alarm_off_consecutive"]),
        int(warning_policy["refractory_consecutive"]),
    )
    prediction_columns["raw_high_probability"] = alarm["raw_high_probability"].to_numpy()
    prediction_columns["alarm_active"] = alarm["alarm_active"].to_numpy()
    prediction_columns["warning"] = prediction_columns["alarm_active"]
    for index in range(config.n_bins):
        prediction_columns[f"p_bin_{index + 1:02d}"] = pmf[:, index]

    warning_metrics = _warning_summary(
        holdout,
        risk,
        threshold,
        config.bin_seconds,
        int(warning_policy["alarm_on_consecutive"]),
        int(warning_policy["alarm_off_consecutive"]),
        int(warning_policy["refractory_consecutive"]),
    )
    horizon_label = f"{config.horizon_seconds / 60:g}-minute"
    auroc = (
        float(roc_auc_score(has_event, risk))
        if np.unique(has_event).size == 2
        else float("nan")
    )
    average_precision = (
        float(average_precision_score(has_event, risk))
        if has_event.any()
        else float("nan")
    )
    timing_mae = (
        float(np.nanmean(timing_error))
        if np.isfinite(timing_error).any()
        else float("nan")
    )
    metrics = [
        {
            "metric": "negative log likelihood",
            "value": float(nll.mean()),
            "interpretation": "Lower is better; proper score for the observed bin/no-event class.",
        },
        {
            "metric": "multicategory Brier score",
            "value": float(categorical_brier.mean()),
            "interpretation": (
                f"Lower is better; probability error across {config.n_bins} "
                "bins plus no-event."
            ),
        },
        {
            "metric": "multicategory Brier skill score",
            "value": float(brier_skill),
            "interpretation": "Above 0 improves on the development-set class-frequency forecast.",
        },
        {
            "metric": "integrated survival Brier score",
            "value": float(integrated_brier.mean()),
            "interpretation": (
                "Lower is better; mean survival-probability error across "
                f"{horizon_label} horizon."
            ),
        },
        {
            "metric": f"{horizon_label} AUROC",
            "value": auroc,
            "interpretation": "Discrimination only; does not assess calibration.",
        },
        {
            "metric": f"{horizon_label} average precision",
            "value": average_precision,
            "interpretation": "Ranking metric sensitive to event prevalence.",
        },
        {
            "metric": "conditional timing MAE (seconds)",
            "value": timing_mae,
            "interpretation": "Error of expected onset time among seizure landmarks.",
        },
        {
            "metric": "seizure sensitivity",
            "value": warning_metrics["sensitivity"],
            "interpretation": (
                f"Fraction of seizure episodes with an alarm after "
                f"{warning_policy['alarm_on_consecutive']} consecutive "
                "high-risk landmarks."
            ),
        },
        {
            "metric": "time in warning",
            "value": warning_metrics["time_in_warning"],
            "interpretation": (
                f"Fraction of evaluated {config.bin_seconds}-second landmarks "
                "under warning."
            ),
        },
        {
            "metric": "false alarms per hour",
            "value": warning_metrics["false_alarms_per_hour"],
            "interpretation": (
                "Rising persistent-alarm edges in interictal episodes per "
                "monitored hour."
            ),
        },
        {
            "metric": "median warning lead (seconds)",
            "value": warning_metrics["median_warning_lead_seconds"],
            "interpretation": "First-warning lead among captured holdout seizures.",
        },
        {
            "metric": "development-selected warning threshold",
            "value": threshold,
            "interpretation": (
                "Chosen without holdout outcomes to meet the configured "
                "development sensitivity/false-alarm targets using a causal "
                "development-only alarm policy."
            ),
        },
    ]
    metrics_frame = pd.DataFrame(metrics)

    patient_rows: list[dict[str, Any]] = []
    for patient_id, group in prediction_columns.groupby("patient_id"):
        group_risk = group["event_risk"].to_numpy()
        group_has_event = group["has_event_in_horizon"].to_numpy(dtype=bool)
        warning = _warning_summary(
            group,
            group_risk,
            threshold,
            config.bin_seconds,
            int(warning_policy["alarm_on_consecutive"]),
            int(warning_policy["alarm_off_consecutive"]),
            int(warning_policy["refractory_consecutive"]),
        )
        patient_rows.append(
            {
                "patient_id": patient_id,
                "landmarks": len(group),
                "seizure_episodes": int(
                    group.loc[group["episode_type"].eq("preictal"), "episode_id"].nunique()
                ),
                "interictal_episodes": int(
                    group.loc[group["episode_type"].eq("interictal"), "episode_id"].nunique()
                ),
                "auroc": (
                    float(roc_auc_score(group_has_event, group_risk))
                    if np.unique(group_has_event).size == 2
                    else np.nan
                ),
                **warning,
            }
        )
    patient_metrics = pd.DataFrame(patient_rows)
    return (
        prediction_columns,
        metrics_frame,
        patient_metrics,
        threshold,
        threshold_curve,
    )


def extract_context_features_from_edf(
    edf_path: str | Path,
    landmark_seconds: float,
    config: ForecastConfig = ForecastConfig(),
) -> pd.DataFrame:
    """Read the preceding context from an arbitrary EDF landmark."""

    data, sample_rate, _ = read_edf_eeg_segment(
        edf_path,
        landmark_seconds - config.context_seconds,
        config.context_seconds,
        min_channels=config.effective_min_eeg_channels,
        included_channels=config.selected_eeg_channels,
    )
    micro = segment_micro_features(data, sample_rate, config)
    aggregated = aggregate_context(
        micro[-config.context_bins :], config.bin_seconds
    )
    return pd.DataFrame([aggregated], columns=MODEL_FEATURE_COLUMNS)


def build_seizure_reading_forecasts(
    model: Any,
    seizure_manifest: pd.DataFrame,
    landmarks: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
    verbose: bool = True,
    onset_cache_csv: str | Path | None = None,
    onset_cache_json: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Forecast every reading from ``-m`` through onset for each seizure.

    Existing cached landmark features provide readings ``-m`` through ``-i``.
    Only the final onset-time context is read from each EDF. All feature rows
    are then forecast in one model batch.
    """

    config.validate()
    seizures = seizure_manifest.loc[
        seizure_manifest["episode_type"].eq("preictal")
    ].copy()
    if seizures.empty:
        raise ValueError("No preictal seizures were supplied.")

    onset_signature = {
        "feature_cache_version": FEATURE_CACHE_VERSION,
        "bin_seconds": config.bin_seconds,
        "context_seconds": config.context_seconds,
        "target_sample_rate": config.target_sample_rate,
        "min_eeg_channels": config.min_eeg_channels,
        "included_eeg_channels": (
            list(config.selected_eeg_channels)
            if config.selected_eeg_channels is not None
            else None
        ),
        "seizures": (
            seizures[
                [
                    "episode_id",
                    "edf_path",
                    "event_onset_seconds",
                ]
            ]
            .sort_values("episode_id")
            .to_dict(orient="records")
        ),
    }
    onset_cache_csv = (
        Path(onset_cache_csv) if onset_cache_csv is not None else None
    )
    onset_cache_json = (
        Path(onset_cache_json) if onset_cache_json is not None else None
    )
    onset_features: pd.DataFrame | None = None
    if (
        onset_cache_csv is not None
        and onset_cache_json is not None
        and onset_cache_csv.exists()
        and onset_cache_json.exists()
    ):
        try:
            saved_signature = json.loads(
                onset_cache_json.read_text(encoding="utf-8")
            )
            cached_onsets = pd.read_csv(onset_cache_csv)
            if (
                saved_signature == onset_signature
                and set(MODEL_FEATURE_COLUMNS).issubset(cached_onsets.columns)
                and cached_onsets["episode_id"].nunique() == len(seizures)
            ):
                onset_features = cached_onsets
                if verbose:
                    print(
                        f"Loaded {len(onset_features)} cached onset contexts."
                    )
        except (OSError, ValueError, json.JSONDecodeError):
            onset_features = None

    if onset_features is None:
        onset_rows: list[dict[str, Any]] = []
        for number, episode in enumerate(seizures.itertuples(), start=1):
            extracted = extract_context_features_from_edf(
                episode.edf_path,
                float(episode.event_onset_seconds),
                config,
            ).iloc[0]
            onset_rows.append(
                {
                    "episode_id": episode.episode_id,
                    **extracted.to_dict(),
                }
            )
            if verbose:
                print(
                    f"[{number:>2}/{len(seizures)}] onset context "
                    f"{episode.source_event_id} ({episode.dataset_split})",
                    flush=True,
                )
        onset_features = pd.DataFrame(onset_rows)
        if onset_cache_csv is not None and onset_cache_json is not None:
            onset_cache_csv.parent.mkdir(parents=True, exist_ok=True)
            onset_features.to_csv(onset_cache_csv, index=False)
            onset_cache_json.write_text(
                json.dumps(onset_signature, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    onset_lookup = onset_features.set_index("episode_id")
    feature_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    for episode in seizures.itertuples():
        cached = landmarks.loc[
            landmarks["episode_id"].eq(episode.episode_id)
        ].sort_values("landmark_step")
        if len(cached) != config.pre_onset_steps:
            raise ValueError(
                f"{episode.episode_id} has {len(cached)} cached readings; "
                f"expected {config.pre_onset_steps}."
            )
        onset_row = pd.DataFrame(
            [onset_lookup.loc[episode.episode_id, MODEL_FEATURE_COLUMNS]],
            columns=MODEL_FEATURE_COLUMNS,
        )
        episode_features = pd.concat(
            [
                cached[MODEL_FEATURE_COLUMNS].reset_index(drop=True),
                onset_row,
            ],
            ignore_index=True,
        )
        feature_frames.append(episode_features)
        for reading_index in range(config.reading_count):
            seconds_before = (
                config.pre_onset_seconds
                - reading_index * config.bin_seconds
            )
            metadata_rows.append(
                {
                    "dataset_split": episode.dataset_split,
                    "patient_id": episode.patient_id,
                    "source_event_id": episode.source_event_id,
                    "episode_id": episode.episode_id,
                    "recording": episode.recording,
                    "reading_index": reading_index,
                    "landmark_seconds": (
                        float(episode.event_onset_seconds) - seconds_before
                    ),
                    "seconds_before_onset": seconds_before,
                    "minutes_before_onset": seconds_before / 60.0,
                }
            )
    reading_metadata = pd.DataFrame(metadata_rows)
    all_features = pd.concat(feature_frames, ignore_index=True)
    all_features["episode_id"] = reading_metadata["episode_id"].to_numpy()
    all_features["landmark_step"] = reading_metadata[
        "reading_index"
    ].to_numpy()
    if len(all_features) != len(reading_metadata):
        raise AssertionError("Reading feature and metadata counts differ.")
    _, pmf, no_event = predict_horizon_distribution(
        model, all_features, config
    )
    risk = 1.0 - no_event
    if (
        isinstance(model, TwoStageForecastModel)
        and model.warning_policy is not None
    ):
        risk = _causal_smooth_risk(
            all_features,
            risk,
            float(model.warning_policy["smoothing_alpha"]),
        )
        pmf, no_event = _rescale_event_distribution(
            pmf, no_event, risk
        )
    conditional = pmf / np.clip(risk[:, None], 1e-12, None)

    repeated_metadata = reading_metadata.loc[
        reading_metadata.index.repeat(config.n_bins)
    ].reset_index(drop=True)
    future_bins = np.tile(
        np.arange(1, config.n_bins + 1, dtype=int), len(reading_metadata)
    )
    repeated_metadata["future_bin"] = future_bins
    repeated_metadata["future_interval_start_seconds"] = (
        future_bins - 1
    ) * config.bin_seconds
    repeated_metadata["future_interval_end_seconds"] = (
        future_bins * config.bin_seconds
    )
    repeated_metadata["onset_probability"] = pmf.reshape(-1)
    repeated_metadata["timing_probability_given_event"] = conditional.reshape(
        -1
    )
    repeated_metadata["event_risk"] = np.repeat(risk, config.n_bins)
    repeated_metadata["no_event_probability"] = np.repeat(
        no_event, config.n_bins
    )

    peak_indices = np.argmax(pmf, axis=1)
    peak_rows = reading_metadata.copy()
    peak_rows["peak_bin"] = peak_indices + 1
    peak_rows["peak_probability"] = pmf[
        np.arange(len(pmf)), peak_indices
    ]
    peak_rows["event_risk"] = risk
    peak_rows["no_event_probability"] = no_event
    peak_rows["peak_entry"] = [
        f"({int(bin_number)}, {probability:.6f})"
        for bin_number, probability in zip(
            peak_rows["peak_bin"],
            peak_rows["peak_probability"],
            strict=True,
        )
    ]
    peak_matrix = peak_rows.pivot(
        index=["reading_index", "minutes_before_onset"],
        columns="source_event_id",
        values="peak_entry",
    ).reset_index()
    peak_matrix.columns.name = None

    total_mass = (
        repeated_metadata.groupby(
            ["source_event_id", "reading_index"], sort=False
        )["onset_probability"].sum().to_numpy()
        + no_event
    )
    if not np.allclose(total_mass, 1.0, atol=1e-8):
        raise AssertionError("A reading forecast does not sum to one.")
    return repeated_metadata, peak_rows, peak_matrix


def moving_horizon_forecast_table(
    pmf: np.ndarray,
    episode_step: int,
    config: ForecastConfig = ForecastConfig(),
) -> tuple[pd.DataFrame, float]:
    """Return r/i future bins, shifted by ``episode_step`` from the anchor."""

    pmf = np.asarray(pmf, dtype=float).reshape(-1)
    if len(pmf) != config.n_bins:
        raise ValueError(f"Expected {config.n_bins} event-bin probabilities.")
    if episode_step < 0:
        raise ValueError("episode_step must be nonnegative.")
    visible = pmf.copy()
    event_risk = float(visible.sum())
    no_event_probability = float(1.0 - event_risk)
    conditional = (
        visible / event_risk if event_risk > 1e-12 else np.zeros_like(visible)
    )
    cumulative = np.cumsum(visible)
    absolute_bins = np.arange(
        episode_step + 1, episode_step + config.n_bins + 1
    )
    table = pd.DataFrame(
        {
            "absolute_episode_bin": absolute_bins,
            "interval_start_seconds": (absolute_bins - 1) * config.bin_seconds,
            "interval_end_seconds": absolute_bins * config.bin_seconds,
            "seconds_ahead_start": (
                np.arange(config.n_bins) * config.bin_seconds
            ),
            "seconds_ahead_end": (
                (np.arange(config.n_bins) + 1) * config.bin_seconds
            ),
            "onset_probability": visible,
            "timing_probability_given_event": conditional,
            "cumulative_onset_probability": cumulative,
            "survival_probability": 1.0 - cumulative,
        }
    )
    return table, no_event_probability


# Backward-compatible name for code that imported the original helper.  Its
# behavior is intentionally the new moving-horizon behavior.
fixed_episode_forecast_table = moving_horizon_forecast_table


class RollingForecastEngine:
    """Optimized offline rolling forecaster with a constant r/i-bin horizon.

    The expensive work is performed once: a contiguous prerecorded EEG replay
    is read once, five-second features are cached, all overlapping 120-second
    contexts are formed, and every r/i-bin forecast is evaluated in one model
    batch.  ``advance_to`` then performs a cached lookup at each boundary.
    """

    def __init__(
        self,
        model: Any,
        edf_path: str | Path,
        episode_anchor_seconds: float,
        config: ForecastConfig = ForecastConfig(),
        replay_duration_seconds: int | None = None,
    ) -> None:
        config.validate()
        if replay_duration_seconds is None:
            replay_duration_seconds = config.pre_onset_seconds
        if replay_duration_seconds < 0:
            raise ValueError("replay_duration_seconds must be nonnegative.")
        if replay_duration_seconds % config.bin_seconds:
            raise ValueError(
                "replay_duration_seconds must be divisible by bin_seconds."
            )
        self.model = model
        self.edf_path = Path(edf_path)
        self.episode_anchor_seconds = float(episode_anchor_seconds)
        self.config = config
        self.replay_duration_seconds = int(replay_duration_seconds)
        self.replay_end_seconds = (
            self.episode_anchor_seconds + self.replay_duration_seconds
        )
        self.max_step = self.replay_duration_seconds // self.config.bin_seconds
        self.last_step = -1
        self.update_count = 0
        self.last_result: dict[str, Any] | None = None
        self._precompute()

    def _precompute(self) -> None:
        started = time.perf_counter()
        read_start = self.episode_anchor_seconds - self.config.context_seconds
        read_duration = (
            self.config.context_seconds + self.replay_duration_seconds
        )
        data, sample_rate, labels = read_edf_eeg_segment(
            self.edf_path,
            read_start,
            read_duration,
            min_channels=self.config.effective_min_eeg_channels,
            included_channels=self.config.selected_eeg_channels,
        )
        micro = segment_micro_features(data, sample_rate, self.config)
        expected_micro = (
            self.config.context_seconds + self.replay_duration_seconds
        ) // self.config.bin_seconds
        if len(micro) != expected_micro:
            raise AssertionError(
                f"Expected {expected_micro} cached micro-windows, found {len(micro)}."
            )
        contexts = np.vstack(
            [
                aggregate_context(
                    micro[step : step + self.config.context_bins],
                    self.config.bin_seconds,
                )
                for step in range(self.max_step + 1)
            ]
        )
        self.context_features = pd.DataFrame(
            contexts, columns=MODEL_FEATURE_COLUMNS
        )
        self.context_features["episode_id"] = "rolling_replay"
        self.context_features["landmark_step"] = np.arange(
            self.max_step + 1
        )
        (
            self.precomputed_hazards,
            self.precomputed_pmf,
            self.precomputed_no_event,
        ) = predict_horizon_distribution(
            self.model, self.context_features, self.config
        )
        if (
            isinstance(self.model, TwoStageForecastModel)
            and self.model.warning_policy is not None
        ):
            precomputed_risk = _causal_smooth_risk(
                self.context_features,
                1.0 - self.precomputed_no_event,
                float(self.model.warning_policy["smoothing_alpha"]),
            )
            (
                self.precomputed_pmf,
                self.precomputed_no_event,
            ) = _rescale_event_distribution(
                self.precomputed_pmf,
                self.precomputed_no_event,
                precomputed_risk,
            )
            survival_before = 1.0 - np.concatenate(
                [
                    np.zeros((len(self.precomputed_pmf), 1), dtype=float),
                    np.cumsum(self.precomputed_pmf[:, :-1], axis=1),
                ],
                axis=1,
            )
            self.precomputed_hazards = np.clip(
                self.precomputed_pmf
                / np.clip(survival_before, 1e-12, None),
                1e-7,
                1.0 - 1e-7,
            )
        self.precomputed_tables = [
            moving_horizon_forecast_table(
                self.precomputed_pmf[step], step, self.config
            )[0]
            for step in range(self.max_step + 1)
        ]
        self.source_sample_rate = float(sample_rate)
        self.eeg_channel_count = len(labels)
        self.precomputation_seconds = time.perf_counter() - started
        self.optimization_stats = {
            "forecast_states_precomputed": self.max_step + 1,
            "eeg_seconds_processed_once": read_duration,
            "naive_eeg_seconds_if_full_context_reread": (
                (self.max_step + 1) * self.config.context_seconds
            ),
            "raw_signal_work_reduction_factor": (
                (self.max_step + 1) * self.config.context_seconds / read_duration
            ),
            "online_edf_reads": 0,
            "online_model_calls": 0,
            "precomputation_seconds": self.precomputation_seconds,
        }

    def advance_to(self, recording_seconds: float) -> dict[str, Any]:
        """Return a new cached r/i-bin forecast only at a crossed boundary."""

        elapsed = float(recording_seconds) - self.episode_anchor_seconds
        if elapsed < 0:
            raise ValueError("recording_seconds precedes the episode anchor.")
        step = int(math.floor(elapsed / self.config.bin_seconds))
        if step > self.max_step:
            raise ValueError(
                f"Requested step {step} exceeds the precomputed replay. "
                "Construct the engine with a longer replay_duration_seconds."
            )
        if step < self.last_step:
            raise ValueError(
                "RollingForecastEngine requires nondecreasing recording times; "
                f"requested step {step} after step {self.last_step}."
            )
        if step == self.last_step and self.last_result is not None:
            return {
                **self.last_result,
                "updated": False,
                "recomputed": False,
            }

        landmark_seconds = (
            self.episode_anchor_seconds + step * self.config.bin_seconds
        )
        self.last_step = step
        self.update_count += 1
        self.last_result = {
            "episode_step": step,
            "landmark_seconds": landmark_seconds,
            "horizon_start_seconds": landmark_seconds,
            "horizon_end_seconds": landmark_seconds + self.config.horizon_seconds,
            "forecast_bins": self.config.n_bins,
            "expired_absolute_bin": step if step > 0 else None,
            "appended_absolute_bin": step + self.config.n_bins,
            "event_risk": float(
                1.0 - self.precomputed_no_event[step]
            ),
            "no_event_probability": float(
                self.precomputed_no_event[step]
            ),
            "hazards": self.precomputed_hazards[step],
            "pmf": self.precomputed_pmf[step],
            "probability_table": self.precomputed_tables[step],
            "update_count": self.update_count,
            "batch_precomputation_count": 1,
            "online_edf_reads": 0,
            "online_model_calls": 0,
            "updated": True,
            "recomputed": True,
        }
        return self.last_result


def readable_feature_name(feature: str) -> str:
    """Convert a model column into a concise human-readable EEG label."""

    aggregation_labels = {
        "mean": "context mean",
        "std": "context variability",
        "last": "most recent interval",
        "slope": "two-minute trend",
    }
    aggregation = next(
        (
            label
            for suffix, label in aggregation_labels.items()
            if feature.endswith(f"_{suffix}")
        ),
        "",
    )
    base = feature.rsplit("_", 1)[0] if aggregation else feature
    replacements = {
        "log_rms": "log RMS amplitude",
        "log_line_length": "log line length",
        "log_mad": "log robust amplitude",
        "log_robust_range": "log robust peak-to-peak range",
        "zero_crossing_rate": "zero-crossing rate",
        "hjorth_mobility": "Hjorth mobility",
        "hjorth_complexity": "Hjorth complexity",
        "usable_channel_fraction": "usable-channel fraction",
    }
    base_label = replacements.get(base, base.replace("_", " "))
    return (
        f"{base_label} — {aggregation}"
        if aggregation
        else base_label
    )


def _forecast_negative_log_likelihood(
    frame: pd.DataFrame,
    pmf: np.ndarray,
    no_event: np.ndarray,
) -> float:
    """Mean categorical log loss for event-bin or no-event targets."""

    has_event = frame["has_event_in_horizon"].to_numpy(dtype=bool)
    event_bin = frame["event_bin"].to_numpy(dtype=int)
    target_probability = no_event.copy()
    event_rows = np.flatnonzero(has_event)
    target_probability[event_rows] = pmf[
        event_rows, event_bin[event_rows]
    ]
    return float(
        -np.log(np.clip(target_probability, 1e-12, None)).mean()
    )


def permutation_feature_importance(
    model: Any,
    validation: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
    n_repeats: int = 2,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Model-agnostic global importance using validation log-loss increase.

    A positive value means that shuffling the feature worsened validation
    negative log likelihood, so the fitted model relied on that feature.
    Correlated EEG features can share or mask importance; this is descriptive,
    not causal.
    """

    if n_repeats < 1:
        raise ValueError("n_repeats must be at least one.")
    seed = config.random_seed if random_seed is None else random_seed
    rng = np.random.default_rng(seed)
    metadata_columns = [
        column
        for column in ("patient_id", "episode_id", "landmark_step")
        if column in validation
    ]
    base = validation[
        metadata_columns + MODEL_FEATURE_COLUMNS
    ].copy().reset_index(drop=True)
    _, baseline_pmf, baseline_no_event = predict_horizon_distribution(
        model, base, config
    )
    baseline_nll = _forecast_negative_log_likelihood(
        validation.reset_index(drop=True),
        baseline_pmf,
        baseline_no_event,
    )

    rows: list[dict[str, Any]] = []
    for feature in MODEL_FEATURE_COLUMNS:
        repeat_importance: list[float] = []
        original = base[feature].to_numpy(copy=True)
        for _ in range(n_repeats):
            permuted = base.copy()
            permuted[feature] = rng.permutation(original)
            _, pmf, no_event = predict_horizon_distribution(
                model, permuted, config
            )
            permuted_nll = _forecast_negative_log_likelihood(
                validation.reset_index(drop=True), pmf, no_event
            )
            repeat_importance.append(permuted_nll - baseline_nll)
        rows.append(
            {
                "feature": feature,
                "readable_feature": readable_feature_name(feature),
                "importance_nll_increase": float(
                    np.mean(repeat_importance)
                ),
                "importance_std": float(np.std(repeat_importance)),
                "baseline_validation_nll": baseline_nll,
                "repeats": n_repeats,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "importance_nll_increase", ascending=False
    ).reset_index(drop=True)


def local_counterfactual_explanation(
    model: Any,
    reading: pd.DataFrame,
    reference: pd.DataFrame | pd.Series,
    config: ForecastConfig = ForecastConfig(),
) -> pd.DataFrame:
    """Explain one forecast by replacing each feature with a reference value.

    ``risk_change_when_observed`` is original risk minus counterfactual risk.
    Positive values mean the observed feature value raised the forecast versus
    its training-median reference. The effects are sensitivity checks and are
    not additive Shapley values or causal effects.
    """

    if len(reading) != 1:
        raise ValueError("Exactly one reading row is required.")
    metadata_columns = [
        column
        for column in ("patient_id", "episode_id", "landmark_step")
        if column in reading
    ]
    base = reading[
        metadata_columns + MODEL_FEATURE_COLUMNS
    ].reset_index(drop=True)
    if isinstance(reference, pd.DataFrame):
        reference_values = reference[MODEL_FEATURE_COLUMNS].median()
    else:
        reference_values = reference.reindex(MODEL_FEATURE_COLUMNS)
    if reference_values.isna().any():
        raise ValueError("Reference feature values contain missing data.")

    _, _, original_no_event = predict_horizon_distribution(
        model, base, config
    )
    original_risk = float(1.0 - original_no_event[0])
    counterfactual = pd.DataFrame(
        np.repeat(
            base[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float),
            len(MODEL_FEATURE_COLUMNS),
            axis=0,
        ),
        columns=MODEL_FEATURE_COLUMNS,
    )
    for metadata_column in metadata_columns:
        counterfactual[metadata_column] = base.at[0, metadata_column]
    for index, feature in enumerate(MODEL_FEATURE_COLUMNS):
        counterfactual.at[index, feature] = float(reference_values[feature])
    _, _, counterfactual_no_event = predict_horizon_distribution(
        model, counterfactual, config
    )
    counterfactual_risk = 1.0 - counterfactual_no_event
    result = pd.DataFrame(
        {
            "feature": MODEL_FEATURE_COLUMNS,
            "readable_feature": [
                readable_feature_name(feature)
                for feature in MODEL_FEATURE_COLUMNS
            ],
            "observed_value": base.loc[
                0, MODEL_FEATURE_COLUMNS
            ].to_numpy(dtype=float),
            "training_median": reference_values.to_numpy(dtype=float),
            "original_event_risk": original_risk,
            "counterfactual_event_risk": counterfactual_risk,
            "risk_change_when_observed": (
                original_risk - counterfactual_risk
            ),
        }
    )
    result["absolute_risk_change"] = result[
        "risk_change_when_observed"
    ].abs()
    return result.sort_values(
        "absolute_risk_change", ascending=False
    ).reset_index(drop=True)


def save_model_bundle(
    path: str | Path,
    model: Any,
    config: ForecastConfig,
    threshold: float,
    split: dict[str, list[str]],
    metrics: pd.DataFrame,
) -> Path:
    """Persist the fitted model and the information needed for inference."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        # Store plain data instead of a custom dataclass instance so joblib.load
        # works even when this module has not already been imported.
        "config": asdict(config),
        "warning_threshold": float(threshold),
        "model_feature_columns": MODEL_FEATURE_COLUMNS.copy(),
        "hazard_feature_columns": HAZARD_FEATURE_COLUMNS.copy(),
        "split": split,
        "metrics": metrics.copy(),
        "module_version": MODULE_VERSION,
        "disclaimer": "Exploratory research model; not for clinical use.",
    }
    joblib.dump(bundle, path)
    return path


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    """Load a saved bundle and reconstruct its validated configuration."""

    bundle = joblib.load(Path(path))
    required = {
        "model",
        "config",
        "warning_threshold",
        "model_feature_columns",
        "hazard_feature_columns",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"Model bundle is missing fields: {missing}")
    raw_config = bundle["config"]
    if isinstance(raw_config, ForecastConfig):
        # Backward compatibility with bundles created before module version 3.2.
        config = raw_config
    elif isinstance(raw_config, dict):
        config = ForecastConfig(**raw_config)
    else:
        raise TypeError("Model bundle config must be a dict or ForecastConfig.")
    config.validate()
    if list(bundle["model_feature_columns"]) != MODEL_FEATURE_COLUMNS:
        raise ValueError("Model bundle feature columns do not match this module.")
    if list(bundle["hazard_feature_columns"]) != HAZARD_FEATURE_COLUMNS:
        raise ValueError("Model bundle hazard columns do not match this module.")
    return {**bundle, "config": config}


def probability_invariants(
    hazards: np.ndarray, pmf: np.ndarray, no_event: np.ndarray
) -> dict[str, bool]:
    """Return explicit numerical checks used by the notebook."""

    return {
        "hazards_are_probabilities": bool(
            np.all((hazards > 0) & (hazards < 1))
        ),
        "pmf_is_nonnegative": bool(np.all(pmf >= 0)),
        "no_event_is_probability": bool(
            np.all((no_event >= 0) & (no_event <= 1))
        ),
        "mass_sums_to_one": bool(
            np.allclose(pmf.sum(axis=1) + no_event, 1.0, atol=1e-8)
        ),
        "survival_is_monotone": bool(
            np.all(np.diff(1.0 - np.cumsum(pmf, axis=1), axis=1) <= 1e-10)
        ),
    }
