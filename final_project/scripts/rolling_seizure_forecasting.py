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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODULE_VERSION = "6.1.0"
FEATURE_CACHE_VERSION = "6.0.0"
BIN_SECONDS = 5
HORIZON_SECONDS = 300
CONTEXT_SECONDS = 120
PRE_ONSET_SECONDS = 300
INTERICTAL_BUFFER_SECONDS = 300
TARGET_SAMPLE_RATE = 64
RANDOM_SEED = 42
MIN_EEG_CHANNELS = 8

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
        "PN10": 8,
        "PN11": 1,
        "PN12": 3,
        "PN13": 2,
        "PN14": 3,
        "PN16": 1,
    },
    "validation": {"PN10": 1, "PN14": 1, "PN16": 1, "PN17": 1},
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
    random_seed: int = RANDOM_SEED
    test_fraction: float = 0.25
    warning_time_target: float = 0.25
    minimum_development_sensitivity: float = 0.90
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
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be in (0, 1).")
        if not 0 <= self.warning_time_target <= 1:
            raise ValueError("warning_time_target must be in [0, 1].")
        if not 0 <= self.minimum_development_sensitivity <= 1:
            raise ValueError(
                "minimum_development_sensitivity must be in [0, 1]."
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
    smoothing_alpha: float = 0.10


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


def _is_eeg_label(label: str) -> bool:
    upper = label.upper().replace("-", " ").strip()
    excluded = ("EKG", "ECG", "SPO2", "STATUS", "EVENT", "EMG", "RESP")
    return upper.startswith("EEG") and not any(token in upper for token in excluded)


def read_edf_eeg_segment(
    path: str | Path,
    start_seconds: float,
    duration_seconds: float,
    min_channels: int = MIN_EEG_CHANNELS,
) -> tuple[np.ndarray, float, list[str]]:
    """Random-access a scaled multichannel EEG interval from an EDF file."""

    path = Path(path)
    if start_seconds < 0 or duration_seconds <= 0:
        raise ValueError("Requested EDF segment has invalid bounds.")
    reader = pyedflib.EdfReader(str(path))
    try:
        labels = [str(label).strip() for label in reader.getSignalLabels()]
        eeg_indices = [i for i, label in enumerate(labels) if _is_eeg_label(label)]
        if len(eeg_indices) < min_channels:
            raise ValueError(
                f"{path.name} has only {len(eeg_indices)} recognized EEG channels."
            )
        rates = np.asarray(
            [reader.getSampleFrequency(i) for i in eeg_indices], dtype=float
        )
        rounded_rates = np.round(rates, 6)
        values, counts = np.unique(rounded_rates, return_counts=True)
        sample_rate = float(values[np.argmax(counts)])
        eeg_indices = [
            index
            for index, rate in zip(eeg_indices, rounded_rates, strict=True)
            if math.isclose(float(rate), sample_rate, rel_tol=0, abs_tol=1e-6)
        ]
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
        return np.vstack(arrays), sample_rate, [labels[i] for i in eeg_indices]
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
    window = window - np.nanmedian(window, axis=0, keepdims=True)
    keep = _quality_mask(window)
    if int(keep.sum()) < min_channels:
        raise ValueError(
            f"Only {int(keep.sum())} usable EEG channels in the configured window."
    )
    window = window[keep]

    frequencies, psd = signal.periodogram(
        window,
        fs=sample_rate,
        window="hann",
        detrend=False,
        scaling="density",
        axis=1,
    )
    analysis_mask = (frequencies >= 0.5) & (frequencies <= 45.0)
    total_power = np.trapz(
        psd[:, analysis_mask], frequencies[analysis_mask], axis=1
    )
    total_power = np.clip(total_power, 1e-12, None)

    relative_values: list[float] = []
    log_power_values: list[float] = []
    for low, high in BANDS.values():
        mask = (frequencies >= low) & (frequencies < high)
        band_power = np.trapz(psd[:, mask], frequencies[mask], axis=1)
        band_power = np.clip(band_power, 1e-12, None)
        relative_values.append(float(np.median(band_power / total_power)))
        log_power_values.append(float(np.median(np.log10(band_power))))

    rms = np.sqrt(np.mean(np.square(window), axis=1))
    line_length = np.mean(np.abs(np.diff(window, axis=1)), axis=1)
    normalized_psd = psd[:, analysis_mask] / np.clip(
        psd[:, analysis_mask].sum(axis=1, keepdims=True), 1e-12, None
    )
    entropy = -np.sum(
        normalized_psd * np.log(np.clip(normalized_psd, 1e-12, None)), axis=1
    )
    entropy /= np.log(normalized_psd.shape[1])

    return np.asarray(
        relative_values
        + log_power_values
        + [
            float(np.median(np.log1p(rms))),
            float(np.median(np.log1p(line_length))),
            float(np.median(entropy)),
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
                min_channels=config.min_eeg_channels,
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
        min_channels=config.min_eeg_channels,
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
    return {
        "module_version": FEATURE_CACHE_VERSION,
        "config": asdict(config),
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
    }

    def canonical(signature: dict[str, Any]) -> dict[str, Any]:
        result = dict(signature)
        result["config"] = {
            key: value
            for key, value in signature.get("config", {}).items()
            if key in data_fields
        }
        return result

    return canonical(cached_signature) == canonical(current_signature)


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

    risk_feature_columns = tuple(
        column
        for column in MODEL_FEATURE_COLUMNS
        if "channel_p90_std" in column
    )
    if not risk_feature_columns:
        raise AssertionError("No channel-p90 variability features are available.")
    risk_model = ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=5,
        max_features=1.0,
        class_weight="balanced",
        n_jobs=-1,
        random_state=config.random_seed,
    )
    risk_model.fit(
        development[list(risk_feature_columns)],
        development["has_event_in_horizon"].to_numpy(dtype=int),
    )
    return TwoStageForecastModel(
        timing_model=timing_model,
        risk_model=risk_model,
        risk_feature_columns=risk_feature_columns,
        smoothing_alpha=0.10,
    )


def model_iteration_count(model: Any) -> int | None:
    """Return fitted iterations for supported estimators."""

    if isinstance(model, TwoStageForecastModel):
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
            feature_frame is not None
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
    for index, is_high in enumerate(high):
        if not is_active:
            high_run = high_run + 1 if is_high else 0
            if high_run >= alarm_on_consecutive:
                is_active = True
                low_run = 0
        else:
            low_run = 0 if is_high else low_run + 1
            if low_run >= alarm_off_consecutive:
                is_active = False
                high_run = 0
        active[index] = is_active
    return high, active


def _alarm_columns(
    frame: pd.DataFrame,
    risk: np.ndarray,
    threshold: float,
    alarm_on_consecutive: int,
    alarm_off_consecutive: int,
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
) -> dict[str, float]:
    scored = frame[
        ["episode_id", "patient_id", "episode_type", "landmark_step", "time_to_event_seconds"]
    ].copy()
    scored["risk"] = risk
    alarm = _alarm_columns(
        scored,
        risk,
        threshold,
        alarm_on_consecutive,
        alarm_off_consecutive,
    )
    scored["raw_high_probability"] = alarm["raw_high_probability"].to_numpy()
    scored["warning"] = alarm["alarm_active"].to_numpy()
    positive = scored.loc[scored["episode_type"].eq("preictal")]
    negative = scored.loc[scored["episode_type"].eq("interictal")]

    captured = positive.groupby("episode_id")["warning"].any()
    sensitivity = float(captured.mean()) if len(captured) else float("nan")
    time_in_warning = float(scored["warning"].mean())

    false_alarms = 0
    for _, group in negative.sort_values(
        ["episode_id", "landmark_step"]
    ).groupby("episode_id"):
        values = group["warning"].to_numpy(dtype=bool)
        false_alarms += int(np.sum(values & ~np.r_[False, values[:-1]]))
    negative_hours = len(negative) * bin_seconds / 3600.0
    false_alarms_per_hour = (
        float(false_alarms / negative_hours) if negative_hours else float("nan")
    )

    lead_times: list[float] = []
    for _, group in positive.groupby("episode_id"):
        warned = group.loc[group["warning"]].sort_values("landmark_step")
        if not warned.empty:
            lead_times.append(float(warned.iloc[0]["time_to_event_seconds"]))
    return {
        "sensitivity": sensitivity,
        "time_in_warning": time_in_warning,
        "false_alarms_per_hour": false_alarms_per_hour,
        "captured_seizures": float(captured.sum()),
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
) -> tuple[float, pd.DataFrame]:
    """Choose a development-only operating point.

    When a minimum sensitivity is requested, choose the qualifying point with
    the fewest false alarms and then the least warning time. Otherwise retain
    the original time-in-warning constrained selection.
    """

    quantiles = np.linspace(0, 1, 201)
    candidates = np.unique(np.r_[0.0, np.quantile(risk, quantiles), 1.0])
    rows: list[dict[str, float]] = []
    for threshold in candidates:
        summary = _warning_summary(
            development,
            risk,
            float(threshold),
            bin_seconds,
            alarm_on_consecutive,
            alarm_off_consecutive,
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, pd.DataFrame]:
    """Evaluate probability, timing, and operational warning performance."""

    _, development_pmf, development_no_event = predict_horizon_distribution(
        model, development, config
    )
    development_risk = 1.0 - development_no_event
    threshold, threshold_curve = select_warning_threshold(
        development,
        development_risk,
        warning_time_target,
        config.bin_seconds,
        config.alarm_on_consecutive,
        config.alarm_off_consecutive,
        config.minimum_development_sensitivity,
    )

    hazards, pmf, no_event = predict_horizon_distribution(
        model, holdout, config
    )
    risk = 1.0 - no_event
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
        config.alarm_on_consecutive,
        config.alarm_off_consecutive,
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
        config.alarm_on_consecutive,
        config.alarm_off_consecutive,
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
                f"{config.alarm_on_consecutive} consecutive high-risk landmarks."
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
                "development-sensitivity target, then minimize false alarms."
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
            config.alarm_on_consecutive,
            config.alarm_off_consecutive,
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
        min_channels=config.min_eeg_channels,
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
            min_channels=self.config.min_eeg_channels,
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
    base = validation[MODEL_FEATURE_COLUMNS].copy().reset_index(drop=True)
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
    base = reading[MODEL_FEATURE_COLUMNS].reset_index(drop=True)
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
        np.repeat(base.to_numpy(dtype=float), len(MODEL_FEATURE_COLUMNS), axis=0),
        columns=MODEL_FEATURE_COLUMNS,
    )
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
            "observed_value": base.iloc[0].to_numpy(dtype=float),
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
