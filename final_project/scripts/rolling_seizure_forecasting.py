"""Discrete-time seizure forecasting from EDF scalp EEG recordings.

This module supports the companion ``rolling_seizure_forecasting.ipynb``.
It implements a fixed five-minute forecast episode that updates only at
five-second landmarks.  A discrete hazard model produces an internally
consistent distribution over 60 possible onset bins plus a separate
``no seizure in the next five minutes`` outcome.

The implementation is an exploratory research model, not a medical device.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import json
import math
import warnings

import joblib
import numpy as np
import pandas as pd
import pyedflib
from scipy import signal
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


MODULE_VERSION = "1.0.0"
BIN_SECONDS = 5
HORIZON_SECONDS = 300
CONTEXT_SECONDS = 120
N_BINS = HORIZON_SECONDS // BIN_SECONDS
MICRO_WINDOWS_PER_CONTEXT = CONTEXT_SECONDS // BIN_SECONDS
INTERICTAL_BUFFER_SECONDS = 300
TARGET_SAMPLE_RATE = 128
RANDOM_SEED = 42
MIN_EEG_CHANNELS = 8

BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 45.0),
}

MICRO_FEATURE_NAMES = (
    [f"relative_{name}" for name in BANDS]
    + [f"log_power_{name}" for name in BANDS]
    + [
        "log_rms",
        "log_line_length",
        "spectral_entropy",
        "usable_channel_fraction",
    ]
)

AGGREGATIONS = ("mean", "std", "last", "slope")
MODEL_FEATURE_COLUMNS = [
    f"{feature}_{aggregation}"
    for feature in MICRO_FEATURE_NAMES
    for aggregation in AGGREGATIONS
]
HAZARD_TIME_COLUMNS = ("lead_bin", "lead_fraction", "log1p_lead_seconds")
HAZARD_FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS + list(HAZARD_TIME_COLUMNS)


@dataclass(frozen=True)
class ForecastConfig:
    """Configuration used to build features and fit the hazard model."""

    bin_seconds: int = BIN_SECONDS
    horizon_seconds: int = HORIZON_SECONDS
    context_seconds: int = CONTEXT_SECONDS
    interictal_buffer_seconds: int = INTERICTAL_BUFFER_SECONDS
    target_sample_rate: int = TARGET_SAMPLE_RATE
    min_eeg_channels: int = MIN_EEG_CHANNELS
    random_seed: int = RANDOM_SEED
    test_fraction: float = 0.25
    warning_time_target: float = 0.25
    max_iter: int = 120
    learning_rate: float = 0.07
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 80
    l2_regularization: float = 1.0

    def validate(self) -> None:
        if self.bin_seconds <= 0:
            raise ValueError("bin_seconds must be positive.")
        if self.horizon_seconds % self.bin_seconds:
            raise ValueError("horizon_seconds must be divisible by bin_seconds.")
        if self.context_seconds % self.bin_seconds:
            raise ValueError("context_seconds must be divisible by bin_seconds.")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be in (0, 1).")
        if not 0 <= self.warning_time_target <= 1:
            raise ValueError("warning_time_target must be in [0, 1].")


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
    """Calculate robust spectral and signal features for one five-second bin."""

    window = signal.detrend(window, axis=1, type="linear")
    window = window - np.nanmedian(window, axis=0, keepdims=True)
    keep = _quality_mask(window)
    if int(keep.sum()) < min_channels:
        raise ValueError(
            f"Only {int(keep.sum())} usable EEG channels in a {BIN_SECONDS}-s window."
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
    total_power = np.trapezoid(
        psd[:, analysis_mask], frequencies[analysis_mask], axis=1
    )
    total_power = np.clip(total_power, 1e-12, None)

    relative_values: list[float] = []
    log_power_values: list[float] = []
    for low, high in BANDS.values():
        mask = (frequencies >= low) & (frequencies < high)
        band_power = np.trapezoid(psd[:, mask], frequencies[mask], axis=1)
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
    if source_sample_rate <= 2 * max(high for _, high in BANDS.values()):
        raise ValueError("Source sample rate is too low for the configured bands.")

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


def aggregate_context(micro_features: np.ndarray) -> np.ndarray:
    """Aggregate 24 consecutive five-second rows into one context vector."""

    if micro_features.shape != (
        MICRO_WINDOWS_PER_CONTEXT,
        len(MICRO_FEATURE_NAMES),
    ):
        raise ValueError(
            "Expected context micro-features with shape "
            f"({MICRO_WINDOWS_PER_CONTEXT}, {len(MICRO_FEATURE_NAMES)}), "
            f"received {micro_features.shape}."
        )
    means = np.mean(micro_features, axis=0)
    stds = np.std(micro_features, axis=0)
    last = micro_features[-1]
    times = np.arange(MICRO_WINDOWS_PER_CONTEXT, dtype=float) * BIN_SECONDS
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
        anchor = float(row.onset_seconds) - config.horizon_seconds
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
                "fixed_end_seconds": anchor + config.horizon_seconds,
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
                math.floor(duration - 2 * config.horizon_seconds)
            )
            for anchor in range(
                config.context_seconds, latest_anchor + 1, 60
            ):
                # Context covers 120 s before the anchor. Labels for the final
                # landmark require another full 300 s of follow-up.
                segment_start = anchor - config.context_seconds
                label_followup_end = anchor + 2 * config.horizon_seconds
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

        needed = len(patient_positives)
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
                    "fixed_end_seconds": float(
                        row.anchor_seconds + config.horizon_seconds
                    ),
                    "event_onset_seconds": np.nan,
                }
            )

    manifest = pd.concat(
        [positives, pd.DataFrame(negative_rows)], ignore_index=True
    ).sort_values(["patient_id", "episode_type", "episode_id"])
    counts = manifest.groupby(["patient_id", "episode_type"]).size().unstack(fill_value=0)
    if not (counts["preictal"] == counts["interictal"]).all():
        raise AssertionError("Positive and interictal episode counts are not matched.")
    return manifest.reset_index(drop=True)


def _episode_landmarks(
    episode: pd.Series,
    config: ForecastConfig,
) -> pd.DataFrame:
    """Extract all 60 landmark contexts from one episode."""

    read_start = float(episode["anchor_seconds"]) - config.context_seconds
    read_duration = config.context_seconds + config.horizon_seconds
    data, sample_rate, labels = read_edf_eeg_segment(
        episode["edf_path"],
        read_start,
        read_duration,
        min_channels=config.min_eeg_channels,
    )
    micro = segment_micro_features(data, sample_rate, config)
    expected_micro_rows = (
        config.context_seconds + config.horizon_seconds
    ) // config.bin_seconds
    if len(micro) != expected_micro_rows:
        raise ValueError(
            f"Expected {expected_micro_rows} micro-windows, obtained {len(micro)}."
        )

    rows: list[dict[str, Any]] = []
    is_event = episode["episode_type"] == "preictal"
    for step in range(N_BINS):
        context = micro[step : step + MICRO_WINDOWS_PER_CONTEXT]
        aggregated = aggregate_context(context)
        landmark = float(episode["anchor_seconds"]) + step * config.bin_seconds
        if is_event:
            time_to_event = float(episode["event_onset_seconds"]) - landmark
            event_bin = int(math.ceil(time_to_event / config.bin_seconds) - 1)
            if not 0 <= event_bin < N_BINS:
                raise AssertionError("Preictal landmark target is outside the horizon.")
        else:
            time_to_event = np.nan
            event_bin = -1
        row = {
            "episode_id": episode["episode_id"],
            "patient_id": episode["patient_id"],
            "source_event_id": episode["source_event_id"],
            "episode_type": episode["episode_type"],
            "recording": episode["recording"],
            "edf_path": episode["edf_path"],
            "anchor_seconds": float(episode["anchor_seconds"]),
            "fixed_end_seconds": float(episode["fixed_end_seconds"]),
            "landmark_step": step,
            "landmark_seconds": landmark,
            "time_to_event_seconds": time_to_event,
            "event_bin": event_bin,
            "has_event_in_5m": int(is_event),
            "source_sample_rate": sample_rate,
            "eeg_channel_count": len(labels),
        }
        row.update(dict(zip(MODEL_FEATURE_COLUMNS, aggregated, strict=True)))
        rows.append(row)
    return pd.DataFrame(rows)


def _cache_signature(
    manifest: pd.DataFrame, config: ForecastConfig
) -> dict[str, Any]:
    return {
        "module_version": MODULE_VERSION,
        "config": asdict(config),
        "episode_ids": manifest["episode_id"].tolist(),
        "recordings": sorted(manifest["recording"].unique().tolist()),
    }


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
                cached_signature == signature
                and set(MODEL_FEATURE_COLUMNS).issubset(cached.columns)
                and len(cached) == len(manifest) * N_BINS
            ):
                if verbose:
                    print(f"Loaded {len(cached):,} cached landmark rows.")
                return cached
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    frames: list[pd.DataFrame] = []
    total = len(manifest)
    for number, (_, episode) in enumerate(manifest.iterrows(), start=1):
        if verbose:
            print(
                f"[{number:>3}/{total}] {episode['episode_id']} "
                f"({episode['recording']})",
                flush=True,
            )
        frames.append(_episode_landmarks(episode, config))
    landmarks = pd.concat(frames, ignore_index=True)
    expected_rows = len(manifest) * N_BINS
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
        if frame["has_event_in_5m"].nunique() != 2:
            raise ValueError(f"{name} split does not contain both outcome classes.")
    return train, test, {
        "development_patients": train_patients,
        "holdout_patients": test_patients,
    }


def make_person_period(landmarks: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand landmarks into at-risk bin rows for discrete-hazard likelihood."""

    base = landmarks[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)
    event_bins = landmarks["event_bin"].to_numpy(dtype=int)
    has_event = landmarks["has_event_in_5m"].to_numpy(dtype=bool)
    counts = np.where(has_event, event_bins + 1, N_BINS).astype(int)
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
        x[cursor:stop, -2] = (lead_bins + 0.5) / N_BINS
        x[cursor:stop, -1] = np.log1p((lead_bins + 0.5) * BIN_SECONDS)
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
) -> HistGradientBoostingClassifier:
    """Fit a nonlinear discrete-time hazard model by weighted log loss."""

    x, y, weights = make_person_period(development)
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=config.random_seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        model.fit(x, y, sample_weight=weights)
    return model


def predict_horizon_distribution(
    model: HistGradientBoostingClassifier,
    landmark_features: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return hazards, 60-bin probability mass, and no-event mass."""

    if isinstance(landmark_features, pd.DataFrame):
        base = landmark_features[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)
    else:
        base = np.asarray(landmark_features, dtype=float)
    if base.ndim != 2 or base.shape[1] != len(MODEL_FEATURE_COLUMNS):
        raise ValueError("Landmark feature matrix has an unexpected shape.")

    n_samples = len(base)
    repeated = np.repeat(base, N_BINS, axis=0).astype(np.float32)
    lead_bins = np.tile(np.arange(N_BINS, dtype=float), n_samples)
    x = np.empty(
        (n_samples * N_BINS, len(HAZARD_FEATURE_COLUMNS)), dtype=np.float32
    )
    x[:, : len(MODEL_FEATURE_COLUMNS)] = repeated
    x[:, -3] = lead_bins
    x[:, -2] = (lead_bins + 0.5) / N_BINS
    x[:, -1] = np.log1p((lead_bins + 0.5) * BIN_SECONDS)

    hazards = model.predict_proba(x)[:, 1].reshape(n_samples, N_BINS)
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
    if not np.allclose(pmf.sum(axis=1) + no_event, 1.0, atol=1e-8):
        raise AssertionError("Forecast probabilities do not sum to one.")
    return hazards, pmf, no_event


def _warning_summary(
    frame: pd.DataFrame,
    risk: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    scored = frame[
        ["episode_id", "patient_id", "episode_type", "landmark_step", "time_to_event_seconds"]
    ].copy()
    scored["risk"] = risk
    scored["warning"] = scored["risk"] >= threshold
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
    negative_hours = len(negative) * BIN_SECONDS / 3600.0
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
) -> tuple[float, pd.DataFrame]:
    """Choose a development-only warning threshold under a TiW target."""

    quantiles = np.linspace(0, 1, 201)
    candidates = np.unique(np.r_[0.0, np.quantile(risk, quantiles), 1.0])
    rows: list[dict[str, float]] = []
    for threshold in candidates:
        summary = _warning_summary(development, risk, float(threshold))
        rows.append({"threshold": float(threshold), **summary})
    curve = pd.DataFrame(rows)
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


def _class_baseline(development: pd.DataFrame) -> np.ndarray:
    targets = np.where(
        development["has_event_in_5m"].to_numpy(dtype=bool),
        development["event_bin"].to_numpy(dtype=int),
        N_BINS,
    )
    counts = np.bincount(targets, minlength=N_BINS + 1).astype(float)
    return counts / counts.sum()


def evaluate_forecasts(
    model: HistGradientBoostingClassifier,
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    warning_time_target: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, pd.DataFrame]:
    """Evaluate probability, timing, and operational warning performance."""

    _, development_pmf, development_no_event = predict_horizon_distribution(
        model, development
    )
    development_risk = 1.0 - development_no_event
    threshold, threshold_curve = select_warning_threshold(
        development, development_risk, warning_time_target
    )

    hazards, pmf, no_event = predict_horizon_distribution(model, holdout)
    risk = 1.0 - no_event
    has_event = holdout["has_event_in_5m"].to_numpy(dtype=bool)
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
            np.arange(N_BINS)[None, :] < event_bin[has_event, None]
        )
    integrated_brier = np.mean(
        np.square(predicted_survival - truth_survival), axis=1
    )

    conditional_pmf = pmf / np.clip(risk[:, None], 1e-12, None)
    bin_midpoints = (np.arange(N_BINS) + 0.5) * BIN_SECONDS
    expected_seconds = conditional_pmf @ bin_midpoints
    timing_error = np.full(len(holdout), np.nan)
    timing_error[has_event] = np.abs(
        expected_seconds[has_event]
        - holdout.loc[has_event, "time_to_event_seconds"].to_numpy(dtype=float)
    )

    baseline = _class_baseline(development)
    reference_brier = np.square(baseline).sum() + 1.0 - 2.0 * np.where(
        has_event, baseline[event_bin], baseline[N_BINS]
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
            "has_event_in_5m",
        ]
    ].reset_index(drop=True)
    prediction_columns["event_risk_5m"] = risk
    prediction_columns["no_event_probability_5m"] = no_event
    prediction_columns["negative_log_likelihood"] = nll
    prediction_columns["categorical_brier"] = categorical_brier
    prediction_columns["binary_brier"] = binary_brier
    prediction_columns["integrated_brier"] = integrated_brier
    prediction_columns["expected_seconds_given_event"] = expected_seconds
    prediction_columns["absolute_timing_error_seconds"] = timing_error
    prediction_columns["warning"] = risk >= threshold
    for index in range(N_BINS):
        prediction_columns[f"p_bin_{index + 1:02d}"] = pmf[:, index]

    warning_metrics = _warning_summary(holdout, risk, threshold)
    metrics = [
        {
            "metric": "negative log likelihood",
            "value": float(nll.mean()),
            "interpretation": "Lower is better; proper score for the observed bin/no-event class.",
        },
        {
            "metric": "multicategory Brier score",
            "value": float(categorical_brier.mean()),
            "interpretation": "Lower is better; probability error across 60 bins plus no-event.",
        },
        {
            "metric": "multicategory Brier skill score",
            "value": float(brier_skill),
            "interpretation": "Above 0 improves on the development-set class-frequency forecast.",
        },
        {
            "metric": "integrated survival Brier score",
            "value": float(integrated_brier.mean()),
            "interpretation": "Lower is better; mean survival-probability error across 5 minutes.",
        },
        {
            "metric": "5-minute AUROC",
            "value": float(roc_auc_score(has_event, risk)),
            "interpretation": "Discrimination only; does not assess calibration.",
        },
        {
            "metric": "5-minute average precision",
            "value": float(average_precision_score(has_event, risk)),
            "interpretation": "Ranking metric sensitive to event prevalence.",
        },
        {
            "metric": "conditional timing MAE (seconds)",
            "value": float(np.nanmean(timing_error)),
            "interpretation": "Error of expected onset time among seizure landmarks.",
        },
        {
            "metric": "seizure sensitivity",
            "value": warning_metrics["sensitivity"],
            "interpretation": "Fraction of seizure episodes with at least one warning.",
        },
        {
            "metric": "time in warning",
            "value": warning_metrics["time_in_warning"],
            "interpretation": "Fraction of evaluated five-second landmarks under warning.",
        },
        {
            "metric": "false alarms per hour",
            "value": warning_metrics["false_alarms_per_hour"],
            "interpretation": "Rising warning edges in interictal episodes per monitored hour.",
        },
        {
            "metric": "median warning lead (seconds)",
            "value": warning_metrics["median_warning_lead_seconds"],
            "interpretation": "First-warning lead among captured holdout seizures.",
        },
        {
            "metric": "development-selected warning threshold",
            "value": threshold,
            "interpretation": "Chosen without holdout outcomes under the configured TiW target.",
        },
    ]
    metrics_frame = pd.DataFrame(metrics)

    patient_rows: list[dict[str, Any]] = []
    for patient_id, group in prediction_columns.groupby("patient_id"):
        group_risk = group["event_risk_5m"].to_numpy()
        group_has_event = group["has_event_in_5m"].to_numpy(dtype=bool)
        warning = _warning_summary(group, group_risk, threshold)
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
    aggregated = aggregate_context(micro[-MICRO_WINDOWS_PER_CONTEXT:])
    return pd.DataFrame([aggregated], columns=MODEL_FEATURE_COLUMNS)


def fixed_episode_forecast_table(
    pmf: np.ndarray,
    episode_step: int,
    config: ForecastConfig = ForecastConfig(),
) -> tuple[pd.DataFrame, float]:
    """Drop elapsed rectangles without extending the original episode end."""

    pmf = np.asarray(pmf, dtype=float).reshape(-1)
    if len(pmf) != N_BINS:
        raise ValueError(f"Expected {N_BINS} event-bin probabilities.")
    if not 0 <= episode_step < N_BINS:
        raise ValueError(f"episode_step must be in [0, {N_BINS - 1}].")
    remaining = N_BINS - episode_step
    visible = pmf[:remaining].copy()
    event_risk = float(visible.sum())
    no_event_by_fixed_end = float(1.0 - event_risk)
    conditional = (
        visible / event_risk if event_risk > 1e-12 else np.zeros_like(visible)
    )
    cumulative = np.cumsum(visible)
    absolute_bins = np.arange(episode_step + 1, N_BINS + 1)
    table = pd.DataFrame(
        {
            "absolute_episode_bin": absolute_bins,
            "interval_start_seconds": (absolute_bins - 1) * config.bin_seconds,
            "interval_end_seconds": absolute_bins * config.bin_seconds,
            "seconds_ahead_start": np.arange(remaining) * config.bin_seconds,
            "seconds_ahead_end": (np.arange(remaining) + 1) * config.bin_seconds,
            "onset_probability": visible,
            "timing_probability_given_event": conditional,
            "cumulative_onset_probability": cumulative,
            "survival_probability": 1.0 - cumulative,
        }
    )
    return table, no_event_by_fixed_end


class RollingForecastEngine:
    """Boundary-triggered EDF forecaster with a fixed, shrinking episode end."""

    def __init__(
        self,
        model: HistGradientBoostingClassifier,
        edf_path: str | Path,
        episode_anchor_seconds: float,
        config: ForecastConfig = ForecastConfig(),
    ) -> None:
        self.model = model
        self.edf_path = Path(edf_path)
        self.episode_anchor_seconds = float(episode_anchor_seconds)
        self.fixed_end_seconds = self.episode_anchor_seconds + config.horizon_seconds
        self.config = config
        self.last_step = -1
        self.recomputation_count = 0
        self.last_result: dict[str, Any] | None = None

    def advance_to(self, recording_seconds: float) -> dict[str, Any]:
        """Update only after a new five-second boundary has been crossed."""

        elapsed = float(recording_seconds) - self.episode_anchor_seconds
        if elapsed < 0:
            raise ValueError("recording_seconds precedes the episode anchor.")
        if elapsed >= self.config.horizon_seconds:
            if self.last_step >= N_BINS and self.last_result is not None:
                return {**self.last_result, "recomputed": False}
            self.last_step = N_BINS
            self.last_result = {
                "episode_step": N_BINS,
                "landmark_seconds": self.fixed_end_seconds,
                "fixed_end_seconds": self.fixed_end_seconds,
                "remaining_bins": 0,
                "event_risk_by_fixed_end": np.nan,
                "no_event_probability_by_fixed_end": np.nan,
                "no_event_probability_next_5m": np.nan,
                "hazards_next_5m": np.asarray([], dtype=float),
                "pmf_next_5m": np.asarray([], dtype=float),
                "probability_table": pd.DataFrame(
                    columns=[
                        "absolute_episode_bin",
                        "interval_start_seconds",
                        "interval_end_seconds",
                        "seconds_ahead_start",
                        "seconds_ahead_end",
                        "onset_probability",
                        "timing_probability_given_event",
                        "cumulative_onset_probability",
                        "survival_probability",
                    ]
                ),
                "recomputation_count": self.recomputation_count,
                "recomputed": False,
                "episode_complete": True,
            }
            return self.last_result

        step = int(math.floor(elapsed / self.config.bin_seconds))
        if step <= self.last_step and self.last_result is not None:
            return {**self.last_result, "recomputed": False}

        landmark_seconds = self.episode_anchor_seconds + step * self.config.bin_seconds
        features = extract_context_features_from_edf(
            self.edf_path, landmark_seconds, self.config
        )
        hazards, pmf, no_event_5m = predict_horizon_distribution(
            self.model, features
        )
        table, no_event_fixed = fixed_episode_forecast_table(
            pmf[0], step, self.config
        )
        self.last_step = step
        self.recomputation_count += 1
        self.last_result = {
            "episode_step": step,
            "landmark_seconds": landmark_seconds,
            "fixed_end_seconds": self.fixed_end_seconds,
            "remaining_bins": N_BINS - step,
            "event_risk_by_fixed_end": 1.0 - no_event_fixed,
            "no_event_probability_by_fixed_end": no_event_fixed,
            "no_event_probability_next_5m": float(no_event_5m[0]),
            "hazards_next_5m": hazards[0],
            "pmf_next_5m": pmf[0],
            "probability_table": table,
            "recomputation_count": self.recomputation_count,
            "recomputed": True,
            "episode_complete": False,
        }
        return self.last_result


def save_model_bundle(
    path: str | Path,
    model: HistGradientBoostingClassifier,
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
        "config": config,
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
