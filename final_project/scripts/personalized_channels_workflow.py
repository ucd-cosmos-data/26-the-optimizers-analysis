"""Patient-specific EEG channel selection for rolling seizure forecasting.

This module backs ``personalized_channels.ipynb`` and its quick-mode companion.
It preserves the five-minute discrete-hazard target from
``rolling_seizure_forecasting.py`` while retaining features per EEG channel.

The code is an exploratory research workflow, not a clinical warning system.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence
import json
import math
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import optimize, signal
from scipy.special import expit
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import rolling_seizure_forecasting as rsf
import split_eeg_channels as split_eeg


MODULE_VERSION = "3.1.0"
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 45.0),
}
PER_CHANNEL_FEATURE_NAMES = (
    *(f"relative_{band}" for band in BANDS),
    *(f"log_power_{band}" for band in BANDS),
    "log_rms",
    "log_line_length",
    "spectral_entropy",
    "usable_channel_fraction",
)
AGGREGATIONS = tuple(rsf.AGGREGATIONS)
COMPACT_FEATURE_NAMES = (
    "relative_delta",
    "relative_theta",
    "relative_alpha",
    "relative_beta",
    "relative_low_gamma",
    "log_rms",
    "log_line_length",
    "usable_channel_fraction",
)
COMPACT_AGGREGATIONS = ("mean", "last", "slope")


@dataclass(frozen=True)
class PersonalizedConfig:
    """Settings intentionally exposed near the top of both notebooks."""

    k: int = 4
    test_fraction: float = 0.20
    random_state: int = 42
    random_baseline_repeats: int = 10
    swap_refinement: bool = True
    max_patients: int | None = None
    patient_ids: tuple[str, ...] | None = None
    force_rebuild_features: bool = False
    selector_max_iter: int = 1_000
    max_threshold_folds: int = 4
    quick_mode: bool = False

    def validate(self) -> None:
        if self.k < 1:
            raise ValueError("k must be at least 1.")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be in (0, 1).")
        if self.random_baseline_repeats < 0:
            raise ValueError("random_baseline_repeats cannot be negative.")
        if self.max_patients is not None and self.max_patients < 1:
            raise ValueError("max_patients must be positive or None.")
        if self.max_threshold_folds < 1:
            raise ValueError("max_threshold_folds must be at least 1.")


@dataclass
class PatientFeatureData:
    """Cached, channel-separated landmark data for one patient."""

    patient_id: str
    frame: pd.DataFrame
    channel_names: list[str]
    channel_feature_columns: dict[str, list[str]]


def quick_config(k: int = 4, patient_id: str | None = None) -> PersonalizedConfig:
    """Return a fast smoke-test configuration for one patient."""

    return PersonalizedConfig(
        k=k,
        random_baseline_repeats=1,
        swap_refinement=False,
        max_patients=1,
        patient_ids=None if patient_id is None else (patient_id,),
        quick_mode=True,
    )


def personalized_paths(notebook_dir: str | Path) -> dict[str, Path]:
    """Resolve shared input/cache paths and mode-specific output roots."""

    paths = rsf.project_paths(notebook_dir)
    paths["feature_cache"] = paths["processed"] / "personalized_channels"
    paths["personalized_results"] = paths["project"] / "results" / "personalized_channels"
    paths["quick_results"] = (
        paths["project"] / "results" / "personalized_channels_quick"
    )
    return paths


def load_manifest(paths: dict[str, Path], forecast_config: rsf.ForecastConfig) -> pd.DataFrame:
    """Load the audited episode manifest, rebuilding it only when possible."""

    saved = paths["results"] / "episode_manifest.csv"
    if paths["event_inventory"].exists():
        events = rsf.load_event_inventory(paths["event_inventory"])
        manifest = rsf.build_episode_manifest(events, paths["raw"], forecast_config)
    elif saved.exists():
        manifest = pd.read_csv(saved)
        manifest = manifest.rename(
            columns={"fixed_end_seconds": "training_episode_end_seconds"}
        )
    else:
        raise FileNotFoundError(
            "Neither results/event_inventory.csv nor "
            "results/rolling_forecast/episode_manifest.csv is available."
        )
    required = {
        "episode_id",
        "patient_id",
        "source_event_id",
        "episode_type",
        "recording",
        "edf_path",
        "anchor_seconds",
        "training_episode_end_seconds",
        "event_onset_seconds",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Episode manifest is missing columns: {missing}")
    # Saved manifests may contain absolute paths from the computer that created
    # them. Always resolve patient/recording names against this checkout.
    manifest["edf_path"] = [
        str(rsf._resolve_edf(paths["raw"], row.patient_id, row.recording))
        for row in manifest.itertuples()
    ]
    # The rolling manifest may construct multiple interictal controls per
    # seizure and leaves their source_event_id blank. Distribute every control
    # as evenly as possible across chronologically ordered seizures so all
    # controls follow a deterministic source seizure into train or test.
    for patient_id, group in manifest.groupby("patient_id", sort=False):
        positive = group.loc[group["episode_type"].eq("preictal")].copy()
        positive["_recording_key"] = positive["recording"].map(
            _natural_recording_key
        )
        positive = positive.sort_values(
            ["_recording_key", "event_onset_seconds", "source_event_id"]
        )
        negative_indices = (
            group.loc[group["episode_type"].eq("interictal")]
            .sort_values("episode_id")
            .index
        )
        event_ids = (
            positive["source_event_id"].astype(str).drop_duplicates().tolist()
        )
        if len(negative_indices) and not event_ids:
            raise ValueError(
                f"{patient_id} has interictal controls but no preictal episodes."
            )
        controls_per_event, remainder = divmod(
            len(negative_indices), len(event_ids) or 1
        )
        assignments = [
            event_id
            for event_number, event_id in enumerate(event_ids)
            for _ in range(
                controls_per_event + int(event_number < remainder)
            )
        ]
        manifest.loc[negative_indices, "source_event_id"] = assignments
    return manifest.reset_index(drop=True)


def _natural_recording_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", str(value))
    )


def chronological_event_split(
    patient_manifest: pd.DataFrame,
    test_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split complete seizure/control pairs by seizure order.

    The last ``ceil(test_fraction * n)`` seizures are held out.  Each matched
    interictal episode follows its source seizure into the same split.
    """

    positive = patient_manifest.loc[
        patient_manifest["episode_type"].eq("preictal")
    ].copy()
    positive["_recording_key"] = positive["recording"].map(_natural_recording_key)
    positive = positive.sort_values(
        ["_recording_key", "event_onset_seconds", "source_event_id"]
    )
    event_ids = positive["source_event_id"].astype(str).drop_duplicates().tolist()
    n_seizures = len(event_ids)
    if n_seizures < 2:
        raise ValueError("Patient has fewer than two usable seizures.")
    n_test = max(1, int(math.ceil(test_fraction * n_seizures)))
    n_test = min(n_test, n_seizures - 1)
    train_ids = event_ids[:-n_test]
    test_ids = event_ids[-n_test:]
    event_as_text = patient_manifest["source_event_id"].astype(str)
    train = patient_manifest.loc[event_as_text.isin(train_ids)].copy()
    test = patient_manifest.loc[event_as_text.isin(test_ids)].copy()
    details = {
        "n_seizures": n_seizures,
        "n_train_seizures": len(train_ids),
        "n_test_seizures": len(test_ids),
        "train_event_ids": train_ids,
        "test_event_ids": test_ids,
    }
    return train, test, details


def canonical_channel_name(label: str) -> str:
    """Normalize common Siena EDF label variants without merging electrodes."""

    name = str(label).upper().strip()
    name = re.sub(r"^EEG[\s:_-]*", "", name)
    name = re.sub(r"[\s:_-]*(REF|LE|AVG)$", "", name)
    name = re.sub(r"\s+", "", name)
    return name


def common_patient_channels(patient_manifest: pd.DataFrame) -> list[str]:
    """Return channels available in every split-data episode recording."""

    channel_sets: list[set[str]] = []
    for edf_path in patient_manifest["edf_path"].drop_duplicates():
        metadata = split_eeg.split_eeg_metadata(edf_path)
        names = {
            canonical_channel_name(label)
            for label in metadata["labels"]
        }
        channel_sets.append(names)
    if not channel_sets:
        return []
    common = set.intersection(*channel_sets)
    return sorted(common, key=_natural_recording_key)


def _per_channel_micro_features(
    window: np.ndarray,
    sample_rate: float,
) -> np.ndarray:
    """Compute the rolling notebook's features separately for each channel."""

    # Detrend each channel independently. Do not apply the rolling notebook's
    # cross-channel common-median reference here: doing so would allow a
    # nominal K-channel model to contain information from excluded electrodes.
    window = signal.detrend(window, axis=1, type="linear")
    usable = rsf._quality_mask(window)
    frequencies, psd = signal.periodogram(
        window,
        fs=sample_rate,
        window="hann",
        detrend=False,
        scaling="density",
        axis=1,
    )
    analysis = (frequencies >= 0.5) & (frequencies <= 45.0)
    total_power = np.trapz(psd[:, analysis], frequencies[analysis], axis=1)
    total_power = np.clip(total_power, 1e-12, None)
    columns: list[np.ndarray] = []
    for low, high in BANDS.values():
        mask = (frequencies >= low) & (frequencies < high)
        band_power = np.trapz(psd[:, mask], frequencies[mask], axis=1)
        band_power = np.clip(band_power, 1e-12, None)
        columns.append(band_power / total_power)
    for low, high in BANDS.values():
        mask = (frequencies >= low) & (frequencies < high)
        band_power = np.trapz(psd[:, mask], frequencies[mask], axis=1)
        columns.append(np.log10(np.clip(band_power, 1e-12, None)))
    rms = np.sqrt(np.mean(np.square(window), axis=1))
    line_length = np.mean(np.abs(np.diff(window, axis=1)), axis=1)
    normalized_psd = psd[:, analysis] / np.clip(
        psd[:, analysis].sum(axis=1, keepdims=True), 1e-12, None
    )
    entropy = -np.sum(
        normalized_psd * np.log(np.clip(normalized_psd, 1e-12, None)), axis=1
    ) / np.log(normalized_psd.shape[1])
    columns.extend(
        [
            np.log1p(rms),
            np.log1p(line_length),
            entropy,
            usable.astype(float),
        ]
    )
    result = np.column_stack(columns).astype(float)
    # Bad channels become missing for physiological features. HistGradientBoosting
    # handles missing values; the final usability feature remains explicit.
    result[~usable, :-1] = np.nan
    return result


def _segment_channel_features(
    data: np.ndarray,
    source_sample_rate: float,
    forecast_config: rsf.ForecastConfig,
) -> np.ndarray:
    """Return ``(micro_windows, channels, per_channel_features)``."""

    if source_sample_rate <= 2 * max(high for _, high in BANDS.values()):
        raise ValueError("Source sample rate is too low for the configured bands.")
    rounded_rate = int(round(source_sample_rate))
    gcd = math.gcd(rounded_rate, forecast_config.target_sample_rate)
    up = forecast_config.target_sample_rate // gcd
    down = rounded_rate // gcd
    if up != down:
        data = signal.resample_poly(data, up=up, down=down, axis=1)
    sample_rate = float(forecast_config.target_sample_rate)
    samples_per_bin = int(round(forecast_config.bin_seconds * sample_rate))
    n_windows = data.shape[1] // samples_per_bin
    data = data[:, : n_windows * samples_per_bin]
    return np.stack(
        [
            _per_channel_micro_features(
                data[:, index * samples_per_bin : (index + 1) * samples_per_bin],
                sample_rate,
            )
            for index in range(n_windows)
        ]
    )


def _aggregate_channel_context(
    micro: np.ndarray,
    *,
    expected_windows: int,
    bin_seconds: int,
) -> np.ndarray:
    """Aggregate context as mean/std/latest/slope for every channel."""

    if micro.shape[0] != expected_windows:
        raise ValueError(
            f"Expected {expected_windows} context windows, "
            f"found {micro.shape[0]}."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        means = np.nanmean(micro, axis=0)
        stds = np.nanstd(micro, axis=0)
        latest = micro[-1]
        times = np.arange(expected_windows, dtype=float) * bin_seconds
        centered = times - times.mean()
        slopes = np.nansum(
            centered[:, None, None] * (micro - means[None, :, :]), axis=0
        ) / np.square(centered).sum()
    # Flatten in channel -> feature -> aggregation order.
    return np.stack([means, stds, latest, slopes], axis=-1).reshape(micro.shape[1], -1)


def _channel_column_map(channel_names: Sequence[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for channel in channel_names:
        result[channel] = [
            f"ch::{channel}::{feature}::{aggregation}"
            for feature in PER_CHANNEL_FEATURE_NAMES
            for aggregation in AGGREGATIONS
        ]
    return result


def compact_channel_column_map(
    channel_feature_columns: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Keep a prespecified compact feature set to reduce patient-level overfit."""

    compact: dict[str, list[str]] = {}
    allowed_features = set(COMPACT_FEATURE_NAMES)
    allowed_aggregations = set(COMPACT_AGGREGATIONS)
    for channel, columns in channel_feature_columns.items():
        kept = []
        for column in columns:
            _, _, feature, aggregation = column.split("::", maxsplit=3)
            if feature in allowed_features and aggregation in allowed_aggregations:
                kept.append(column)
        if len(kept) != len(COMPACT_FEATURE_NAMES) * len(COMPACT_AGGREGATIONS):
            raise AssertionError(f"Unexpected compact feature width for {channel}.")
        compact[channel] = kept
    return compact


def _episode_channel_landmarks(
    episode: pd.Series,
    channel_names: Sequence[str],
    forecast_config: rsf.ForecastConfig,
) -> pd.DataFrame:
    read_start = float(episode["anchor_seconds"]) - forecast_config.context_seconds
    read_duration = forecast_config.context_seconds + forecast_config.horizon_seconds
    data, sample_rate, labels = split_eeg.read_split_eeg_segment(
        episode["edf_path"],
        read_start,
        read_duration,
        min_channels=1,
    )
    lookup = {canonical_channel_name(label): index for index, label in enumerate(labels)}
    missing = [name for name in channel_names if name not in lookup]
    if missing:
        raise ValueError(f"Episode {episode['episode_id']} lacks channels {missing}.")
    data = data[[lookup[name] for name in channel_names]]
    micro = _segment_channel_features(data, sample_rate, forecast_config)
    expected_rows = (
        forecast_config.context_seconds + forecast_config.horizon_seconds
    ) // forecast_config.bin_seconds
    if len(micro) != expected_rows:
        raise ValueError(f"Expected {expected_rows} micro-windows, found {len(micro)}.")
    column_map = _channel_column_map(channel_names)
    feature_columns = [column for name in channel_names for column in column_map[name]]
    is_event = episode["episode_type"] == "preictal"
    rows: list[dict[str, Any]] = []
    context_windows = (
        forecast_config.context_seconds // forecast_config.bin_seconds
    )
    for step in range(forecast_config.n_bins):
        context = micro[step : step + context_windows]
        features = _aggregate_channel_context(
            context,
            expected_windows=context_windows,
            bin_seconds=forecast_config.bin_seconds,
        ).reshape(-1)
        landmark = float(episode["anchor_seconds"]) + step * forecast_config.bin_seconds
        if is_event:
            time_to_event = float(episode["event_onset_seconds"]) - landmark
            event_bin = int(math.ceil(time_to_event / forecast_config.bin_seconds) - 1)
        else:
            time_to_event = np.nan
            event_bin = -1
        row = {
            "episode_id": episode["episode_id"],
            "patient_id": episode["patient_id"],
            "source_event_id": str(episode["source_event_id"]),
            "episode_type": episode["episode_type"],
            "recording": episode["recording"],
            "landmark_step": step,
            "landmark_seconds": landmark,
            "time_to_event_seconds": time_to_event,
            "event_bin": event_bin,
            "has_event_in_5m": int(is_event),
        }
        row.update(dict(zip(feature_columns, features, strict=True)))
        rows.append(row)
    return pd.DataFrame(rows)


def _patient_cache_signature(
    patient_manifest: pd.DataFrame,
    channel_names: Sequence[str],
    forecast_config: rsf.ForecastConfig,
) -> dict[str, Any]:
    split_manifests = []
    for edf_path in patient_manifest["edf_path"].drop_duplicates():
        manifest = (
            split_eeg.split_recording_dir(edf_path) / "channel_manifest.csv"
        )
        stat = manifest.stat()
        split_manifests.append(
            {
                "path": str(manifest),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return {
        "module_version": MODULE_VERSION,
        "rolling_module_version": rsf.MODULE_VERSION,
        "signal_source": "raw/splitdata",
        "split_manifests": split_manifests,
        "forecast_config": asdict(forecast_config),
        "episode_ids": patient_manifest["episode_id"].astype(str).tolist(),
        "channel_names": list(channel_names),
    }


def build_patient_feature_data(
    patient_manifest: pd.DataFrame,
    cache_dir: str | Path,
    forecast_config: rsf.ForecastConfig,
    *,
    force: bool = False,
    verbose: bool = True,
) -> PatientFeatureData:
    """Build or reuse channel-separated landmark features for one patient."""

    patient_ids = patient_manifest["patient_id"].astype(str).unique()
    if len(patient_ids) != 1:
        raise ValueError("patient_manifest must contain exactly one patient.")
    patient_id = patient_ids[0]
    channels = common_patient_channels(patient_manifest)
    if not channels:
        raise ValueError(f"No common EEG channels found for {patient_id}.")
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{patient_id}_channel_landmarks.joblib"
    signature_path = cache_dir / f"{patient_id}_channel_landmarks.json"
    signature = _patient_cache_signature(patient_manifest, channels, forecast_config)
    if not force and cache_path.exists() and signature_path.exists():
        try:
            if json.loads(signature_path.read_text("utf-8")) == signature:
                cached = joblib.load(cache_path)
                if verbose:
                    print(f"{patient_id}: loaded cached channel features.")
                return PatientFeatureData(
                    patient_id=patient_id,
                    frame=cached["frame"],
                    channel_names=cached["channel_names"],
                    channel_feature_columns=cached["channel_feature_columns"],
                )
        except (OSError, ValueError, json.JSONDecodeError, KeyError):
            pass
    frames = []
    for number, (_, episode) in enumerate(patient_manifest.iterrows(), start=1):
        if verbose:
            print(
                f"{patient_id} [{number}/{len(patient_manifest)}] "
                f"{episode['episode_id']}",
                flush=True,
            )
        frames.append(_episode_channel_landmarks(episode, channels, forecast_config))
    frame = pd.concat(frames, ignore_index=True)
    channel_map = _channel_column_map(channels)
    cache_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "frame": frame,
            "channel_names": channels,
            "channel_feature_columns": channel_map,
        },
        cache_path,
        compress=3,
    )
    signature_path.write_text(
        json.dumps(signature, indent=2, sort_keys=True), encoding="utf-8"
    )
    return PatientFeatureData(patient_id, frame, channels, channel_map)


def _subset_columns(
    channel_feature_columns: dict[str, list[str]],
    channels: Iterable[str],
) -> list[str]:
    return [
        column
        for channel in channels
        for column in channel_feature_columns[channel]
    ]


def _selector_estimator(config: PersonalizedConfig):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=config.selector_max_iter,
            solver="liblinear",
            random_state=config.random_state,
        ),
    )


def _chronological_validation_splits(
    train: pd.DataFrame,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    events = (
        train.loc[train["episode_type"].eq("preictal")]
        .sort_values(["recording", "landmark_seconds"])
        ["source_event_id"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    event_values = train["source_event_id"].astype(str).to_numpy()
    if len(events) < 2:
        index = np.arange(len(train))
        return [(index, index)], "training_resubstitution_fallback"
    splits = []
    for split_at in range(1, len(events)):
        fit_events = events[:split_at]
        valid_event = events[split_at]
        fit = np.flatnonzero(np.isin(event_values, fit_events))
        valid = np.flatnonzero(event_values == valid_event)
        splits.append((fit, valid))
    return splits, "expanding_chronological_validation"


def _selection_score(
    train: pd.DataFrame,
    columns: Sequence[str],
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    config: PersonalizedConfig,
) -> tuple[float, float, float]:
    """AUPRC-dominant score with a small calibration penalty."""

    x = train[list(columns)].to_numpy(dtype=float)
    # LogisticRegression cannot consume NaN. Imputation values are derived only
    # from each fitting fold.
    y = train["has_event_in_5m"].to_numpy(dtype=int)
    prediction_parts: list[np.ndarray] = []
    truth_parts: list[np.ndarray] = []
    for fit, valid in splits:
        fill = np.nanmedian(x[fit], axis=0)
        fill = np.where(np.isfinite(fill), fill, 0.0)
        x_fit = np.where(np.isfinite(x[fit]), x[fit], fill)
        x_valid = np.where(np.isfinite(x[valid]), x[valid], fill)
        model = _selector_estimator(config)
        model.fit(x_fit, y[fit])
        prediction_parts.append(model.predict_proba(x_valid)[:, 1])
        truth_parts.append(y[valid])
    probability = np.concatenate(prediction_parts)
    truth = np.concatenate(truth_parts)
    ap = float(average_precision_score(truth, probability))
    brier = float(brier_score_loss(truth, probability))
    return ap - 0.25 * brier, ap, brier


def select_fixed_k_channels(
    train: pd.DataFrame,
    channel_feature_columns: dict[str, list[str]],
    config: PersonalizedConfig,
) -> tuple[list[str], pd.DataFrame, str]:
    """Greedy forward selection with optional one-for-one swap refinement."""

    channels = list(channel_feature_columns)
    if config.k > len(channels):
        raise ValueError(
            f"Requested K={config.k}, but only {len(channels)} channels are common."
        )
    splits, validation_method = _chronological_validation_splits(train)
    cache: dict[tuple[str, ...], tuple[float, float, float]] = {}

    def evaluate(subset: Iterable[str]) -> tuple[float, float, float]:
        key = tuple(sorted(subset, key=_natural_recording_key))
        if key not in cache:
            cache[key] = _selection_score(
                train,
                _subset_columns(channel_feature_columns, key),
                splits,
                config,
            )
        return cache[key]

    chosen: list[str] = []
    remaining = set(channels)
    trace_rows: list[dict[str, Any]] = []
    for step in range(1, config.k + 1):
        candidates = []
        for channel in sorted(remaining, key=_natural_recording_key):
            subset = [*chosen, channel]
            score, ap, brier = evaluate(subset)
            candidates.append((score, channel, ap, brier))
        score, added, ap, brier = max(candidates, key=lambda value: value[0])
        chosen.append(added)
        remaining.remove(added)
        trace_rows.append(
            {
                "step": step,
                "action": "add",
                "channel": added,
                "channels": ", ".join(chosen),
                "selection_score": score,
                "validation_auprc": ap,
                "validation_brier": brier,
            }
        )
    if config.swap_refinement and remaining and config.k > 1:
        improved = True
        while improved:
            improved = False
            current = evaluate(chosen)[0]
            swaps = []
            for old in chosen:
                for new in sorted(remaining, key=_natural_recording_key):
                    subset = [channel for channel in chosen if channel != old] + [new]
                    score, ap, brier = evaluate(subset)
                    swaps.append((score, old, new, subset, ap, brier))
            score, old, new, subset, ap, brier = max(swaps, key=lambda value: value[0])
            if score > current + 1e-12:
                chosen = subset
                remaining.remove(new)
                remaining.add(old)
                improved = True
                trace_rows.append(
                    {
                        "step": config.k,
                        "action": f"swap {old} -> {new}",
                        "channel": new,
                        "channels": ", ".join(chosen),
                        "selection_score": score,
                        "validation_auprc": ap,
                        "validation_brier": brier,
                    }
                )
    return chosen, pd.DataFrame(trace_rows), validation_method


def _make_person_period(
    landmarks: pd.DataFrame,
    feature_columns: Sequence[str],
    forecast_config: rsf.ForecastConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = landmarks[list(feature_columns)].to_numpy(dtype=float)
    event_bins = landmarks["event_bin"].to_numpy(dtype=int)
    has_event = landmarks["has_event_in_5m"].to_numpy(dtype=bool)
    counts = np.where(
        has_event, event_bins + 1, forecast_config.n_bins
    ).astype(int)
    total = int(counts.sum())
    x = np.empty((total, len(feature_columns) + 3), dtype=np.float32)
    y = np.zeros(total, dtype=np.uint8)
    weights = np.empty(total, dtype=np.float32)
    episode_totals = (
        pd.Series(counts, index=landmarks.index)
        .groupby(landmarks["episode_id"])
        .transform("sum")
        .to_numpy(dtype=float)
    )
    cursor = 0
    for row_index, count in enumerate(counts):
        stop = cursor + count
        x[cursor:stop, : len(feature_columns)] = base[row_index]
        lead = np.arange(count, dtype=float)
        x[cursor:stop, -3] = lead
        x[cursor:stop, -2] = (lead + 0.5) / forecast_config.n_bins
        x[cursor:stop, -1] = np.log1p(
            (lead + 0.5) * forecast_config.bin_seconds
        )
        if has_event[row_index]:
            y[stop - 1] = 1
        weights[cursor:stop] = 1.0 / episode_totals[row_index]
        cursor = stop
    weights *= len(weights) / weights.sum()
    return x, y, weights


def fit_hazard_model(
    train: pd.DataFrame,
    feature_columns: Sequence[str],
    forecast_config: rsf.ForecastConfig,
) -> HistGradientBoostingClassifier:
    x, y, weights = _make_person_period(
        train, feature_columns, forecast_config
    )
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=forecast_config.learning_rate,
        max_iter=forecast_config.max_iter,
        max_leaf_nodes=forecast_config.max_leaf_nodes,
        min_samples_leaf=forecast_config.min_samples_leaf,
        l2_regularization=forecast_config.l2_regularization,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=forecast_config.random_seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        model.fit(x, y, sample_weight=weights)
    return model


def predict_horizon_distribution(
    model: HistGradientBoostingClassifier,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    forecast_config: rsf.ForecastConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = frame[list(feature_columns)].to_numpy(dtype=float)
    n = len(base)
    repeated = np.repeat(base, forecast_config.n_bins, axis=0).astype(
        np.float32
    )
    lead = np.tile(np.arange(forecast_config.n_bins, dtype=float), n)
    x = np.empty(
        (n * forecast_config.n_bins, len(feature_columns) + 3),
        dtype=np.float32,
    )
    x[:, : len(feature_columns)] = repeated
    x[:, -3] = lead
    x[:, -2] = (lead + 0.5) / forecast_config.n_bins
    x[:, -1] = np.log1p((lead + 0.5) * forecast_config.bin_seconds)
    hazards = np.clip(
        model.predict_proba(x)[:, 1].reshape(n, forecast_config.n_bins),
        1e-7,
        1 - 1e-7,
    )
    survival_before = np.concatenate(
        [np.ones((n, 1)), np.cumprod(1.0 - hazards[:, :-1], axis=1)], axis=1
    )
    pmf = hazards * survival_before
    no_event = np.prod(1.0 - hazards, axis=1)
    return hazards, pmf, no_event


def _chronological_forecast_splits(
    train: pd.DataFrame,
    max_folds: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Bounded expanding-window splits over complete seizure/control pairs."""

    events = (
        train.loc[train["episode_type"].eq("preictal"), "source_event_id"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    if len(events) < 2:
        return []
    first_split = 2 if len(events) >= 3 else 1
    candidates = np.arange(first_split, len(events), dtype=int)
    if len(candidates) > max_folds:
        positions = np.linspace(0, len(candidates) - 1, max_folds).round().astype(int)
        candidates = candidates[np.unique(positions)]
    event_values = train["source_event_id"].astype(str).to_numpy()
    return [
        (
            np.flatnonzero(np.isin(event_values, events[:split_at])),
            np.flatnonzero(event_values == events[split_at]),
        )
        for split_at in candidates
    ]


def _episode_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    sizes = frame.groupby("episode_id")["episode_id"].transform("size").to_numpy(float)
    weights = 1.0 / sizes
    return weights * len(weights) / weights.sum()


def _oof_hazard_risk(
    train: pd.DataFrame,
    feature_columns: Sequence[str],
    forecast_config: rsf.ForecastConfig,
    max_folds: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    frames: list[pd.DataFrame] = []
    risks: list[np.ndarray] = []
    for fit_index, valid_index in _chronological_forecast_splits(train, max_folds):
        fold_model = fit_hazard_model(
            train.iloc[fit_index], feature_columns, forecast_config
        )
        _, _, no_event = predict_horizon_distribution(
            fold_model,
            train.iloc[valid_index],
            feature_columns,
            forecast_config,
        )
        frames.append(train.iloc[valid_index].copy())
        risks.append(1.0 - no_event)
    if not frames:
        return train.iloc[0:0].copy(), np.asarray([], dtype=float)
    return pd.concat(frames, ignore_index=True), np.concatenate(risks)


@dataclass
class RiskCalibrator:
    """Training-only positive-slope Platt calibration for five-minute risk."""

    slope: float | None
    intercept: float
    method: str

    def transform(self, risk: np.ndarray) -> np.ndarray:
        values = np.asarray(risk, dtype=float)
        if self.slope is None:
            return np.clip(values, 1e-5, 1 - 1e-5)
        logit = np.log(np.clip(values, 1e-7, 1 - 1e-7)) - np.log(
            np.clip(1.0 - values, 1e-7, 1 - 1e-7)
        )
        return np.clip(expit(self.slope * logit + self.intercept), 1e-5, 1 - 1e-5)


def fit_risk_calibrator(
    frame: pd.DataFrame,
    raw_risk: np.ndarray,
) -> RiskCalibrator:
    y = frame["has_event_in_5m"].to_numpy(dtype=int)
    risk = np.asarray(raw_risk, dtype=float)
    if len(risk) < 2 or np.unique(y).size < 2 or np.unique(risk).size < 3:
        return RiskCalibrator(None, 0.0, "identity_fallback")
    x = np.log(np.clip(risk, 1e-7, 1 - 1e-7)) - np.log(
        np.clip(1.0 - risk, 1e-7, 1 - 1e-7)
    )
    weights = _episode_balanced_weights(frame)
    prevalence = np.average(y, weights=weights)
    initial_intercept = float(
        np.log(np.clip(prevalence, 1e-5, 1 - 1e-5) / (1 - prevalence))
        - np.average(x, weights=weights)
    )

    def objective(parameters: np.ndarray) -> float:
        slope = float(np.exp(parameters[0]))
        probability = np.clip(expit(slope * x + parameters[1]), 1e-8, 1 - 1e-8)
        loss = -(y * np.log(probability) + (1 - y) * np.log(1 - probability))
        return float(np.average(loss, weights=weights))

    fitted = optimize.minimize(
        objective,
        x0=np.asarray([0.0, initial_intercept]),
        method="L-BFGS-B",
        bounds=[(-5.0, 5.0), (-20.0, 20.0)],
    )
    if not fitted.success:
        return RiskCalibrator(None, 0.0, "identity_calibration_failure")
    return RiskCalibrator(
        float(np.exp(fitted.x[0])),
        float(fitted.x[1]),
        "positive_platt_oof",
    )


def _calibrate_distribution(
    pmf: np.ndarray,
    no_event: np.ndarray,
    calibrator: RiskCalibrator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_risk = np.clip(1.0 - no_event, 1e-12, 1.0)
    calibrated_risk = calibrator.transform(raw_risk)
    conditional_timing = pmf / raw_risk[:, None]
    calibrated_pmf = conditional_timing * calibrated_risk[:, None]
    calibrated_no_event = 1.0 - calibrated_risk
    return calibrated_pmf, calibrated_no_event, calibrated_risk


def _evaluate_risk(
    test: pd.DataFrame,
    risk: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    has_event = test["has_event_in_5m"].to_numpy(dtype=bool)
    warning = rsf._warning_summary(test, risk, threshold)
    predictions = test[
        [
            "patient_id",
            "source_event_id",
            "episode_id",
            "episode_type",
            "recording",
            "landmark_step",
            "time_to_event_seconds",
            "has_event_in_5m",
        ]
    ].copy()
    predictions["event_risk_5m"] = risk
    predictions["warning"] = risk >= threshold
    metrics = {
        "auprc": float(average_precision_score(has_event, risk)),
        "auroc": float(roc_auc_score(has_event, risk)),
        "binary_brier": float(brier_score_loss(has_event, risk)),
        "seizure_sensitivity": float(warning["sensitivity"]),
        "time_in_warning": float(warning["time_in_warning"]),
        "false_alarms_per_hour": float(warning["false_alarms_per_hour"]),
        "median_warning_lead_seconds": float(
            warning["median_warning_lead_seconds"]
        ),
        "warning_threshold": float(threshold),
    }
    return metrics, predictions


def evaluate_model(
    model: HistGradientBoostingClassifier,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Sequence[str],
    forecast_config: rsf.ForecastConfig,
    *,
    max_threshold_folds: int = 4,
    use_oof_calibration: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calibrate/threshold on chronological OOF train predictions, then test."""

    if use_oof_calibration:
        threshold_frame, raw_threshold_risk = _oof_hazard_risk(
            train, feature_columns, forecast_config, max_threshold_folds
        )
    else:
        threshold_frame = train.iloc[0:0].copy()
        raw_threshold_risk = np.asarray([], dtype=float)
    if threshold_frame.empty:
        _, _, train_no_event = predict_horizon_distribution(
            model, train, feature_columns, forecast_config
        )
        threshold_frame = train
        raw_threshold_risk = 1.0 - train_no_event
        calibrator = RiskCalibrator(
            None, 0.0, "identity_single_seizure_fallback"
        )
        threshold_method = "training_resubstitution_fallback"
    else:
        calibrator = fit_risk_calibrator(threshold_frame, raw_threshold_risk)
        threshold_method = "chronological_oof"
    calibrated_threshold_risk = calibrator.transform(raw_threshold_risk)
    threshold, _ = rsf.select_warning_threshold(
        threshold_frame,
        calibrated_threshold_risk,
        forecast_config.warning_time_target,
    )
    _, raw_pmf, raw_no_event = predict_horizon_distribution(
        model, test, feature_columns, forecast_config
    )
    pmf, no_event, risk = _calibrate_distribution(
        raw_pmf, raw_no_event, calibrator
    )
    has_event = test["has_event_in_5m"].to_numpy(dtype=bool)
    event_bin = test["event_bin"].to_numpy(dtype=int)
    rows = np.arange(len(test))
    observed_probability = np.where(has_event, pmf[rows, event_bin], no_event)
    nll = -np.log(np.clip(observed_probability, 1e-12, None))
    squared = np.square(pmf).sum(axis=1) + np.square(no_event)
    categorical_brier = squared.copy()
    categorical_brier[has_event] += 1.0 - 2.0 * pmf[
        rows[has_event], event_bin[has_event]
    ]
    categorical_brier[~has_event] += 1.0 - 2.0 * no_event[~has_event]
    risk_metrics, predictions = _evaluate_risk(test, risk, threshold)
    predictions["raw_event_risk_5m"] = 1.0 - raw_no_event
    metrics: dict[str, Any] = {
        **risk_metrics,
        "categorical_brier": float(categorical_brier.mean()),
        "negative_log_likelihood": float(nll.mean()),
        "calibration_method": calibrator.method,
        "threshold_method": threshold_method,
    }
    return metrics, predictions


def _fit_and_evaluate_subset(
    train: pd.DataFrame,
    test: pd.DataFrame,
    channel_map: dict[str, list[str]],
    channels: Sequence[str],
    forecast_config: rsf.ForecastConfig,
    *,
    max_threshold_folds: int = 4,
    use_oof_calibration: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    columns = _subset_columns(channel_map, channels)
    model = fit_hazard_model(train, columns, forecast_config)
    return evaluate_model(
        model,
        train,
        test,
        columns,
        forecast_config,
        max_threshold_folds=max_threshold_folds,
        use_oof_calibration=use_oof_calibration,
    )


def fit_binary_risk_model(
    train: pd.DataFrame,
    feature_columns: Sequence[str],
    random_state: int,
):
    """Simple regularized comparator for event-in-five-minutes risk."""

    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=2_000,
            solver="liblinear",
            random_state=random_state,
        ),
    )
    model.fit(
        train[list(feature_columns)].to_numpy(dtype=float),
        train["has_event_in_5m"].to_numpy(dtype=int),
        logisticregression__sample_weight=_episode_balanced_weights(train),
    )
    return model


def _oof_binary_risk(
    train: pd.DataFrame,
    feature_columns: Sequence[str],
    random_state: int,
    max_folds: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    frames: list[pd.DataFrame] = []
    risks: list[np.ndarray] = []
    for fold, (fit_index, valid_index) in enumerate(
        _chronological_forecast_splits(train, max_folds)
    ):
        model = fit_binary_risk_model(
            train.iloc[fit_index], feature_columns, random_state + fold
        )
        risks.append(
            model.predict_proba(
                train.iloc[valid_index][list(feature_columns)].to_numpy(float)
            )[:, 1]
        )
        frames.append(train.iloc[valid_index].copy())
    if not frames:
        return train.iloc[0:0].copy(), np.asarray([], dtype=float)
    return pd.concat(frames, ignore_index=True), np.concatenate(risks)


def fit_and_evaluate_binary(
    train: pd.DataFrame,
    test: pd.DataFrame,
    channel_map: dict[str, list[str]],
    channels: Sequence[str],
    config: PersonalizedConfig,
    forecast_config: rsf.ForecastConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    columns = _subset_columns(channel_map, channels)
    model = fit_binary_risk_model(train, columns, config.random_state)
    threshold_frame, raw_threshold_risk = _oof_binary_risk(
        train, columns, config.random_state, config.max_threshold_folds
    )
    if threshold_frame.empty:
        threshold_frame = train
        raw_threshold_risk = model.predict_proba(
            train[columns].to_numpy(float)
        )[:, 1]
        calibrator = RiskCalibrator(
            None, 0.0, "identity_single_seizure_fallback"
        )
        threshold_method = "training_resubstitution_fallback"
    else:
        calibrator = fit_risk_calibrator(threshold_frame, raw_threshold_risk)
        threshold_method = "chronological_oof"
    calibrated_threshold_risk = calibrator.transform(raw_threshold_risk)
    threshold, _ = rsf.select_warning_threshold(
        threshold_frame,
        calibrated_threshold_risk,
        forecast_config.warning_time_target,
    )
    raw_test_risk = model.predict_proba(test[columns].to_numpy(float))[:, 1]
    test_risk = calibrator.transform(raw_test_risk)
    metrics, predictions = _evaluate_risk(test, test_risk, threshold)
    metrics.update(
        {
            "calibration_method": calibrator.method,
            "threshold_method": threshold_method,
        }
    )
    predictions["raw_event_risk_5m"] = raw_test_risk
    return metrics, predictions


def analyze_patient(
    patient_manifest: pd.DataFrame,
    paths: dict[str, Path],
    config: PersonalizedConfig,
    forecast_config: rsf.ForecastConfig,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    patient_id = str(patient_manifest["patient_id"].iloc[0])
    train_manifest, test_manifest, split = chronological_event_split(
        patient_manifest, config.test_fraction
    )
    features = build_patient_feature_data(
        patient_manifest,
        paths["feature_cache"],
        forecast_config,
        force=config.force_rebuild_features,
    )
    train_ids = set(train_manifest["episode_id"].astype(str))
    test_ids = set(test_manifest["episode_id"].astype(str))
    train = features.frame.loc[
        features.frame["episode_id"].astype(str).isin(train_ids)
    ].copy()
    test = features.frame.loc[
        features.frame["episode_id"].astype(str).isin(test_ids)
    ].copy()
    channel_map = compact_channel_column_map(features.channel_feature_columns)
    selected, trace, validation_method = select_fixed_k_channels(
        train, channel_map, config
    )
    selected_metrics, selected_predictions = _fit_and_evaluate_subset(
        train,
        test,
        channel_map,
        selected,
        forecast_config,
        max_threshold_folds=config.max_threshold_folds,
    )
    all_metrics, all_predictions = _fit_and_evaluate_subset(
        train,
        test,
        channel_map,
        features.channel_names,
        forecast_config,
        max_threshold_folds=config.max_threshold_folds,
    )
    binary_selected_metrics, binary_selected_predictions = fit_and_evaluate_binary(
        train, test, channel_map, selected, config, forecast_config
    )
    binary_all_metrics, binary_all_predictions = fit_and_evaluate_binary(
        train, test, channel_map, features.channel_names, config, forecast_config
    )
    rng = np.random.default_rng(config.random_state)
    random_rows: list[dict[str, Any]] = []
    for repeat in range(config.random_baseline_repeats):
        random_channels = sorted(
            rng.choice(features.channel_names, size=config.k, replace=False).tolist(),
            key=_natural_recording_key,
        )
        metrics, _ = _fit_and_evaluate_subset(
            train,
            test,
            channel_map,
            random_channels,
            replace(forecast_config, random_seed=config.random_state + repeat + 1),
            use_oof_calibration=False,
        )
        random_rows.append(
            {
                "repeat": repeat + 1,
                "channels": ", ".join(random_channels),
                "auprc": metrics["auprc"],
                "auroc": metrics["auroc"],
            }
        )
    random_frame = pd.DataFrame(random_rows)
    random_auprc = (
        float(random_frame["auprc"].mean()) if not random_frame.empty else np.nan
    )
    random_auprc_std = (
        float(random_frame["auprc"].std(ddof=1))
        if len(random_frame) > 1
        else np.nan
    )
    no_skill_auprc = float(test["has_event_in_5m"].mean())
    summary = {
        "patient_id": patient_id,
        "status": "included",
        "requested_k": config.k,
        "available_channels": len(features.channel_names),
        "features_per_channel": len(next(iter(channel_map.values()))),
        "selected_model_feature_count": config.k * len(next(iter(channel_map.values()))),
        "selected_channels": ", ".join(selected),
        "selection_validation": validation_method,
        "test_no_skill_auprc": no_skill_auprc,
        **split,
        **{f"selected_{key}": value for key, value in selected_metrics.items()},
        **{f"all_{key}": value for key, value in all_metrics.items()},
        **{
            f"binary_selected_{key}": value
            for key, value in binary_selected_metrics.items()
        },
        **{f"binary_all_{key}": value for key, value in binary_all_metrics.items()},
        "random_k_mean_auprc": random_auprc,
        "random_k_sd_auprc": random_auprc_std,
        "random_k_repeats": len(random_frame),
        "selected_beats_random_fraction": (
            float((random_frame["auprc"] < selected_metrics["auprc"]).mean())
            if not random_frame.empty
            else np.nan
        ),
        "auprc_retained_fraction": (
            selected_metrics["auprc"] / all_metrics["auprc"]
            if all_metrics["auprc"] > 0
            else np.nan
        ),
        "selected_minus_all_auprc": (
            selected_metrics["auprc"] - all_metrics["auprc"]
        ),
        "selected_minus_random_auprc": selected_metrics["auprc"] - random_auprc,
        "selected_minus_no_skill_auprc": (
            selected_metrics["auprc"] - no_skill_auprc
        ),
        "binary_selected_minus_no_skill_auprc": (
            binary_selected_metrics["auprc"] - no_skill_auprc
        ),
    }
    predictions = pd.concat(
        [
            selected_predictions.assign(model="hazard_selected_k"),
            all_predictions.assign(model="hazard_all_channels"),
            binary_selected_predictions.assign(model="binary_selected_k"),
            binary_all_predictions.assign(model="binary_all_channels"),
        ],
        ignore_index=True,
    )
    trace.insert(0, "patient_id", patient_id)
    if not random_frame.empty:
        random_frame.insert(0, "patient_id", patient_id)
        trace = pd.concat(
            [
                trace,
                random_frame.rename(
                    columns={
                        "repeat": "random_repeat",
                        "auprc": "random_test_auprc",
                        "auroc": "random_test_auroc",
                    }
                ),
            ],
            ignore_index=True,
            sort=False,
        )
    return summary, trace, predictions


def run_analysis(
    manifest: pd.DataFrame,
    paths: dict[str, Path],
    config: PersonalizedConfig,
    forecast_config: rsf.ForecastConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all requested patients and save readable result tables."""

    config.validate()
    patients = sorted(manifest["patient_id"].astype(str).unique())
    if config.patient_ids is not None:
        requested = set(config.patient_ids)
        patients = [patient for patient in patients if patient in requested]
    if config.max_patients is not None:
        patients = patients[: config.max_patients]
    summaries: list[dict[str, Any]] = []
    traces: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    for patient_id in patients:
        patient_manifest = manifest.loc[
            manifest["patient_id"].astype(str).eq(patient_id)
        ].copy()
        n_seizures = patient_manifest.loc[
            patient_manifest["episode_type"].eq("preictal"), "source_event_id"
        ].nunique()
        if n_seizures < 2:
            summaries.append(
                {
                    "patient_id": patient_id,
                    "status": "excluded",
                    "exclusion_reason": "fewer than two usable seizures",
                    "n_seizures": int(n_seizures),
                    "requested_k": config.k,
                }
            )
            continue
        try:
            summary, trace, patient_predictions = analyze_patient(
                patient_manifest, paths, config, forecast_config
            )
            summaries.append(summary)
            traces.append(trace)
            predictions.append(patient_predictions)
        except Exception as error:
            summaries.append(
                {
                    "patient_id": patient_id,
                    "status": "excluded",
                    "exclusion_reason": f"{type(error).__name__}: {error}",
                    "n_seizures": int(n_seizures),
                    "requested_k": config.k,
                }
            )
            print(f"{patient_id}: excluded ({type(error).__name__}: {error})")
    summary_frame = pd.DataFrame(summaries)
    trace_frame = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    prediction_frame = (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    )
    output_dir = paths["quick_results"] if config.quick_mode else paths[
        "personalized_results"
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(output_dir / "patient_channel_summary.csv", index=False)
    trace_frame.to_csv(output_dir / "channel_selection_trace.csv", index=False)
    prediction_frame.to_csv(output_dir / "test_landmark_predictions.csv", index=False)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "personalized_config": asdict(config),
                "forecast_config": asdict(forecast_config),
                "module_version": MODULE_VERSION,
                "research_only": True,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return summary_frame, trace_frame, prediction_frame
