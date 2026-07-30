"""Build the compact per-channel feature caches used by the final analysis.

Each episode contributes sixty five-second landmarks.  Every landmark uses
only its preceding 120 seconds of EEG.  The cache contains exactly the seven
per-channel features and three temporal aggregations consumed by
``src.reduced_sensor_pipeline``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import signal

from src import eeg_io


MODULE_VERSION = "compact-channel-features-1.0.0"
ROLLING_MODULE_VERSION = "not-used-compact-cache-1.0.0"

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
FEATURE_NAMES = (
    *(f"relative_{band}" for band in BANDS),
    "log_rms",
    "log_line_length",
    "usable_channel_fraction",
)
AGGREGATIONS = ("mean", "last", "slope")
MANIFEST_COLUMNS = (
    "episode_id",
    "patient_id",
    "source_event_id",
    "episode_type",
    "recording",
    "edf_path",
    "anchor_seconds",
    "event_onset_seconds",
)
METADATA_COLUMNS = (
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


@dataclass(frozen=True)
class FeatureConfig:
    """Scientifically relevant settings for compact feature extraction."""

    bin_seconds: int = 5
    horizon_seconds: int = 300
    context_seconds: int = 120
    target_sample_rate: int = 64
    total_power_low_hz: float = 0.5
    total_power_high_hz: float = 32.0
    flat_mad_uv: float = 0.05
    extreme_mad_uv: float = 250.0
    extreme_peak_uv: float = 1500.0

    @property
    def n_landmarks(self) -> int:
        return self.horizon_seconds // self.bin_seconds

    @property
    def context_windows(self) -> int:
        return self.context_seconds // self.bin_seconds

    def validate(self) -> None:
        if self.bin_seconds <= 0:
            raise ValueError("bin_seconds must be positive.")
        if self.horizon_seconds % self.bin_seconds:
            raise ValueError("horizon_seconds must be divisible by bin_seconds.")
        if self.context_seconds % self.bin_seconds:
            raise ValueError("context_seconds must be divisible by bin_seconds.")
        if self.context_windows < 2:
            raise ValueError("At least two context windows are required.")
        highest_modeled_hz = max(high for _, high in BANDS.values())
        if self.target_sample_rate <= 2 * highest_modeled_hz:
            raise ValueError(
                "target_sample_rate must exceed twice the highest modeled "
                "frequency."
            )
        if not (
            0 < self.total_power_low_hz < highest_modeled_hz
            <= self.total_power_high_hz
            <= self.target_sample_rate / 2
        ):
            raise ValueError(
                "Total-power bounds must cover every modeled band without "
                "exceeding the target Nyquist frequency."
            )
        if not (
            0
            < self.flat_mad_uv
            < self.extreme_mad_uv
            < self.extreme_peak_uv
        ):
            raise ValueError("Artifact thresholds are not ordered correctly.")


@dataclass
class PatientFeatureData:
    """One patient's cached landmark table and channel schema."""

    patient_id: str
    frame: pd.DataFrame
    channel_names: list[str]
    channel_feature_columns: dict[str, list[str]]


def channel_columns(channel: str) -> list[str]:
    """Return the exact 21-column schema consumed for one EEG channel."""

    return [
        f"ch::{channel}::{feature}::{aggregation}"
        for feature in FEATURE_NAMES
        for aggregation in AGGREGATIONS
    ]


def channel_column_map(
    channel_names: Sequence[str],
) -> dict[str, list[str]]:
    return {channel: channel_columns(channel) for channel in channel_names}


def quality_mask(
    detrended_window: np.ndarray,
    config: FeatureConfig = FeatureConfig(),
) -> np.ndarray:
    """Return usable channels for one five-second microvolt window."""

    if detrended_window.ndim != 2:
        raise ValueError("EEG window must have shape channels x samples.")
    finite = np.isfinite(detrended_window).all(axis=1)
    centered = detrended_window - np.nanmedian(
        detrended_window, axis=1, keepdims=True
    )
    mad = 1.4826 * np.nanmedian(np.abs(centered), axis=1)
    peak = np.nanmax(np.abs(centered), axis=1)
    return (
        finite
        & (mad >= config.flat_mad_uv)
        & (mad <= config.extreme_mad_uv)
        & (peak <= config.extreme_peak_uv)
    )


def per_channel_micro_features(
    window: np.ndarray,
    sample_rate: float,
    config: FeatureConfig = FeatureConfig(),
) -> np.ndarray:
    """Compute seven compact features independently for every channel."""

    config.validate()
    if window.ndim != 2:
        raise ValueError("EEG window must have shape channels x samples.")
    if window.shape[1] < 3:
        raise ValueError("EEG window is too short.")
    finite_input = np.isfinite(window).all(axis=1)
    safe_window = np.asarray(window, dtype=float).copy()
    safe_window[~finite_input] = 0.0
    detrended = signal.detrend(safe_window, axis=1, type="linear")
    usable = quality_mask(detrended, config) & finite_input
    frequencies, psd = signal.periodogram(
        detrended,
        fs=sample_rate,
        window="hann",
        detrend=False,
        scaling="density",
        axis=1,
    )
    analysis = (
        (frequencies >= config.total_power_low_hz)
        & (frequencies <= config.total_power_high_hz)
    )
    if int(analysis.sum()) < 2:
        raise ValueError("No usable total-power frequency range.")
    total_power = np.trapezoid(
        psd[:, analysis], frequencies[analysis], axis=1
    )
    total_power = np.clip(total_power, 1e-12, None)

    columns: list[np.ndarray] = []
    for low, high in BANDS.values():
        mask = (frequencies >= low) & (frequencies < high)
        if int(mask.sum()) < 2:
            raise ValueError(f"No usable spectral bins for {low:g}-{high:g} Hz.")
        band_power = np.trapezoid(
            psd[:, mask], frequencies[mask], axis=1
        )
        columns.append(np.clip(band_power, 1e-12, None) / total_power)
    rms = np.sqrt(np.mean(np.square(detrended), axis=1))
    line_length = np.mean(np.abs(np.diff(detrended, axis=1)), axis=1)
    columns.extend(
        [
            np.log1p(rms),
            np.log1p(line_length),
            usable.astype(float),
        ]
    )
    result = np.column_stack(columns).astype(float)
    result[~usable, :-1] = np.nan
    if result.shape[1] != len(FEATURE_NAMES):
        raise AssertionError("Unexpected per-channel feature width.")
    return result


def segment_channel_features(
    data: np.ndarray,
    source_sample_rate: float,
    config: FeatureConfig = FeatureConfig(),
) -> np.ndarray:
    """Return ``micro-windows x channels x seven features``."""

    config.validate()
    if data.ndim != 2:
        raise ValueError("EEG data must have shape channels x samples.")
    rounded_source_rate = int(round(source_sample_rate))
    if not math.isclose(
        source_sample_rate,
        rounded_source_rate,
        rel_tol=0,
        abs_tol=1e-6,
    ):
        raise ValueError("Non-integer source sample rates are unsupported.")
    if rounded_source_rate < config.target_sample_rate:
        raise ValueError("Source sample rate is below target_sample_rate.")
    divisor = math.gcd(
        rounded_source_rate, int(config.target_sample_rate)
    )
    up = config.target_sample_rate // divisor
    down = rounded_source_rate // divisor
    if up != down:
        data = signal.resample_poly(data, up=up, down=down, axis=1)
    samples_per_bin = int(
        round(config.bin_seconds * config.target_sample_rate)
    )
    n_windows = data.shape[1] // samples_per_bin
    if n_windows < 1:
        raise ValueError("EEG segment is shorter than one feature window.")
    data = data[:, : n_windows * samples_per_bin]
    return np.stack(
        [
            per_channel_micro_features(
                data[
                    :,
                    index * samples_per_bin : (index + 1) * samples_per_bin,
                ],
                float(config.target_sample_rate),
                config,
            )
            for index in range(n_windows)
        ]
    )


def aggregate_channel_context(
    micro: np.ndarray,
    *,
    expected_windows: int,
    bin_seconds: int,
) -> np.ndarray:
    """Aggregate one causal context as mean, latest value, and slope."""

    if micro.ndim != 3:
        raise ValueError(
            "Micro-features must have shape windows x channels x features."
        )
    if micro.shape[0] != expected_windows:
        raise ValueError(
            f"Expected {expected_windows} context windows, "
            f"found {micro.shape[0]}."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        means = np.nanmean(micro, axis=0)
        latest = micro[-1]
        times = np.arange(expected_windows, dtype=float) * bin_seconds
        centered = times - times.mean()
        slopes = np.nansum(
            centered[:, None, None] * (micro - means[None, :, :]),
            axis=0,
        ) / np.square(centered).sum()
    return np.stack([means, latest, slopes], axis=-1).reshape(
        micro.shape[1], -1
    )


def episode_channel_landmarks(
    episode: pd.Series,
    channel_names: Sequence[str],
    config: FeatureConfig = FeatureConfig(),
    *,
    use_split_cache: bool = False,
    split_root: str | Path | None = None,
) -> pd.DataFrame:
    """Convert one manifest episode into sixty causal landmark rows."""

    config.validate()
    read_start = float(episode["anchor_seconds"]) - config.context_seconds
    read_duration = config.context_seconds + config.horizon_seconds
    data, sample_rate, labels = eeg_io.read_eeg_segment(
        episode["edf_path"],
        read_start,
        read_duration,
        channels=channel_names,
        use_split_cache=use_split_cache,
        split_root=split_root,
    )
    if list(labels) != list(channel_names):
        raise ValueError("EEG reader did not preserve requested channel order.")
    micro = segment_channel_features(data, sample_rate, config)
    expected_micro_windows = (
        config.context_seconds + config.horizon_seconds
    ) // config.bin_seconds
    if len(micro) != expected_micro_windows:
        raise ValueError(
            f"Expected {expected_micro_windows} micro-windows, "
            f"found {len(micro)}."
        )
    feature_columns = [
        column
        for channel in channel_names
        for column in channel_columns(channel)
    ]
    is_event = str(episode["episode_type"]) == "preictal"
    rows: list[dict[str, Any]] = []
    for step in range(config.n_landmarks):
        features = aggregate_channel_context(
            micro[step : step + config.context_windows],
            expected_windows=config.context_windows,
            bin_seconds=config.bin_seconds,
        ).reshape(-1)
        landmark = float(episode["anchor_seconds"]) + (
            step * config.bin_seconds
        )
        if is_event:
            time_to_event = (
                float(episode["event_onset_seconds"]) - landmark
            )
            event_bin = int(
                math.ceil(time_to_event / config.bin_seconds) - 1
            )
        else:
            time_to_event = np.nan
            event_bin = -1
        row: dict[str, Any] = {
            "episode_id": str(episode["episode_id"]),
            "patient_id": str(episode["patient_id"]),
            "source_event_id": str(episode["source_event_id"]),
            "episode_type": str(episode["episode_type"]),
            "recording": str(episode["recording"]),
            "landmark_step": step,
            "landmark_seconds": landmark,
            "time_to_event_seconds": time_to_event,
            "event_bin": event_bin,
            "has_event_in_5m": int(is_event),
        }
        row.update(dict(zip(feature_columns, features, strict=True)))
        rows.append(row)
    return pd.DataFrame(rows)


def _natural_recording_key(value: str) -> tuple[object, ...]:
    return eeg_io.natural_key(value)


def _is_blank_source_event(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "<na>"}


def assign_control_source_events(manifest: pd.DataFrame) -> pd.DataFrame:
    """Give every control a deterministic seizure group when it lacks one.

    This preserves the historical grouping needed for compatibility.  The
    final analysis subsequently merges raw-time-overlapping control groups.
    """

    result = manifest.copy()
    for patient_id, indices in result.groupby(
        "patient_id", sort=False
    ).groups.items():
        group = result.loc[indices]
        positive = group.loc[group["episode_type"].eq("preictal")].copy()
        positive["_recording_key"] = positive["recording"].map(
            _natural_recording_key
        )
        positive = positive.sort_values(
            ["_recording_key", "event_onset_seconds", "source_event_id"]
        )
        event_ids = (
            positive["source_event_id"].astype(str).drop_duplicates().tolist()
        )
        controls = group.loc[group["episode_type"].eq("interictal")]
        if len(controls) and not event_ids:
            raise ValueError(
                f"{patient_id} has controls but no preictal seizure."
            )
        unknown = sorted(
            {
                str(value)
                for value in controls["source_event_id"]
                if not _is_blank_source_event(value)
                and str(value) not in event_ids
            }
        )
        if unknown:
            raise ValueError(
                f"{patient_id} controls reference unknown seizures {unknown}."
            )
        blank_indices = [
            index
            for index, value in controls["source_event_id"].items()
            if _is_blank_source_event(value)
        ]
        if not blank_indices:
            continue
        ordered_blank_indices = (
            controls.loc[blank_indices].sort_values("episode_id").index.tolist()
        )
        if len(blank_indices) == len(controls):
            controls_per_event, remainder = divmod(
                len(ordered_blank_indices), len(event_ids)
            )
            assignments = [
                event_id
                for event_number, event_id in enumerate(event_ids)
                for _ in range(
                    controls_per_event + int(event_number < remainder)
                )
            ]
            result.loc[
                ordered_blank_indices, "source_event_id"
            ] = assignments
            continue
        existing_counts = {
            event_id: int(
                controls["source_event_id"].astype(str).eq(event_id).sum()
            )
            for event_id in event_ids
        }
        assignments: list[str] = []
        for _ in ordered_blank_indices:
            selected = min(
                event_ids,
                key=lambda event_id: (
                    existing_counts[event_id],
                    event_ids.index(event_id),
                ),
            )
            assignments.append(selected)
            existing_counts[selected] += 1
        result.loc[ordered_blank_indices, "source_event_id"] = assignments
    return result


def _resolve_manifest_edf(
    row: pd.Series,
    *,
    project_root: Path,
) -> Path:
    supplied = Path(str(row["edf_path"]))
    candidates = []
    if supplied.is_absolute():
        candidates.append(supplied)
    else:
        candidates.append(project_root / supplied)
    candidates.append(
        project_root
        / "data"
        / "raw"
        / str(row["patient_id"])
        / str(row["recording"])
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve EDF for {row['patient_id']}/"
        f"{row['recording']} from {row['edf_path']!r}."
    )


def load_episode_manifest(
    manifest_path: str | Path,
    *,
    project_root: str | Path,
) -> pd.DataFrame:
    """Load, validate, rebase, and group the canonical episode manifest."""

    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Canonical episode manifest not found: {path}."
        )
    manifest = pd.read_csv(path)
    missing = sorted(set(MANIFEST_COLUMNS) - set(manifest.columns))
    if missing:
        raise ValueError(f"Episode manifest is missing columns {missing}.")
    manifest = manifest.copy()
    for column in (
        "episode_id",
        "patient_id",
        "episode_type",
        "recording",
    ):
        if manifest[column].isna().any():
            raise ValueError(f"Manifest column {column} contains missing values.")
        manifest[column] = manifest[column].astype(str)
    allowed_types = {"preictal", "interictal"}
    unexpected = sorted(set(manifest["episode_type"]) - allowed_types)
    if unexpected:
        raise ValueError(f"Unexpected episode types: {unexpected}.")
    if manifest["episode_id"].duplicated().any():
        duplicates = manifest.loc[
            manifest["episode_id"].duplicated(keep=False), "episode_id"
        ].tolist()
        raise ValueError(f"Duplicate episode IDs: {duplicates}.")
    manifest["anchor_seconds"] = pd.to_numeric(
        manifest["anchor_seconds"], errors="raise"
    )
    manifest["event_onset_seconds"] = pd.to_numeric(
        manifest["event_onset_seconds"], errors="coerce"
    )
    if manifest["anchor_seconds"].isna().any():
        raise ValueError("Manifest anchor_seconds contains missing values.")
    preictal = manifest["episode_type"].eq("preictal")
    if manifest.loc[preictal, "event_onset_seconds"].isna().any():
        raise ValueError("A preictal episode lacks event_onset_seconds.")
    root = Path(project_root).resolve()
    manifest["edf_path"] = [
        str(_resolve_manifest_edf(row, project_root=root))
        for _, row in manifest.iterrows()
    ]
    return assign_control_source_events(manifest).reset_index(drop=True)


def common_patient_channels(
    patient_manifest: pd.DataFrame,
    *,
    use_split_cache: bool = False,
    split_root: str | Path | None = None,
) -> list[str]:
    """Return channels present in every recording used by one patient."""

    channel_sets = [
        set(
            eeg_io.available_eeg_channels(
                edf_path,
                use_split_cache=use_split_cache,
                split_root=split_root,
            )
        )
        for edf_path in patient_manifest["edf_path"].drop_duplicates()
    ]
    if not channel_sets:
        return []
    return sorted(
        set.intersection(*channel_sets), key=eeg_io.natural_key
    )


def _json_ready_manifest(
    patient_manifest: pd.DataFrame,
    *,
    project_root: Path,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for _, row in patient_manifest[list(MANIFEST_COLUMNS)].iterrows():
        edf_path = Path(str(row["edf_path"])).resolve()
        try:
            logical_path = edf_path.relative_to(project_root).as_posix()
        except ValueError:
            logical_path = f"{edf_path.parent.name}/{edf_path.name}"
        record: dict[str, object] = {}
        for column in MANIFEST_COLUMNS:
            value = row[column]
            if column == "edf_path":
                record[column] = logical_path
            elif pd.isna(value):
                record[column] = None
            elif isinstance(value, (np.integer, np.floating)):
                record[column] = value.item()
            else:
                record[column] = value
        records.append(record)
    return records


def _stable_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _input_signature(
    patient_manifest: pd.DataFrame,
    channel_names: Sequence[str],
    config: FeatureConfig,
    *,
    project_root: Path,
    use_split_cache: bool,
    split_root: str | Path | None,
    source_hash_memo: dict[Path, str] | None,
) -> dict[str, object]:
    manifest_records = _json_ready_manifest(
        patient_manifest, project_root=project_root
    )
    source_records = eeg_io.source_content_records(
        patient_manifest["edf_path"].drop_duplicates(),
        project_root=project_root,
        channels=channel_names,
        use_split_cache=use_split_cache,
        split_root=split_root,
        memo=source_hash_memo,
    )
    return {
        "module_version": MODULE_VERSION,
        "rolling_module_version": ROLLING_MODULE_VERSION,
        "signal_source": (
            "optional_split_eeg_cache" if use_split_cache else "raw_edf"
        ),
        "forecast_config": asdict(config),
        "feature_names": list(FEATURE_NAMES),
        "aggregations": list(AGGREGATIONS),
        "features_per_channel": len(FEATURE_NAMES) * len(AGGREGATIONS),
        "patient_id": str(patient_manifest["patient_id"].iloc[0]),
        "channel_names": list(channel_names),
        "manifest_sha256": hashlib.sha256(
            _stable_json_bytes(manifest_records)
        ).hexdigest(),
        "manifest_records": manifest_records,
        "source_files": source_records,
    }


def _signature_digest(signature: dict[str, object]) -> str:
    return hashlib.sha256(_stable_json_bytes(signature)).hexdigest()


def _valid_cached_payload(
    cache_path: Path,
    signature_path: Path,
    expected_signature: dict[str, object],
) -> dict[str, object] | None:
    if not cache_path.exists() or not signature_path.exists():
        return None
    try:
        stored = json.loads(signature_path.read_text(encoding="utf-8"))
        if stored.get("input_signature_sha256") != _signature_digest(
            expected_signature
        ):
            return None
        if stored.get("cache_payload_sha256") != eeg_io.sha256_file(
            cache_path
        ):
            return None
        payload = joblib.load(cache_path)
        required = {"frame", "channel_names", "channel_feature_columns"}
        if not isinstance(payload, dict) or not required.issubset(payload):
            return None
        return payload
    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ):
        return None


def build_patient_feature_data(
    patient_manifest: pd.DataFrame,
    cache_dir: str | Path,
    config: FeatureConfig = FeatureConfig(),
    *,
    project_root: str | Path,
    force: bool = False,
    verbose: bool = True,
    use_split_cache: bool = False,
    split_root: str | Path | None = None,
    source_hash_memo: dict[Path, str] | None = None,
) -> PatientFeatureData:
    """Build or content-validate one patient's compact feature cache."""

    config.validate()
    patient_ids = patient_manifest["patient_id"].astype(str).unique()
    if len(patient_ids) != 1:
        raise ValueError("patient_manifest must contain exactly one patient.")
    patient_id = str(patient_ids[0])
    channels = common_patient_channels(
        patient_manifest,
        use_split_cache=use_split_cache,
        split_root=split_root,
    )
    if not channels:
        raise ValueError(f"No common EEG channels found for {patient_id}.")
    project = Path(project_root).resolve()
    expected_signature = _input_signature(
        patient_manifest,
        channels,
        config,
        project_root=project,
        use_split_cache=use_split_cache,
        split_root=split_root,
        source_hash_memo=source_hash_memo,
    )
    output_dir = Path(cache_dir)
    cache_path = output_dir / f"{patient_id}_channel_landmarks.joblib"
    signature_path = output_dir / f"{patient_id}_channel_landmarks.json"
    if not force:
        cached = _valid_cached_payload(
            cache_path, signature_path, expected_signature
        )
        if cached is not None:
            if verbose:
                print(f"{patient_id}: loaded content-validated cache.")
            return PatientFeatureData(
                patient_id=patient_id,
                frame=cached["frame"],
                channel_names=list(cached["channel_names"]),
                channel_feature_columns=dict(
                    cached["channel_feature_columns"]
                ),
            )

    frames = []
    for number, (_, episode) in enumerate(
        patient_manifest.iterrows(), start=1
    ):
        if verbose:
            print(
                f"{patient_id} [{number}/{len(patient_manifest)}] "
                f"{episode['episode_id']}",
                flush=True,
            )
        frames.append(
            episode_channel_landmarks(
                episode,
                channels,
                config,
                use_split_cache=use_split_cache,
                split_root=split_root,
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    column_map = channel_column_map(channels)
    expected_columns = [
        *METADATA_COLUMNS,
        *[
            column
            for channel in channels
            for column in column_map[channel]
        ],
    ]
    if list(frame.columns) != expected_columns:
        raise AssertionError("Compact cache columns do not match the schema.")
    payload = {
        "frame": frame,
        "channel_names": channels,
        "channel_feature_columns": column_map,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_cache = cache_path.with_name(cache_path.name + ".part")
    temporary_signature = signature_path.with_name(
        signature_path.name + ".part"
    )
    joblib.dump(payload, temporary_cache, compress=3)
    payload_sha256 = eeg_io.sha256_file(temporary_cache)
    completed_signature = {
        **expected_signature,
        "input_signature_sha256": _signature_digest(expected_signature),
        "cache_payload_sha256": payload_sha256,
    }
    temporary_cache.replace(cache_path)
    temporary_signature.write_text(
        json.dumps(completed_signature, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_signature.replace(signature_path)
    return PatientFeatureData(patient_id, frame, channels, column_map)


def build_feature_caches(
    manifest: pd.DataFrame,
    cache_dir: str | Path,
    *,
    project_root: str | Path,
    patient_ids: Iterable[str] | None = None,
    config: FeatureConfig = FeatureConfig(),
    force: bool = False,
    verbose: bool = True,
    use_split_cache: bool = False,
    split_root: str | Path | None = None,
) -> list[PatientFeatureData]:
    """Build compact caches for selected patients in natural order."""

    available = sorted(
        manifest["patient_id"].astype(str).unique(),
        key=eeg_io.natural_key,
    )
    selected = available if patient_ids is None else list(patient_ids)
    missing = sorted(set(selected) - set(available), key=eeg_io.natural_key)
    if missing:
        raise ValueError(f"Patients not present in the manifest: {missing}.")
    digest_memo: dict[Path, str] = {}
    results = []
    for patient_id in selected:
        patient_manifest = manifest.loc[
            manifest["patient_id"].astype(str).eq(patient_id)
        ].copy()
        results.append(
            build_patient_feature_data(
                patient_manifest,
                cache_dir,
                config,
                project_root=project_root,
                force=force,
                verbose=verbose,
                use_split_cache=use_split_cache,
                split_root=split_root,
                source_hash_memo=digest_memo,
            )
        )
    return results
