"""Small, explicit EEG input layer for the reduced-sensor analysis.

Raw EDF files are the canonical signal source.  Channel-separated ``.npy``
files may be used as an optional read cache; they are not a second dataset.
All returned samples are physical microvolts in ``channels x samples`` order.
"""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import math
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np
import pyedflib


EEG_EXCLUDED_LABEL_TOKENS = (
    "EKG",
    "ECG",
    "SPO2",
    "STATUS",
    "EVENT",
    "EMG",
    "RESP",
)
MICROVOLT_UNITS = {"UV", "ΜV", "ΜVOLT", "MICROVOLT", "MICROVOLTS"}


def natural_key(value: str | Path) -> tuple[object, ...]:
    """Return a key that sorts embedded numbers in human order."""

    return tuple(
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", str(value))
    )


def is_eeg_label(label: str) -> bool:
    """Return whether an EDF signal label represents an EEG electrode."""

    upper = str(label).upper().replace("-", " ").strip()
    return upper.startswith("EEG") and not any(
        token in upper for token in EEG_EXCLUDED_LABEL_TOKENS
    )


def canonical_channel_name(label: str) -> str:
    """Normalize Siena labels such as ``EEG Fp2`` to ``FP2``."""

    text = str(label).strip().upper()
    text = re.sub(r"^EEG[\s:_-]*", "", text)
    text = re.sub(r"[\s:_-]*(REF|LE|RE|AVG)$", "", text)
    canonical = re.sub(r"[^A-Z0-9]", "", text)
    if not canonical:
        raise ValueError(f"Cannot canonicalize EEG label {label!r}.")
    return canonical


def _normalized_unit(unit: str) -> str:
    return (
        str(unit)
        .strip()
        .upper()
        .replace("µ", "Μ")
        .replace(" ", "")
        .replace("_", "")
    )


def _require_microvolts(units: Sequence[str], source: Path) -> None:
    unexpected = sorted(
        {
            str(unit)
            for unit in units
            if _normalized_unit(unit) not in MICROVOLT_UNITS
        }
    )
    if unexpected:
        raise ValueError(
            f"{source} uses unsupported EEG units {unexpected}; artifact "
            "thresholds require microvolts."
        )


def _modal_sample_rate(rates: Sequence[float]) -> float:
    if not rates:
        raise ValueError("No EEG sample rates were supplied.")
    rounded = [round(float(rate), 6) for rate in rates]
    return float(Counter(rounded).most_common(1)[0][0])


def _edf_channel_table(edf_path: str | Path) -> list[dict[str, object]]:
    """Read EEG channel metadata without loading signal samples."""

    path = Path(edf_path)
    if not path.exists():
        raise FileNotFoundError(path)
    reader = pyedflib.EdfReader(str(path))
    try:
        labels = [str(label).strip() for label in reader.getSignalLabels()]
        rows: list[dict[str, object]] = []
        seen: set[str] = set()
        for index, label in enumerate(labels):
            if not is_eeg_label(label):
                continue
            canonical = canonical_channel_name(label)
            if canonical in seen:
                raise ValueError(
                    f"{path.name} contains duplicate EEG channel {canonical}."
                )
            seen.add(canonical)
            rows.append(
                {
                    "index": index,
                    "label": label,
                    "channel": canonical,
                    "sample_rate_hz": float(reader.getSampleFrequency(index)),
                    "sample_count": int(reader.getNSamples()[index]),
                    "physical_unit": str(
                        reader.getPhysicalDimension(index)
                    ).strip(),
                }
            )
        if not rows:
            raise ValueError(f"No EEG channels were recognized in {path}.")
        return rows
    finally:
        reader.close()


def edf_eeg_channels(edf_path: str | Path) -> list[str]:
    """Return modal-rate EEG channel names available in an EDF."""

    rows = _edf_channel_table(edf_path)
    modal_rate = _modal_sample_rate(
        [float(row["sample_rate_hz"]) for row in rows]
    )
    selected = [
        row
        for row in rows
        if math.isclose(
            float(row["sample_rate_hz"]),
            modal_rate,
            rel_tol=0,
            abs_tol=1e-6,
        )
    ]
    _require_microvolts(
        [str(row["physical_unit"]) for row in selected], Path(edf_path)
    )
    return [str(row["channel"]) for row in selected]


def read_edf_segment(
    edf_path: str | Path,
    start_seconds: float,
    duration_seconds: float,
    *,
    channels: Sequence[str] | None = None,
) -> tuple[np.ndarray, float, list[str]]:
    """Read a calibrated EEG interval directly from an EDF."""

    path = Path(edf_path)
    if start_seconds < 0 or duration_seconds <= 0:
        raise ValueError("Requested EDF segment has invalid bounds.")
    rows = _edf_channel_table(path)
    by_channel = {str(row["channel"]): row for row in rows}
    if channels is None:
        modal_rate = _modal_sample_rate(
            [float(row["sample_rate_hz"]) for row in rows]
        )
        selected_names = [
            str(row["channel"])
            for row in rows
            if math.isclose(
                float(row["sample_rate_hz"]),
                modal_rate,
                rel_tol=0,
                abs_tol=1e-6,
            )
        ]
    else:
        selected_names = [
            canonical_channel_name(channel) for channel in channels
        ]
        if len(set(selected_names)) != len(selected_names):
            raise ValueError("Requested EEG channels contain duplicates.")
        missing = [
            channel for channel in selected_names if channel not in by_channel
        ]
        if missing:
            raise ValueError(
                f"{path.name} is missing EEG channels {missing}."
            )
    selected = [by_channel[channel] for channel in selected_names]
    rates = [float(row["sample_rate_hz"]) for row in selected]
    sample_rate = rates[0]
    if any(
        not math.isclose(rate, sample_rate, rel_tol=0, abs_tol=1e-6)
        for rate in rates[1:]
    ):
        raise ValueError(
            f"Requested channels in {path.name} do not share a sample rate."
        )
    _require_microvolts(
        [str(row["physical_unit"]) for row in selected], path
    )
    start_sample = int(round(start_seconds * sample_rate))
    sample_count = int(round(duration_seconds * sample_rate))
    if start_sample + sample_count > min(
        int(row["sample_count"]) for row in selected
    ):
        raise ValueError(
            f"Requested [{start_seconds}, "
            f"{start_seconds + duration_seconds}) s extends beyond {path.name}."
        )

    reader = pyedflib.EdfReader(str(path))
    try:
        arrays = [
            np.asarray(
                reader.readSignal(
                    int(row["index"]),
                    start=start_sample,
                    n=sample_count,
                ),
                dtype=np.float64,
            )
            for row in selected
        ]
    finally:
        reader.close()
    if any(len(values) != sample_count for values in arrays):
        raise ValueError(f"Incomplete EDF read from {path.name}.")
    return np.vstack(arrays), sample_rate, selected_names


def split_recording_dir(
    edf_path: str | Path,
    split_root: str | Path | None = None,
) -> Path:
    """Return the split-cache directory corresponding to one EDF."""

    path = Path(edf_path)
    root = (
        Path(split_root)
        if split_root is not None
        else path.parent.parent / "splitdata"
    )
    return root / path.parent.name / path.stem


def _read_split_manifest(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Split EEG manifest not found: {path}. Use raw EDF input or "
            "generate the optional split cache."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError(f"Split EEG manifest is empty: {path}.")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in raw_rows:
        channel = canonical_channel_name(
            raw.get("canonical_name") or raw["channel_label"]
        )
        if channel in seen:
            raise ValueError(
                f"{path} contains duplicate EEG channel {channel}."
            )
        seen.add(channel)
        rows.append(
            {
                "channel": channel,
                "label": raw["channel_label"],
                "filename": raw["filename"],
                "sample_rate_hz": float(raw["sample_rate_hz"]),
                "sample_count": int(raw["sample_count"]),
                "physical_unit": raw["physical_unit"],
                "scale_to_physical": float(raw["scale_to_physical"]),
                "offset_to_physical": float(raw["offset_to_physical"]),
            }
        )
    return rows


def split_eeg_channels(
    edf_path: str | Path,
    *,
    split_root: str | Path | None = None,
) -> list[str]:
    """Return modal-rate channels available in the optional split cache."""

    directory = split_recording_dir(edf_path, split_root)
    rows = _read_split_manifest(directory / "channel_manifest.csv")
    modal_rate = _modal_sample_rate(
        [float(row["sample_rate_hz"]) for row in rows]
    )
    selected = [
        row
        for row in rows
        if math.isclose(
            float(row["sample_rate_hz"]),
            modal_rate,
            rel_tol=0,
            abs_tol=1e-6,
        )
    ]
    _require_microvolts(
        [str(row["physical_unit"]) for row in selected],
        directory / "channel_manifest.csv",
    )
    return [str(row["channel"]) for row in selected]


def read_split_segment(
    edf_path: str | Path,
    start_seconds: float,
    duration_seconds: float,
    *,
    channels: Sequence[str] | None = None,
    split_root: str | Path | None = None,
) -> tuple[np.ndarray, float, list[str]]:
    """Read an EEG interval from the optional channel-separated cache."""

    if start_seconds < 0 or duration_seconds <= 0:
        raise ValueError("Requested split EEG segment has invalid bounds.")
    directory = split_recording_dir(edf_path, split_root)
    rows = _read_split_manifest(directory / "channel_manifest.csv")
    by_channel = {str(row["channel"]): row for row in rows}
    if channels is None:
        sample_rate = _modal_sample_rate(
            [float(row["sample_rate_hz"]) for row in rows]
        )
        selected_names = [
            str(row["channel"])
            for row in rows
            if math.isclose(
                float(row["sample_rate_hz"]),
                sample_rate,
                rel_tol=0,
                abs_tol=1e-6,
            )
        ]
    else:
        selected_names = [
            canonical_channel_name(channel) for channel in channels
        ]
        if len(set(selected_names)) != len(selected_names):
            raise ValueError("Requested EEG channels contain duplicates.")
        missing = [
            channel for channel in selected_names if channel not in by_channel
        ]
        if missing:
            raise ValueError(
                f"Split cache for {Path(edf_path).name} is missing {missing}."
            )
        sample_rate = float(
            by_channel[selected_names[0]]["sample_rate_hz"]
        )
    selected = [by_channel[channel] for channel in selected_names]
    if any(
        not math.isclose(
            float(row["sample_rate_hz"]),
            sample_rate,
            rel_tol=0,
            abs_tol=1e-6,
        )
        for row in selected
    ):
        raise ValueError("Requested split EEG channels have mixed rates.")
    _require_microvolts(
        [str(row["physical_unit"]) for row in selected],
        directory / "channel_manifest.csv",
    )
    start_sample = int(round(start_seconds * sample_rate))
    sample_count = int(round(duration_seconds * sample_rate))
    if start_sample + sample_count > min(
        int(row["sample_count"]) for row in selected
    ):
        raise ValueError(
            f"Requested [{start_seconds}, "
            f"{start_seconds + duration_seconds}) s extends beyond the split "
            f"cache for {Path(edf_path).name}."
        )

    arrays: list[np.ndarray] = []
    for row in selected:
        digital = np.load(
            directory / str(row["filename"]), mmap_mode="r"
        )
        segment = np.asarray(
            digital[start_sample : start_sample + sample_count],
            dtype=np.float64,
        )
        if len(segment) != sample_count:
            raise ValueError(
                f"Incomplete split-data read from {row['filename']}."
            )
        arrays.append(
            segment * float(row["scale_to_physical"])
            + float(row["offset_to_physical"])
        )
    return np.vstack(arrays), sample_rate, selected_names


def available_eeg_channels(
    edf_path: str | Path,
    *,
    use_split_cache: bool = False,
    split_root: str | Path | None = None,
) -> list[str]:
    """Return channels from the selected canonical signal source."""

    if use_split_cache:
        return split_eeg_channels(edf_path, split_root=split_root)
    return edf_eeg_channels(edf_path)


def read_eeg_segment(
    edf_path: str | Path,
    start_seconds: float,
    duration_seconds: float,
    *,
    channels: Sequence[str] | None = None,
    use_split_cache: bool = False,
    split_root: str | Path | None = None,
) -> tuple[np.ndarray, float, list[str]]:
    """Read from raw EDF by default, or from the optional split cache."""

    if use_split_cache:
        return read_split_segment(
            edf_path,
            start_seconds,
            duration_seconds,
            channels=channels,
            split_root=split_root,
        )
    return read_edf_segment(
        edf_path,
        start_seconds,
        duration_seconds,
        channels=channels,
    )


def sha256_file(
    path: str | Path,
    *,
    memo: dict[Path, str] | None = None,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> str:
    """Return a streaming SHA-256 digest, optionally memoized for this run."""

    source = Path(path).resolve()
    if memo is not None and source in memo:
        return memo[source]
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    if memo is not None:
        memo[source] = value
    return value


def source_content_records(
    edf_paths: Iterable[str | Path],
    *,
    project_root: str | Path,
    channels: Sequence[str],
    use_split_cache: bool = False,
    split_root: str | Path | None = None,
    memo: dict[Path, str] | None = None,
) -> list[dict[str, object]]:
    """Describe every signal source with portable names and content hashes."""

    root = Path(project_root).resolve()
    records: list[dict[str, object]] = []
    for raw_path in sorted(
        {Path(path).resolve() for path in edf_paths}, key=natural_key
    ):
        try:
            logical_edf = raw_path.relative_to(root).as_posix()
        except ValueError:
            logical_edf = f"{raw_path.parent.name}/{raw_path.name}"
        if not use_split_cache:
            records.append(
                {
                    "recording": logical_edf,
                    "size_bytes": raw_path.stat().st_size,
                    "sha256": sha256_file(raw_path, memo=memo),
                }
            )
            continue

        directory = split_recording_dir(raw_path, split_root)
        manifest_path = directory / "channel_manifest.csv"
        rows = _read_split_manifest(manifest_path)
        by_channel = {str(row["channel"]): row for row in rows}
        missing = [channel for channel in channels if channel not in by_channel]
        if missing:
            raise ValueError(
                f"Split cache for {raw_path.name} is missing {missing}."
            )
        files = [manifest_path] + [
            directory / str(by_channel[channel]["filename"])
            for channel in channels
        ]
        members = []
        for path in files:
            members.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path, memo=memo),
                }
            )
        combined = hashlib.sha256()
        for member in members:
            combined.update(
                (
                    f"{member['path']}\0{member['size_bytes']}\0"
                    f"{member['sha256']}\n"
                ).encode("utf-8")
            )
        records.append(
            {
                "recording": logical_edf,
                "split_cache_sha256": combined.hexdigest(),
                "members": members,
            }
        )
    return records
