"""Split every Siena EDF recording into EEG-channel arrays and overview plots.

The output mirrors the raw-data hierarchy:

    data/raw/splitdata/PN00/PN00-1/
        channel_manifest.csv
        channel_overview.png
        channel_01_Fp1.npy
        ...

The ``.npy`` files contain the original EDF digital samples losslessly.  Scale
and offset values in the manifest convert those samples back to physical units.
Recordings in this cohort contain either 29 or 31 EEG channels; the 8-by-4
overview page leaves unused panels blank rather than inventing missing signals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyedflib


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = ROOT / "data" / "raw"
DEFAULT_SPLIT_ROOT = DEFAULT_RAW_ROOT / "splitdata"
MANIFEST_FIELDS = (
    "channel_number",
    "edf_signal_index",
    "channel_label",
    "canonical_name",
    "filename",
    "sample_rate_hz",
    "sample_count",
    "duration_seconds",
    "digital_minimum",
    "digital_maximum",
    "physical_minimum",
    "physical_maximum",
    "physical_unit",
    "scale_to_physical",
    "offset_to_physical",
)


def is_eeg_label(label: str) -> bool:
    """Match the EEG definition used by the forecasting pipeline."""

    upper = str(label).upper().replace("-", " ").strip()
    excluded = ("EKG", "ECG", "SPO2", "STATUS", "EVENT", "EMG", "RESP")
    return upper.startswith("EEG") and not any(token in upper for token in excluded)


def canonical_channel_name(label: str) -> str:
    """Normalize common Siena label variants without merging electrodes."""

    name = str(label).upper().strip()
    name = re.sub(r"^EEG[\s:_-]*", "", name)
    name = re.sub(r"[\s:_-]*(REF|LE|AVG)$", "", name)
    return re.sub(r"\s+", "", name)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return cleaned or "unnamed"


def _natural_key(value: str | Path) -> tuple[Any, ...]:
    return tuple(
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", str(value))
    )


def split_recording_dir(
    edf_path: str | Path,
    split_root: str | Path | None = None,
) -> Path:
    """Return the mirrored split directory for one ``raw/PNxx/name.edf`` file."""

    edf_path = Path(edf_path)
    destination = (
        Path(split_root)
        if split_root is not None
        else edf_path.parent.parent / "splitdata"
    )
    return destination / edf_path.parent.name / edf_path.stem


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Split EEG manifest not found: {path}. "
            "Run split_eeg_channels.py first."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Split EEG manifest is empty: {path}")
    numeric_float = {
        "sample_rate_hz",
        "duration_seconds",
        "physical_minimum",
        "physical_maximum",
        "scale_to_physical",
        "offset_to_physical",
    }
    numeric_int = {
        "channel_number",
        "edf_signal_index",
        "sample_count",
        "digital_minimum",
        "digital_maximum",
    }
    for row in rows:
        for key in numeric_float:
            row[key] = float(row[key])
        for key in numeric_int:
            row[key] = int(row[key])
    return rows


def split_eeg_metadata(
    edf_path: str | Path,
    split_root: str | Path | None = None,
) -> dict[str, Any]:
    """Read EDF-equivalent metadata from a split recording manifest."""

    directory = split_recording_dir(edf_path, split_root)
    rows = _read_manifest(directory / "channel_manifest.csv")
    return {
        "path": str(Path(edf_path)),
        "split_directory": str(directory),
        "labels": [row["channel_label"] for row in rows],
        "sample_rates": [row["sample_rate_hz"] for row in rows],
        "sample_counts": [row["sample_count"] for row in rows],
        "duration_seconds": min(row["duration_seconds"] for row in rows),
    }


def read_split_eeg_segment(
    edf_path: str | Path,
    start_seconds: float,
    duration_seconds: float,
    min_channels: int = 1,
    split_root: str | Path | None = None,
) -> tuple[np.ndarray, float, list[str]]:
    """Read a physical-unit EEG interval from channel-separated ``.npy`` files."""

    if start_seconds < 0 or duration_seconds <= 0:
        raise ValueError("Requested split EEG segment has invalid bounds.")
    directory = split_recording_dir(edf_path, split_root)
    rows = _read_manifest(directory / "channel_manifest.csv")
    rounded_rates = [round(float(row["sample_rate_hz"]), 6) for row in rows]
    sample_rate = Counter(rounded_rates).most_common(1)[0][0]
    rows = [
        row
        for row, rate in zip(rows, rounded_rates, strict=True)
        if math.isclose(rate, sample_rate, rel_tol=0, abs_tol=1e-6)
    ]
    if len(rows) < min_channels:
        raise ValueError(
            f"{Path(edf_path).name} has only {len(rows)} split EEG channels "
            f"at {sample_rate:g} Hz."
        )
    start_sample = int(round(start_seconds * sample_rate))
    sample_count = int(round(duration_seconds * sample_rate))
    if start_sample + sample_count > min(int(row["sample_count"]) for row in rows):
        raise ValueError(
            f"Requested [{start_seconds}, {start_seconds + duration_seconds}) s "
            f"extends beyond split data for {Path(edf_path).name}."
        )
    arrays = []
    for row in rows:
        digital = np.load(directory / row["filename"], mmap_mode="r")
        segment = np.asarray(
            digital[start_sample : start_sample + sample_count], dtype=np.float64
        )
        if len(segment) != sample_count:
            raise ValueError(f"Incomplete split-data read from {row['filename']}.")
        arrays.append(
            segment * float(row["scale_to_physical"])
            + float(row["offset_to_physical"])
        )
    return np.vstack(arrays), float(sample_rate), [
        str(row["channel_label"]) for row in rows
    ]


def minmax_envelope(
    values: np.ndarray,
    sample_rate: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summarize consecutive samples while preserving every local extreme."""

    block_size = max(1, int(math.ceil(values.size / max_points)))
    complete = values.size // block_size
    if complete:
        blocks = values[: complete * block_size].reshape(complete, block_size)
        lower = blocks.min(axis=1)
        upper = blocks.max(axis=1)
    else:
        lower = upper = np.asarray(values)
    if complete * block_size < values.size:
        tail = values[complete * block_size :]
        lower = np.append(lower, tail.min())
        upper = np.append(upper, tail.max())
    midpoint = (
        np.arange(len(lower), dtype=float) * block_size
        + (block_size - 1) / 2
    )
    return midpoint / sample_rate / 60.0, lower, upper


def split_recording(
    edf_path: Path,
    split_root: Path,
    *,
    force: bool = False,
) -> tuple[Path, list[dict[str, Any]]]:
    """Losslessly export every recognized EEG signal from one EDF."""

    output_dir = split_recording_dir(edf_path, split_root)
    manifest_path = output_dir / "channel_manifest.csv"
    if manifest_path.exists() and not force:
        rows = _read_manifest(manifest_path)
        if all((output_dir / row["filename"]).exists() for row in rows):
            return output_dir, rows

    output_dir.mkdir(parents=True, exist_ok=True)
    reader = pyedflib.EdfReader(str(edf_path))
    rows: list[dict[str, Any]] = []
    try:
        labels = [str(label).strip() for label in reader.getSignalLabels()]
        indices = [i for i, label in enumerate(labels) if is_eeg_label(label)]
        if not indices:
            raise ValueError(f"No EEG channels recognized in {edf_path}.")
        for channel_number, signal_index in enumerate(indices, start=1):
            label = labels[signal_index]
            canonical = canonical_channel_name(label)
            filename = (
                f"channel_{channel_number:02d}_{_safe_name(canonical)}.npy"
            )
            digital = np.asarray(
                reader.readSignal(signal_index, digital=True), dtype=np.int32
            )
            digital_minimum = int(reader.getDigitalMinimum(signal_index))
            digital_maximum = int(reader.getDigitalMaximum(signal_index))
            if (
                digital_minimum < np.iinfo(np.int16).min
                or digital_maximum > np.iinfo(np.int16).max
            ):
                storage = digital
            else:
                storage = digital.astype("<i2", copy=False)
            temporary = output_dir / f".{filename}.part"
            with temporary.open("wb") as handle:
                np.save(handle, storage, allow_pickle=False)
            temporary.replace(output_dir / filename)
            physical_minimum = float(reader.getPhysicalMinimum(signal_index))
            physical_maximum = float(reader.getPhysicalMaximum(signal_index))
            scale = (physical_maximum - physical_minimum) / (
                digital_maximum - digital_minimum
            )
            offset = physical_minimum - digital_minimum * scale
            sample_rate = float(reader.getSampleFrequency(signal_index))
            rows.append(
                {
                    "channel_number": channel_number,
                    "edf_signal_index": signal_index,
                    "channel_label": label,
                    "canonical_name": canonical,
                    "filename": filename,
                    "sample_rate_hz": sample_rate,
                    "sample_count": len(storage),
                    "duration_seconds": len(storage) / sample_rate,
                    "digital_minimum": digital_minimum,
                    "digital_maximum": digital_maximum,
                    "physical_minimum": physical_minimum,
                    "physical_maximum": physical_maximum,
                    "physical_unit": str(reader.getPhysicalDimension(signal_index)),
                    "scale_to_physical": scale,
                    "offset_to_physical": offset,
                }
            )
    finally:
        reader.close()
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "source_edf": str(edf_path.resolve()),
        "source_size_bytes": edf_path.stat().st_size,
        "source_mtime_ns": edf_path.stat().st_mtime_ns,
        "eeg_channel_count": len(rows),
        "storage": "lossless EDF digital samples in NumPy .npy format",
    }
    (output_dir / "source_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return output_dir, rows


def plot_recording(
    edf_path: Path,
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    max_points: int = 3_000,
) -> Path:
    """Save one 8-by-4 page containing every available EEG channel."""

    figure, axes = plt.subplots(8, 4, figsize=(19, 22), sharex=True)
    axes = axes.ravel()
    for axis, row in zip(axes, rows, strict=False):
        digital = np.load(output_dir / row["filename"], mmap_mode="r")
        minutes, digital_low, digital_high = minmax_envelope(
            digital, float(row["sample_rate_hz"]), max_points
        )
        scale = float(row["scale_to_physical"])
        offset = float(row["offset_to_physical"])
        low = digital_low.astype(np.float64) * scale + offset
        high = digital_high.astype(np.float64) * scale + offset
        axis.fill_between(
            minutes, low, high, color="#2a6fbb", alpha=0.55, linewidth=0
        )
        axis.plot(
            minutes, (low + high) / 2, color="#123a63", linewidth=0.35
        )
        axis.set_title(
            f"{int(row['channel_number']):02d}. {row['channel_label']}",
            fontsize=9,
            loc="left",
            pad=3,
        )
        axis.set_ylabel(str(row["physical_unit"]), fontsize=7)
        axis.tick_params(axis="both", labelsize=7, length=2)
        axis.grid(axis="x", color="0.90", linewidth=0.5)
    for axis in axes[len(rows) :]:
        axis.set_visible(False)
    for axis in axes:
        if axis.get_visible():
            axis.set_xlabel("Minutes from recording start", fontsize=8)
    figure.suptitle(
        f"{edf_path.stem}: all {len(rows)} available EEG channels\n"
        "Consecutive min/max envelope of the complete recording",
        fontsize=15,
        y=0.995,
    )
    output = output_dir / "channel_overview.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def _selected_edfs(
    raw_root: Path,
    patients: Iterable[str] | None,
    recordings: Iterable[str] | None,
) -> list[Path]:
    edfs = sorted(raw_root.glob("PN*/*.edf"), key=_natural_key)
    patient_set = set(patients or ())
    recording_set = {
        Path(value).stem for value in (recordings or ())
    }
    if patient_set:
        edfs = [path for path in edfs if path.parent.name in patient_set]
    if recording_set:
        edfs = [path for path in edfs if path.stem in recording_set]
    return edfs


def export_dataset(
    raw_root: Path = DEFAULT_RAW_ROOT,
    split_root: Path = DEFAULT_SPLIT_ROOT,
    *,
    patients: Iterable[str] | None = None,
    recordings: Iterable[str] | None = None,
    force_split: bool = False,
    force_plots: bool = False,
    plots_only: bool = False,
    max_points: int = 3_000,
) -> list[dict[str, Any]]:
    """Export selected recordings and write a root-level inventory."""

    edfs = _selected_edfs(raw_root, patients, recordings)
    if not edfs:
        raise FileNotFoundError(f"No matching EDF files found below {raw_root}.")
    inventory = []
    for number, edf_path in enumerate(edfs, start=1):
        output_dir = split_recording_dir(edf_path, split_root)
        if plots_only:
            rows = _read_manifest(output_dir / "channel_manifest.csv")
        else:
            output_dir, rows = split_recording(
                edf_path, split_root, force=force_split
            )
        overview = output_dir / "channel_overview.png"
        if force_plots or not overview.exists():
            plot_recording(
                edf_path, output_dir, rows, max_points=max_points
            )
        print(
            f"[{number:02d}/{len(edfs):02d}] {edf_path.parent.name}/"
            f"{edf_path.name}: {len(rows)} EEG channels",
            flush=True,
        )
        inventory.append(
            {
                "patient_id": edf_path.parent.name,
                "recording": edf_path.stem,
                "source_edf": str(edf_path.relative_to(raw_root)),
                "channel_count": len(rows),
                "manifest": str(
                    (output_dir / "channel_manifest.csv").relative_to(split_root)
                ),
                "overview": str(overview.relative_to(split_root)),
            }
        )
    split_root.mkdir(parents=True, exist_ok=True)
    inventory_path = split_root / "split_inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=inventory[0].keys())
        writer.writeheader()
        writer.writerows(inventory)
    (split_root / "README.md").write_text(
        "# Channel-separated Siena EEG data\n\n"
        "Each patient/recording directory mirrors `data/raw/PNxx/*.edf` and "
        "contains one lossless digital `.npy` array per available EEG channel, "
        "a calibration manifest, source metadata, and a one-page overview. "
        "The source recordings contain 29 or 31 EEG channels; blank panels "
        "represent channels that do not exist in that recording.\n\n"
        "Regenerate missing outputs:\n\n"
        "```bash\n"
        "python final_project/scripts/split_eeg_channels.py\n"
        "```\n\n"
        "Regenerate every graph without deleting existing files:\n\n"
        "```bash\n"
        "python final_project/scripts/split_eeg_channels.py "
        "--plots-only --force-plots\n"
        "```\n",
        encoding="utf-8",
    )
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--patient", action="append")
    parser.add_argument("--recording", action="append")
    parser.add_argument("--force-split", action="store_true")
    parser.add_argument("--force-plots", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--max-points-per-channel", type=int, default=3_000)
    args = parser.parse_args()
    if args.max_points_per_channel < 2:
        parser.error("--max-points-per-channel must be at least 2.")
    export_dataset(
        args.raw_root,
        args.split_root,
        patients=args.patient,
        recordings=args.recording,
        force_split=args.force_split,
        force_plots=args.force_plots,
        plots_only=args.plots_only,
        max_points=args.max_points_per_channel,
    )


if __name__ == "__main__":
    main()
