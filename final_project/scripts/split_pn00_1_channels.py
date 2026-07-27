"""Split the 31 requested PN00-1 physiological channels into individual files.

PN00-1 has 35 EDF signals.  This exporter writes the 31 requested channels:
the 29 labels beginning ``EEG``, ``EKG EKG``, and ``SPO2``.  It deliberately
excludes the heart-rate stream and the ``1``, ``2``, and ``MK`` technical or
marker streams.  Every exported ``.npy`` contains the lossless original EDF
digital samples (little-endian signed 16-bit), one channel per file.

The adjacent ``channel_manifest.csv`` contains physical calibration values.
To obtain physical units for an exported array, use

    physical_values = digital_samples * scale_to_physical + offset_to_physical

Run from the repository root:

    .\\.venv\\Scripts\\python.exe final_project/scripts/split_pn00_1_channels.py
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "PN00" / "PN00-1.edf"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "PN00-1_split_channels"
EXPECTED_CHANNEL_COUNT = 31


def _decode(value: bytes) -> str:
    return value.decode("ascii", errors="ignore").strip().replace("\x00", "")


@dataclass(frozen=True)
class EDFHeader:
    header_bytes: int
    n_records: int
    record_seconds: float
    labels: tuple[str, ...]
    physical_dimensions: tuple[str, ...]
    physical_min: np.ndarray
    physical_max: np.ndarray
    digital_min: np.ndarray
    digital_max: np.ndarray
    samples_per_record: np.ndarray

    @property
    def duration_seconds(self) -> float:
        return self.n_records * self.record_seconds


def read_header(path: Path) -> EDFHeader:
    """Read the EDF signal headers necessary for lossless splitting."""
    with path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) != 256:
            raise ValueError(f"Incomplete EDF header: {path}")
        header_bytes = int(_decode(fixed[184:192]))
        n_records = int(_decode(fixed[236:244]))
        record_seconds = float(_decode(fixed[244:252]))
        n_signals = int(_decode(fixed[252:256]))
        labels = tuple(_decode(handle.read(16)) for _ in range(n_signals))
        handle.seek(80 * n_signals, 1)  # transducer type
        dimensions = tuple(_decode(handle.read(8)) for _ in range(n_signals))
        physical_min = np.array(
            [float(_decode(handle.read(8))) for _ in range(n_signals)]
        )
        physical_max = np.array(
            [float(_decode(handle.read(8))) for _ in range(n_signals)]
        )
        digital_min = np.array(
            [float(_decode(handle.read(8))) for _ in range(n_signals)]
        )
        digital_max = np.array(
            [float(_decode(handle.read(8))) for _ in range(n_signals)]
        )
        handle.seek(80 * n_signals, 1)  # prefiltering
        samples_per_record = np.array(
            [int(_decode(handle.read(8))) for _ in range(n_signals)], dtype=int
        )
    return EDFHeader(
        header_bytes=header_bytes,
        n_records=n_records,
        record_seconds=record_seconds,
        labels=labels,
        physical_dimensions=dimensions,
        physical_min=physical_min,
        physical_max=physical_max,
        digital_min=digital_min,
        digital_max=digital_max,
        samples_per_record=samples_per_record,
    )


def _safe_filename(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def select_requested_channels(header: EDFHeader) -> tuple[np.ndarray, np.ndarray]:
    """Return the requested 31 channels and the omitted technical channels."""
    selected = np.array(
        [
            index
            for index, label in enumerate(header.labels)
            if label.upper().startswith("EEG ") or label in {"EKG EKG", "SPO2"}
        ],
        dtype=int,
    )
    omitted = np.array(
        [index for index in range(len(header.labels)) if index not in set(selected)],
        dtype=int,
    )
    if len(selected) != EXPECTED_CHANNEL_COUNT:
        labels = ", ".join(header.labels[index] for index in selected)
        raise ValueError(
            f"Expected {EXPECTED_CHANNEL_COUNT} requested channels, found {len(selected)}: {labels}"
        )
    return selected, omitted


def write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(
        "# PN00-1 split channels\n\n"
        "This folder contains the 31 requested PN00-1 physiological channels: "
        "29 scalp EEG leads, EKG, and SpO2. Each `channel_*.npy` file stores "
        "one lossless EDF digital signal as a one-dimensional little-endian "
        "signed-16-bit NumPy array.\n\n"
        "Use `channel_manifest.csv` for the channel label, sample rate, unit, "
        "and calibration. Convert a digital array to the physical unit with:\n\n"
        "```python\n"
        "physical_values = digital_samples * scale_to_physical + offset_to_physical\n"
        "```\n\n"
        "The source uses 512 Hz sampling; time in seconds is `sample_index / 512`. "
        "The original EDF is unchanged. The omitted EDF signals are HR, 1, 2, "
        "and MK, listed in `omitted_signals.csv`.\n",
        encoding="utf-8",
    )


def export_channels(source: Path, output_dir: Path) -> None:
    header = read_header(source)
    selected, omitted = select_requested_channels(header)
    if not np.all(header.samples_per_record == header.samples_per_record[0]):
        raise ValueError("PN00-1 signals have unequal samples per EDF record")
    samples_per_record = int(header.samples_per_record[0])
    sample_rate = samples_per_record / header.record_seconds
    n_signals = len(header.labels)
    expected_samples = header.n_records * n_signals * samples_per_record
    output_dir.mkdir(parents=True, exist_ok=True)

    # EDF records store all signals consecutively. A memmap avoids duplicating
    # the 94 MB source file in memory while splitting channels one by one.
    raw = np.memmap(
        source,
        dtype="<i2",
        mode="r",
        offset=header.header_bytes,
        shape=(header.n_records, n_signals, samples_per_record),
    )
    if raw.size != expected_samples:
        raise ValueError("EDF payload has an unexpected size")

    manifest_rows: list[dict[str, object]] = []
    for export_index, source_index in enumerate(selected, start=1):
        label = header.labels[source_index]
        filename = f"channel_{export_index:02d}_{_safe_filename(label)}.npy"
        # np.asarray makes an independent contiguous one-channel vector, while
        # preserving every original signed-16-bit EDF sample without rescaling.
        values = np.asarray(raw[:, source_index, :]).reshape(-1).copy()
        np.save(output_dir / filename, values, allow_pickle=False)
        scale = (header.physical_max[source_index] - header.physical_min[source_index]) / (
            header.digital_max[source_index] - header.digital_min[source_index]
        )
        offset = header.physical_min[source_index] - header.digital_min[source_index] * scale
        manifest_rows.append(
            {
                "export_channel": export_index,
                "edf_signal_index": source_index + 1,
                "channel_label": label,
                "is_eeg": label.upper().startswith("EEG "),
                "filename": filename,
                "sample_rate_hz": sample_rate,
                "duration_seconds": header.duration_seconds,
                "sample_count": values.size,
                "stored_dtype": values.dtype.str,
                "physical_unit": header.physical_dimensions[source_index],
                "digital_min": header.digital_min[source_index],
                "digital_max": header.digital_max[source_index],
                "physical_min": header.physical_min[source_index],
                "physical_max": header.physical_max[source_index],
                "scale_to_physical": scale,
                "offset_to_physical": offset,
            }
        )
    with (output_dir / "channel_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (output_dir / "omitted_signals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["edf_signal_index", "channel_label", "physical_unit", "reason"])
        writer.writeheader()
        for source_index in omitted:
            writer.writerow(
                {
                    "edf_signal_index": int(source_index) + 1,
                    "channel_label": header.labels[source_index],
                    "physical_unit": header.physical_dimensions[source_index],
                    "reason": "not among requested 29 EEG + EKG + SPO2 channels",
                }
            )
    write_readme(output_dir)
    print(f"Exported {len(selected)} channels from {source.name} to {output_dir}")
    print(f"Each channel contains {header.n_records * samples_per_record:,} samples at {sample_rate:g} Hz.")
    print("Omitted signals: " + ", ".join(header.labels[index] for index in omitted))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export_channels(args.source, args.output_dir)


if __name__ == "__main__":
    main()
