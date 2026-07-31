"""Create time-aligned EEG arrays with samples outside ±500 µV excluded.

The original EDF files are read-only. For each recording this script writes a
float32 NumPy array under ``data/interim/eeg_threshold_cleaned`` with shape
``(EEG channels, samples)``. Values strictly above +500 µV or strictly below
-500 µV are replaced by NaN. The array shape and sample positions do not
change, so seizure annotations remain aligned. NaN itself is the exclusion
mask and no second full-size mask is needed.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "interim" / "eeg_threshold_cleaned"
DEFAULT_QC_PATH = ROOT / "data" / "interim" / "eeg_threshold_cleaning_qc.csv"
LOWER_UV = -500.0
UPPER_UV = 500.0


@dataclass(frozen=True)
class EDFHeader:
    header_bytes: int
    n_records: int
    record_seconds: float
    labels: tuple[str, ...]
    units: tuple[str, ...]
    physical_min: np.ndarray
    physical_max: np.ndarray
    digital_min: np.ndarray
    digital_max: np.ndarray
    samples_per_record: np.ndarray

    @property
    def n_signals(self) -> int:
        return len(self.labels)

    @property
    def samples_per_data_record(self) -> int:
        return int(self.samples_per_record.sum())

    @property
    def bytes_per_data_record(self) -> int:
        return 2 * self.samples_per_data_record


def _parse_ascii_number(value: bytes, kind: type[int] | type[float]) -> int | float:
    text = value.decode("ascii", errors="strict").strip()
    if not text:
        raise ValueError("Required numeric EDF header field is blank.")
    return kind(text)


def read_edf_header(path: Path) -> EDFHeader:
    """Read the fixed and per-signal EDF header without loading signal data."""
    with path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) != 256:
            raise ValueError(f"{path} has a truncated fixed EDF header.")
        header_bytes = int(_parse_ascii_number(fixed[184:192], int))
        n_records = int(_parse_ascii_number(fixed[236:244], int))
        record_seconds = float(_parse_ascii_number(fixed[244:252], float))
        n_signals = int(_parse_ascii_number(fixed[252:256], int))
        if n_records <= 0 or record_seconds <= 0 or n_signals <= 0:
            raise ValueError(f"{path} declares invalid EDF dimensions.")
        signal_header = handle.read(n_signals * 256)
        if len(signal_header) != n_signals * 256:
            raise ValueError(f"{path} has a truncated per-signal EDF header.")

    offset = 0

    def fields(width: int) -> list[str]:
        nonlocal offset
        block = signal_header[offset : offset + width * n_signals]
        offset += width * n_signals
        return [
            block[index * width : (index + 1) * width]
            .decode("ascii", errors="replace")
            .strip()
            for index in range(n_signals)
        ]

    labels = fields(16)
    fields(80)  # transducer
    units = fields(8)
    physical_min = np.asarray(fields(8), dtype=float)
    physical_max = np.asarray(fields(8), dtype=float)
    digital_min = np.asarray(fields(8), dtype=float)
    digital_max = np.asarray(fields(8), dtype=float)
    fields(80)  # prefiltering
    samples_per_record = np.asarray(fields(8), dtype=int)
    fields(32)  # reserved

    if offset != len(signal_header):
        raise AssertionError("EDF per-signal header parser did not consume the header.")
    if header_bytes != 256 + n_signals * 256:
        raise ValueError(f"{path} has an unexpected EDF header byte count.")
    if np.any(samples_per_record <= 0):
        raise ValueError(f"{path} contains a non-positive signal sample count.")
    if np.any(digital_max == digital_min):
        raise ValueError(f"{path} contains a zero digital range.")

    return EDFHeader(
        header_bytes=header_bytes,
        n_records=n_records,
        record_seconds=record_seconds,
        labels=tuple(labels),
        units=tuple(units),
        physical_min=physical_min,
        physical_max=physical_max,
        digital_min=digital_min,
        digital_max=digital_max,
        samples_per_record=samples_per_record,
    )


def _unit_to_microvolt_factor(unit: str) -> float:
    normalized = unit.strip().replace("µ", "u").lower()
    factors = {"uv": 1.0, "mv": 1_000.0, "v": 1_000_000.0}
    if normalized not in factors:
        raise ValueError(f"Unsupported EEG physical unit: {unit!r}")
    return factors[normalized]


def _normalized_electrode(label: str) -> str:
    return label[3:].strip().upper()


def clean_recording(
    source_path: Path,
    relative_path: str,
    output_dir: Path,
    chunk_records: int = 60,
    force: bool = False,
) -> list[dict[str, object]]:
    """Threshold-clean one EDF and return one QC row per EEG channel."""
    header = read_edf_header(source_path)
    eeg_indices = [
        index for index, label in enumerate(header.labels)
        if label.upper().startswith("EEG")
    ]
    if not eeg_indices:
        raise ValueError(f"{relative_path} has no labels beginning with EEG.")

    eeg_samples_per_record = header.samples_per_record[eeg_indices]
    if not np.all(eeg_samples_per_record == eeg_samples_per_record[0]):
        raise ValueError(f"{relative_path} has unequal scalp-EEG sample rates.")
    samples_per_record = int(eeg_samples_per_record[0])
    total_samples = header.n_records * samples_per_record
    sample_rate_hz = samples_per_record / header.record_seconds

    relative = Path(relative_path)
    array_path = output_dir / relative.parent / f"{relative.stem}.npy"
    metadata_path = output_dir / relative.parent / f"{relative.stem}.json"
    partial_path = array_path.with_suffix(".partial.npy")
    array_path.parent.mkdir(parents=True, exist_ok=True)

    if array_path.exists() and metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_shape = [len(eeg_indices), total_samples]
        existing = np.load(array_path, mmap_mode="r")
        if list(existing.shape) == expected_shape and existing.dtype == np.float32:
            return list(metadata["channel_qc"])

    if partial_path.exists():
        partial_path.unlink()

    output = np.lib.format.open_memmap(
        partial_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(eeg_indices), total_samples),
    )
    excluded_counts = np.zeros(len(eeg_indices), dtype=np.int64)
    retained_min = np.full(len(eeg_indices), np.inf)
    retained_max = np.full(len(eeg_indices), -np.inf)

    signal_offsets = np.concatenate(([0], np.cumsum(header.samples_per_record)))
    with source_path.open("rb") as handle:
        for first_record in range(0, header.n_records, chunk_records):
            records_to_read = min(chunk_records, header.n_records - first_record)
            byte_offset = (
                header.header_bytes
                + first_record * header.bytes_per_data_record
            )
            handle.seek(byte_offset)
            payload = handle.read(records_to_read * header.bytes_per_data_record)
            expected_bytes = records_to_read * header.bytes_per_data_record
            if len(payload) != expected_bytes:
                raise ValueError(
                    f"Short EDF read in {relative_path} at record {first_record}."
                )
            records = np.frombuffer(payload, dtype="<i2").reshape(
                records_to_read, header.samples_per_data_record
            )
            sample_start = first_record * samples_per_record
            sample_stop = sample_start + records_to_read * samples_per_record

            for output_index, signal_index in enumerate(eeg_indices):
                digital = records[
                    :,
                    signal_offsets[signal_index] : signal_offsets[signal_index + 1],
                ].reshape(-1)
                scale = (
                    (header.physical_max[signal_index] - header.physical_min[signal_index])
                    / (header.digital_max[signal_index] - header.digital_min[signal_index])
                )
                physical = (
                    (digital.astype(np.float64) - header.digital_min[signal_index])
                    * scale
                    + header.physical_min[signal_index]
                )
                physical *= _unit_to_microvolt_factor(header.units[signal_index])
                excluded = (physical < LOWER_UV) | (physical > UPPER_UV)
                excluded_counts[output_index] += int(excluded.sum())
                retained = physical[~excluded]
                if retained.size:
                    retained_min[output_index] = min(
                        retained_min[output_index], float(retained.min())
                    )
                    retained_max[output_index] = max(
                        retained_max[output_index], float(retained.max())
                    )
                physical[excluded] = np.nan
                output[output_index, sample_start:sample_stop] = physical.astype(
                    np.float32
                )

    output.flush()
    del output
    os.replace(partial_path, array_path)

    channel_qc: list[dict[str, object]] = []
    for output_index, signal_index in enumerate(eeg_indices):
        excluded_count = int(excluded_counts[output_index])
        channel_qc.append(
            {
                "file": relative_path.replace("/", "\\"),
                "patient_id": relative.parts[0],
                "channel_index": output_index,
                "channel_original": header.labels[signal_index],
                "channel_normalized": _normalized_electrode(
                    header.labels[signal_index]
                ),
                "sample_rate_hz": sample_rate_hz,
                "total_samples": total_samples,
                "excluded_samples": excluded_count,
                "excluded_fraction": excluded_count / total_samples,
                "retained_min_uv": (
                    float(retained_min[output_index])
                    if np.isfinite(retained_min[output_index])
                    else None
                ),
                "retained_max_uv": (
                    float(retained_max[output_index])
                    if np.isfinite(retained_max[output_index])
                    else None
                ),
                "clean_array": str(array_path.relative_to(ROOT)),
            }
        )

    metadata = {
        "source_edf": relative_path,
        "clean_array": str(array_path.relative_to(ROOT)),
        "shape": [len(eeg_indices), total_samples],
        "dtype": "float32",
        "lower_bound_uv_inclusive": LOWER_UV,
        "upper_bound_uv_inclusive": UPPER_UV,
        "excluded_representation": "NaN",
        "sample_rate_hz": sample_rate_hz,
        "record_seconds": header.record_seconds,
        "channel_qc": channel_qc,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return channel_qc


def validate_clean_array(
    array_path: Path, chunk_values: int = 10_000_000
) -> dict[str, int | float]:
    """Validate bounds in chunks without allocating a recording-sized mask."""
    clean = np.load(array_path, mmap_mode="r")
    flat = clean.reshape(-1)
    excluded_samples = 0
    clean_min = np.inf
    clean_max = -np.inf
    for start in range(0, flat.size, chunk_values):
        block = np.asarray(flat[start : start + chunk_values])
        excluded_samples += int(np.isnan(block).sum())
        finite = block[np.isfinite(block)]
        if finite.size:
            clean_min = min(clean_min, float(finite.min()))
            clean_max = max(clean_max, float(finite.max()))
    if np.isfinite(clean_min):
        if clean_min < LOWER_UV or clean_max > UPPER_UV:
            raise AssertionError(
                f"{array_path} contains a retained value outside ±500 µV."
            )
    else:
        clean_min = np.nan
        clean_max = np.nan
    return {
        "samples": int(clean.size),
        "excluded_samples": excluded_samples,
        "retained_min_uv": clean_min,
        "retained_max_uv": clean_max,
    }


def run(
    raw_dir: Path = RAW_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    qc_path: Path = DEFAULT_QC_PATH,
    force: bool = False,
) -> pd.DataFrame:
    records_path = raw_dir / "RECORDS"
    recordings = [
        line.strip()
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    missing = [
        relative for relative in recordings if not (raw_dir / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} raw EDF recordings are missing; first: {missing[0]}"
        )

    all_qc: list[dict[str, object]] = []
    for number, relative_path in enumerate(recordings, start=1):
        print(f"[{number:02d}/{len(recordings)}] Cleaning {relative_path}", flush=True)
        rows = clean_recording(
            raw_dir / relative_path,
            relative_path,
            output_dir,
            force=force,
        )
        all_qc.extend(rows)

    qc = pd.DataFrame(all_qc)
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    qc.to_csv(qc_path, index=False)

    for array_path in sorted(output_dir.rglob("*.npy")):
        validate_clean_array(array_path)
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--qc-path", type=Path, default=DEFAULT_QC_PATH)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qc = run(args.raw_dir, args.output_dir, args.qc_path, args.force)
    print(
        f"Finished: {len(qc):,} EEG channel/recording rows; "
        f"{int(qc['excluded_samples'].sum()):,} samples excluded."
    )


if __name__ == "__main__":
    main()
