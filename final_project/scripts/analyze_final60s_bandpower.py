"""Patient-aware EEG band-power trajectories during the final minute before seizure.

This script is intentionally separate from ``analyze_preictal_bandpower.py``.
It implements the final-60-second analysis requested for this project:

* ten consecutive 6-second preictal windows (-60 to 0 s);
* Welch PSDs calculated from every sample in each window;
* per-channel relative band power followed by a median across usable channels;
* patient-specific, clean interictal baselines and z scores; and
* seizure- and patient-level (rather than pooled-window) summaries.

Run from the repository root:

    .\\.venv\\Scripts\\python.exe final_project/scripts/analyze_final60s_bandpower.py

Raw EDF files are read in place and are never changed.  Outputs are written to
``final_project/data/processed/preictal_60s_zscore`` and
``final_project/results/preictal_60s_zscore`` by default.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats


ROOT = Path(__file__).resolve().parents[1]

# The exact ranges requested for the analysis.  A recording/channel is used for
# a band only when its EDF-declared acquisition filters preserve the full range.
BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
BAND_ORDER = list(BANDS)
PREICTAL_SECONDS = 60
WINDOW_SECONDS = 6
WINDOW_MIDPOINTS = np.arange(-57, 0, WINDOW_SECONDS, dtype=int)
INTERICTAL_BLOCK_SECONDS = 60
INTERICTAL_BUFFER_MINUTES = 15
BLOCKS_PER_SEIZURE = 3
MIN_USABLE_CHANNELS = 10
WELCH_SEGMENT_SECONDS = 4
NOTCH_FREQUENCY_HZ = 60.0
NOTCH_QUALITY_FACTOR = 30.0
ARTIFACT_MAX_ABS_UV = 1_000.0
ARTIFACT_MIN_STD_UV = 0.5
ARTIFACT_MAX_CLIPPED_FRACTION = 0.01


def _decode(value: bytes) -> str:
    return value.decode("ascii", errors="ignore").strip().replace("\x00", "")


@dataclass(frozen=True)
class EDFHeader:
    """The EDF metadata needed for random-access, filter-aware reading."""

    path: Path
    header_bytes: int
    n_records: int
    record_seconds: float
    labels: tuple[str, ...]
    physical_dimensions: tuple[str, ...]
    physical_min: np.ndarray
    physical_max: np.ndarray
    digital_min: np.ndarray
    digital_max: np.ndarray
    prefilters: tuple[str, ...]
    samples_per_record: np.ndarray
    bytes_per_record: int

    @property
    def duration_seconds(self) -> float:
        return self.n_records * self.record_seconds


@dataclass(frozen=True)
class SeizureEvent:
    patient_id: str
    event_id: str
    edf_path: Path
    onset_seconds: int
    end_seconds: int
    annotation_note: str = ""


@dataclass(frozen=True)
class InterictalBlock:
    patient_id: str
    block_id: str
    edf_path: Path
    start_seconds: int
    source_candidate_count: int


def read_edf_header(path: Path) -> EDFHeader:
    """Read EDF metadata without loading its signal records."""
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
        physical_dimensions = tuple(
            _decode(handle.read(8)) for _ in range(n_signals)
        )
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
        prefilters = tuple(_decode(handle.read(80)) for _ in range(n_signals))
        samples_per_record = np.array(
            [int(_decode(handle.read(8))) for _ in range(n_signals)], dtype=int
        )
    if n_records <= 0 or record_seconds <= 0 or np.any(samples_per_record <= 0):
        raise ValueError(f"Invalid EDF recording metadata: {path}")
    return EDFHeader(
        path=path,
        header_bytes=header_bytes,
        n_records=n_records,
        record_seconds=record_seconds,
        labels=labels,
        physical_dimensions=physical_dimensions,
        physical_min=physical_min,
        physical_max=physical_max,
        digital_min=digital_min,
        digital_max=digital_max,
        prefilters=prefilters,
        samples_per_record=samples_per_record,
        bytes_per_record=int(samples_per_record.sum() * 2),
    )


def _parse_clock(text: str) -> int | None:
    """Parse the source clock notation, including the known PN10 '1 6' typo."""
    match = re.search(
        r"(?<!\d)(\d{1,2}(?:\s+\d)?)\s*[\.:]\s*(\d{2})\s*[\.:]\s*(\d{2})",
        text,
    )
    if not match:
        return None
    hour_text, minute_text, second_text = match.groups()
    hour = int(hour_text.replace(" ", ""))
    minute = int(minute_text)
    second = int(second_text)
    if hour > 23 or minute > 59 or second > 59:
        return None
    return hour * 3_600 + minute * 60 + second


def parse_annotations(path: Path) -> dict[str, list[tuple[int, int]]]:
    """Return annotated seizure intervals in seconds from each EDF start."""
    intervals: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None
    registration_start: int | None = None
    seizure_start: int | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        lowered = line.lower()
        if "file name:" in lowered:
            current_file = line.split(":", 1)[1].strip().split()[0]
            intervals.setdefault(current_file, [])
            seizure_start = None
        elif "registration start time:" in lowered:
            registration_start = _parse_clock(line)
        elif "seizure start time:" in lowered or re.match(r"\s*start time:", lowered):
            seizure_start = _parse_clock(line)
        elif "seizure end time:" in lowered or re.match(r"\s*end time:", lowered):
            seizure_end = _parse_clock(line)
            if (
                current_file
                and registration_start is not None
                and seizure_start is not None
                and seizure_end is not None
            ):
                start = (seizure_start - registration_start) % 86_400
                end = (seizure_end - registration_start) % 86_400
                if end <= start:
                    end += 86_400
                intervals[current_file].append((start, end))
            seizure_start = None
    return intervals


def _normalized_filename(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower()).replace("o", "0")


def resolve_edf(annotation_name: str, edf_files: list[Path]) -> Path:
    exact = [path for path in edf_files if path.name.lower() == annotation_name.lower()]
    if len(exact) == 1:
        return exact[0]
    normalized = _normalized_filename(annotation_name)
    matches = [path for path in edf_files if _normalized_filename(path.name) == normalized]
    if len(matches) == 1:
        return matches[0]
    if len(edf_files) == 1:
        return edf_files[0]
    raise FileNotFoundError(f"Could not match annotation filename {annotation_name!r}")


def load_study(raw_dir: Path) -> tuple[list[SeizureEvent], dict[Path, list[tuple[int, int]]]]:
    """Load, resolve, and validate the supplied seizure-list annotations."""
    events: list[SeizureEvent] = []
    intervals_by_file: dict[Path, list[tuple[int, int]]] = {}
    for annotation_path in sorted(raw_dir.glob("PN*/Seizures-list-PN*.txt")):
        patient_id = annotation_path.parent.name
        edf_files = sorted(annotation_path.parent.glob("*.edf"))
        if not edf_files:
            raise FileNotFoundError(f"No EDF files found for {patient_id}")
        for edf_path in edf_files:
            intervals_by_file.setdefault(edf_path, [])
        event_number = 0
        for annotation_name, intervals in parse_annotations(annotation_path).items():
            edf_path = resolve_edf(annotation_name, edf_files)
            duration = read_edf_header(edf_path).duration_seconds
            for onset, end in intervals:
                note = ""
                # The PN00 source list contains one end hour that is exactly one
                # hour beyond the recording.  The repair does not alter onset.
                if end > duration and onset < end - 3_600 <= duration:
                    end -= 3_600
                    note = "end_time_minus_one_hour_to_fit_recording"
                event_number += 1
                events.append(
                    SeizureEvent(
                        patient_id=patient_id,
                        event_id=f"{patient_id}_S{event_number:02d}",
                        edf_path=edf_path,
                        onset_seconds=onset,
                        end_seconds=end,
                        annotation_note=note,
                    )
                )
                intervals_by_file[edf_path].append((onset, end))
    if not events:
        raise FileNotFoundError(f"No seizure annotations found under {raw_dir}")
    return events, intervals_by_file


def is_eeg(label: str) -> bool:
    return label.strip().upper().startswith("EEG ")


def parse_prefilter(prefilter: str) -> tuple[float | None, float | None]:
    """Extract EDF-declared high- and low-pass cutoffs, conservatively."""
    high_match = re.search(r"\bHP\s*:\s*(-?\d+(?:\.\d+)?)\s*Hz", prefilter, re.I)
    low_match = re.search(r"\bLP\s*:\s*(-?\d+(?:\.\d+)?)\s*Hz", prefilter, re.I)
    high = float(high_match.group(1)) if high_match else None
    low = float(low_match.group(1)) if low_match else None
    # -1 denotes no declared value, not a physically valid cutoff.
    if high is not None and high < 0:
        high = None
    if low is not None and low < 0:
        low = None
    return high, low


def channel_band_availability(header: EDFHeader, eeg_indices: np.ndarray) -> dict[str, np.ndarray]:
    """Whether every requested Hz in a band survives each channel's filters."""
    rates = header.samples_per_record[eeg_indices] / header.record_seconds
    high_pass = np.array(
        [
            np.nan if parse_prefilter(header.prefilters[i])[0] is None else parse_prefilter(header.prefilters[i])[0]
            for i in eeg_indices
        ],
        dtype=float,
    )
    low_pass = np.array(
        [
            np.nan if parse_prefilter(header.prefilters[i])[1] is None else parse_prefilter(header.prefilters[i])[1]
            for i in eeg_indices
        ],
        dtype=float,
    )
    availability: dict[str, np.ndarray] = {}
    for band, (low, high) in BANDS.items():
        availability[band] = (
            np.isfinite(high_pass)
            & np.isfinite(low_pass)
            & (high_pass <= low + 1e-9)
            & (low_pass >= high - 1e-9)
            & (rates >= 2 * high)
        )
    return availability


def recording_metadata_audit(raw_dir: Path) -> pd.DataFrame:
    """Verify sample rates, physical units, and band-specific filter validity."""
    rows: list[dict[str, object]] = []
    for edf_path in sorted(raw_dir.glob("PN*/*.edf")):
        header = read_edf_header(edf_path)
        eeg_indices = np.array(
            [i for i, label in enumerate(header.labels) if is_eeg(label)], dtype=int
        )
        if not len(eeg_indices):
            raise ValueError(f"No EEG-labelled channels in {edf_path}")
        rates = header.samples_per_record[eeg_indices] / header.record_seconds
        available = channel_band_availability(header, eeg_indices)
        hps = [parse_prefilter(header.prefilters[i])[0] for i in eeg_indices]
        lps = [parse_prefilter(header.prefilters[i])[1] for i in eeg_indices]
        row: dict[str, object] = {
            "patient_id": edf_path.parent.name,
            "recording": edf_path.name,
            "duration_seconds": header.duration_seconds,
            "eeg_channel_count": len(eeg_indices),
            "eeg_sample_rates_hz": ";".join(map(str, sorted(set(rates)))),
            "equal_eeg_sample_rate": bool(np.allclose(rates, rates[0])),
            "physical_dimensions": ";".join(
                sorted({header.physical_dimensions[i] for i in eeg_indices})
            ),
            "declared_highpass_hz": ";".join(
                sorted({str(value) for value in hps if value is not None})
            ),
            "declared_lowpass_hz": ";".join(
                sorted({str(value) for value in lps if value is not None})
            ),
            "filters_parsed_for_all_eeg_channels": bool(
                all(value is not None for value in hps + lps)
            ),
        }
        for band in BAND_ORDER:
            row[f"{band}_filter_usable_channels"] = int(available[band].sum())
            row[f"{band}_filter_usable_recording"] = bool(available[band].any())
        rows.append(row)
    return pd.DataFrame(rows)


def plan_events(
    events: list[SeizureEvent],
    intervals_by_file: dict[Path, list[tuple[int, int]]],
) -> pd.DataFrame:
    """Check the 60-second interval and annotations for each seizure."""
    rows: list[dict[str, object]] = []
    for event in events:
        header = read_edf_header(event.edf_path)
        preictal_start = event.onset_seconds - PREICTAL_SECONDS
        other_intervals = [
            interval
            for interval in intervals_by_file[event.edf_path]
            if interval != (event.onset_seconds, event.end_seconds)
        ]
        overlaps_other = any(
            preictal_start < other_end and event.onset_seconds > other_start
            for other_start, other_end in other_intervals
        )
        reason = ""
        if preictal_start < 0:
            reason = "less_than_60_seconds_before_onset"
        elif event.onset_seconds > header.duration_seconds:
            reason = "onset_after_recording_end"
        elif event.end_seconds <= event.onset_seconds:
            reason = "invalid_annotation_interval"
        elif overlaps_other:
            reason = "preictal_period_overlaps_another_seizure"
        rows.append(
            {
                "patient_id": event.patient_id,
                "event_id": event.event_id,
                "recording": event.edf_path.name,
                "onset_seconds": event.onset_seconds,
                "end_seconds": event.end_seconds,
                "seizure_duration_seconds": event.end_seconds - event.onset_seconds,
                "annotation_note": event.annotation_note,
                "eligible_preictal": not reason,
                "exclusion_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def choose_interictal_blocks(
    events: list[SeizureEvent],
    intervals_by_file: dict[Path, list[tuple[int, int]]],
    buffer_seconds: int,
    blocks_per_seizure: int,
) -> tuple[list[InterictalBlock], set[str]]:
    """Choose evenly-spaced clean 60-s patient controls without random sampling.

    Candidate blocks are aligned on 60-s recording boundaries and must be at
    least ``buffer_seconds`` from every annotated seizure in their recording.
    Each patient contributes three blocks per analyzed seizure, sampled evenly
    across all of that patient's clean candidates.  These windows form one
    patient-level baseline; they are not matched to, or reused as, individual
    seizure baselines.
    """
    by_patient: dict[str, list[SeizureEvent]] = {}
    for event in events:
        by_patient.setdefault(event.patient_id, []).append(event)

    selected: list[InterictalBlock] = []
    patients_without_controls: set[str] = set()
    for patient_id, patient_events in sorted(by_patient.items()):
        candidates: list[tuple[Path, int]] = []
        patient_files = sorted(
            path for path in intervals_by_file if path.parent.name == patient_id
        )
        for edf_path in patient_files:
            duration = int(read_edf_header(edf_path).duration_seconds)
            for start in range(0, duration - INTERICTAL_BLOCK_SECONDS + 1, INTERICTAL_BLOCK_SECONDS):
                end = start + INTERICTAL_BLOCK_SECONDS
                clean = all(
                    end <= seizure_start - buffer_seconds
                    or start >= seizure_end + buffer_seconds
                    for seizure_start, seizure_end in intervals_by_file[edf_path]
                )
                if clean:
                    candidates.append((edf_path, start))
        target = blocks_per_seizure * len(patient_events)
        if not candidates:
            patients_without_controls.add(patient_id)
            continue
        n_to_select = min(target, len(candidates))
        # This is a reproducible systematic sample, not a random raw-signal
        # selection.  Unique integer indices avoid duplicated controls.
        indices = np.unique(np.linspace(0, len(candidates) - 1, n_to_select, dtype=int))
        if len(indices) < n_to_select:
            remaining = [i for i in range(len(candidates)) if i not in set(indices)]
            indices = np.concatenate([indices, np.array(remaining[: n_to_select - len(indices)])])
        for sequence, candidate_index in enumerate(sorted(indices), start=1):
            edf_path, start = candidates[int(candidate_index)]
            selected.append(
                InterictalBlock(
                    patient_id=patient_id,
                    block_id=f"{patient_id}_I{sequence:03d}",
                    edf_path=edf_path,
                    start_seconds=start,
                    source_candidate_count=len(candidates),
                )
            )
    return selected, patients_without_controls


def read_eeg_window(
    path: Path, start_seconds: int, duration_seconds: int
) -> tuple[np.ndarray, float, np.ndarray, EDFHeader]:
    """Read and scale all EEG channels in a record-aligned EDF interval."""
    header = read_edf_header(path)
    eeg_indices = np.array(
        [i for i, label in enumerate(header.labels) if is_eeg(label)], dtype=int
    )
    if not len(eeg_indices):
        raise ValueError(f"No EEG-labelled channels in {path}")
    rates = header.samples_per_record[eeg_indices] / header.record_seconds
    if not np.allclose(rates, rates[0]):
        raise ValueError(f"EEG channels have unequal sample rates in {path}")
    if not np.all(header.samples_per_record == header.samples_per_record[0]):
        raise ValueError(f"Signals have unequal sample rates; cannot use random reader: {path}")
    if start_seconds % header.record_seconds or duration_seconds % header.record_seconds:
        raise ValueError("Requested EDF window must align with data records")
    start_record = int(start_seconds / header.record_seconds)
    n_records = int(duration_seconds / header.record_seconds)
    if start_record < 0 or start_record + n_records > header.n_records:
        raise ValueError(f"Requested window lies outside {path.name}")

    with path.open("rb") as handle:
        handle.seek(header.header_bytes + start_record * header.bytes_per_record)
        raw = np.frombuffer(
            handle.read(n_records * header.bytes_per_record), dtype="<i2"
        )
    expected = n_records * header.bytes_per_record // 2
    if raw.size != expected:
        raise ValueError(f"Short signal read in {path.name}")
    raw = raw.reshape(n_records, len(header.labels), -1)
    raw = raw[:, eeg_indices, :].transpose(1, 0, 2).reshape(len(eeg_indices), -1)
    dmin = header.digital_min[eeg_indices, None]
    dmax = header.digital_max[eeg_indices, None]
    pmin = header.physical_min[eeg_indices, None]
    pmax = header.physical_max[eeg_indices, None]
    data = raw.astype(np.float64) * ((pmax - pmin) / (dmax - dmin))
    data += pmin - dmin * ((pmax - pmin) / (dmax - dmin))
    return data, float(rates[0]), eeg_indices, header


def _integrate_band(psd: np.ndarray, frequencies: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = (frequencies >= low) & (frequencies <= high)
    if mask.sum() < 2:
        return np.full(psd.shape[0], np.nan)
    return np.trapezoid(psd[:, mask], frequencies[mask], axis=1)


def window_bandpowers(
    data: np.ndarray,
    sample_rate: float,
    eeg_indices: np.ndarray,
    header: EDFHeader,
    min_channels: int,
) -> tuple[dict[str, float], dict[str, dict[str, object]], dict[str, object]]:
    """Calculate filter-aware median relative powers from all window samples."""
    original = data
    data = signal.detrend(data, axis=-1, type="linear")
    # The identical robust common reference is used for every preictal and
    # interictal window.  It lessens common offsets without one lead dominating.
    data = data - np.median(data, axis=0, keepdims=True)
    std = np.std(data, axis=1)
    max_abs = np.max(np.abs(data), axis=1)
    physical_min = header.physical_min[eeg_indices]
    physical_max = header.physical_max[eeg_indices]
    tolerance = np.maximum((physical_max - physical_min) * 0.001, 1e-6)
    clipped = np.mean(
        (original <= physical_min[:, None] + tolerance[:, None])
        | (original >= physical_max[:, None] - tolerance[:, None]),
        axis=1,
    )
    artifact_usable = (
        np.isfinite(data).all(axis=1)
        & (std >= ARTIFACT_MIN_STD_UV)
        & (max_abs <= ARTIFACT_MAX_ABS_UV)
        & (clipped <= ARTIFACT_MAX_CLIPPED_FRACTION)
    )
    channel_available = channel_band_availability(header, eeg_indices)
    global_qc = {
        "eeg_channel_count": int(data.shape[0]),
        "artifact_usable_channel_count": int(artifact_usable.sum()),
        "artifact_rejected_channel_count": int((~artifact_usable).sum()),
        "artifact_channel_fraction": float((~artifact_usable).mean()),
        "sample_rate_hz": sample_rate,
    }

    # Estimate Welch PSD once for all artifact-usable channels, then use each
    # band's valid subset.  This uses every raw EEG sample in the 6-s window.
    if not artifact_usable.any():
        values = {band: np.nan for band in BAND_ORDER}
        band_qc = {
            band: {
                "filter_usable_channel_count": int(channel_available[band].sum()),
                "usable_channel_count": 0,
                "band_qc_pass": False,
                "relative_denominator_bands": "",
            }
            for band in BAND_ORDER
        }
        return values, band_qc, global_qc

    clean = data[artifact_usable]
    clean_available = {band: channel_available[band][artifact_usable] for band in BAND_ORDER}
    if sample_rate > 2 * NOTCH_FREQUENCY_HZ:
        notch_b, notch_a = signal.iirnotch(
            NOTCH_FREQUENCY_HZ, NOTCH_QUALITY_FACTOR, fs=sample_rate
        )
        clean = signal.filtfilt(notch_b, notch_a, clean, axis=-1)
    nperseg = min(int(round(WELCH_SEGMENT_SECONDS * sample_rate)), clean.shape[1])
    noverlap = min(nperseg // 2, nperseg - 1)
    frequencies, psd = signal.welch(
        clean,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="linear",
        axis=-1,
        scaling="density",
    )
    powers = {
        band: _integrate_band(psd, frequencies, low, high)
        for band, (low, high) in BANDS.items()
    }
    availability_matrix = np.column_stack([clean_available[band] for band in BAND_ORDER])
    power_matrix = np.column_stack([powers[band] for band in BAND_ORDER])
    values: dict[str, float] = {}
    band_qc: dict[str, dict[str, object]] = {}
    for band_index, band in enumerate(BAND_ORDER):
        valid_channels = availability_matrix[:, band_index]
        usable_count = int(valid_channels.sum())
        # Relative power is the fraction of a channel's PSD in this band among
        # all bands whose source filters preserve their complete stated ranges.
        denominators = np.nansum(
            np.where(availability_matrix, power_matrix, np.nan), axis=1
        )
        relative = power_matrix[:, band_index] / np.maximum(denominators, 1e-12)
        qc_pass = usable_count >= min_channels
        values[band] = float(np.median(relative[valid_channels])) if qc_pass else np.nan
        available_bands = sorted(
            {
                ";".join(np.array(BAND_ORDER)[availability_matrix[row]])
                for row in range(availability_matrix.shape[0])
                if valid_channels[row]
            }
        )
        band_qc[band] = {
            "filter_usable_channel_count": int(channel_available[band].sum()),
            "usable_channel_count": usable_count,
            "band_qc_pass": qc_pass,
            "relative_denominator_bands": " | ".join(available_bands),
        }
    return values, band_qc, global_qc


def extract_window(
    *,
    edf_path: Path,
    start_seconds: int,
    condition: str,
    patient_id: str,
    event_id: str | None,
    block_id: str | None,
    time_bin: int,
    min_channels: int,
) -> list[dict[str, object]]:
    """Create one tidy row per band from one complete 6-second EEG window."""
    data, sample_rate, eeg_indices, header = read_eeg_window(
        edf_path, start_seconds, WINDOW_SECONDS
    )
    powers, band_qc, global_qc = window_bandpowers(
        data, sample_rate, eeg_indices, header, min_channels
    )
    time_start = -PREICTAL_SECONDS + (time_bin - 1) * WINDOW_SECONDS
    time_end = time_start + WINDOW_SECONDS
    rows: list[dict[str, object]] = []
    for band in BAND_ORDER:
        rows.append(
            {
                "patient_id": patient_id,
                "event_id": event_id,
                "interictal_block_id": block_id,
                "condition": condition,
                "recording": edf_path.name,
                "window_start_seconds": start_seconds,
                "time_bin": time_bin,
                "time_to_onset_start_seconds": time_start if condition == "preictal" else np.nan,
                "time_to_onset_end_seconds": time_end if condition == "preictal" else np.nan,
                "time_to_onset_midpoint_seconds": int((time_start + time_end) / 2)
                if condition == "preictal"
                else np.nan,
                "band": band,
                "relative_power": powers[band],
                **band_qc[band],
                **global_qc,
            }
        )
    return rows


def extract_features(
    events: list[SeizureEvent],
    controls: list[InterictalBlock],
    min_channels: int,
) -> pd.DataFrame:
    """Calculate all preictal and baseline 6-second rows."""
    rows: list[dict[str, object]] = []
    n_windows = PREICTAL_SECONDS // WINDOW_SECONDS
    total_jobs = len(events) * n_windows + len(controls) * (INTERICTAL_BLOCK_SECONDS // WINDOW_SECONDS)
    completed = 0
    for event in events:
        for time_bin in range(1, n_windows + 1):
            start = event.onset_seconds - PREICTAL_SECONDS + (time_bin - 1) * WINDOW_SECONDS
            rows.extend(
                extract_window(
                    edf_path=event.edf_path,
                    start_seconds=start,
                    condition="preictal",
                    patient_id=event.patient_id,
                    event_id=event.event_id,
                    block_id=None,
                    time_bin=time_bin,
                    min_channels=min_channels,
                )
            )
            completed += 1
        print(f"Preictal: {event.event_id} ({completed}/{total_jobs} windows)", flush=True)
    for control in controls:
        for time_bin in range(1, INTERICTAL_BLOCK_SECONDS // WINDOW_SECONDS + 1):
            start = control.start_seconds + (time_bin - 1) * WINDOW_SECONDS
            rows.extend(
                extract_window(
                    edf_path=control.edf_path,
                    start_seconds=start,
                    condition="interictal",
                    patient_id=control.patient_id,
                    event_id=None,
                    block_id=control.block_id,
                    time_bin=time_bin,
                    min_channels=min_channels,
                )
            )
            completed += 1
        print(f"Interictal: {control.block_id} ({completed}/{total_jobs} windows)", flush=True)
    return pd.DataFrame(rows)


def add_patient_baselines(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use clean interictal values to create patient-by-band z scores."""
    baseline_source = features[
        (features["condition"] == "interictal")
        & features["band_qc_pass"]
        & features["relative_power"].notna()
    ]
    baselines = (
        baseline_source.groupby(["patient_id", "band"], as_index=False)["relative_power"]
        .agg(
            interictal_mean="mean",
            interictal_sd=lambda values: values.std(ddof=1),
            interictal_window_count="count",
        )
    )
    baselines["baseline_usable"] = (
        (baselines["interictal_window_count"] >= 2)
        & baselines["interictal_sd"].notna()
        & (baselines["interictal_sd"] > 0)
    )
    features = features.merge(baselines, on=["patient_id", "band"], how="left")
    baseline_usable = features["baseline_usable"].fillna(False).astype(bool)
    features["z_score"] = np.where(
        features["band_qc_pass"] & baseline_usable,
        (features["relative_power"] - features["interictal_mean"])
        / features["interictal_sd"],
        np.nan,
    )
    features["outside_interictal_range"] = features["z_score"].abs() > 2
    return features, baselines


def _correlation_row(
    values: pd.DataFrame,
    id_fields: dict[str, object],
) -> dict[str, object]:
    subset = values.sort_values("time_to_onset_midpoint_seconds")
    x = subset["time_to_onset_midpoint_seconds"].to_numpy(dtype=float)
    y = subset["z_score"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    output = {**id_fields, "n_windows": int(valid.sum())}
    if valid.sum() < 3 or np.unique(x[valid]).size < 3 or np.unique(y[valid]).size < 2:
        output.update({"pearson_r": np.nan, "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan})
        return output
    pearson = stats.pearsonr(x[valid], y[valid])
    spearman = stats.spearmanr(x[valid], y[valid])
    output.update(
        {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
        }
    )
    return output


def correlation_tables(preictal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute correlations within seizures and within patient-median paths."""
    seizure_rows = [
        _correlation_row(group, {"patient_id": patient, "event_id": event, "band": band})
        for (patient, event, band), group in preictal.groupby(["patient_id", "event_id", "band"])
    ]
    seizure_correlations = pd.DataFrame(seizure_rows)
    patient_trajectory = (
        preictal.groupby(["patient_id", "band", "time_bin", "time_to_onset_midpoint_seconds"], as_index=False)["z_score"]
        .median()
    )
    patient_rows = [
        _correlation_row(group, {"patient_id": patient, "band": band})
        for (patient, band), group in patient_trajectory.groupby(["patient_id", "band"])
    ]
    patient_correlations = pd.DataFrame(patient_rows)
    seizure_summary = correlation_summary(seizure_correlations, "seizure")
    patient_summary = correlation_summary(patient_correlations, "patient")
    return seizure_correlations, patient_trajectory, patient_correlations, pd.concat(
        [seizure_summary, patient_summary], ignore_index=True
    )


def _distribution_metrics(values: pd.Series, prefix: str) -> dict[str, float | int]:
    values = values.dropna()
    count = len(values)
    return {
        f"{prefix}_n": count,
        f"{prefix}_median": float(values.median()) if count else np.nan,
        f"{prefix}_iqr_low": float(values.quantile(0.25)) if count else np.nan,
        f"{prefix}_iqr_high": float(values.quantile(0.75)) if count else np.nan,
        f"{prefix}_positive_n": int((values > 0).sum()),
        f"{prefix}_positive_percent": float(100 * (values > 0).mean()) if count else np.nan,
        f"{prefix}_negative_n": int((values < 0).sum()),
        f"{prefix}_negative_percent": float(100 * (values < 0).mean()) if count else np.nan,
    }


def correlation_summary(correlations: pd.DataFrame, level: str) -> pd.DataFrame:
    rows = []
    for band, group in correlations.groupby("band", sort=False):
        rows.append(
            {
                "level": level,
                "band": band,
                **_distribution_metrics(group["pearson_r"], "pearson_r"),
                **_distribution_metrics(group["spearman_rho"], "spearman_rho"),
            }
        )
    return pd.DataFrame(rows)


def _sustained_change(group: pd.DataFrame) -> tuple[bool, float]:
    """Find the first midpoint of a pair of adjacent abnormal windows."""
    group = group.sort_values("time_bin")
    bins = group["time_bin"].to_numpy(dtype=int)
    mids = group["time_to_onset_midpoint_seconds"].to_numpy(dtype=float)
    abnormal = group["z_score"].abs().to_numpy(dtype=float) > 2
    finite = np.isfinite(group["z_score"].to_numpy(dtype=float))
    for index in range(len(group) - 1):
        if (
            bins[index + 1] == bins[index] + 1
            and finite[index]
            and finite[index + 1]
            and abnormal[index]
            and abnormal[index + 1]
        ):
            return True, float(mids[index])
    return False, np.nan


def trajectory_and_sustained_summary(preictal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Patient-equal trajectory summaries and sustained-abnormality summaries."""
    patient_trajectory = (
        preictal.groupby(["patient_id", "band", "time_bin", "time_to_onset_midpoint_seconds"], as_index=False)["z_score"]
        .median()
    )
    trajectory_rows = []
    for (band, time_bin, midpoint), group in patient_trajectory.groupby(
        ["band", "time_bin", "time_to_onset_midpoint_seconds"], sort=False
    ):
        values = group["z_score"].dropna()
        trajectory_rows.append(
            {
                "band": band,
                "time_bin": time_bin,
                "time_to_onset_midpoint_seconds": midpoint,
                "n_patients": len(values),
                "patient_equal_median_z": float(values.median()) if len(values) else np.nan,
                "patient_iqr_low_z": float(values.quantile(0.25)) if len(values) else np.nan,
                "patient_iqr_high_z": float(values.quantile(0.75)) if len(values) else np.nan,
                "pooled_seizure_median_z": float(
                    preictal.loc[
                        (preictal["band"] == band) & (preictal["time_bin"] == time_bin), "z_score"
                    ].median()
                ),
            }
        )
    trajectory = pd.DataFrame(trajectory_rows)

    seizure_rows = []
    for (patient, event, band), group in preictal.groupby(["patient_id", "event_id", "band"]):
        sustained, first_time = _sustained_change(group)
        seizure_rows.append(
            {
                "patient_id": patient,
                "event_id": event,
                "band": band,
                "sustained_abnormal": sustained,
                "first_sustained_midpoint_seconds": first_time,
                "n_valid_windows": int(group["z_score"].notna().sum()),
            }
        )
    sustained_by_seizure = pd.DataFrame(seizure_rows)
    sustained_rows = []
    for band, group in sustained_by_seizure.groupby("band", sort=False):
        valid = group[group["n_valid_windows"] >= 2]
        sustained = valid[valid["sustained_abnormal"]]
        patient_any = (
            valid.groupby("patient_id")["sustained_abnormal"].any()
            if len(valid)
            else pd.Series(dtype=bool)
        )
        band_trajectory = trajectory[trajectory["band"] == band]
        absolute_median = band_trajectory["patient_equal_median_z"].abs()
        max_index = absolute_median.idxmax() if absolute_median.notna().any() else None
        sustained_rows.append(
            {
                "band": band,
                "valid_seizure_count": len(valid),
                "seizures_with_sustained_change_n": len(sustained),
                "seizures_with_sustained_change_percent": 100 * len(sustained) / len(valid) if len(valid) else np.nan,
                "median_first_sustained_midpoint_seconds": float(sustained["first_sustained_midpoint_seconds"].median()) if len(sustained) else np.nan,
                "valid_patient_count": len(patient_any),
                "patients_with_any_sustained_change_n": int(patient_any.sum()),
                "patients_with_any_sustained_change_percent": 100 * float(patient_any.mean()) if len(patient_any) else np.nan,
                "max_abs_patient_equal_median_z": float(absolute_median.max()) if max_index is not None else np.nan,
                "max_abs_change_midpoint_seconds": float(band_trajectory.loc[max_index, "time_to_onset_midpoint_seconds"]) if max_index is not None else np.nan,
            }
        )
    return trajectory, sustained_by_seizure, pd.DataFrame(sustained_rows)


def plot_trajectories(
    preictal: pd.DataFrame,
    patient_trajectory: pd.DataFrame,
    trajectory: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Make one requested patient-aware trajectory figure per frequency band."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    patient_ids = sorted(preictal["patient_id"].dropna().unique())
    patient_colors = {
        patient: plt.get_cmap("tab20")(index % 20)
        for index, patient in enumerate(patient_ids)
    }
    for band in BAND_ORDER:
        fig, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
        data = preictal[preictal["band"] == band]
        patient_data = patient_trajectory[patient_trajectory["band"] == band]
        summary = trajectory[trajectory["band"] == band].sort_values("time_to_onset_midpoint_seconds")
        for _, seizure in data.groupby(["patient_id", "event_id"], sort=False):
            seizure = seizure.sort_values("time_to_onset_midpoint_seconds")
            axis.plot(
                seizure["time_to_onset_midpoint_seconds"],
                seizure["z_score"],
                color="0.60",
                alpha=0.18,
                linewidth=0.8,
                zorder=1,
            )
            inside = seizure["z_score"].abs() <= 2
            outside = seizure["z_score"].abs() > 2
            axis.scatter(
                seizure.loc[inside, "time_to_onset_midpoint_seconds"],
                seizure.loc[inside, "z_score"],
                s=11,
                marker="o",
                color="0.50",
                alpha=0.22,
                linewidths=0,
                zorder=2,
            )
            axis.scatter(
                seizure.loc[outside, "time_to_onset_midpoint_seconds"],
                seizure.loc[outside, "z_score"],
                s=25,
                marker="X",
                color="#c83e4d",
                alpha=0.65,
                linewidths=0.25,
                zorder=3,
            )
        for patient, path in patient_data.groupby("patient_id", sort=False):
            path = path.sort_values("time_to_onset_midpoint_seconds")
            axis.plot(
                path["time_to_onset_midpoint_seconds"],
                path["z_score"],
                color=patient_colors[patient],
                alpha=0.75,
                linewidth=1.4,
                zorder=4,
            )
        if not summary.empty:
            x = summary["time_to_onset_midpoint_seconds"].to_numpy(dtype=float)
            median = summary["patient_equal_median_z"].to_numpy(dtype=float)
            lower = summary["patient_iqr_low_z"].to_numpy(dtype=float)
            upper = summary["patient_iqr_high_z"].to_numpy(dtype=float)
            axis.fill_between(x, lower, upper, color="#1f4e79", alpha=0.20, label="Patient IQR", zorder=5)
            axis.plot(x, median, color="#0b1f33", marker="D", markersize=5, linewidth=2.7, label="Overall patient-equal median", zorder=6)
        axis.axhline(0, color="black", linewidth=1.1, linestyle="-", label="Interictal mean (z = 0)")
        axis.axhline(2, color="#c83e4d", linewidth=1.0, linestyle="--", label="Abnormal threshold (|z| > 2)")
        axis.axhline(-2, color="#c83e4d", linewidth=1.0, linestyle="--")
        axis.set(
            xlim=(-60, 0),
            xticks=WINDOW_MIDPOINTS,
            xlabel="Seconds before annotated seizure onset (window midpoint)",
            ylabel="Standardized change from patient-specific interictal baseline (z)",
            title=f"{band.capitalize()} relative-power trajectory in the final 60 s before seizure onset",
        )
        axis.grid(axis="x", color="0.90", linewidth=0.8)
        axis.legend(loc="best", frameon=True, fontsize=9)
        fig.savefig(figures_dir / f"preictal_{band}_trajectory.png", dpi=220)
        plt.close(fig)


def _fmt(value: object, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "not estimable"
    return f"{float(value):.{digits}f}"


def write_report(
    path: Path,
    settings: dict[str, object],
    event_plan: pd.DataFrame,
    metadata: pd.DataFrame,
    baselines: pd.DataFrame,
    correlation_summary_table: pd.DataFrame,
    trajectory: pd.DataFrame,
    sustained: pd.DataFrame,
) -> None:
    """Write a concise interpretation-ready report alongside the CSV outputs."""
    analyzed = event_plan[event_plan["eligible_preictal"]]
    lines = [
        "# Final-60-second EEG frequency-band analysis",
        "",
        "## Design and verification",
        "",
        f"- **Analyzed seizures:** {len(analyzed)} from {analyzed['patient_id'].nunique()} patients.",
        f"- **Preictal interval:** ten consecutive {WINDOW_SECONDS}-second windows from -60 to 0 seconds; plotted at midpoints -57 to -3 seconds.",
        "- **Signal calculation:** every sample in every window was detrended, robustly common-median referenced, quality screened, notched at 60 Hz when possible, and analyzed with 4-second Hann Welch segments with 50% overlap.",
        "- **Band powers:** calculated per usable EEG channel, transformed to channel-relative power, then aggregated by the median across channels.",
        f"- **Interictal controls:** clean, non-overlapping 60-second blocks at least {settings['interictal']['buffer_minutes']} minutes from every annotated seizure; their 6-second windows create one patient-and-band baseline.",
        "- **Filter validity:** an EDF channel was analyzed for a band only if its declared high-pass cutoff was at or below the band lower edge and its low-pass cutoff was at or above the band upper edge. Delta is therefore only reported when 1-4 Hz was preserved.",
        "- **Dependence warning:** neighboring windows from the same seizure and seizures from the same patient are correlated. The total number of window rows is not the statistical sample size; correlations are calculated within seizure and patient trajectories are summarized with equal patient weight.",
        "",
        "## Recording checks",
        "",
        f"All {len(metadata)} EDF recordings were audited for EEG sampling rate, units, and acquisition-filter passband. Sampling-rate and filter details are in `recording_metadata_audit.csv`; resolved onset annotations and any exclusions are in `seizure_annotation_audit.csv`.",
        "",
        "## Band-level findings",
        "",
        "| Band | Valid seizure paths | Seizure Pearson r, median [IQR] | Patient Pearson r, median [IQR] | Max |patient-median z| | Sustained seizures | Median first sustained time | Patients with sustained change |",
        "|---|---:|---|---|---:|---:|---|---:|",
    ]
    for band in BAND_ORDER:
        seizure_corr = correlation_summary_table[
            (correlation_summary_table["level"] == "seizure") & (correlation_summary_table["band"] == band)
        ]
        patient_corr = correlation_summary_table[
            (correlation_summary_table["level"] == "patient") & (correlation_summary_table["band"] == band)
        ]
        sustained_row = sustained[sustained["band"] == band]
        sc = seizure_corr.iloc[0] if len(seizure_corr) else pd.Series(dtype=float)
        pc = patient_corr.iloc[0] if len(patient_corr) else pd.Series(dtype=float)
        ss = sustained_row.iloc[0] if len(sustained_row) else pd.Series(dtype=float)
        lines.append(
            "| "
            + " | ".join(
                [
                    band.capitalize(),
                    str(int(ss.get("valid_seizure_count", 0))),
                    f"{_fmt(sc.get('pearson_r_median'))} [{_fmt(sc.get('pearson_r_iqr_low'))}, {_fmt(sc.get('pearson_r_iqr_high'))}]",
                    f"{_fmt(pc.get('pearson_r_median'))} [{_fmt(pc.get('pearson_r_iqr_low'))}, {_fmt(pc.get('pearson_r_iqr_high'))}]",
                    _fmt(ss.get("max_abs_patient_equal_median_z")),
                    f"{int(ss.get('seizures_with_sustained_change_n', 0))}/{int(ss.get('valid_seizure_count', 0))} ({_fmt(ss.get('seizures_with_sustained_change_percent'), 1)}%)",
                    _fmt(ss.get("median_first_sustained_midpoint_seconds"), 1) + " s",
                    f"{int(ss.get('patients_with_any_sustained_change_n', 0))}/{int(ss.get('valid_patient_count', 0))} ({_fmt(ss.get('patients_with_any_sustained_change_percent'), 1)}%)",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Negative correlations mean standardized relative power tended to fall as the annotated onset approached (time moves from -57 to -3 s); positive correlations mean it tended to rise. Spearman results are reported alongside Pearson results in `correlation_summary.csv`.",
            "",
            "## Patient-equal median standardized power by 6-second window",
            "",
            "Each entry is the median of the patient-level median values at that window, so each patient has equal weight. The accompanying patient IQR is in `trajectory_summary.csv` and shown in the plots.",
            "",
            "| Band | -57 | -51 | -45 | -39 | -33 | -27 | -21 | -15 | -9 | -3 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for band in BAND_ORDER:
        values = trajectory[trajectory["band"] == band].set_index("time_to_onset_midpoint_seconds")["patient_equal_median_z"]
        lines.append(
            "| " + band.capitalize() + " | " + " | ".join(
                _fmt(values.get(float(midpoint))) for midpoint in WINDOW_MIDPOINTS
            ) + " |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `preictal_window_features.csv`: one band row per seizure-window, with patient ID, seizure ID, QC, source recording, patient baseline, and z score.",
            "- `patient_band_baselines.csv`: patient-specific interictal means, SDs, and clean-window counts.",
            "- `seizure_correlations.csv`, `patient_correlations.csv`, and `correlation_summary.csv`: repeated-measures-aware temporal correlations.",
            "- `trajectory_summary.csv` and `sustained_change_summary.csv`: requested median/IQR and abnormal-run measures.",
            "- `figures/preictal_<band>_trajectory.png`: one trajectory graph for each band.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument(
        "--processed-dir", type=Path, default=ROOT / "data" / "processed" / "preictal_60s_zscore"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=ROOT / "results" / "preictal_60s_zscore"
    )
    parser.add_argument("--interictal-buffer-minutes", type=int, default=INTERICTAL_BUFFER_MINUTES)
    parser.add_argument("--blocks-per-seizure", type=int, default=BLOCKS_PER_SEIZURE)
    parser.add_argument("--min-channels", type=int, default=MIN_USABLE_CHANNELS)
    args = parser.parse_args()
    if args.interictal_buffer_minutes < 0 or args.blocks_per_seizure < 1 or args.min_channels < 1:
        parser.error("Buffer must be nonnegative; blocks-per-seizure and min-channels must be positive")

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    metadata = recording_metadata_audit(args.raw_dir)
    events, intervals_by_file = load_study(args.raw_dir)
    event_plan = plan_events(events, intervals_by_file)
    analysis_events = [
        event for event in events if event.event_id in set(event_plan.loc[event_plan["eligible_preictal"], "event_id"])
    ]
    controls, no_controls = choose_interictal_blocks(
        analysis_events,
        intervals_by_file,
        args.interictal_buffer_minutes * 60,
        args.blocks_per_seizure,
    )
    if no_controls:
        event_plan.loc[event_plan["patient_id"].isin(no_controls), "eligible_preictal"] = False
        event_plan.loc[event_plan["patient_id"].isin(no_controls), "exclusion_reason"] = "no_clean_patient_interictal_blocks"
        analysis_events = [event for event in analysis_events if event.patient_id not in no_controls]
        controls = [control for control in controls if control.patient_id not in no_controls]
    if not analysis_events:
        raise RuntimeError("No seizures have both an eligible preictal minute and patient controls")

    print(f"Audited {len(metadata)} EDF recordings and {len(events)} annotations.", flush=True)
    print(f"Analyzing {len(analysis_events)} seizures and {len(controls)} patient-level baseline blocks.", flush=True)
    features = extract_features(analysis_events, controls, args.min_channels)
    features, baselines = add_patient_baselines(features)
    preictal = features[(features["condition"] == "preictal") & features["z_score"].notna()].copy()
    seizure_correlations, patient_trajectory, patient_correlations, correlation_summary_table = correlation_tables(preictal)
    trajectory, sustained_by_seizure, sustained_summary = trajectory_and_sustained_summary(preictal)
    plot_trajectories(preictal, patient_trajectory, trajectory, args.results_dir / "figures")

    controls_frame = pd.DataFrame([asdict(control) for control in controls])
    controls_frame["end_seconds"] = controls_frame["start_seconds"] + INTERICTAL_BLOCK_SECONDS
    controls_frame["seizure_exclusion_buffer_seconds"] = args.interictal_buffer_minutes * 60
    controls_frame["no_overlap_with_annotated_seizure_or_preictal_window"] = True
    metadata.to_csv(args.results_dir / "recording_metadata_audit.csv", index=False)
    event_plan.to_csv(args.results_dir / "seizure_annotation_audit.csv", index=False)
    controls_frame.to_csv(args.results_dir / "interictal_baseline_blocks.csv", index=False)
    baselines.to_csv(args.results_dir / "patient_band_baselines.csv", index=False)
    features.to_csv(args.processed_dir / "all_window_features.csv", index=False)
    features[features["condition"] == "preictal"].to_csv(
        args.processed_dir / "preictal_window_features.csv", index=False
    )
    seizure_correlations.to_csv(args.results_dir / "seizure_correlations.csv", index=False)
    patient_trajectory.to_csv(args.results_dir / "patient_median_trajectories.csv", index=False)
    patient_correlations.to_csv(args.results_dir / "patient_correlations.csv", index=False)
    correlation_summary_table.to_csv(args.results_dir / "correlation_summary.csv", index=False)
    trajectory.to_csv(args.results_dir / "trajectory_summary.csv", index=False)
    sustained_by_seizure.to_csv(args.results_dir / "sustained_change_by_seizure.csv", index=False)
    sustained_summary.to_csv(args.results_dir / "sustained_change_summary.csv", index=False)
    settings = {
        "bands_hz": BANDS,
        "preictal_seconds": PREICTAL_SECONDS,
        "window_seconds": WINDOW_SECONDS,
        "window_midpoints_seconds": WINDOW_MIDPOINTS.tolist(),
        "welch": {
            "window": "Hann",
            "segment_seconds": WELCH_SEGMENT_SECONDS,
            "overlap_percent": 50,
            "scaling": "density",
        },
        "artifact_handling": {
            "minimum_channels_per_band_window": args.min_channels,
            "minimum_channel_std_uv": ARTIFACT_MIN_STD_UV,
            "maximum_channel_absolute_uv": ARTIFACT_MAX_ABS_UV,
            "maximum_clipped_fraction": ARTIFACT_MAX_CLIPPED_FRACTION,
        },
        "reference": "common median across EEG-labelled channels per window",
        "line_noise_filter": "60 Hz IIR notch (Q=30) when sampling rate permits",
        "relative_power": "per-channel band power divided by summed power in EDF-filter-valid requested bands; median across valid channels",
        "filter_validity": "declared HP <= band lower edge and declared LP >= band upper edge, plus Nyquist >= band upper edge",
        "interictal": {
            "block_seconds": INTERICTAL_BLOCK_SECONDS,
            "window_seconds": WINDOW_SECONDS,
            "buffer_minutes": args.interictal_buffer_minutes,
            "blocks_per_seizure_target": args.blocks_per_seizure,
            "selection": "deterministic evenly-spaced sample of clean patient-specific 60-s candidate blocks",
        },
        "normalization": "z = (window relative power - patient interictal mean) / patient interictal SD",
        "annotated_seizures": len(events),
        "analyzed_seizures": len(analysis_events),
        "analyzed_patients": len({event.patient_id for event in analysis_events}),
        "baseline_blocks": len(controls),
    }
    (args.results_dir / "analysis_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    write_report(
        args.results_dir / "PREICTAL_60S_ANALYSIS_REPORT.md",
        settings,
        event_plan,
        metadata,
        baselines,
        correlation_summary_table,
        trajectory,
        sustained_summary,
    )
    print(f"Completed: {len(preictal)} valid preictal band-window values from {len(analysis_events)} seizures.")
    print(f"Results: {args.results_dir}")


if __name__ == "__main__":
    main()
