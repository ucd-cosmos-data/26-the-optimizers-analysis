"""Extract onset-aligned EEG band powers and answer the preictal research question.

The analysis unit is a 10-second window. For every usable seizure, the script
extracts the six windows from -60 seconds to seizure onset and patient-matched
interictal control blocks. It then writes tidy features, temporal summaries,
and consistency summaries. Raw EDF files are read in place and never changed.

Run from the repository root:
    python final_project/scripts/analyze_preictal_bandpower.py
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats


ROOT = Path(__file__).resolve().parents[1]
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}
BAND_ORDER = list(BANDS)
DEFAULT_WINDOW_SECONDS = 10
DEFAULT_PREICTAL_SECONDS = 60
DEFAULT_INTERICTAL_BUFFER_MINUTES = 15
DEFAULT_CONTROLS_PER_SEIZURE = 3
DEFAULT_MIN_CHANNELS = 10
ARTIFACT_MAX_ABS_UV = 1_000.0
ARTIFACT_MIN_STD_UV = 0.5
ARTIFACT_MAX_CLIPPED_FRACTION = 0.01


def _decode(value: bytes) -> str:
    return value.decode("ascii", errors="ignore").strip().replace("\x00", "")


@dataclass(frozen=True)
class EDFHeader:
    path: Path
    header_bytes: int
    n_records: int
    record_seconds: float
    labels: tuple[str, ...]
    samples_per_record: np.ndarray
    physical_min: np.ndarray
    physical_max: np.ndarray
    digital_min: np.ndarray
    digital_max: np.ndarray
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
class ControlBlock:
    patient_id: str
    event_id: str
    control_id: int
    edf_path: Path
    start_seconds: int


def read_edf_header(path: Path) -> EDFHeader:
    """Read the EDF fields needed for random-access signal extraction."""
    with path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) != 256:
            raise ValueError(f"Incomplete EDF header: {path}")
        header_bytes = int(_decode(fixed[184:192]))
        n_records = int(_decode(fixed[236:244]))
        record_seconds = float(_decode(fixed[244:252]))
        n_signals = int(_decode(fixed[252:256]))
        labels = tuple(_decode(handle.read(16)) for _ in range(n_signals))
        handle.seek(80 * n_signals, 1)
        handle.seek(8 * n_signals, 1)
        physical_min = np.array([float(_decode(handle.read(8))) for _ in range(n_signals)])
        physical_max = np.array([float(_decode(handle.read(8))) for _ in range(n_signals)])
        digital_min = np.array([float(_decode(handle.read(8))) for _ in range(n_signals)])
        digital_max = np.array([float(_decode(handle.read(8))) for _ in range(n_signals)])
        handle.seek(80 * n_signals, 1)
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
        samples_per_record=samples_per_record,
        physical_min=physical_min,
        physical_max=physical_max,
        digital_min=digital_min,
        digital_max=digital_max,
        bytes_per_record=int(samples_per_record.sum() * 2),
    )


def _parse_clock(text: str) -> int | None:
    # One supplied PN10 line writes 16 as "1 6"; accept that known spacing typo.
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
    return hour * 3600 + minute * 60 + second


def parse_annotations(path: Path) -> dict[str, list[tuple[int, int]]]:
    """Return seizure intervals as recording-relative seconds, grouped by filename."""
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
    """Load all seizure events and a complete recording-to-interval map."""
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
            validated_intervals = []
            for onset, end in intervals:
                note = ""
                # PN00 seizure 3 has an end-hour typo (19 instead of 18). Correct
                # only the unambiguous one-hour case and record the correction.
                if end > duration and onset < end - 3_600 <= duration:
                    end -= 3_600
                    note = "end_time_minus_one_hour_to_fit_recording"
                validated_intervals.append((onset, end))
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
            intervals_by_file[edf_path].extend(validated_intervals)
    if not events:
        raise FileNotFoundError(f"No seizure events found beneath {raw_dir}")
    return events, intervals_by_file


def plan_events(
    events: list[SeizureEvent],
    intervals_by_file: dict[Path, list[tuple[int, int]]],
    preictal_seconds: int,
) -> pd.DataFrame:
    """Check that each seizure has a clean, complete preictal interval."""
    rows = []
    for event in events:
        header = read_edf_header(event.edf_path)
        preictal_start = event.onset_seconds - preictal_seconds
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
        elif overlaps_other:
            reason = "preictal_period_overlaps_another_seizure"
        rows.append(
            {
                "patient_id": event.patient_id,
                "event_id": event.event_id,
                "recording": event.edf_path.name,
                "onset_seconds": event.onset_seconds,
                "seizure_duration_seconds": event.end_seconds - event.onset_seconds,
                "annotation_note": event.annotation_note,
                "eligible_preictal": not reason,
                "exclusion_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def choose_controls(
    eligible_events: list[SeizureEvent],
    intervals_by_file: dict[Path, list[tuple[int, int]]],
    block_seconds: int,
    buffer_seconds: int,
    controls_per_seizure: int,
    seed: int,
) -> tuple[list[ControlBlock], set[str]]:
    """Select non-overlapping, patient-matched blocks far from annotated seizures."""
    rng = np.random.default_rng(seed)
    by_patient: dict[str, list[SeizureEvent]] = {}
    for event in eligible_events:
        by_patient.setdefault(event.patient_id, []).append(event)

    selected: list[ControlBlock] = []
    events_without_controls: set[str] = set()
    for patient_id, patient_events in sorted(by_patient.items()):
        patient_files = sorted(
            path for path in intervals_by_file if path.parent.name == patient_id
        )
        candidates: list[tuple[Path, int]] = []
        for edf_path in patient_files:
            duration = int(read_edf_header(edf_path).duration_seconds)
            intervals = intervals_by_file[edf_path]
            for start in range(0, duration - block_seconds + 1, block_seconds):
                end = start + block_seconds
                far_from_seizures = all(
                    end <= seizure_start - buffer_seconds
                    or start >= seizure_end + buffer_seconds
                    for seizure_start, seizure_end in intervals
                )
                if far_from_seizures:
                    candidates.append((edf_path, start))
        rng.shuffle(candidates)
        used: set[tuple[Path, int]] = set()
        for event in patient_events:
            same_recording = [
                item for item in candidates if item[0] == event.edf_path and item not in used
            ]
            other_recording = [
                item for item in candidates if item[0] != event.edf_path and item not in used
            ]
            choices = same_recording + other_recording
            if len(choices) < controls_per_seizure:
                events_without_controls.add(event.event_id)
                continue
            for control_id, (edf_path, start) in enumerate(
                choices[:controls_per_seizure], start=1
            ):
                used.add((edf_path, start))
                selected.append(
                    ControlBlock(
                        patient_id=patient_id,
                        event_id=event.event_id,
                        control_id=control_id,
                        edf_path=edf_path,
                        start_seconds=start,
                    )
                )
    return selected, events_without_controls


def _is_eeg(label: str) -> bool:
    return label.strip().upper().startswith("EEG ")


def read_eeg_window(
    path: Path, start_seconds: int, duration_seconds: int
) -> tuple[np.ndarray, float, list[str], EDFHeader]:
    """Read one record-aligned EEG window and scale it to physical units."""
    header = read_edf_header(path)
    eeg_indices = np.array(
        [index for index, label in enumerate(header.labels) if _is_eeg(label)], dtype=int
    )
    if not len(eeg_indices):
        raise ValueError(f"No EEG-labelled channels in {path}")
    rates = header.samples_per_record[eeg_indices] / header.record_seconds
    if not np.allclose(rates, rates[0]):
        raise ValueError(f"EEG channels have unequal sample rates in {path}")
    if not np.all(header.samples_per_record == header.samples_per_record[0]):
        raise ValueError(f"Random reader requires equal per-signal sample counts in {path}")
    if start_seconds % header.record_seconds or duration_seconds % header.record_seconds:
        raise ValueError("Requested window must align with EDF data records")
    start_record = int(start_seconds / header.record_seconds)
    n_records = int(duration_seconds / header.record_seconds)
    if start_record < 0 or start_record + n_records > header.n_records:
        raise ValueError(f"Requested window lies outside {path.name}")

    with path.open("rb") as handle:
        handle.seek(header.header_bytes + start_record * header.bytes_per_record)
        block = np.frombuffer(
            handle.read(n_records * header.bytes_per_record), dtype="<i2"
        )
    expected = n_records * header.bytes_per_record // 2
    if block.size != expected:
        raise ValueError(f"Short signal read in {path.name}")
    block = block.reshape(n_records, len(header.labels), -1)
    raw = block[:, eeg_indices, :].transpose(1, 0, 2).reshape(len(eeg_indices), -1)
    dmin = header.digital_min[eeg_indices, None]
    dmax = header.digital_max[eeg_indices, None]
    pmin = header.physical_min[eeg_indices, None]
    pmax = header.physical_max[eeg_indices, None]
    data = raw.astype(np.float64) * ((pmax - pmin) / (dmax - dmin))
    data += pmin - dmin * ((pmax - pmin) / (dmax - dmin))
    labels = [header.labels[index] for index in eeg_indices]
    return data, float(rates[0]), labels, header


def window_bandpowers(
    data: np.ndarray,
    sample_rate: float,
    physical_min: np.ndarray,
    physical_max: np.ndarray,
    min_channels: int,
) -> tuple[dict[str, float], dict[str, float | int | bool]]:
    """Return robust scalp-level relative powers and quality information."""
    original = data
    data = signal.detrend(data, axis=-1, type="linear")
    # Robust common-median reference reduces shared offsets while tolerating bad leads.
    data = data - np.median(data, axis=0, keepdims=True)
    std = np.std(data, axis=1)
    max_abs = np.max(np.abs(data), axis=1)
    tolerance = np.maximum((physical_max - physical_min) * 0.001, 1e-6)
    clipped = np.mean(
        (original <= physical_min[:, None] + tolerance[:, None])
        | (original >= physical_max[:, None] - tolerance[:, None]),
        axis=1,
    )
    usable = (
        np.isfinite(data).all(axis=1)
        & (std >= ARTIFACT_MIN_STD_UV)
        & (max_abs <= ARTIFACT_MAX_ABS_UV)
        & (clipped <= ARTIFACT_MAX_CLIPPED_FRACTION)
    )
    total_channels = int(data.shape[0])
    usable_channels = int(usable.sum())
    qc = {
        "eeg_channel_count": total_channels,
        "usable_channel_count": usable_channels,
        "rejected_channel_count": total_channels - usable_channels,
        "artifact_channel_fraction": (total_channels - usable_channels) / total_channels,
        "qc_pass": usable_channels >= min_channels,
    }
    if usable_channels < min_channels:
        return {band: np.nan for band in BANDS}, qc
    clean = data[usable]
    if sample_rate > 122:
        notch_b, notch_a = signal.iirnotch(60.0, 30.0, fs=sample_rate)
        clean = signal.filtfilt(notch_b, notch_a, clean, axis=-1)
    nperseg = min(int(round(4 * sample_rate)), clean.shape[1])
    frequencies, psd = signal.welch(
        clean,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="linear",
        axis=-1,
        scaling="density",
    )
    powers = {}
    for band, (low, high) in BANDS.items():
        mask = (frequencies >= low) & (frequencies < high)
        powers[band] = np.trapezoid(psd[:, mask], frequencies[mask], axis=1)
    stacked = np.stack([powers[band] for band in BAND_ORDER], axis=1)
    total_power = stacked.sum(axis=1, keepdims=True)
    relative = stacked / np.maximum(total_power, 1e-12)
    # Median across leads prevents a few high-amplitude electrodes dominating the scalp value.
    values = {
        band: float(np.median(relative[:, index]))
        for index, band in enumerate(BAND_ORDER)
    }
    return values, qc


def extract_window(
    edf_path: Path,
    start_seconds: int,
    condition: str,
    patient_id: str,
    event_id: str,
    bin_index: int,
    window_seconds: int,
    preictal_seconds: int,
    control_id: int | None,
    min_channels: int,
) -> list[dict[str, object]]:
    data, sample_rate, _, header = read_eeg_window(
        edf_path, start_seconds, window_seconds
    )
    eeg_indices = np.array(
        [index for index, label in enumerate(header.labels) if _is_eeg(label)], dtype=int
    )
    relative_powers, qc = window_bandpowers(
        data,
        sample_rate,
        header.physical_min[eeg_indices],
        header.physical_max[eeg_indices],
        min_channels,
    )
    relative_start = -preictal_seconds + (bin_index - 1) * window_seconds
    relative_end = relative_start + window_seconds
    return [
        {
            "patient_id": patient_id,
            "event_id": event_id,
            "condition": condition,
            "control_id": control_id,
            "recording": edf_path.name,
            "window_start_seconds": start_seconds,
            "time_to_onset_start": relative_start,
            "time_to_onset_end": relative_end,
            "time_bin": bin_index,
            "band": band,
            "relative_power": relative_powers[band],
            **qc,
        }
        for band in BAND_ORDER
    ]


def extract_features(
    events: list[SeizureEvent],
    controls: list[ControlBlock],
    window_seconds: int,
    preictal_seconds: int,
    min_channels: int,
) -> pd.DataFrame:
    controls_by_event: dict[str, list[ControlBlock]] = {}
    for control in controls:
        controls_by_event.setdefault(control.event_id, []).append(control)
    rows: list[dict[str, object]] = []
    n_bins = preictal_seconds // window_seconds
    for index, event in enumerate(events, start=1):
        print(
            f"[{index}/{len(events)}] {event.event_id}: "
            f"{event.edf_path.name} onset {event.onset_seconds}s",
            flush=True,
        )
        preictal_start = event.onset_seconds - preictal_seconds
        for bin_index in range(1, n_bins + 1):
            start = preictal_start + (bin_index - 1) * window_seconds
            rows.extend(
                extract_window(
                    event.edf_path,
                    start,
                    "preictal",
                    event.patient_id,
                    event.event_id,
                    bin_index,
                    window_seconds,
                    preictal_seconds,
                    None,
                    min_channels,
                )
            )
        for control in controls_by_event[event.event_id]:
            for bin_index in range(1, n_bins + 1):
                start = control.start_seconds + (bin_index - 1) * window_seconds
                rows.extend(
                    extract_window(
                        control.edf_path,
                        start,
                        "interictal",
                        event.patient_id,
                        event.event_id,
                        bin_index,
                        window_seconds,
                        preictal_seconds,
                        control.control_id,
                        min_channels,
                    )
                )
    features = pd.DataFrame(rows)
    valid_controls = features[
        (features["condition"] == "interictal") & features["qc_pass"]
    ]
    baseline = (
        valid_controls.groupby(["event_id", "band"], as_index=False)["relative_power"]
        .mean()
        .rename(columns={"relative_power": "matched_interictal_mean"})
    )
    features = features.merge(baseline, on=["event_id", "band"], how="left")
    features["log2_change_from_interictal"] = np.log2(
        np.maximum(features["relative_power"], 1e-12)
        / np.maximum(features["matched_interictal_mean"], 1e-12)
    )
    return features


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    samples = rng.choice(values, size=(2_000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def temporal_summary(features: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Summarize patients equally so one patient with many seizures cannot dominate."""
    preictal = features[
        (features["condition"] == "preictal")
        & features["qc_pass"]
        & features["log2_change_from_interictal"].notna()
    ]
    patient_means = (
        preictal.groupby(
            ["patient_id", "band", "time_bin", "time_to_onset_start", "time_to_onset_end"],
            as_index=False,
        )["log2_change_from_interictal"]
        .mean()
    )
    rng = np.random.default_rng(seed)
    rows = []
    for keys, group in patient_means.groupby(
        ["band", "time_bin", "time_to_onset_start", "time_to_onset_end"],
        sort=False,
    ):
        band, time_bin, time_start, time_end = keys
        values = group["log2_change_from_interictal"].to_numpy()
        ci_low, ci_high = bootstrap_mean_ci(values, rng)
        rows.append(
            {
                "band": band,
                "time_bin": time_bin,
                "time_to_onset_start": time_start,
                "time_to_onset_end": time_end,
                "n_patients": len(values),
                "patient_mean_log2_change": float(np.mean(values)),
                "patient_sd_log2_change": float(np.std(values, ddof=1))
                if len(values) > 1
                else np.nan,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def _direction_agreement(values: pd.Series) -> float:
    values = values.dropna()
    if values.empty:
        return np.nan
    positive = float((values > 0).mean())
    return max(positive, 1 - positive)


def consistency_summary(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare late-preictal (-20 to 0 s) variability across bands."""
    late = features[
        (features["condition"] == "preictal")
        & features["qc_pass"]
        & (features["time_to_onset_start"] >= -20)
        & features["log2_change_from_interictal"].notna()
    ]
    event_values = (
        late.groupby(["patient_id", "event_id", "band"], as_index=False)[
            "log2_change_from_interictal"
        ]
        .mean()
        .rename(columns={"log2_change_from_interictal": "late_log2_change"})
    )
    patient_values = (
        event_values.groupby(["patient_id", "band"], as_index=False)["late_log2_change"]
        .mean()
    )
    within_patient = (
        event_values.groupby(["patient_id", "band"])["late_log2_change"]
        .agg(
            seizure_count="count",
            within_patient_seizure_sd="std",
            patient_mean="mean",
            positive_fraction=lambda values: float((values > 0).mean()),
        )
        .reset_index()
    )
    within_patient["within_patient_direction_agreement"] = np.maximum(
        within_patient["positive_fraction"], 1 - within_patient["positive_fraction"]
    )
    rows = []
    for band in BAND_ORDER:
        event_band = event_values.loc[
            event_values["band"] == band, "late_log2_change"
        ]
        patient_band = patient_values.loc[
            patient_values["band"] == band, "late_log2_change"
        ]
        within_band = within_patient[within_patient["band"] == band]
        repeated = within_band[within_band["seizure_count"] >= 2]
        rows.append(
            {
                "band": band,
                "n_seizures": len(event_band),
                "n_patients": len(patient_band),
                "n_patients_with_multiple_seizures": len(repeated),
                "mean_late_log2_change": event_band.mean(),
                "pooled_event_sd": event_band.std(ddof=1),
                "pooled_event_median_absolute_deviation": stats.median_abs_deviation(
                    event_band, nan_policy="omit", scale="normal"
                ),
                "mean_within_patient_seizure_sd": repeated[
                    "within_patient_seizure_sd"
                ].mean(),
                "patient_balanced_event_direction_agreement": repeated[
                    "within_patient_direction_agreement"
                ].mean(),
                "patient_sd": patient_band.std(ddof=1),
                "patient_direction_agreement": _direction_agreement(patient_band),
            }
        )
    summary = pd.DataFrame(rows)

    comparisons = []
    for high_band in ("beta", "gamma"):
        high = summary.loc[summary["band"] == high_band].iloc[0]
        for lower_band in ("delta", "theta", "alpha"):
            lower = summary.loc[summary["band"] == lower_band].iloc[0]
            comparisons.append(
                {
                    "higher_frequency_band": high_band,
                    "lower_frequency_band": lower_band,
                    "higher_band_mean_within_patient_sd": high[
                        "mean_within_patient_seizure_sd"
                    ],
                    "lower_band_mean_within_patient_sd": lower[
                        "mean_within_patient_seizure_sd"
                    ],
                    "higher_band_less_variable_within_patients": high[
                        "mean_within_patient_seizure_sd"
                    ]
                    < lower["mean_within_patient_seizure_sd"],
                    "higher_band_patient_sd": high["patient_sd"],
                    "lower_band_patient_sd": lower["patient_sd"],
                    "higher_band_less_variable_across_patients": high["patient_sd"]
                    < lower["patient_sd"],
                }
            )
    comparisons_frame = pd.DataFrame(comparisons)
    return summary, comparisons_frame


def write_outputs(
    processed_dir: Path,
    results_dir: Path,
    event_plan: pd.DataFrame,
    controls: list[ControlBlock],
    features: pd.DataFrame,
    temporal: pd.DataFrame,
    consistency: pd.DataFrame,
    comparisons: pd.DataFrame,
    settings: dict[str, object],
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    event_plan.to_csv(results_dir / "event_inventory.csv", index=False)
    pd.DataFrame(
        [
            {
                "patient_id": control.patient_id,
                "event_id": control.event_id,
                "control_id": control.control_id,
                "recording": control.edf_path.name,
                "start_seconds": control.start_seconds,
            }
            for control in controls
        ]
    ).to_csv(results_dir / "interictal_control_blocks.csv", index=False)
    features.to_csv(processed_dir / "bandpower_features.csv", index=False)
    temporal.to_csv(results_dir / "temporal_summary.csv", index=False)
    consistency.to_csv(results_dir / "consistency_summary.csv", index=False)
    comparisons.to_csv(results_dir / "consistency_comparisons.csv", index=False)
    (results_dir / "analysis_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument(
        "--processed-dir", type=Path, default=ROOT / "data" / "processed"
    )
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--window-seconds", type=int, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument(
        "--preictal-seconds", type=int, default=DEFAULT_PREICTAL_SECONDS
    )
    parser.add_argument(
        "--interictal-buffer-minutes",
        type=int,
        default=DEFAULT_INTERICTAL_BUFFER_MINUTES,
    )
    parser.add_argument(
        "--controls-per-seizure", type=int, default=DEFAULT_CONTROLS_PER_SEIZURE
    )
    parser.add_argument("--min-channels", type=int, default=DEFAULT_MIN_CHANNELS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.preictal_seconds % args.window_seconds:
        parser.error("--preictal-seconds must be divisible by --window-seconds")

    events, intervals_by_file = load_study(args.raw_dir)
    event_plan = plan_events(events, intervals_by_file, args.preictal_seconds)
    eligible_ids = set(
        event_plan.loc[event_plan["eligible_preictal"], "event_id"].tolist()
    )
    preictal_eligible = [event for event in events if event.event_id in eligible_ids]
    controls, no_controls = choose_controls(
        preictal_eligible,
        intervals_by_file,
        args.preictal_seconds,
        args.interictal_buffer_minutes * 60,
        args.controls_per_seizure,
        args.seed,
    )
    if no_controls:
        event_plan.loc[
            event_plan["event_id"].isin(no_controls), "eligible_preictal"
        ] = False
        event_plan.loc[
            event_plan["event_id"].isin(no_controls), "exclusion_reason"
        ] = "insufficient_patient_matched_interictal_time"
    analysis_ids = set(
        event_plan.loc[event_plan["eligible_preictal"], "event_id"].tolist()
    )
    analysis_events = [event for event in events if event.event_id in analysis_ids]
    controls = [control for control in controls if control.event_id in analysis_ids]
    if not analysis_events:
        raise RuntimeError("No seizures have both preictal data and matched controls")

    features = extract_features(
        analysis_events,
        controls,
        args.window_seconds,
        args.preictal_seconds,
        args.min_channels,
    )
    temporal = temporal_summary(features, args.seed)
    consistency, comparisons = consistency_summary(features)
    settings = {
        "research_question": (
            "Temporal ordering and magnitude of relative EEG band-power changes "
            "during the final 60 seconds before seizure onset versus interictal periods"
        ),
        "bands_hz": BANDS,
        "window_seconds": args.window_seconds,
        "preictal_seconds": args.preictal_seconds,
        "interictal_buffer_minutes": args.interictal_buffer_minutes,
        "controls_per_seizure": args.controls_per_seizure,
        "minimum_usable_eeg_channels": args.min_channels,
        "artifact_max_abs_uv": ARTIFACT_MAX_ABS_UV,
        "artifact_min_std_uv": ARTIFACT_MIN_STD_UV,
        "artifact_max_clipped_fraction": ARTIFACT_MAX_CLIPPED_FRACTION,
        "reference": "common median across EEG-labelled channels",
        "line_noise_filter": "60 Hz IIR notch, Q=30",
        "scalp_aggregation": "median channel relative power",
        "normalization": "log2(preictal relative power / matched interictal mean)",
        "random_seed": args.seed,
        "annotated_seizures": len(events),
        "analyzed_seizures": len(analysis_events),
        "analyzed_patients": len({event.patient_id for event in analysis_events}),
    }
    write_outputs(
        args.processed_dir,
        args.results_dir,
        event_plan,
        controls,
        features,
        temporal,
        consistency,
        comparisons,
        settings,
    )
    print(
        f"\nAnalyzed {len(analysis_events)} seizures from "
        f"{settings['analyzed_patients']} patients."
    )
    print(f"Saved window-level features to {args.processed_dir}")
    print(f"Saved research summaries to {args.results_dir}")


if __name__ == "__main__":
    main()
