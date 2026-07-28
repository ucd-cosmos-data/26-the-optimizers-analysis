"""Sample time-resolved band power before every seizure and fit regressions.

For each eligible seizure, this script:

* reads the final 60 preictal seconds;
* estimates a continuous relative-power envelope for the five project bands;
* splits the minute into ten non-overlapping 6-second bins;
* samples ten time points per bin (100 points per seizure);
* expresses each point as log2 deviation from that patient's interictal mean;
* classifies it against that patient's empirical 95% interictal range; and
* fits a descriptive point-level linear regression for each band.

The script uses local EDF bytes when available. If a requested byte range is
not present locally, it retrieves only that range from PhysioNet's public S3
mirror instead of downloading the complete 20.3 GB dataset.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, signal, stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_preictal_bandpower as core  # noqa: E402


DEFAULT_BASE_URL = (
    "https://physionet-open.s3.amazonaws.com/siena-scalp-eeg/1.0.0"
)
DEFAULT_SEED = 42
PREICTAL_SECONDS = 60
N_TIME_BINS = 10
SAMPLES_PER_BIN = 10
INTERICTAL_REFERENCE_FRACTION = 0.95
POWER_SMOOTHING_SECONDS = 1.0

BAND_ORDER = list(core.BANDS)
BAND_COLORS = {
    "delta": "#4C78A8",
    "theta": "#72B7B2",
    "alpha": "#E0AC00",
    "beta": "#F58518",
    "gamma": "#E45756",
}
IN_RANGE_COLOR = "#2A9D8F"
OUT_OF_RANGE_COLOR = "#D1495B"


@dataclass(frozen=True)
class EDFSource:
    """Local/remote source for one EDF recording."""

    patient_id: str
    recording: str
    local_path: Path
    remote_url: str

    @property
    def relative_path(self) -> str:
        return f"{self.patient_id}/{self.recording}"


class EDFRangeReader:
    """Read exact EDF byte ranges, preferring a complete local range."""

    def __init__(self, raw_dir: Path, base_url: str, timeout_seconds: int = 90):
        self.raw_dir = raw_dir
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.remote_bytes = 0
        self.local_bytes = 0

    def source(self, patient_id: str, recording: str) -> EDFSource:
        return EDFSource(
            patient_id=patient_id,
            recording=recording,
            local_path=self.raw_dir / patient_id / recording,
            remote_url=f"{self.base_url}/{patient_id}/{recording}",
        )

    def read(self, source: EDFSource, start: int, length: int) -> bytes:
        if start < 0 or length <= 0:
            raise ValueError("EDF byte ranges require start >= 0 and length > 0")
        end = start + length
        if source.local_path.exists() and source.local_path.stat().st_size >= end:
            with source.local_path.open("rb") as handle:
                handle.seek(start)
                payload = handle.read(length)
            if len(payload) == length:
                self.local_bytes += len(payload)
                return payload

        request = urllib.request.Request(
            source.remote_url,
            headers={
                "Range": f"bytes={start}-{end - 1}",
                "User-Agent": "preictal-bandpower-research/1.0",
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                payload = response.read()
                status = getattr(response, "status", None)
                content_range = response.headers.get("Content-Range", "")
        except urllib.error.URLError as error:
            raise ConnectionError(
                f"Could not read {source.relative_path} from PhysioNet: {error}"
            ) from error

        if len(payload) != length:
            raise IOError(
                f"Expected {length:,} bytes from {source.relative_path}, "
                f"received {len(payload):,} (HTTP {status}, "
                f"Content-Range={content_range!r})"
            )
        if status != 206 and start != 0:
            raise IOError(
                f"Server ignored byte-range request for {source.relative_path} "
                f"(HTTP {status})"
            )
        self.remote_bytes += len(payload)
        return payload


def _parse_repeated_text(
    block: bytes, offset: int, width: int, count: int
) -> tuple[list[str], int]:
    values = [
        core._decode(block[offset + index * width : offset + (index + 1) * width])
        for index in range(count)
    ]
    return values, offset + width * count


def parse_edf_header_bytes(source: EDFSource, block: bytes) -> core.EDFHeader:
    """Parse the EDF metadata used by the project from a complete header."""
    if len(block) < 256:
        raise ValueError(f"Incomplete EDF header for {source.relative_path}")
    header_bytes = int(core._decode(block[184:192]))
    n_records = int(core._decode(block[236:244]))
    record_seconds = float(core._decode(block[244:252]))
    n_signals = int(core._decode(block[252:256]))
    if len(block) != header_bytes:
        raise ValueError(
            f"Expected {header_bytes} header bytes for {source.relative_path}, "
            f"received {len(block)}"
        )

    offset = 256
    labels, offset = _parse_repeated_text(block, offset, 16, n_signals)
    _, offset = _parse_repeated_text(block, offset, 80, n_signals)
    _, offset = _parse_repeated_text(block, offset, 8, n_signals)
    physical_min_text, offset = _parse_repeated_text(block, offset, 8, n_signals)
    physical_max_text, offset = _parse_repeated_text(block, offset, 8, n_signals)
    digital_min_text, offset = _parse_repeated_text(block, offset, 8, n_signals)
    digital_max_text, offset = _parse_repeated_text(block, offset, 8, n_signals)
    _, offset = _parse_repeated_text(block, offset, 80, n_signals)
    samples_text, offset = _parse_repeated_text(block, offset, 8, n_signals)
    _, offset = _parse_repeated_text(block, offset, 32, n_signals)
    if offset != header_bytes:
        raise ValueError(
            f"EDF header layout mismatch for {source.relative_path}: "
            f"parsed {offset} of {header_bytes} bytes"
        )

    samples_per_record = np.array([int(value) for value in samples_text], dtype=int)
    header = core.EDFHeader(
        path=source.local_path,
        header_bytes=header_bytes,
        n_records=n_records,
        record_seconds=record_seconds,
        labels=tuple(labels),
        samples_per_record=samples_per_record,
        physical_min=np.array(physical_min_text, dtype=float),
        physical_max=np.array(physical_max_text, dtype=float),
        digital_min=np.array(digital_min_text, dtype=float),
        digital_max=np.array(digital_max_text, dtype=float),
        bytes_per_record=int(samples_per_record.sum() * 2),
    )
    if (
        header.n_records <= 0
        or header.record_seconds <= 0
        or np.any(header.samples_per_record <= 0)
    ):
        raise ValueError(f"Invalid EDF metadata for {source.relative_path}")
    return header


def get_header(reader: EDFRangeReader, source: EDFSource) -> core.EDFHeader:
    fixed = reader.read(source, 0, 256)
    header_bytes = int(core._decode(fixed[184:192]))
    if header_bytes < 256:
        raise ValueError(f"Invalid header size in {source.relative_path}")
    remainder = reader.read(source, 256, header_bytes - 256)
    return parse_edf_header_bytes(source, fixed + remainder)


def read_eeg_window(
    reader: EDFRangeReader,
    source: EDFSource,
    start_seconds: int,
    duration_seconds: int,
) -> tuple[np.ndarray, float, core.EDFHeader, np.ndarray]:
    """Read and physically scale one record-aligned EEG interval."""
    header = get_header(reader, source)
    eeg_indices = np.array(
        [
            index
            for index, label in enumerate(header.labels)
            if core._is_eeg(label)
        ],
        dtype=int,
    )
    if not len(eeg_indices):
        raise ValueError(f"No EEG-labelled channels in {source.relative_path}")
    rates = header.samples_per_record[eeg_indices] / header.record_seconds
    if not np.allclose(rates, rates[0]):
        raise ValueError(f"Unequal EEG sample rates in {source.relative_path}")
    if not np.all(header.samples_per_record == header.samples_per_record[0]):
        raise ValueError(
            f"Range reader requires equal per-signal samples in "
            f"{source.relative_path}"
        )
    if (
        start_seconds % header.record_seconds
        or duration_seconds % header.record_seconds
    ):
        raise ValueError("Requested window must align with EDF data records")

    start_record = int(start_seconds / header.record_seconds)
    n_records = int(duration_seconds / header.record_seconds)
    if start_record < 0 or start_record + n_records > header.n_records:
        raise ValueError(
            f"Requested {start_seconds}:{start_seconds + duration_seconds}s "
            f"lies outside {source.relative_path}"
        )

    byte_start = header.header_bytes + start_record * header.bytes_per_record
    byte_length = n_records * header.bytes_per_record
    payload = reader.read(source, byte_start, byte_length)
    block = np.frombuffer(payload, dtype="<i2")
    expected = byte_length // 2
    if block.size != expected:
        raise ValueError(f"Short signal read in {source.relative_path}")
    block = block.reshape(n_records, len(header.labels), -1)
    raw = (
        block[:, eeg_indices, :]
        .transpose(1, 0, 2)
        .reshape(len(eeg_indices), -1)
    )

    dmin = header.digital_min[eeg_indices, None]
    dmax = header.digital_max[eeg_indices, None]
    pmin = header.physical_min[eeg_indices, None]
    pmax = header.physical_max[eeg_indices, None]
    scale = (pmax - pmin) / (dmax - dmin)
    data = raw.astype(np.float64) * scale + pmin - dmin * scale
    return data, float(rates[0]), header, eeg_indices


def usable_eeg_channels(
    data: np.ndarray,
    physical_min: np.ndarray,
    physical_max: np.ndarray,
    min_channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the project's artifact screen and return clean re-referenced EEG."""
    original = data
    clean = signal.detrend(data, axis=-1, type="linear")
    clean = clean - np.median(clean, axis=0, keepdims=True)
    standard_deviation = np.std(clean, axis=1)
    maximum_absolute = np.max(np.abs(clean), axis=1)
    tolerance = np.maximum((physical_max - physical_min) * 0.001, 1e-6)
    clipped_fraction = np.mean(
        (original <= physical_min[:, None] + tolerance[:, None])
        | (original >= physical_max[:, None] - tolerance[:, None]),
        axis=1,
    )
    usable = (
        np.isfinite(clean).all(axis=1)
        & (standard_deviation >= core.ARTIFACT_MIN_STD_UV)
        & (maximum_absolute <= core.ARTIFACT_MAX_ABS_UV)
        & (clipped_fraction <= core.ARTIFACT_MAX_CLIPPED_FRACTION)
    )
    if int(usable.sum()) < min_channels:
        raise ValueError(
            f"Only {int(usable.sum())} EEG channels passed QC; "
            f"{min_channels} required"
        )
    return clean[usable], usable


def time_resolved_relative_power(
    data: np.ndarray,
    sample_rate: float,
    physical_min: np.ndarray,
    physical_max: np.ndarray,
    min_channels: int,
    smoothing_seconds: float,
) -> tuple[dict[str, np.ndarray], int]:
    """Estimate continuous relative band-power envelopes at the EEG sample rate."""
    clean, usable = usable_eeg_channels(
        data, physical_min, physical_max, min_channels
    )
    if sample_rate > 122:
        notch_b, notch_a = signal.iirnotch(60.0, 30.0, fs=sample_rate)
        clean = signal.filtfilt(notch_b, notch_a, clean, axis=-1)

    smoothing_samples = max(1, int(round(smoothing_seconds * sample_rate)))
    smoothed_absolute: list[np.ndarray] = []
    for low, high in core.BANDS.values():
        sos = signal.butter(
            4, [low, high], btype="bandpass", fs=sample_rate, output="sos"
        )
        filtered = signal.sosfiltfilt(sos, clean, axis=-1)
        instantaneous_power = np.square(
            np.abs(signal.hilbert(filtered, axis=-1))
        )
        smoothed = ndimage.uniform_filter1d(
            instantaneous_power,
            size=smoothing_samples,
            axis=-1,
            mode="reflect",
        )
        smoothed_absolute.append(smoothed.astype(np.float32))

    total = np.maximum(np.sum(smoothed_absolute, axis=0), 1e-12)
    relative = {
        band: np.median(smoothed_absolute[index] / total, axis=0)
        for index, band in enumerate(BAND_ORDER)
    }
    for band, values in relative.items():
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError(f"Invalid {band} relative-power envelope")
    return relative, int(usable.sum())


def stratified_sample_indices(
    sample_rate: float,
    rng: np.random.Generator,
    n_bins: int = N_TIME_BINS,
    samples_per_bin: int = SAMPLES_PER_BIN,
) -> list[tuple[int, int]]:
    """Choose an equal number of distinct sample indices from every time bin."""
    bin_seconds = PREICTAL_SECONDS / n_bins
    selected: list[tuple[int, int]] = []
    for bin_index in range(n_bins):
        start = int(round(bin_index * bin_seconds * sample_rate))
        end = int(round((bin_index + 1) * bin_seconds * sample_rate))
        if end - start < samples_per_bin:
            raise ValueError("Too few EEG samples for requested stratified sampling")
        indices = np.sort(
            rng.choice(
                np.arange(start, end, dtype=int),
                size=samples_per_bin,
                replace=False,
            )
        )
        selected.extend((bin_index + 1, int(index)) for index in indices)
    return selected


def process_block(
    reader: EDFRangeReader,
    patient_id: str,
    recording: str,
    start_seconds: int,
    min_channels: int,
    smoothing_seconds: float,
) -> tuple[dict[str, np.ndarray], float, int]:
    source = reader.source(patient_id, recording)
    data, sample_rate, header, eeg_indices = read_eeg_window(
        reader, source, start_seconds, PREICTAL_SECONDS
    )
    relative, usable_count = time_resolved_relative_power(
        data,
        sample_rate,
        header.physical_min[eeg_indices],
        header.physical_max[eeg_indices],
        min_channels,
        smoothing_seconds,
    )
    expected_samples = int(round(PREICTAL_SECONDS * sample_rate))
    if any(len(values) != expected_samples for values in relative.values()):
        raise ValueError(f"Unexpected envelope length for {source.relative_path}")
    return relative, sample_rate, usable_count


def build_interictal_reference(
    controls: pd.DataFrame,
    reader: EDFRangeReader,
    seed: int,
    min_channels: int,
    smoothing_seconds: float,
    reference_fraction: float,
) -> pd.DataFrame:
    """Build patient-specific band means and empirical central ranges."""
    rows: list[dict[str, object]] = []
    for index, control in controls.reset_index(drop=True).iterrows():
        print(
            f"[interictal {index + 1:03d}/{len(controls)}] "
            f"{control.patient_id} {control.recording} "
            f"{int(control.start_seconds)}s",
            flush=True,
        )
        envelope, sample_rate, usable_count = process_block(
            reader,
            str(control.patient_id),
            str(control.recording),
            int(control.start_seconds),
            min_channels,
            smoothing_seconds,
        )
        block_rng = np.random.default_rng(
            np.random.SeedSequence(
                [seed, index, int(control.control_id), int(control.start_seconds)]
            )
        )
        sampled = stratified_sample_indices(sample_rate, block_rng)
        for bin_index, sample_index in sampled:
            for band in BAND_ORDER:
                rows.append(
                    {
                        "patient_id": control.patient_id,
                        "event_id": control.event_id,
                        "control_id": int(control.control_id),
                        "recording": control.recording,
                        "start_seconds": int(control.start_seconds),
                        "time_bin": bin_index,
                        "band": band,
                        "relative_power": float(envelope[band][sample_index]),
                        "usable_eeg_channels": usable_count,
                    }
                )

    samples = pd.DataFrame(rows)
    alpha = (1 - reference_fraction) / 2
    reference = (
        samples.groupby(["patient_id", "band"])["relative_power"]
        .agg(
            interictal_mean="mean",
            interictal_sd="std",
            interictal_q_low=lambda values: values.quantile(alpha),
            interictal_q_high=lambda values: values.quantile(1 - alpha),
            interictal_sample_count="count",
        )
        .reset_index()
    )
    return reference


def sample_preictal(
    events: pd.DataFrame,
    reader: EDFRangeReader,
    reference: pd.DataFrame,
    seed: int,
    min_channels: int,
    smoothing_seconds: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, event in events.reset_index(drop=True).iterrows():
        onset = int(event.onset_seconds)
        print(
            f"[preictal {index + 1:02d}/{len(events)}] "
            f"{event.event_id} {event.recording} onset {onset}s",
            flush=True,
        )
        envelope, sample_rate, usable_count = process_block(
            reader,
            str(event.patient_id),
            str(event.recording),
            onset - PREICTAL_SECONDS,
            min_channels,
            smoothing_seconds,
        )
        event_rng = np.random.default_rng(
            np.random.SeedSequence([seed, index, onset])
        )
        sampled = stratified_sample_indices(sample_rate, event_rng)
        for bin_index, sample_index in sampled:
            seconds_to_onset = -PREICTAL_SECONDS + sample_index / sample_rate
            for band in BAND_ORDER:
                rows.append(
                    {
                        "patient_id": event.patient_id,
                        "event_id": event.event_id,
                        "recording": event.recording,
                        "onset_seconds": onset,
                        "time_bin": bin_index,
                        "sample_index": sample_index,
                        "seconds_to_onset": seconds_to_onset,
                        "band": band,
                        "relative_power": float(envelope[band][sample_index]),
                        "usable_eeg_channels": usable_count,
                    }
                )

    sampled = pd.DataFrame(rows).merge(
        reference, on=["patient_id", "band"], how="left", validate="many_to_one"
    )
    if sampled[
        ["interictal_mean", "interictal_q_low", "interictal_q_high"]
    ].isna().any().any():
        raise ValueError("Missing patient-specific interictal reference values")
    sampled["log2_deviation_from_interictal_mean"] = np.log2(
        np.maximum(sampled["relative_power"], 1e-12)
        / np.maximum(sampled["interictal_mean"], 1e-12)
    )
    sampled["interictal_log2_low"] = np.log2(
        np.maximum(sampled["interictal_q_low"], 1e-12)
        / np.maximum(sampled["interictal_mean"], 1e-12)
    )
    sampled["interictal_log2_high"] = np.log2(
        np.maximum(sampled["interictal_q_high"], 1e-12)
        / np.maximum(sampled["interictal_mean"], 1e-12)
    )
    sampled["within_patient_interictal_range"] = (
        sampled["relative_power"].between(
            sampled["interictal_q_low"], sampled["interictal_q_high"]
        )
    )
    return sampled


def regression_summary(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for band in BAND_ORDER:
        band_data = samples[samples["band"] == band]
        time = band_data["seconds_to_onset"].to_numpy(dtype=float)
        deviation = band_data[
            "log2_deviation_from_interictal_mean"
        ].to_numpy(dtype=float)
        regression = stats.linregress(time, deviation)
        rows.append(
            {
                "band": band,
                "n_points": len(band_data),
                "n_seizures": band_data["event_id"].nunique(),
                "n_patients": band_data["patient_id"].nunique(),
                "pearson_r": regression.rvalue,
                "r_squared": regression.rvalue**2,
                "slope_log2_per_second": regression.slope,
                "intercept_log2_at_onset": regression.intercept,
                "p_value_descriptive": regression.pvalue,
                "standard_error": regression.stderr,
                "within_interictal_range_fraction": band_data[
                    "within_patient_interictal_range"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def save_band_plots(
    samples: pd.DataFrame, summary: pd.DataFrame, figures_dir: Path
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for band_index, band in enumerate(BAND_ORDER, start=1):
        band_data = samples[samples["band"] == band].copy()
        result = summary[summary["band"] == band].iloc[0]
        in_range = band_data["within_patient_interictal_range"]

        figure, axis = plt.subplots(figsize=(8.2, 8.2))
        axis.scatter(
            band_data.loc[in_range, "log2_deviation_from_interictal_mean"],
            band_data.loc[in_range, "seconds_to_onset"],
            s=11,
            alpha=0.32,
            color=IN_RANGE_COLOR,
            edgecolors="none",
            label="Within patient-specific interictal 95% range",
            rasterized=True,
        )
        axis.scatter(
            band_data.loc[~in_range, "log2_deviation_from_interictal_mean"],
            band_data.loc[~in_range, "seconds_to_onset"],
            s=13,
            alpha=0.48,
            color=OUT_OF_RANGE_COLOR,
            edgecolors="none",
            label="Outside patient-specific interictal 95% range",
            rasterized=True,
        )

        time_line = np.linspace(-60, 0, 200)
        fitted_deviation = (
            float(result["intercept_log2_at_onset"])
            + float(result["slope_log2_per_second"]) * time_line
        )
        axis.plot(
            fitted_deviation,
            time_line,
            color=BAND_COLORS[band],
            linewidth=2.6,
            label="Point-level linear fit",
        )
        axis.axvline(0, color="#667085", linewidth=1.1, linestyle="--")
        for boundary in range(-54, 0, 6):
            axis.axhline(boundary, color="#E5E7EB", linewidth=0.65, zorder=0)

        axis.set_ylim(-60, 0)
        axis.set_yticks(np.arange(-60, 1, 6))
        axis.set_ylabel("Seconds relative to seizure onset")
        axis.set_xlabel(
            "Log2 deviation from the patient's interictal mean relative power"
        )
        axis.set_title(
            f"{band.title()} relative-power samples before seizure onset",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )
        axis.text(
            0.02,
            0.98,
            f"n = {int(result['n_points']):,} points "
            f"({int(result['n_seizures'])} seizures)\n"
            f"Pearson r = {result['pearson_r']:+.3f}  •  "
            f"R² = {result['r_squared']:.3f}\n"
            f"Slope = {result['slope_log2_per_second']:+.4f} log2/s",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "white",
                "edgecolor": "#D0D5DD",
                "alpha": 0.94,
            },
        )
        axis.grid(axis="x", color="#E5E7EB", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="lower right", fontsize=8.5)
        figure.tight_layout()
        figure.savefig(
            figures_dir
            / f"11_{band_index:02d}_{band}_sampled_preictal_regression.png",
            dpi=220,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)


def validate_outputs(
    samples: pd.DataFrame,
    reference: pd.DataFrame,
    summary: pd.DataFrame,
    expected_events: int,
) -> None:
    expected_per_band = expected_events * N_TIME_BINS * SAMPLES_PER_BIN
    counts = samples.groupby("band").size().reindex(BAND_ORDER)
    if not (counts == expected_per_band).all():
        raise AssertionError(
            f"Expected {expected_per_band} points per band; got {counts.to_dict()}"
        )
    event_counts = samples.groupby(["event_id", "band"]).size()
    if not (event_counts == N_TIME_BINS * SAMPLES_PER_BIN).all():
        raise AssertionError("Each seizure-band must contain exactly 100 points")
    bin_counts = samples.groupby(["event_id", "band", "time_bin"]).size()
    if not (bin_counts == SAMPLES_PER_BIN).all():
        raise AssertionError("Each seizure-band-bin must contain exactly 10 points")
    if not samples["seconds_to_onset"].between(
        -PREICTAL_SECONDS, 0, inclusive="left"
    ).all():
        raise AssertionError("A sampled point lies outside [-60, 0) seconds")
    if not samples["relative_power"].between(0, 1).all():
        raise AssertionError("Relative power must lie in [0, 1]")
    if not (
        (reference["interictal_q_low"] <= reference["interictal_mean"])
        & (reference["interictal_mean"] <= reference["interictal_q_high"])
    ).all():
        raise AssertionError("An interictal mean lies outside its reference range")
    if (
        samples[
            [
                "relative_power",
                "log2_deviation_from_interictal_mean",
                "seconds_to_onset",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise AssertionError("Sampled output contains missing numeric values")
    if len(summary) != len(BAND_ORDER):
        raise AssertionError("Regression summary is missing a band")


def validate_band_filters() -> pd.DataFrame:
    """Confirm that canonical sine waves are assigned to the intended bands."""
    sample_rate = 512.0
    seconds = 20
    time = np.arange(int(sample_rate * seconds)) / sample_rate
    test_frequencies = {
        "delta": 2.0,
        "theta": 6.0,
        "alpha": 10.0,
        "beta": 20.0,
        "gamma": 50.0,
    }
    rows = []
    for expected_band, frequency in test_frequencies.items():
        wave = np.sin(2 * np.pi * frequency * time)
        powers = {}
        for band, (low, high) in core.BANDS.items():
            sos = signal.butter(
                4, [low, high], btype="bandpass", fs=sample_rate, output="sos"
            )
            filtered = signal.sosfiltfilt(sos, wave)
            powers[band] = float(np.mean(np.square(filtered)))
        represented_band = max(powers, key=powers.get)
        rows.append(
            {
                "test_frequency_hz": frequency,
                "expected_band": expected_band,
                "represented_band": represented_band,
                "expected_band_power_fraction": powers[expected_band]
                / max(sum(powers.values()), 1e-12),
                "passed": represented_band == expected_band,
            }
        )
    validation = pd.DataFrame(rows)
    if not validation["passed"].all():
        raise AssertionError(
            "Band-filter validation failed:\n" + validation.to_string(index=False)
        )
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--processed-dir", type=Path, default=ROOT / "data" / "processed"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-channels", type=int, default=core.DEFAULT_MIN_CHANNELS)
    parser.add_argument(
        "--smoothing-seconds", type=float, default=POWER_SMOOTHING_SECONDS
    )
    parser.add_argument(
        "--interictal-reference-fraction",
        type=float,
        default=INTERICTAL_REFERENCE_FRACTION,
    )
    args = parser.parse_args()
    if not 0 < args.interictal_reference_fraction < 1:
        parser.error("--interictal-reference-fraction must be between 0 and 1")
    if args.smoothing_seconds <= 0:
        parser.error("--smoothing-seconds must be positive")
    return args


def main() -> None:
    args = parse_args()
    event_path = args.results_dir / "event_inventory.csv"
    control_path = args.results_dir / "interictal_control_blocks.csv"
    if not event_path.exists() or not control_path.exists():
        raise FileNotFoundError(
            "The validated event inventory and interictal control table are required"
        )

    events = pd.read_csv(event_path)
    controls = pd.read_csv(control_path)
    events = events[events["eligible_preictal"]].copy()
    if events.empty:
        raise ValueError("No eligible seizures found")
    if set(events["event_id"]) != set(controls["event_id"]):
        raise ValueError("Control table does not cover the eligible seizure inventory")
    if not (controls.groupby("event_id").size() == 3).all():
        raise ValueError("Expected exactly three matched controls per seizure")

    band_validation = validate_band_filters()
    reader = EDFRangeReader(args.raw_dir, args.base_url)
    reference = build_interictal_reference(
        controls,
        reader,
        args.seed,
        args.min_channels,
        args.smoothing_seconds,
        args.interictal_reference_fraction,
    )
    samples = sample_preictal(
        events,
        reader,
        reference,
        args.seed,
        args.min_channels,
        args.smoothing_seconds,
    )
    summary = regression_summary(samples)
    validate_outputs(samples, reference, summary, len(events))

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.results_dir / "figures"
    samples.to_csv(
        args.processed_dir / "preictal_sampled_bandpower.csv", index=False
    )
    reference.to_csv(
        args.results_dir / "patient_interictal_bandpower_ranges.csv", index=False
    )
    summary.to_csv(
        args.results_dir / "sampled_bandpower_regression.csv", index=False
    )
    band_validation.to_csv(
        args.results_dir / "band_filter_validation.csv", index=False
    )
    save_band_plots(samples, summary, figures_dir)

    print("\nRegression results:")
    print(
        summary[
            [
                "band",
                "n_points",
                "pearson_r",
                "r_squared",
                "slope_log2_per_second",
            ]
        ].to_string(index=False)
    )
    print(
        f"\nRead {reader.local_bytes / 1024**2:.1f} MiB locally and "
        f"{reader.remote_bytes / 1024**2:.1f} MiB from PhysioNet."
    )
    print(
        f"Saved {len(samples):,} long-format rows "
        f"({len(samples) // len(BAND_ORDER):,} sampled times × "
        f"{len(BAND_ORDER)} bands)."
    )


if __name__ == "__main__":
    main()
