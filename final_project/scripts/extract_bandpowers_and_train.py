"""Extract EEG band-power features from EDF files and train seizure models.

The script is intentionally dependency-light: it reads the EDF binary layout
directly and uses NumPy/SciPy/scikit-learn already present in the project.

Example
-------
python scripts/extract_bandpowers_and_train.py

The default run writes features to ``data/processed`` and model results to
``results/models``. The raw EDF files are never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}
DEFAULT_WINDOW_SECONDS = 10.0
DEFAULT_SEGMENT_SECONDS = 4.0


def _decode(value: bytes) -> str:
    return value.decode("ascii", errors="ignore").strip().replace("\x00", "")


def _header_number(value: bytes, cast=float) -> float:
    text = _decode(value).replace("\x00", "").strip()
    return cast(text)


@dataclass
class EDFHeader:
    path: Path
    header_bytes: int
    n_records: int
    record_seconds: float
    labels: list[str]
    samples_per_record: np.ndarray
    physical_min: np.ndarray
    physical_max: np.ndarray
    digital_min: np.ndarray
    digital_max: np.ndarray
    bytes_per_record: int

    @property
    def n_signals(self) -> int:
        return len(self.labels)

    @property
    def sample_rate(self) -> float:
        rates = self.samples_per_record / self.record_seconds
        return float(rates[0])


def read_edf_header(path: Path) -> EDFHeader:
    """Read the fixed and per-channel EDF header fields."""
    with path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) != 256:
            raise ValueError(f"EDF header is incomplete: {path}")
        header_bytes = int(_decode(fixed[184:192]))
        n_records = int(_decode(fixed[236:244]))
        record_seconds = float(_decode(fixed[244:252]))
        n_signals = int(_decode(fixed[252:256]))
        if header_bytes < 256 + 256 * n_signals:
            raise ValueError(f"Invalid EDF header length in {path}")

        labels = [_decode(handle.read(16)) for _ in range(n_signals)]
        handle.seek(80 * n_signals, 1)  # transducer
        handle.seek(8 * n_signals, 1)  # physical dimension
        physical_min = np.array([_header_number(handle.read(8)) for _ in range(n_signals)])
        physical_max = np.array([_header_number(handle.read(8)) for _ in range(n_signals)])
        digital_min = np.array([_header_number(handle.read(8)) for _ in range(n_signals)])
        digital_max = np.array([_header_number(handle.read(8)) for _ in range(n_signals)])
        handle.seek(80 * n_signals, 1)  # prefilter
        samples_per_record = np.array(
            [_header_number(handle.read(8), int) for _ in range(n_signals)], dtype=int
        )
        bytes_per_record = int(samples_per_record.sum() * 2)

    if np.any(samples_per_record <= 0) or record_seconds <= 0:
        raise ValueError(f"Invalid EDF sampling metadata in {path}")
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
        bytes_per_record=bytes_per_record,
    )


def _is_eeg_label(label: str) -> bool:
    return label.strip().upper().startswith("EEG ")


def _scale_signal(raw: np.ndarray, header: EDFHeader, indices: np.ndarray) -> np.ndarray:
    dmin = header.digital_min[indices].astype(np.float32)
    dmax = header.digital_max[indices].astype(np.float32)
    pmin = header.physical_min[indices].astype(np.float32)
    pmax = header.physical_max[indices].astype(np.float32)
    scale = (pmax - pmin) / (dmax - dmin)
    offset = pmin - scale * dmin
    return raw.astype(np.float32) * scale[:, None] + offset[:, None]


def _band_power_matrix(
    data: np.ndarray, sample_rate: float, segment_seconds: float
) -> np.ndarray:
    """Return channel x band integrated PSD in physical units (uV^2)."""
    n_channels, n_samples = data.shape
    nperseg = int(round(segment_seconds * sample_rate))
    if n_samples < nperseg:
        raise ValueError("Window is shorter than the requested PSD segment")
    step = nperseg // 2
    starts = np.arange(0, n_samples - nperseg + 1, step, dtype=int)
    if len(starts) == 0:
        raise ValueError("No complete PSD segments in window")

    # Detrend every segment before FFT so DC offsets do not overwhelm delta.
    segments = np.stack([data[:, start : start + nperseg] for start in starts], axis=1)
    segments = signal.detrend(segments, axis=-1, type="linear")
    taper = signal.windows.hann(nperseg, sym=False).astype(np.float32)
    fft = np.fft.rfft(segments * taper, axis=-1)
    psd = (np.abs(fft) ** 2) / (sample_rate * np.sum(taper**2))
    if psd.shape[-1] > 2:
        psd[..., 1:-1] *= 2.0
    psd = psd.mean(axis=1)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sample_rate)

    powers = np.empty((n_channels, len(BANDS)), dtype=np.float64)
    for band_index, (low, high) in enumerate(BANDS.values()):
        mask = (freqs >= low) & (freqs <= high)
        powers[:, band_index] = np.trapezoid(psd[:, mask], freqs[mask], axis=1)
    return powers


def _parse_clock(text: str) -> int | None:
    match = re.search(r"(?<!\d)(\d{1,2})\s*[\.:]\s*(\d{2})\s*[\.:]\s*(\d{2})", text)
    if not match:
        return None
    hour, minute, second = map(int, match.groups())
    if hour > 23 or minute > 59 or second > 59:
        return None
    return hour * 3600 + minute * 60 + second


def _annotation_intervals(annotation_path: Path) -> dict[str, list[tuple[int, int]]]:
    """Parse seizure intervals and convert clock times to recording offsets."""
    lines = annotation_path.read_text(encoding="utf-8", errors="replace").splitlines()
    intervals: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None
    registration_start: int | None = None
    pending_start: int | None = None

    for line in lines:
        lowered = line.lower()
        if "file name:" in lowered:
            current_file = line.split(":", 1)[1].strip().split()[0]
            intervals.setdefault(current_file, [])
            pending_start = None
        elif "registration start time:" in lowered:
            registration_start = _parse_clock(line)
        elif "seizure start time:" in lowered or re.match(r"\s*start time:", lowered):
            pending_start = _parse_clock(line)
        elif "seizure end time:" in lowered or re.match(r"\s*end time:", lowered):
            seizure_end = _parse_clock(line)
            if current_file and registration_start is not None and pending_start is not None and seizure_end is not None:
                start_offset = (pending_start - registration_start) % 86400
                end_offset = (seizure_end - registration_start) % 86400
                if end_offset <= start_offset:
                    end_offset += 86400
                intervals[current_file].append((start_offset, end_offset))
            pending_start = None
    return intervals


def _resolve_annotation_file(annotation_name: str, edf_files: list[Path]) -> Path:
    """Resolve known filename typos in the supplied seizure lists."""
    exact = [p for p in edf_files if p.name.lower() == annotation_name.lower()]
    if exact:
        return exact[0]
    normalized = re.sub(r"[^a-z0-9]", "", annotation_name.lower()).replace("o", "0")
    candidates = [p for p in edf_files if re.sub(r"[^a-z0-9]", "", p.name.lower()).replace("o", "0") == normalized]
    if len(candidates) == 1:
        return candidates[0]
    if len(edf_files) == 1:
        return edf_files[0]
    raise FileNotFoundError(f"Could not resolve annotation file {annotation_name}")


def _window_rows(
    edf_path: Path,
    intervals: list[tuple[int, int]],
    window_seconds: float,
    segment_seconds: float,
) -> Iterable[dict[str, object]]:
    header = read_edf_header(edf_path)
    eeg_indices = np.array([i for i, label in enumerate(header.labels) if _is_eeg_label(label)], dtype=int)
    if len(eeg_indices) == 0:
        raise ValueError(f"No EEG channels found in {edf_path}")
    rates = header.samples_per_record[eeg_indices] / header.record_seconds
    if not np.allclose(rates, rates[0]):
        raise ValueError(f"EEG channels have different rates in {edf_path}")
    sample_rate = float(rates[0])
    records_per_window = int(round(window_seconds / header.record_seconds))
    window_samples = int(round(window_seconds * sample_rate))
    samples_per_record = int(header.samples_per_record[eeg_indices[0]])
    if records_per_window <= 0 or records_per_window * samples_per_record != window_samples:
        raise ValueError(f"Window must align to EDF records in {edf_path}")
    selected_bytes = header.samples_per_record[eeg_indices] * 2

    with edf_path.open("rb") as handle:
        handle.seek(header.header_bytes)
        for window_index, record_start in enumerate(range(0, header.n_records, records_per_window)):
            n_records = min(records_per_window, header.n_records - record_start)
            if n_records != records_per_window:
                break
            block = np.frombuffer(
                handle.read(header.bytes_per_record * n_records), dtype="<i2"
            )
            if block.size != (header.bytes_per_record * n_records) // 2:
                break
            block = block.reshape(n_records, header.n_signals, -1)
            raw = block[:, eeg_indices, :].transpose(1, 0, 2).reshape(len(eeg_indices), -1)
            data = _scale_signal(raw, header, eeg_indices)
            powers = _band_power_matrix(data, sample_rate, segment_seconds)
            start_seconds = window_index * window_seconds
            end_seconds = start_seconds + window_seconds
            label = int(any(start_seconds < end and end_seconds > start for start, end in intervals))

            row: dict[str, object] = {
                "patient_id": edf_path.parent.name,
                "recording": edf_path.name,
                "window_start_seconds": round(start_seconds, 3),
                "window_end_seconds": round(end_seconds, 3),
                "seizure": label,
                "eeg_channel_count": len(eeg_indices),
            }
            total = powers.sum(axis=1)
            for band_index, band in enumerate(BANDS):
                values = powers[:, band_index]
                row[f"{band}_power_mean"] = float(np.mean(values))
                row[f"{band}_power_median"] = float(np.median(values))
                row[f"{band}_power_std"] = float(np.std(values))
                row[f"{band}_relative_power"] = float(np.mean(values / np.maximum(total, 1e-12)))
                row[f"{band}_log10_power"] = float(np.log10(max(np.mean(values), 1e-12)))
            yield row


def extract_features(raw_dir: Path, output_path: Path, window_seconds: float, segment_seconds: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seizure_files = sorted(raw_dir.glob("PN*/Seizures-list-PN*.txt"))
    if not seizure_files:
        raise FileNotFoundError(f"No seizure annotation files found under {raw_dir}")
    for annotation_path in seizure_files:
        patient_id = annotation_path.stem.split("-")[-1]
        edf_files = sorted((raw_dir / patient_id).glob("*.edf"))
        if not edf_files:
            raise FileNotFoundError(f"No EDF files found for {patient_id}")
        parsed = _annotation_intervals(annotation_path)
        resolved: dict[Path, list[tuple[int, int]]] = {}
        for annotation_name, intervals in parsed.items():
            resolved.setdefault(_resolve_annotation_file(annotation_name, edf_files), []).extend(intervals)
        # Files without listed seizures are retained as negative examples.
        for edf_path in edf_files:
            file_intervals = resolved.get(edf_path, [])
            print(f"Extracting {edf_path.relative_to(raw_dir)} ({len(file_intervals)} seizure intervals)", flush=True)
            rows.extend(_window_rows(edf_path, file_intervals, window_seconds, segment_seconds))
    features = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False, float_format="%.8g")
    return features


def _model_metrics(model_name: str, model, x_test, y_test, groups_test) -> dict[str, object]:
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics: dict[str, object] = {
        "model": model_name,
        "n_test_windows": int(len(y_test)),
        "n_test_subjects": int(pd.Series(groups_test).nunique()),
        "positive_test_windows": int(y_test.sum()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if len(np.unique(y_test)) == 2 else None,
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    return metrics


def train_models(features: pd.DataFrame, models_dir: Path, results_path: Path) -> pd.DataFrame:
    excluded = {"patient_id", "recording", "window_start_seconds", "window_end_seconds", "seizure"}
    feature_columns = [column for column in features.columns if column not in excluded]
    x = features[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = features["seizure"].astype(int)
    groups = features["patient_id"]
    if y.nunique() < 2:
        raise ValueError("Both seizure and non-seizure windows are required for modeling")

    splitter = GroupShuffleSplit(n_splits=100, test_size=0.25, random_state=42)
    for train_index, test_index in splitter.split(x, y, groups):
        if y.iloc[train_index].nunique() == 2 and y.iloc[test_index].nunique() == 2:
            break
    else:
        raise ValueError("Could not find a subject-level split containing both classes")

    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    groups_test = groups.iloc[test_index]
    models = {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    }
    sample_weight = np.where(y_train.to_numpy() == 1, len(y_train) / (2 * y_train.sum()), len(y_train) / (2 * (len(y_train) - y_train.sum())))
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if name == "hist_gradient_boosting":
                model.fit(x_train, y_train, sample_weight=sample_weight)
            else:
                model.fit(x_train, y_train)
        joblib.dump({"model": model, "feature_columns": feature_columns}, models_dir / f"{name}.joblib")
        metrics_rows.append(_model_metrics(name, model, x_test, y_test, groups_test))

    results_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metrics_rows).sort_values("average_precision", ascending=False)
    metrics.to_csv(results_path, index=False)
    split_info = {
        "train_subjects": sorted(groups.iloc[train_index].unique().tolist()),
        "test_subjects": sorted(groups.iloc[test_index].unique().tolist()),
        "train_windows": int(len(train_index)),
        "test_windows": int(len(test_index)),
        "feature_columns": feature_columns,
    }
    results_path.with_name("subject_split.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    print("\nModel results:")
    print(metrics[["model", "balanced_accuracy", "f1", "roc_auc", "average_precision"]].to_string(index=False))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("final_project/data/raw"))
    parser.add_argument("--features", type=Path, default=Path("final_project/data/processed/eeg_bandpowers.csv"))
    parser.add_argument("--models-dir", type=Path, default=Path("final_project/models"))
    parser.add_argument("--results", type=Path, default=Path("final_project/results/model_metrics.csv"))
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--segment-seconds", type=float, default=DEFAULT_SEGMENT_SECONDS)
    parser.add_argument("--reuse-features", action="store_true", help="Skip extraction if the feature CSV exists")
    args = parser.parse_args()
    if args.reuse_features and args.features.exists():
        features = pd.read_csv(args.features)
    else:
        features = extract_features(args.raw_dir, args.features, args.window_seconds, args.segment_seconds)
    train_models(features, args.models_dir, args.results)


if __name__ == "__main__":
    main()
