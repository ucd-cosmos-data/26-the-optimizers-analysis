"""Build a chronological seizure/band-response table from the final-project EEG data.

The seizure start/end fields come from the raw ``Seizures-list-PN*.txt`` files.
Band-response times are derived from the 10-second band-power windows extracted
from the raw EDF recordings in ``data/processed/eeg_bandpowers.csv``.

For each recording and frequency band, an interictal reference is the median
log10 power across all windows that do not overlap an annotated seizure. A
band is considered to deviate when its absolute robust z-score is at least
2.5 (median/MAD scale). The table reports the first such window that occurs
from 60 seconds before onset through 10 seconds after the annotated seizure;
the recovery time is the first later 10-second window at or below that
threshold after the annotated seizure has ended. All timing fields are stored
as integer seconds relative to the annotated seizure start (which is always
zero). ``N/A`` means no qualifying deviation was detected in that interval
(or no later recovery window exists).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BANDS = ("alpha", "beta", "delta", "theta", "gamma")
ROBUST_Z_THRESHOLD = 2.5
PRE_ONSET_SECONDS = 60
POST_END_SECONDS = 10


@dataclass(frozen=True)
class Seizure:
    patient_id: str
    recording: str
    source_number: int | None
    start_seconds: int
    end_seconds: int


def parse_clock_seconds(text: str) -> int | None:
    """Parse HH.MM.SS or HH:MM:SS, including the source typo ``1 6.49.25``."""
    match = re.search(
        r"(?<!\d)(\d(?:\s*\d)?)\s*[:.]\s*(\d{2})\s*[:.]\s*(\d{2})",
        text,
    )
    if not match:
        return None
    hour = int(re.sub(r"\s+", "", match.group(1)))
    minute, second = map(int, match.group(2, 3))
    if hour > 23 or minute > 59 or second > 59:
        return None
    return hour * 3600 + minute * 60 + second


def normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower()).replace("o", "0")


def read_edf_duration_seconds(path: Path) -> float:
    with path.open("rb") as handle:
        header = handle.read(256)
    n_records = int(header[236:244].decode("ascii", errors="ignore").strip())
    record_seconds = float(header[244:252].decode("ascii", errors="ignore").strip())
    return n_records * record_seconds


def resolve_edf(annotation_name: str, edf_files: list[Path]) -> Path:
    exact = [path for path in edf_files if path.name.lower() == annotation_name.lower()]
    if exact:
        return exact[0]
    normalized = [path for path in edf_files if normalized_name(path.name) == normalized_name(annotation_name)]
    if len(normalized) == 1:
        return normalized[0]
    if len(edf_files) == 1:
        return edf_files[0]
    raise FileNotFoundError(f"Could not resolve annotation filename {annotation_name!r}")


def corrected_end_offset(
    start_clock: int,
    end_clock: int,
    registration_start: int,
    start_offset: int,
    end_offset: int,
    duration_seconds: float,
) -> int:
    """Repair an impossible end time only when a one-hour source typo is unambiguous.

    PN00-3 lists an end time beyond the EDF duration (19:29:29) although its
    registration ends at 18:57:13 and the onset is 18:28:29. Reusing the
    onset hour gives the only valid in-recording end time, 00:13:45.
    """
    if end_offset <= duration_seconds:
        return end_offset
    same_hour_end_clock = (start_clock // 3600) * 3600 + (end_clock % 3600)
    if same_hour_end_clock < start_clock:
        same_hour_end_clock += 3600
    candidate = (same_hour_end_clock - registration_start) % 86400
    if start_offset < candidate <= duration_seconds:
        return candidate
    return end_offset


def parse_raw_seizures(raw_dir: Path) -> list[Seizure]:
    seizures: list[Seizure] = []
    for annotation_path in sorted(raw_dir.glob("PN*/Seizures-list-PN*.txt")):
        patient_id = annotation_path.parent.name
        edf_files = sorted(annotation_path.parent.glob("*.edf"))
        current_file: str | None = None
        registration_start: int | None = None
        source_number: int | None = None
        pending_start: int | None = None

        for line in annotation_path.read_text(encoding="utf-8", errors="replace").splitlines():
            lowered = line.lower()
            seizure_number = re.search(r"seizure\s+n\s*(\d+)", lowered)
            if seizure_number:
                source_number = int(seizure_number.group(1))
            if "file name:" in lowered:
                current_file = line.split(":", 1)[1].strip().split()[0]
            elif "registration start time:" in lowered:
                parsed = parse_clock_seconds(line)
                if parsed is not None:
                    registration_start = parsed
            elif "seizure start time:" in lowered or re.match(r"\s*start time:", lowered):
                pending_start = parse_clock_seconds(line)
            elif "seizure end time:" in lowered or re.match(r"\s*end time:", lowered):
                end_clock = parse_clock_seconds(line)
                if (
                    current_file is not None
                    and registration_start is not None
                    and pending_start is not None
                    and end_clock is not None
                ):
                    edf_path = resolve_edf(current_file, edf_files)
                    start_offset = (pending_start - registration_start) % 86400
                    end_offset = (end_clock - registration_start) % 86400
                    if end_offset <= start_offset:
                        end_offset += 86400
                    end_offset = corrected_end_offset(
                        pending_start,
                        end_clock,
                        registration_start,
                        start_offset,
                        end_offset,
                        read_edf_duration_seconds(edf_path),
                    )
                    if end_offset > read_edf_duration_seconds(edf_path):
                        raise ValueError(
                            f"Annotated seizure ends outside its EDF recording: {edf_path.name} "
                            f"({start_offset}–{end_offset} seconds)"
                        )
                    seizures.append(
                        Seizure(
                            patient_id=patient_id,
                            recording=edf_path.name,
                            source_number=source_number,
                            start_seconds=start_offset,
                            end_seconds=end_offset,
                        )
                    )
                pending_start = None
    return seizures


def relative_seconds(seconds: float | int | None, seizure_start: int) -> int | str:
    if seconds is None or not np.isfinite(seconds):
        return "N/A"
    return int(round(float(seconds) - seizure_start))


def add_robust_band_z_scores(features: pd.DataFrame, seizures: list[Seizure]) -> pd.DataFrame:
    intervals: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for seizure in seizures:
        intervals.setdefault((seizure.patient_id, seizure.recording), []).append(
            (seizure.start_seconds, seizure.end_seconds)
        )

    output = features.copy()
    for band in BANDS:
        output[f"{band}_robust_z"] = np.nan

    for key, group in output.groupby(["patient_id", "recording"], sort=False):
        starts = group["window_start_seconds"].to_numpy()
        ends = group["window_end_seconds"].to_numpy()
        interictal = np.ones(len(group), dtype=bool)
        for seizure_start, seizure_end in intervals.get(key, []):
            interictal &= ~((starts < seizure_end) & (ends > seizure_start))

        for band in BANDS:
            values = group[f"{band}_log10_power"].to_numpy(dtype=float)
            baseline = values[interictal]
            median = np.median(baseline)
            scale = 1.4826 * np.median(np.abs(baseline - median))
            if not np.isfinite(scale) or scale < 1e-8:
                scale = np.std(baseline)
            if not np.isfinite(scale) or scale < 1e-8:
                scale = 1.0
            output.loc[group.index, f"{band}_robust_z"] = np.abs((values - median) / scale)
    return output


def response_times(recording: pd.DataFrame, seizure: Seizure, band: str) -> tuple[float | None, float | None]:
    starts = recording["window_start_seconds"].to_numpy(dtype=float)
    ends = recording["window_end_seconds"].to_numpy(dtype=float)
    z_scores = recording[f"{band}_robust_z"].to_numpy(dtype=float)
    near_seizure = (starts < seizure.end_seconds + POST_END_SECONDS) & (
        ends > seizure.start_seconds - PRE_ONSET_SECONDS
    )
    deviating = near_seizure & (z_scores >= ROBUST_Z_THRESHOLD)
    if not deviating.any():
        return None, None

    deviation = float(starts[np.flatnonzero(deviating)[0]])
    recovered = (starts >= seizure.end_seconds) & (z_scores < ROBUST_Z_THRESHOLD)
    recovery = float(starts[np.flatnonzero(recovered)[0]]) if recovered.any() else None
    return deviation, recovery


def build_timeline(raw_dir: Path, features_path: Path, output_path: Path) -> pd.DataFrame:
    seizures = parse_raw_seizures(raw_dir)
    features = pd.read_csv(features_path)
    features = add_robust_band_z_scores(features, seizures)

    rows: list[dict[str, int | str]] = []
    for sequence, seizure in enumerate(seizures, start=1):
        recording = features.loc[
            (features["patient_id"] == seizure.patient_id)
            & (features["recording"] == seizure.recording)
        ]
        if recording.empty:
            raise ValueError(f"No band-power windows available for {seizure.recording}")
        row: dict[str, int | str] = {
            "Seizure number": str(sequence),
            "Seizure start (seconds relative to seizure start)": 0,
            "Seizure end (seconds relative to seizure start)": relative_seconds(
                seizure.end_seconds, seizure.start_seconds
            ),
        }
        for band in BANDS:
            deviation, recovery = response_times(recording, seizure, band)
            title = band.capitalize()
            row[f"{title} deviates from interictal (seconds relative to seizure start)"] = (
                relative_seconds(deviation, seizure.start_seconds)
            )
            row[f"{title} comes back (seconds relative to seizure start)"] = relative_seconds(
                recovery, seizure.start_seconds
            )
        rows.append(row)

    output = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return output


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    output_path = project_dir / "data" / "interim" / "seizure_band_timeline.csv"
    table = build_timeline(
        project_dir / "data" / "raw",
        project_dir / "data" / "processed" / "eeg_bandpowers.csv",
        output_path,
    )
    print(f"Wrote {len(table)} seizures to {output_path}")


if __name__ == "__main__":
    main()
