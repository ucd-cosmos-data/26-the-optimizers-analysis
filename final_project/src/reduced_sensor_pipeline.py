"""Leakage-aware reduced-sensor seizure-prediction experiments.

The three study stages share one logical model:

1. fit one regularized logistic classifier per EEG sensor;
2. average the selected sensors' probabilities; and
3. calibrate that average using out-of-fold training predictions.

K-finder estimates only the number of sensors. K-suiter ranks sensor identities.
Personalized (P) and generalized (G) models differ only in their training
population. This is exploratory research software, not a clinical detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import hashlib
import itertools
import json
import math
import re

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import optimize
from scipy.special import expit
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
TARGET_PATIENTS = ("PN00", "PN06", "PN10", "PN12", "PN14")
ALLOWED_AUPRC_LOSS = 0.03
TEST_FRACTION = 0.20
MIN_TEST_SEIZURES = 2
CONTEXT_SECONDS = 120

COMPACT_FEATURES = (
    "relative_delta",
    "relative_theta",
    "relative_alpha",
    "relative_beta",
    "log_rms",
    "log_line_length",
    "usable_channel_fraction",
)
COMPACT_AGGREGATIONS = ("mean", "last", "slope")
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


def natural_key(value: str) -> tuple[object, ...]:
    """Sort strings containing numbers in human order."""

    return tuple(
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", str(value))
    )


def channel_columns(channel: str) -> list[str]:
    """Return the fixed 21-feature schema for one channel."""

    return [
        f"ch::{channel}::{feature}::{aggregation}"
        for feature in COMPACT_FEATURES
        for aggregation in COMPACT_AGGREGATIONS
    ]


def make_sensor_model() -> object:
    """Return the common Gen1/P/G sensor-level classifier."""

    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=2_000,
            solver="liblinear",
            random_state=RANDOM_SEED,
        ),
    )


@dataclass
class PatientData:
    patient_id: str
    frame: pd.DataFrame
    channels: list[str]


def load_patient_cache(cache_dir: str | Path, patient_id: str) -> PatientData:
    """Load and validate one cached patient feature table."""

    path = Path(cache_dir) / f"{patient_id}_channel_landmarks.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Build the channel-level features before analysis."
        )
    payload = joblib.load(path)
    required = {"frame", "channel_names", "channel_feature_columns"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"{path} is not a valid patient feature cache.")
    frame = payload["frame"].copy()
    missing_metadata = sorted(set(METADATA_COLUMNS) - set(frame.columns))
    if missing_metadata:
        raise ValueError(f"{path} is missing metadata: {missing_metadata}")
    if frame[list(METADATA_COLUMNS)].isna().any().drop(
        labels=["time_to_event_seconds"], errors="ignore"
    ).any():
        raise ValueError(f"{path} contains missing required metadata.")
    channels = sorted(
        [str(channel) for channel in payload["channel_names"]],
        key=natural_key,
    )
    for channel in channels:
        missing = sorted(set(channel_columns(channel)) - set(frame.columns))
        if missing:
            raise ValueError(
                f"{path} is missing {len(missing)} compact features for {channel}."
            )
    feature_values = frame[
        [column for channel in channels for column in channel_columns(channel)]
    ].to_numpy(dtype=float)
    if np.isinf(feature_values).any():
        raise ValueError(f"{path} contains infinite feature values.")
    frame["source_event_id"] = frame["source_event_id"].astype(str)
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame, leakage_audit = reassign_overlapping_controls(frame, channels)
    frame.attrs["leakage_audit"] = leakage_audit
    return PatientData(patient_id, frame, channels)


def load_patient_cohort(
    cache_dir: str | Path,
    patient_ids: Sequence[str] = TARGET_PATIENTS,
) -> dict[str, PatientData]:
    return {
        patient_id: load_patient_cache(cache_dir, patient_id)
        for patient_id in patient_ids
    }


def build_clean_cohort_matrix(
    cache_dir: str | Path,
    output_npz: str | Path,
    *,
    expected_patients: int = 14,
) -> Path:
    """Build one version-checked K-finder matrix from current patient caches."""

    cache_dir = Path(cache_dir)
    cache_paths = sorted(
        cache_dir.glob("PN??_channel_landmarks.joblib"),
        key=lambda path: natural_key(path.name),
    )
    patient_ids = [path.name.split("_", maxsplit=1)[0] for path in cache_paths]
    if len(patient_ids) != expected_patients or len(set(patient_ids)) != len(
        patient_ids
    ):
        raise ValueError(
            f"Expected {expected_patients} unique patient caches; "
            f"found {len(set(patient_ids))}."
        )

    signature_records: list[dict[str, object]] = []
    compatibility_keys: list[str] = []
    for patient_id in patient_ids:
        signature_path = cache_dir / f"{patient_id}_channel_landmarks.json"
        if not signature_path.exists():
            raise FileNotFoundError(f"Missing cache signature {signature_path}.")
        raw = signature_path.read_bytes()
        signature = json.loads(raw)
        compatibility = json.dumps(
            {
                "module_version": signature.get("module_version"),
                "rolling_module_version": signature.get(
                    "rolling_module_version"
                ),
                "forecast_config": signature.get("forecast_config"),
                "signal_source": signature.get("signal_source"),
            },
            sort_keys=True,
        )
        compatibility_keys.append(compatibility)
        signature_records.append(
            {
                "patient_id": patient_id,
                "signature_file": signature_path.name,
                "signature_sha256": hashlib.sha256(raw).hexdigest(),
                "module_version": signature.get("module_version"),
                "rolling_module_version": signature.get(
                    "rolling_module_version"
                ),
            }
        )
    if len(set(compatibility_keys)) != 1:
        raise ValueError(
            "Patient caches were built by incompatible feature-pipeline "
            "versions/configurations. Rebuild all caches before K-finder."
        )

    patients = load_patient_cohort(cache_dir, patient_ids)
    channels = shared_channels(patients.values())
    feature_blocks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    subjects: list[np.ndarray] = []
    source_events: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    cleaning_records: list[dict[str, object]] = []
    for patient_id in patient_ids:
        patient = patients[patient_id]
        feature_blocks.append(
            np.stack(
                [
                    patient.frame[channel_columns(channel)].to_numpy(
                        dtype=np.float32
                    )
                    for channel in channels
                ],
                axis=1,
            )
        )
        targets.append(
            patient.frame["has_event_in_5m"].to_numpy(dtype=np.uint8)
        )
        subjects.append(np.repeat(patient_id, len(patient.frame)))
        source_events.append(
            patient.frame["source_event_id"].astype(str).to_numpy()
        )
        episode_ids.append(patient.frame["episode_id"].astype(str).to_numpy())
        cleaning_records.append(
            {
                "patient_id": patient_id,
                **patient.frame.attrs.get("leakage_audit", {}),
            }
        )

    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    X = np.concatenate(feature_blocks)
    y = np.concatenate(targets)
    subject_array = np.concatenate(subjects)
    np.savez_compressed(
        output_npz,
        X=X,
        y=y,
        subjects=subject_array,
        channels=np.asarray(channels),
        source_event_ids=np.concatenate(source_events),
        episode_ids=np.concatenate(episode_ids),
    )
    metadata = {
        "rows": len(y),
        "patients": len(patient_ids),
        "channels": channels,
        "features_per_channel": X.shape[2],
        "positive_rows": int(y.sum()),
        "negative_rows": int((1 - y).sum()),
        "missing_feature_values": int(np.isnan(X).sum()),
        "infinite_feature_values": int(np.isinf(X).sum()),
        "cache_signatures": signature_records,
        "leakage_cleaning": cleaning_records,
    }
    output_npz.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return output_npz


def shared_channels(patients: Iterable[PatientData]) -> list[str]:
    """Return channels available to every supplied patient."""

    channel_sets = [set(patient.channels) for patient in patients]
    if not channel_sets:
        return []
    return sorted(set.intersection(*channel_sets), key=natural_key)


def ordered_seizure_ids(frame: pd.DataFrame) -> list[str]:
    """Return seizure IDs in recording/event order."""

    events = (
        frame.loc[frame["episode_type"].eq("preictal")]
        [["recording", "source_event_id", "landmark_seconds"]]
        .drop_duplicates(["source_event_id"])
        .copy()
    )
    events["_recording_key"] = events["recording"].map(natural_key)
    events["_event_key"] = events["source_event_id"].map(natural_key)
    events = events.sort_values(
        ["_recording_key", "landmark_seconds", "_event_key"]
    )
    return events["source_event_id"].astype(str).tolist()


def _intervals_overlap(first: pd.Series, second: pd.Series) -> bool:
    return bool(
        first["recording"] == second["recording"]
        and first["raw_start_seconds"] < second["raw_end_seconds"]
        and second["raw_start_seconds"] < first["raw_end_seconds"]
    )


def _episode_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the raw causal interval represented by every cached episode."""

    intervals = (
        frame.groupby(
            ["episode_id", "episode_type", "source_event_id", "recording"],
            as_index=False,
        )
        .agg(
            first_landmark_seconds=("landmark_seconds", "min"),
            last_landmark_seconds=("landmark_seconds", "max"),
        )
        .copy()
    )
    intervals["raw_start_seconds"] = (
        intervals["first_landmark_seconds"] - CONTEXT_SECONDS
    )
    # A landmark at t contains context ending at t; no future EEG is used.
    intervals["raw_end_seconds"] = intervals["last_landmark_seconds"]
    return intervals


def reassign_overlapping_controls(
    frame: pd.DataFrame,
    channels: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Prevent raw EEG from appearing in two event-level CV groups.

    The historical manifest placed controls only 60--120 seconds apart even
    though each cached episode represents 420 seconds of raw EEG. Controls are
    not intrinsically tied to a particular seizure, so overlapping control
    components are reassigned together to one seizure group. A control that
    overlaps a positive preictal interval is removed. This preserves all
    available non-conflicting raw data while making event-grouped validation
    honest.
    """

    cleaned = frame.copy()
    intervals = _episode_intervals(cleaned)
    positives = intervals.loc[intervals["episode_type"].eq("preictal")]
    controls = intervals.loc[intervals["episode_type"].eq("interictal")]
    event_ids = ordered_seizure_ids(cleaned)

    dropped_controls: list[str] = []
    valid_control_indices: list[int] = []
    for index, control in controls.iterrows():
        if any(
            _intervals_overlap(control, positive)
            for _, positive in positives.iterrows()
        ):
            dropped_controls.append(str(control["episode_id"]))
        else:
            valid_control_indices.append(index)
    controls = controls.loc[valid_control_indices].copy()

    # Union-find creates connected components of controls that share raw EEG.
    control_indices = controls.index.tolist()
    parent = {index: index for index in control_indices}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for position, first_index in enumerate(control_indices):
        first = controls.loc[first_index]
        for second_index in control_indices[position + 1 :]:
            if _intervals_overlap(first, controls.loc[second_index]):
                union(first_index, second_index)
    components: dict[int, list[int]] = {}
    for index in control_indices:
        components.setdefault(find(index), []).append(index)

    # Largest-first assignment keeps control counts reasonably balanced while
    # ensuring every raw-overlap component belongs to only one event group.
    assignments: dict[str, str] = {}
    loads = {event_id: 0 for event_id in event_ids}
    ordered_components = sorted(
        components.values(),
        key=lambda indices: (
            -len(indices),
            natural_key(str(controls.loc[indices[0], "episode_id"])),
        ),
    )
    unresolved_components: list[list[int]] = []
    for component in ordered_components:
        historical_groups = set(
            controls.loc[component, "source_event_id"].astype(str)
        )
        if len(historical_groups) == 1:
            event_id = historical_groups.pop()
            if event_id in loads:
                for index in component:
                    assignments[str(controls.loc[index, "episode_id"])] = event_id
                loads[event_id] += len(component)
                continue
        unresolved_components.append(component)
    for component in unresolved_components:
        event_id = min(event_ids, key=lambda value: (loads[value], natural_key(value)))
        for index in component:
            assignments[str(controls.loc[index, "episode_id"])] = event_id
        loads[event_id] += len(component)

    if dropped_controls:
        cleaned = cleaned.loc[
            ~cleaned["episode_id"].astype(str).isin(dropped_controls)
        ].copy()
    original_groups = cleaned.set_index("episode_id")["source_event_id"].to_dict()
    control_mask = cleaned["episode_type"].eq("interictal")
    cleaned.loc[control_mask, "source_event_id"] = (
        cleaned.loc[control_mask, "episode_id"].astype(str).map(assignments)
    )
    if cleaned.loc[control_mask, "source_event_id"].isna().any():
        raise AssertionError("Every retained control needs one event group.")

    reassigned = sum(
        str(original_groups.get(episode_id)) != event_id
        for episode_id, event_id in assignments.items()
    )
    post_intervals = _episode_intervals(cleaned)
    for first_position, (_, first) in enumerate(post_intervals.iterrows()):
        later = post_intervals.iloc[first_position + 1 :]
        for _, second in later.iterrows():
            if (
                first["source_event_id"] != second["source_event_id"]
                and _intervals_overlap(first, second)
            ):
                raise AssertionError(
                    "Raw EEG interval leakage remains between "
                    f"{first['episode_id']} and {second['episode_id']}."
                )

    # Overlapping historical control episodes can contain the exact same
    # causal landmark more than once. Retain one copy so duplicated raw windows
    # cannot receive extra statistical weight.
    raw_window_key = ["recording", "landmark_seconds"]
    conflicting_labels = (
        cleaned.groupby(raw_window_key)["has_event_in_5m"].nunique() > 1
    )
    if conflicting_labels.any():
        raise AssertionError(
            "The same raw EEG landmark has conflicting positive/negative labels."
        )
    conflicting_groups = (
        cleaned.groupby(raw_window_key)["source_event_id"].nunique() > 1
    )
    if conflicting_groups.any():
        raise AssertionError(
            "The same raw EEG landmark remains in multiple event groups."
        )
    rows_before_deduplication = len(cleaned)
    cleaned = cleaned.drop_duplicates(raw_window_key, keep="first").copy()
    duplicate_raw_landmark_rows_removed = rows_before_deduplication - len(cleaned)

    # The earlier PN00 cache contained exact feature duplicates in different
    # folds. This content-level assertion catches that failure even if interval
    # metadata is later changed.
    feature_columns = [
        column for channel in channels for column in channel_columns(channel)
    ]
    feature_hash = pd.util.hash_pandas_object(
        cleaned[feature_columns], index=False
    )
    hash_groups = (
        pd.DataFrame(
            {
                "feature_hash": feature_hash.to_numpy(),
                "source_event_id": cleaned["source_event_id"].to_numpy(),
            }
        )
        .drop_duplicates()
        .groupby("feature_hash")["source_event_id"]
        .nunique()
    )
    cross_group_duplicates = int((hash_groups > 1).sum())
    if cross_group_duplicates:
        raise AssertionError(
            f"{cross_group_duplicates} exact feature patterns cross event groups."
        )
    audit: dict[str, object] = {
        "controls_before": int(len(intervals) - len(positives)),
        "controls_after": int(
            cleaned.loc[cleaned["episode_type"].eq("interictal"), "episode_id"].nunique()
        ),
        "controls_dropped_for_preictal_overlap": len(dropped_controls),
        "controls_reassigned_to_nonoverlapping_groups": reassigned,
        "duplicate_raw_landmark_rows_removed": (
            duplicate_raw_landmark_rows_removed
        ),
        "raw_interval_cross_group_overlaps_after": 0,
        "exact_feature_hashes_crossing_groups_after": cross_group_duplicates,
        "controls_per_event_after": loads,
    }
    return cleaned.reset_index(drop=True), audit


def _assign_controls_within_partition(
    frame: pd.DataFrame,
    event_ids: Sequence[str],
) -> pd.DataFrame:
    """Assign each overlapping-control component to one event in a partition."""

    assigned = frame.copy()
    controls = _episode_intervals(assigned).loc[
        lambda table: table["episode_type"].eq("interictal")
    ]
    positives = _episode_intervals(assigned).loc[
        lambda table: table["episode_type"].eq("preictal")
    ]
    if controls.empty:
        return assigned

    control_indices = controls.index.tolist()
    parent = {index: index for index in control_indices}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for position, first_index in enumerate(control_indices):
        for second_index in control_indices[position + 1 :]:
            if _intervals_overlap(
                controls.loc[first_index], controls.loc[second_index]
            ):
                union(first_index, second_index)
    components: dict[int, list[int]] = {}
    for index in control_indices:
        components.setdefault(find(index), []).append(index)

    event_locations = (
        positives.groupby(["source_event_id", "recording"], as_index=False)
        .agg(event_landmark=("last_landmark_seconds", "max"))
        .set_index("source_event_id")
    )
    loads = {str(event_id): 0 for event_id in event_ids}
    episode_assignments: dict[str, str] = {}
    ordered_components = sorted(
        components.values(),
        key=lambda indices: (
            -len(indices),
            natural_key(str(controls.loc[indices[0], "recording"])),
            float(controls.loc[indices[0], "raw_start_seconds"]),
        ),
    )
    for component in ordered_components:
        representative = controls.loc[component[0]]
        same_recording = [
            str(event_id)
            for event_id in event_ids
            if str(event_locations.loc[str(event_id), "recording"])
            == str(representative["recording"])
        ]
        if not same_recording:
            raise AssertionError(
                "A retained control recording has no seizure event in its "
                "train/test partition."
            )
        candidates = same_recording

        def assignment_key(event_id: str) -> tuple[object, ...]:
            event_landmark = float(
                event_locations.loc[event_id, "event_landmark"]
            )
            midpoint = 0.5 * (
                float(representative["raw_start_seconds"])
                + float(representative["raw_end_seconds"])
            )
            return (
                loads[event_id],
                abs(midpoint - event_landmark),
                natural_key(event_id),
            )

        chosen = min(candidates, key=assignment_key)
        for index in component:
            episode_assignments[str(controls.loc[index, "episode_id"])] = chosen
        loads[chosen] += len(component)

    control_mask = assigned["episode_type"].eq("interictal")
    assigned.loc[control_mask, "source_event_id"] = (
        assigned.loc[control_mask, "episode_id"]
        .astype(str)
        .map(episode_assignments)
    )
    if assigned.loc[control_mask, "source_event_id"].isna().any():
        raise AssertionError("Every retained control needs one event group.")
    return assigned


def chronological_patient_split(
    frame: pd.DataFrame,
    *,
    test_fraction: float = TEST_FRACTION,
    min_test_seizures: int = MIN_TEST_SEIZURES,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Hold out the last seizures in recording-disjoint test sessions.

    A prior version assigned controls to synthetic seizure groups before the
    split. That allowed train and test rows from the same recording session.
    Here seizure order chooses the requested test events, recordings containing
    those events are reserved wholly for test, and controls follow their
    recording partition. Earlier seizures in a reserved test recording are
    excluded rather than leaked into personalized training.
    """

    event_ids = ordered_seizure_ids(frame)
    n_seizures = len(event_ids)
    if n_seizures < min_test_seizures + 1:
        raise ValueError(
            f"Need at least {min_test_seizures + 1} seizures; found {n_seizures}."
        )
    n_test = max(min_test_seizures, int(math.ceil(test_fraction * n_seizures)))
    n_test = min(n_test, n_seizures - 1)
    nominal_train_ids = event_ids[:-n_test]
    test_ids = event_ids[-n_test:]

    positives = frame.loc[frame["episode_type"].eq("preictal")]
    event_recordings = (
        positives[["source_event_id", "recording"]]
        .drop_duplicates("source_event_id")
        .set_index("source_event_id")["recording"]
        .astype(str)
        .to_dict()
    )
    test_recordings = {
        event_recordings[event_id] for event_id in test_ids
    }
    train_ids = [
        event_id
        for event_id in nominal_train_ids
        if event_recordings[event_id] not in test_recordings
    ]
    if not train_ids:
        raise ValueError(
            "Recording-disjoint split leaves no personalized training seizures."
        )
    train_recordings = {event_recordings[event_id] for event_id in train_ids}
    if train_recordings & test_recordings:
        raise AssertionError("Train and test recordings must be disjoint.")

    positive_mask = frame["episode_type"].eq("preictal")
    control_mask = frame["episode_type"].eq("interictal")
    train = frame.loc[
        (positive_mask & frame["source_event_id"].isin(train_ids))
        | (control_mask & frame["recording"].astype(str).isin(train_recordings))
    ].copy()
    test = frame.loc[
        (positive_mask & frame["source_event_id"].isin(test_ids))
        | (control_mask & frame["recording"].astype(str).isin(test_recordings))
    ].copy()
    train = _assign_controls_within_partition(train, train_ids)
    test = _assign_controls_within_partition(test, test_ids)

    if set(train["recording"].astype(str)) & set(test["recording"].astype(str)):
        raise AssertionError("A recording session leaked across the split.")
    if set(train["source_event_id"]) & set(test["source_event_id"]):
        raise AssertionError("A seizure/control group leaked across the split.")
    for partition_name, partition in (("train", train), ("test", test)):
        if partition.empty or partition["has_event_in_5m"].nunique() != 2:
            raise ValueError(
                f"Recording-disjoint {partition_name} partition needs both classes."
            )

    dropped_train_ids = sorted(
        set(nominal_train_ids) - set(train_ids), key=natural_key
    )
    details: dict[str, object] = {
        "n_seizures": n_seizures,
        "n_train_seizures": len(train_ids),
        "n_nominal_train_seizures": len(nominal_train_ids),
        "n_test_seizures": len(test_ids),
        "train_event_ids": train_ids,
        "test_event_ids": test_ids,
        "train_recordings": sorted(train_recordings, key=natural_key),
        "test_recordings": sorted(test_recordings, key=natural_key),
        "train_events_excluded_for_recording_isolation": dropped_train_ids,
    }
    return (
        train.reset_index(drop=True),
        test.reset_index(drop=True),
        details,
    )


def _fit_predict_channel(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    channel: str,
) -> np.ndarray:
    model = make_sensor_model()
    columns = channel_columns(channel)
    model.fit(
        train[columns].to_numpy(dtype=float),
        train["has_event_in_5m"].to_numpy(dtype=int),
    )
    return model.predict_proba(
        validation[columns].to_numpy(dtype=float)
    )[:, 1]


def grouped_oof_sensor_probabilities(
    frame: pd.DataFrame,
    channels: Sequence[str],
    *,
    groups: Sequence[str] | np.ndarray,
    n_jobs: int = -1,
) -> np.ndarray:
    """Generate out-of-fold predictions without crossing a supplied group."""

    group_values = np.asarray(groups)
    if len(group_values) != len(frame):
        raise ValueError("groups must align with frame rows.")
    unique_groups = np.unique(group_values)
    if len(unique_groups) < 2:
        raise ValueError("At least two non-overlapping groups are required.")
    splits = list(
        LeaveOneGroupOut().split(
            np.zeros(len(frame)),
            frame["has_event_in_5m"].to_numpy(dtype=int),
            group_values,
        )
    )

    def predict_one(channel: str) -> np.ndarray:
        prediction = np.full(len(frame), np.nan, dtype=float)
        for train_index, validation_index in splits:
            train = frame.iloc[train_index]
            validation = frame.iloc[validation_index]
            if train["has_event_in_5m"].nunique() != 2:
                raise ValueError("Every training fold must contain both classes.")
            prediction[validation_index] = _fit_predict_channel(
                train, validation, channel
            )
        return prediction

    columns = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(predict_one)(channel) for channel in channels
    )
    probabilities = np.column_stack(columns)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("OOF prediction did not cover every row and channel.")
    return probabilities


def _weighted_binary_log_loss(
    truth: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    truth = np.asarray(truth, dtype=float)
    losses = -(truth * np.log(probability) + (1 - truth) * np.log(1 - probability))
    return float(np.average(losses, weights=weights))


def greedy_channel_ranking(
    truth: Sequence[int],
    sensor_probabilities: np.ndarray,
    channels: Sequence[str],
) -> tuple[list[str], pd.DataFrame]:
    """Rank channels by forward cross-validated cross-entropy improvement."""

    y = np.asarray(truth, dtype=int)
    if sensor_probabilities.shape != (len(y), len(channels)):
        raise ValueError("Probability matrix shape does not match rows/channels.")
    selected: list[int] = []
    remaining = set(range(len(channels)))
    probability_sum = np.zeros(len(y), dtype=float)
    rows: list[dict[str, object]] = []
    while remaining:
        step = len(selected) + 1
        candidates = []
        for index in sorted(remaining, key=lambda i: natural_key(channels[i])):
            ensemble = (probability_sum + sensor_probabilities[:, index]) / step
            loss = _weighted_binary_log_loss(y, ensemble)
            candidates.append((loss, natural_key(channels[index]), index))
        loss, _, added = min(candidates)
        selected.append(added)
        remaining.remove(added)
        probability_sum += sensor_probabilities[:, added]
        ensemble = probability_sum / step
        rows.append(
            {
                "rank": step,
                "channel": channels[added],
                "selected_channels": ", ".join(channels[i] for i in selected),
                "validation_cross_entropy": loss,
                "validation_auprc": float(average_precision_score(y, ensemble)),
                "validation_brier": float(brier_score_loss(y, ensemble)),
            }
        )
    return [channels[index] for index in selected], pd.DataFrame(rows)


@dataclass(frozen=True)
class PositivePlattCalibrator:
    """One-dimensional, positive-slope probability calibration."""

    slope: float = 1.0
    intercept: float = 0.0
    method: str = "identity"

    def transform(self, probability: Sequence[float]) -> np.ndarray:
        values = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
        logit = np.log(values) - np.log1p(-values)
        return np.clip(
            expit(self.slope * logit + self.intercept),
            1e-6,
            1 - 1e-6,
        )


def fit_positive_platt(
    truth: Sequence[int],
    probability: Sequence[float],
) -> PositivePlattCalibrator:
    """Fit calibration on training-only out-of-fold probabilities."""

    y = np.asarray(truth, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    if len(y) < 2 or np.unique(y).size != 2 or np.unique(p).size < 3:
        return PositivePlattCalibrator()
    x = np.log(p) - np.log1p(-p)
    prevalence = float(np.mean(y))
    initial_intercept = math.log(prevalence / (1 - prevalence)) - float(
        np.mean(x)
    )

    def objective(parameters: np.ndarray) -> float:
        slope = float(np.exp(parameters[0]))
        calibrated = expit(slope * x + parameters[1])
        return _weighted_binary_log_loss(y, calibrated)

    result = optimize.minimize(
        objective,
        np.asarray([0.0, initial_intercept]),
        method="L-BFGS-B",
        bounds=[(-5.0, 5.0), (-20.0, 20.0)],
    )
    if not result.success:
        return PositivePlattCalibrator(method="identity_fit_failure")
    return PositivePlattCalibrator(
        slope=float(np.exp(result.x[0])),
        intercept=float(result.x[1]),
        method="positive_platt_oof",
    )


@dataclass
class SensorEnsemble:
    channels: list[str]
    models: dict[str, object]
    calibrator: PositivePlattCalibrator

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        sensor_predictions = []
        for channel in self.channels:
            sensor_predictions.append(
                self.models[channel].predict_proba(
                    frame[channel_columns(channel)].to_numpy(dtype=float)
                )[:, 1]
            )
        raw = np.mean(np.column_stack(sensor_predictions), axis=1)
        return self.calibrator.transform(raw)


def fit_sensor_ensemble(
    train: pd.DataFrame,
    channels: Sequence[str],
    oof_sensor_probabilities: np.ndarray,
    all_ranked_channels: Sequence[str],
    *,
    n_jobs: int = -1,
) -> SensorEnsemble:
    """Fit selected sensor models and training-only ensemble calibration."""

    lookup = {channel: index for index, channel in enumerate(all_ranked_channels)}
    indices = [lookup[channel] for channel in channels]
    raw_oof = np.mean(oof_sensor_probabilities[:, indices], axis=1)
    calibrator = fit_positive_platt(
        train["has_event_in_5m"].to_numpy(dtype=int), raw_oof
    )

    def fit_one(channel: str) -> tuple[str, object]:
        model = make_sensor_model()
        model.fit(
            train[channel_columns(channel)].to_numpy(dtype=float),
            train["has_event_in_5m"].to_numpy(dtype=int),
        )
        return channel, model

    fitted = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(fit_one)(channel) for channel in channels
    )
    return SensorEnsemble(list(channels), dict(fitted), calibrator)


def prediction_metrics(
    frame: pd.DataFrame,
    probability: Sequence[float],
) -> dict[str, float]:
    """Return landmark- and episode-level probability metrics."""

    truth = frame["has_event_in_5m"].to_numpy(dtype=int)
    risk = np.asarray(probability, dtype=float)
    episode = (
        frame[["episode_id", "has_event_in_5m"]]
        .assign(probability=risk)
        .groupby("episode_id", as_index=False)
        .agg(
            has_event_in_5m=("has_event_in_5m", "first"),
            probability=("probability", "mean"),
        )
    )
    episode_truth = episode["has_event_in_5m"].to_numpy(dtype=int)
    episode_risk = episode["probability"].to_numpy(dtype=float)
    return {
        "cross_entropy": float(log_loss(truth, risk, labels=[0, 1])),
        "auprc": float(average_precision_score(truth, risk)),
        "auroc": float(roc_auc_score(truth, risk)),
        "brier": float(brier_score_loss(truth, risk)),
        "episode_cross_entropy": float(
            log_loss(episode_truth, episode_risk, labels=[0, 1])
        ),
        "episode_auprc": float(
            average_precision_score(episode_truth, episode_risk)
        ),
        "episode_auroc": float(roc_auc_score(episode_truth, episode_risk)),
        "episode_brier": float(
            brier_score_loss(episode_truth, episode_risk)
        ),
    }


def _evaluate_selected_and_full(
    train: pd.DataFrame,
    test: pd.DataFrame,
    channels: Sequence[str],
    selected: Sequence[str],
    oof: np.ndarray,
    *,
    n_jobs: int,
) -> tuple[dict[str, float], np.ndarray, dict[str, float], np.ndarray]:
    selected_model = fit_sensor_ensemble(
        train,
        selected,
        oof,
        channels,
        n_jobs=n_jobs,
    )
    selected_risk = selected_model.predict_proba(test)
    selected_metrics = prediction_metrics(test, selected_risk)
    full_model = fit_sensor_ensemble(
        train,
        channels,
        oof,
        channels,
        n_jobs=n_jobs,
    )
    full_risk = full_model.predict_proba(test)
    full_metrics = prediction_metrics(test, full_risk)
    return selected_metrics, selected_risk, full_metrics, full_risk


def run_personalized_generalized(
    patients: dict[str, PatientData],
    *,
    k: int,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train five P models and five leave-one-patient-out G models."""

    expected = set(TARGET_PATIENTS)
    if set(patients) != expected:
        raise ValueError(
            f"Expected exactly {sorted(expected)}; found {sorted(patients)}."
        )
    channels = shared_channels(patients.values())
    if k > len(channels):
        raise ValueError(f"K={k} exceeds {len(channels)} shared channels.")

    splits: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]] = {
        patient_id: chronological_patient_split(patient.frame)
        for patient_id, patient in patients.items()
    }
    result_rows: list[dict[str, object]] = []
    ranking_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    event_loss_rows: list[dict[str, object]] = []

    for target in TARGET_PATIENTS:
        p_train, target_test, split = splits[target]
        p_oof = grouped_oof_sensor_probabilities(
            p_train,
            channels,
            groups=p_train["source_event_id"].to_numpy(),
            n_jobs=n_jobs,
        )
        p_ranking, p_trace = greedy_channel_ranking(
            p_train["has_event_in_5m"].to_numpy(dtype=int),
            p_oof,
            channels,
        )
        p_selected = p_ranking[:k]
        p_metrics, p_risk, p_full_metrics, p_full_risk = (
            _evaluate_selected_and_full(
                p_train,
                target_test,
                channels,
                p_selected,
                p_oof,
                n_jobs=n_jobs,
            )
        )
        p_trace.insert(0, "selection_population", "personalized_training_events")
        p_trace.insert(0, "model", "P")
        p_trace.insert(0, "target_patient", target)

        g_train = pd.concat(
            [
                patient.frame
                for patient_id, patient in patients.items()
                if patient_id != target
            ],
            ignore_index=True,
        )
        if target in set(g_train["patient_id"]):
            raise AssertionError(f"{target} leaked into its G training data.")
        g_oof = grouped_oof_sensor_probabilities(
            g_train,
            channels,
            groups=g_train["patient_id"].to_numpy(),
            n_jobs=n_jobs,
        )
        g_ranking, g_trace = greedy_channel_ranking(
            g_train["has_event_in_5m"].to_numpy(dtype=int),
            g_oof,
            channels,
        )
        g_selected = g_ranking[:k]
        g_metrics, g_risk, g_full_metrics, g_full_risk = (
            _evaluate_selected_and_full(
                g_train,
                target_test,
                channels,
                g_selected,
                g_oof,
                n_jobs=n_jobs,
            )
        )
        g_trace.insert(0, "selection_population", "other_four_patients")
        g_trace.insert(0, "model", "G")
        g_trace.insert(0, "target_patient", target)
        ranking_frames.extend([p_trace, g_trace])

        test_truth = target_test["has_event_in_5m"].to_numpy(dtype=int)

        def constant_cross_entropy(probability: float) -> float:
            return float(
                log_loss(
                    test_truth,
                    np.repeat(probability, len(target_test)),
                    labels=[0, 1],
                )
            )

        test_prevalence = float(test_truth.mean())
        p_train_prevalence = float(p_train["has_event_in_5m"].mean())
        g_train_prevalence = float(g_train["has_event_in_5m"].mean())
        result_rows.append(
            {
                "patient_id": target,
                "n_seizures": split["n_seizures"],
                "n_train_seizures_P": split["n_train_seizures"],
                "n_nominal_train_seizures_P": split[
                    "n_nominal_train_seizures"
                ],
                "n_test_seizures": split["n_test_seizures"],
                "test_event_ids": ", ".join(split["test_event_ids"]),
                "train_recordings_P": ", ".join(split["train_recordings"]),
                "test_recordings": ", ".join(split["test_recordings"]),
                "train_events_excluded_for_recording_isolation": ", ".join(
                    split["train_events_excluded_for_recording_isolation"]
                ),
                "n_train_patients_G": int(g_train["patient_id"].nunique()),
                "n_train_seizures_G": int(
                    g_train.loc[
                        g_train["episode_type"].eq("preictal"),
                        ["patient_id", "source_event_id"],
                    ]
                    .drop_duplicates()
                    .shape[0]
                ),
                "k": k,
                "personalized_channels": ", ".join(p_selected),
                "generalized_channels": ", ".join(g_selected),
                "test_positive_fraction": test_prevalence,
                "test_prevalence_oracle_cross_entropy": (
                    constant_cross_entropy(test_prevalence)
                ),
                "P_training_prevalence": p_train_prevalence,
                "P_training_prevalence_baseline_cross_entropy": (
                    constant_cross_entropy(p_train_prevalence)
                ),
                "G_training_prevalence": g_train_prevalence,
                "G_training_prevalence_baseline_cross_entropy": (
                    constant_cross_entropy(g_train_prevalence)
                ),
                **{f"P_{key}": value for key, value in p_metrics.items()},
                **{f"G_{key}": value for key, value in g_metrics.items()},
                **{f"P_full_{key}": value for key, value in p_full_metrics.items()},
                **{f"G_full_{key}": value for key, value in g_full_metrics.items()},
                "G_minus_P_cross_entropy": (
                    g_metrics["cross_entropy"] - p_metrics["cross_entropy"]
                ),
                "P_reduced_minus_full_cross_entropy": (
                    p_metrics["cross_entropy"] - p_full_metrics["cross_entropy"]
                ),
                "G_reduced_minus_full_cross_entropy": (
                    g_metrics["cross_entropy"] - g_full_metrics["cross_entropy"]
                ),
            }
        )

        base_prediction = target_test[
            [
                "patient_id",
                "source_event_id",
                "episode_id",
                "episode_type",
                "recording",
                "landmark_step",
                "time_to_event_seconds",
                "has_event_in_5m",
            ]
        ].reset_index(drop=True)
        for model_name, selected_risk, full_risk in (
            ("P", p_risk, p_full_risk),
            ("G", g_risk, g_full_risk),
        ):
            prediction_frames.append(
                base_prediction.assign(
                    model=model_name,
                    selected_k_probability=selected_risk,
                    full_29_probability=full_risk,
                )
            )

        y = target_test["has_event_in_5m"].to_numpy(dtype=int)
        p_losses = -(
            y * np.log(np.clip(p_risk, 1e-7, 1 - 1e-7))
            + (1 - y) * np.log(np.clip(1 - p_risk, 1e-7, 1 - 1e-7))
        )
        g_losses = -(
            y * np.log(np.clip(g_risk, 1e-7, 1 - 1e-7))
            + (1 - y) * np.log(np.clip(1 - g_risk, 1e-7, 1 - 1e-7))
        )
        loss_frame = target_test[["source_event_id"]].copy()
        loss_frame["P_cross_entropy"] = p_losses
        loss_frame["G_cross_entropy"] = g_losses
        for event_id, group in loss_frame.groupby("source_event_id", sort=False):
            event_loss_rows.append(
                {
                    "patient_id": target,
                    "source_event_id": event_id,
                    "P_cross_entropy": float(group["P_cross_entropy"].mean()),
                    "G_cross_entropy": float(group["G_cross_entropy"].mean()),
                    "G_minus_P_cross_entropy": float(
                        (
                            group["G_cross_entropy"]
                            - group["P_cross_entropy"]
                        ).mean()
                    ),
                }
            )

    results = pd.DataFrame(result_rows)
    rankings = pd.concat(ranking_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    event_losses = pd.DataFrame(event_loss_rows)
    return results, rankings, predictions, event_losses


def data_quality_summary(
    patients: dict[str, PatientData],
) -> pd.DataFrame:
    """Summarize cached artifact filtering for the five comparison patients."""

    rows = []
    for patient_id in TARGET_PATIENTS:
        patient = patients[patient_id]
        frame = patient.frame
        shared = shared_channels(patients.values())
        feature_columns = [
            column for channel in shared for column in channel_columns(channel)
        ]
        quality_columns = [
            f"ch::{channel}::usable_channel_fraction::mean"
            for channel in shared
        ]
        values = frame[feature_columns].to_numpy(dtype=float)
        quality = frame[quality_columns].to_numpy(dtype=float)
        leakage_audit = frame.attrs.get("leakage_audit", {})
        rows.append(
            {
                "patient_id": patient_id,
                "landmark_rows": len(frame),
                "episodes": int(frame["episode_id"].nunique()),
                "seizures": int(
                    frame.loc[
                        frame["episode_type"].eq("preictal"),
                        "source_event_id",
                    ].nunique()
                ),
                "available_channels": len(patient.channels),
                "shared_channels_used": len(shared),
                "sensor_contexts_with_any_rejected_5s_window_fraction": float(
                    np.mean(quality < 1.0)
                ),
                "feature_value_missing_fraction": float(np.isnan(values).mean()),
                "landmarks_with_any_missing_feature_fraction": float(
                    np.mean(np.isnan(values).any(axis=1))
                ),
                "infinite_feature_values": int(np.isinf(values).sum()),
                "controls_reassigned_to_prevent_split_leakage": int(
                    leakage_audit.get(
                        "controls_reassigned_to_nonoverlapping_groups", 0
                    )
                ),
                "controls_dropped_for_preictal_overlap": int(
                    leakage_audit.get(
                        "controls_dropped_for_preictal_overlap", 0
                    )
                ),
                "duplicate_raw_landmark_rows_removed": int(
                    leakage_audit.get(
                        "duplicate_raw_landmark_rows_removed", 0
                    )
                ),
                "raw_interval_cross_group_overlaps_after": int(
                    leakage_audit.get(
                        "raw_interval_cross_group_overlaps_after", 0
                    )
                ),
                "exact_feature_hashes_crossing_groups_after": int(
                    leakage_audit.get(
                        "exact_feature_hashes_crossing_groups_after", 0
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def channel_set_summary(
    results: pd.DataFrame,
    all_channels: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return selected-channel frequencies and strict set intersections."""

    frequency_rows = []
    sets: dict[str, list[set[str]]] = {"P": [], "G": []}
    for model, column in (
        ("P", "personalized_channels"),
        ("G", "generalized_channels"),
    ):
        for row in results.itertuples(index=False):
            selected = {
                part.strip()
                for part in str(getattr(row, column)).split(",")
                if part.strip()
            }
            sets[model].append(selected)
        for channel in all_channels:
            frequency_rows.append(
                {
                    "model": model,
                    "channel": channel,
                    "patients_selected": sum(
                        channel in selected for selected in sets[model]
                    ),
                    "selection_fraction": float(
                        np.mean([channel in selected for selected in sets[model]])
                    ),
                }
            )
    core_rows = []
    for model in ("P", "G"):
        intersection = set.intersection(*sets[model]) if sets[model] else set()
        core_rows.append(
            {
                "model": model,
                "definition": "intersection across all five selected channel sets",
                "n_core_channels": len(intersection),
                "core_channels": ", ".join(
                    sorted(intersection, key=natural_key)
                ),
            }
        )
    combined_sets = [*sets["P"], *sets["G"]]
    combined_intersection = (
        set.intersection(*combined_sets) if combined_sets else set()
    )
    core_rows.append(
        {
            "model": "P_and_G",
            "definition": "intersection across all ten P and G channel sets",
            "n_core_channels": len(combined_intersection),
            "core_channels": ", ".join(
                sorted(combined_intersection, key=natural_key)
            ),
        }
    )
    return pd.DataFrame(frequency_rows), pd.DataFrame(core_rows)


def paired_patient_bootstrap(
    results: pd.DataFrame,
    *,
    repeats: int = 10_000,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Bootstrap the mean paired G-minus-P cross-entropy difference by patient."""

    differences = results["G_minus_P_cross_entropy"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        differences,
        size=(repeats, len(differences)),
        replace=True,
    ).mean(axis=1)
    observed_absolute_mean = abs(float(differences.mean()))
    sign_flip_means = np.asarray(
        [
            abs(float(np.mean(differences * np.asarray(signs))))
            for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
        ]
    )
    return {
        "mean_P_cross_entropy": float(results["P_cross_entropy"].mean()),
        "mean_G_cross_entropy": float(results["G_cross_entropy"].mean()),
        "median_P_cross_entropy": float(results["P_cross_entropy"].median()),
        "median_G_cross_entropy": float(results["G_cross_entropy"].median()),
        "mean_G_minus_P_cross_entropy": float(differences.mean()),
        "median_G_minus_P_cross_entropy": float(np.median(differences)),
        "bootstrap_95_ci_lower": float(np.percentile(samples, 2.5)),
        "bootstrap_95_ci_upper": float(np.percentile(samples, 97.5)),
        "exact_two_sided_sign_flip_p": float(
            np.mean(sign_flip_means >= observed_absolute_mean - 1e-12)
        ),
        "patients_P_lower_loss": int(np.sum(differences > 0)),
        "patients_G_lower_loss": int(np.sum(differences < 0)),
        "patients_tied": int(np.sum(np.isclose(differences, 0))),
        "n_patients": len(differences),
        "bootstrap_repeats": repeats,
        "random_seed": seed,
        "uncertainty_note": (
            "Only five patients are available. The bootstrap interval is "
            "descriptive; the exact sign-flip test is the safer small-sample check."
        ),
    }


@dataclass(frozen=True)
class KFinderOutput:
    k: int
    channels: list[str]
    subject_scores: pd.DataFrame
    loss_curve: pd.DataFrame
    channel_paths: pd.DataFrame


def _macro_subject_ap(
    truth: np.ndarray,
    probability: np.ndarray,
    subjects: np.ndarray,
) -> float:
    return float(
        np.mean(
            [
                average_precision_score(
                    truth[subjects == subject],
                    probability[subjects == subject],
                )
                for subject in np.unique(subjects)
            ]
        )
    )


def _greedy_ap_path(
    truth: np.ndarray,
    sensor_probabilities: np.ndarray,
    subjects: np.ndarray,
) -> list[tuple[int, ...]]:
    chosen: tuple[int, ...] = ()
    remaining = set(range(sensor_probabilities.shape[1]))
    probability_sum = np.zeros(len(truth), dtype=float)
    path = []
    while remaining:
        k = len(chosen) + 1
        candidates = []
        for sensor in sorted(remaining):
            probability = (
                probability_sum + sensor_probabilities[:, sensor]
            ) / k
            candidates.append(
                (
                    _macro_subject_ap(truth, probability, subjects),
                    -sensor,
                    sensor,
                )
            )
        _, _, added = max(candidates)
        chosen = (*chosen, added)
        remaining.remove(added)
        probability_sum += sensor_probabilities[:, added]
        path.append(chosen)
    return path


def run_k_finder(
    feature_cache_npz: str | Path,
    *,
    allowed_loss: float = ALLOWED_AUPRC_LOSS,
    inner_splits: int = 4,
    n_jobs: int = -1,
) -> KFinderOutput:
    """Find the smallest K with observed AUPRC loss <= limit for every patient."""

    cache = np.load(feature_cache_npz, allow_pickle=False)
    X = np.asarray(cache["X"], dtype=float)
    y = np.asarray(cache["y"], dtype=int)
    subjects = np.asarray(cache["subjects"]).astype(str)
    channels = np.asarray(cache["channels"]).astype(str).tolist()
    if X.ndim != 3 or X.shape[1] != len(channels):
        raise ValueError("K-finder cache must have rows × sensors × features.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("K-finder target must contain 0 and 1.")
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) < 3:
        raise ValueError("K-finder needs at least three patients.")
    scores = np.full((len(unique_subjects), len(channels)), np.nan, dtype=float)
    subject_position = {
        subject: index for index, subject in enumerate(unique_subjects)
    }
    channel_path_rows: list[dict[str, object]] = []

    outer = LeaveOneGroupOut()
    for train_index, test_index in outer.split(X, y, subjects):
        train_subjects = subjects[train_index]
        n_splits = min(inner_splits, len(np.unique(train_subjects)))
        inner = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_SEED,
        )
        inner_folds = list(
            inner.split(X[train_index], y[train_index], train_subjects)
        )

        def inner_sensor(sensor: int) -> np.ndarray:
            prediction = np.full(len(train_index), np.nan, dtype=float)
            for fit, validation in inner_folds:
                model = clone(make_sensor_model())
                model.fit(
                    X[train_index][fit, sensor, :],
                    y[train_index][fit],
                )
                prediction[validation] = model.predict_proba(
                    X[train_index][validation, sensor, :]
                )[:, 1]
            return prediction

        inner_predictions = np.column_stack(
            Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(inner_sensor)(sensor)
                for sensor in range(len(channels))
            )
        )
        path = _greedy_ap_path(
            y[train_index], inner_predictions, train_subjects
        )
        test_subject = np.unique(subjects[test_index])
        if len(test_subject) != 1:
            raise AssertionError("LOSO fold must contain one patient.")
        for rank, sensor_indices in enumerate(path, start=1):
            channel_path_rows.append(
                {
                    "held_out_patient": str(test_subject[0]),
                    "rank": rank,
                    "channel_added": channels[sensor_indices[-1]],
                    "selected_channels": ", ".join(
                        channels[index] for index in sensor_indices
                    ),
                }
            )

        def outer_sensor(sensor: int) -> np.ndarray:
            model = clone(make_sensor_model())
            model.fit(X[train_index, sensor, :], y[train_index])
            return model.predict_proba(X[test_index, sensor, :])[:, 1]

        outer_predictions = np.column_stack(
            Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(outer_sensor)(sensor)
                for sensor in range(len(channels))
            )
        )
        row = subject_position[test_subject[0]]
        for count, sensor_indices in enumerate(path, start=1):
            scores[row, count - 1] = average_precision_score(
                y[test_index],
                np.mean(outer_predictions[:, sensor_indices], axis=1),
            )

    if not np.isfinite(scores).all():
        raise RuntimeError("K-finder did not score every patient/count.")
    full = scores[:, -1]
    losses = full[:, None] - scores
    worst = np.max(losses, axis=0)
    acceptable = worst <= allowed_loss
    candidates = np.flatnonzero(acceptable)
    if not len(candidates):
        raise RuntimeError("The full montage should always satisfy the limit.")
    selected_k = int(candidates[0] + 1)
    subject_scores = pd.DataFrame(
        scores,
        index=unique_subjects,
        columns=np.arange(1, len(channels) + 1),
    )
    subject_scores.index.name = "patient_id"
    subject_scores.insert(
        0,
        "positive_fraction",
        [
            float(np.mean(y[subjects == subject]))
            for subject in unique_subjects
        ],
    )
    loss_curve = pd.DataFrame(
        {
            "k": np.arange(1, len(channels) + 1),
            "mean_held_out_auprc": scores.mean(axis=0),
            "full_montage_mean_auprc": float(full.mean()),
            "worst_patient_auprc_loss_from_full": worst,
            "meets_0.03_worst_patient_limit": acceptable,
        }
    )
    return KFinderOutput(
        selected_k,
        channels,
        subject_scores.reset_index(),
        loss_curve,
        pd.DataFrame(channel_path_rows),
    )
