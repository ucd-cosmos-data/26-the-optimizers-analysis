from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import channel_features as cf
from src import eeg_io


def test_quality_mask_rejects_flat_extreme_peak_scale_and_nonfinite() -> None:
    samples = 320
    phase = np.linspace(0, 8 * np.pi, samples, endpoint=False)
    healthy = 20 * np.sin(phase)
    flat = np.zeros(samples)
    extreme_scale = 500 * np.sin(phase)
    extreme_peak = healthy.copy()
    extreme_peak[samples // 2] = 2_000
    nonfinite = healthy.copy()
    nonfinite[10] = np.nan

    keep = cf.quality_mask(
        np.vstack(
            [healthy, flat, extreme_scale, extreme_peak, nonfinite]
        )
    )

    assert keep.tolist() == [True, False, False, False, False]


def test_micro_features_have_exact_final_schema_and_mask_artifacts() -> None:
    sample_rate = 64
    seconds = 5
    time = np.arange(sample_rate * seconds) / sample_rate
    healthy = (
        15 * np.sin(2 * np.pi * 3 * time)
        + 5 * np.sin(2 * np.pi * 10 * time)
    )
    flat = np.zeros_like(healthy)
    nonfinite = healthy.copy()
    nonfinite[0] = np.inf

    features = cf.per_channel_micro_features(
        np.vstack([healthy, flat, nonfinite]), sample_rate
    )

    assert len(cf.FEATURE_NAMES) == 7
    assert "relative_low_gamma" not in cf.FEATURE_NAMES
    assert features.shape == (3, 7)
    assert np.isfinite(features[0]).all()
    assert np.isnan(features[1, :-1]).all()
    assert features[1, -1] == 0
    assert np.isnan(features[2, :-1]).all()
    assert features[2, -1] == 0
    assert len(cf.channel_columns("C3")) == 21


def test_episode_landmarks_are_sixty_rows_and_use_causal_context(
    monkeypatch,
) -> None:
    channels = ["C3", "C4"]
    micro = np.empty((84, 2, 7), dtype=float)
    for window in range(84):
        micro[window] = window

    def fake_read(
        edf_path,
        start_seconds,
        duration_seconds,
        *,
        channels,
        use_split_cache,
        split_root,
    ):
        assert edf_path == "recording.edf"
        assert start_seconds == 80
        assert duration_seconds == 420
        assert channels == ["C3", "C4"]
        assert not use_split_cache
        return np.zeros((2, 1)), 512.0, list(channels)

    monkeypatch.setattr(eeg_io, "read_eeg_segment", fake_read)
    monkeypatch.setattr(
        cf,
        "segment_channel_features",
        lambda data, source_sample_rate, config: micro,
    )
    episode = pd.Series(
        {
            "episode_id": "PN00_S01_preictal",
            "patient_id": "PN00",
            "source_event_id": "PN00_S01",
            "episode_type": "preictal",
            "recording": "PN00-1.edf",
            "edf_path": "recording.edf",
            "anchor_seconds": 200.0,
            "event_onset_seconds": 500.0,
        }
    )

    frame = cf.episode_channel_landmarks(episode, channels)

    assert frame.shape == (60, 10 + 2 * 21)
    assert frame["landmark_step"].iloc[[0, -1]].tolist() == [0, 59]
    assert frame["landmark_seconds"].iloc[[0, -1]].tolist() == [
        200.0,
        495.0,
    ]
    assert frame["time_to_event_seconds"].iloc[[0, -1]].tolist() == [
        300.0,
        5.0,
    ]
    assert frame["event_bin"].iloc[[0, -1]].tolist() == [59, 0]
    assert frame["has_event_in_5m"].eq(1).all()
    # The first context contains windows 0..23; the next contains 1..24.
    assert frame["ch::C3::relative_delta::mean"].iloc[:2].tolist() == [
        11.5,
        12.5,
    ]
    assert frame["ch::C3::relative_delta::last"].iloc[:2].tolist() == [
        23.0,
        24.0,
    ]


def test_manifest_rebases_edfs_and_reproduces_historical_control_groups(
    tmp_path,
) -> None:
    project = tmp_path / "project"
    raw = project / "data" / "raw" / "PN00"
    raw.mkdir(parents=True)
    recording = raw / "PN00-1.edf"
    recording.write_bytes(b"edf")
    rows = [
        {
            "episode_id": "PN00_S01_preictal",
            "patient_id": "PN00",
            "source_event_id": "PN00_S01",
            "episode_type": "preictal",
            "recording": recording.name,
            "edf_path": r"C:\stale\PN00-1.edf",
            "anchor_seconds": 100,
            "event_onset_seconds": 400,
        },
        {
            "episode_id": "PN00_S02_preictal",
            "patient_id": "PN00",
            "source_event_id": "PN00_S02",
            "episode_type": "preictal",
            "recording": recording.name,
            "edf_path": r"C:\stale\PN00-1.edf",
            "anchor_seconds": 500,
            "event_onset_seconds": 800,
        },
    ]
    rows.extend(
        {
            "episode_id": f"PN00_interictal_{index:02d}",
            "patient_id": "PN00",
            "source_event_id": "",
            "episode_type": "interictal",
            "recording": recording.name,
            "edf_path": r"C:\stale\PN00-1.edf",
            "anchor_seconds": 1_000 + 500 * index,
            "event_onset_seconds": np.nan,
        }
        for index in range(1, 5)
    )
    manifest_path = project / "metadata" / "episode_manifest.csv"
    manifest_path.parent.mkdir()
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    manifest = cf.load_episode_manifest(
        manifest_path, project_root=project
    )

    assert manifest["edf_path"].eq(str(recording.resolve())).all()
    controls = manifest.loc[
        manifest["episode_type"].eq("interictal")
    ].sort_values("episode_id")
    assert controls["source_event_id"].tolist() == [
        "PN00_S01",
        "PN00_S01",
        "PN00_S02",
        "PN00_S02",
    ]


def _fake_landmark_frame(
    episode: pd.Series, channels: list[str]
) -> pd.DataFrame:
    rows = []
    for step in range(60):
        landmark = float(episode["anchor_seconds"]) + 5 * step
        row = {
            "episode_id": str(episode["episode_id"]),
            "patient_id": str(episode["patient_id"]),
            "source_event_id": str(episode["source_event_id"]),
            "episode_type": str(episode["episode_type"]),
            "recording": str(episode["recording"]),
            "landmark_step": step,
            "landmark_seconds": landmark,
            "time_to_event_seconds": (
                float(episode["event_onset_seconds"]) - landmark
            ),
            "event_bin": 59 - step,
            "has_event_in_5m": 1,
        }
        for channel in channels:
            for column in cf.channel_columns(channel):
                row[column] = landmark
        rows.append(row)
    return pd.DataFrame(rows)


def test_cache_contract_reuse_and_manifest_content_invalidation(
    tmp_path, monkeypatch
) -> None:
    project = tmp_path / "project"
    recording = project / "data" / "raw" / "PN00" / "PN00-1.edf"
    recording.parent.mkdir(parents=True)
    recording.write_bytes(b"small deterministic source")
    manifest = pd.DataFrame(
        [
            {
                "episode_id": "PN00_S01_preictal",
                "patient_id": "PN00",
                "source_event_id": "PN00_S01",
                "episode_type": "preictal",
                "recording": recording.name,
                "edf_path": str(recording),
                "anchor_seconds": 100.0,
                "event_onset_seconds": 400.0,
            }
        ]
    )
    calls = []
    channels = ["C3", "C4"]
    monkeypatch.setattr(
        cf,
        "common_patient_channels",
        lambda *args, **kwargs: list(channels),
    )

    def fake_episode(episode, channel_names, config, **kwargs):
        calls.append(float(episode["anchor_seconds"]))
        return _fake_landmark_frame(episode, list(channel_names))

    monkeypatch.setattr(cf, "episode_channel_landmarks", fake_episode)
    cache_dir = project / "cache"

    first = cf.build_patient_feature_data(
        manifest,
        cache_dir,
        project_root=project,
        verbose=False,
    )
    second = cf.build_patient_feature_data(
        manifest,
        cache_dir,
        project_root=project,
        verbose=False,
    )

    assert calls == [100.0]
    assert first.frame.equals(second.frame)
    cache_path = cache_dir / "PN00_channel_landmarks.joblib"
    signature_path = cache_dir / "PN00_channel_landmarks.json"
    payload = joblib.load(cache_path)
    assert set(payload) == {
        "frame",
        "channel_names",
        "channel_feature_columns",
    }
    assert payload["frame"].shape == (60, 10 + 2 * 21)
    signature = json.loads(signature_path.read_text())
    assert signature["features_per_channel"] == 21
    assert signature["signal_source"] == "raw_edf"
    assert len(signature["manifest_sha256"]) == 64
    assert len(signature["source_files"][0]["sha256"]) == 64
    assert len(signature["cache_payload_sha256"]) == 64
    assert signature["source_files"][0]["recording"] == (
        "data/raw/PN00/PN00-1.edf"
    )

    changed = manifest.copy()
    changed.loc[0, "anchor_seconds"] = 105.0
    third = cf.build_patient_feature_data(
        changed,
        cache_dir,
        project_root=project,
        verbose=False,
    )

    assert calls == [100.0, 105.0]
    assert third.frame["landmark_seconds"].iloc[0] == 105.0

    source_bytes = bytearray(recording.read_bytes())
    source_bytes[0] ^= 1
    recording.write_bytes(source_bytes)
    cf.build_patient_feature_data(
        changed,
        cache_dir,
        project_root=project,
        verbose=False,
    )

    # Same manifest and file size, different source bytes: rebuild required.
    assert calls == [100.0, 105.0, 105.0]


def test_optional_split_cache_applies_physical_calibration(tmp_path) -> None:
    edf = tmp_path / "data" / "raw" / "PN00" / "PN00-1.edf"
    edf.parent.mkdir(parents=True)
    edf.write_bytes(b"not read")
    split_root = tmp_path / "splitdata"
    directory = split_root / "PN00" / "PN00-1"
    directory.mkdir(parents=True)
    digital = np.arange(20, dtype=np.int16)
    np.save(directory / "channel_01_C3.npy", digital)
    fields = [
        "channel_label",
        "canonical_name",
        "filename",
        "sample_rate_hz",
        "sample_count",
        "physical_unit",
        "scale_to_physical",
        "offset_to_physical",
    ]
    with (directory / "channel_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "channel_label": "EEG C3",
                "canonical_name": "C3",
                "filename": "channel_01_C3.npy",
                "sample_rate_hz": 10,
                "sample_count": 20,
                "physical_unit": "uV",
                "scale_to_physical": 0.5,
                "offset_to_physical": -2.0,
            }
        )

    values, rate, labels = eeg_io.read_split_segment(
        edf,
        0.5,
        0.4,
        channels=["C3"],
        split_root=split_root,
    )

    assert rate == 10
    assert labels == ["C3"]
    np.testing.assert_array_equal(
        values[0], digital[5:9].astype(float) * 0.5 - 2.0
    )
