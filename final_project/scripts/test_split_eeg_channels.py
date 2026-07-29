import csv
from pathlib import Path

import numpy as np

from split_eeg_channels import (
    MANIFEST_FIELDS,
    minmax_envelope,
    read_split_eeg_segment,
)


def _write_split_recording(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    edf = raw / "PN99" / "PN99-1.edf"
    edf.parent.mkdir(parents=True)
    split = raw / "splitdata" / "PN99" / "PN99-1"
    split.mkdir(parents=True)
    rows = []
    for number, values in enumerate(
        (np.arange(20, dtype=np.int16), np.arange(20, dtype=np.int16) + 10),
        start=1,
    ):
        filename = f"channel_{number:02d}_X{number}.npy"
        np.save(split / filename, values)
        rows.append(
            {
                "channel_number": number,
                "edf_signal_index": number - 1,
                "channel_label": f"EEG X{number}",
                "canonical_name": f"X{number}",
                "filename": filename,
                "sample_rate_hz": 4,
                "sample_count": len(values),
                "duration_seconds": 5,
                "digital_minimum": -32768,
                "digital_maximum": 32767,
                "physical_minimum": -3276.8,
                "physical_maximum": 3276.7,
                "physical_unit": "uV",
                "scale_to_physical": 0.1,
                "offset_to_physical": 0,
            }
        )
    with (split / "channel_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return edf


def test_split_segment_reader_uses_manifest_calibration(tmp_path):
    edf = _write_split_recording(tmp_path)
    data, sample_rate, labels = read_split_eeg_segment(edf, 1.0, 2.0)

    assert data.shape == (2, 8)
    assert sample_rate == 4
    assert labels == ["EEG X1", "EEG X2"]
    np.testing.assert_allclose(data[0], np.arange(4, 12) * 0.1)
    np.testing.assert_allclose(data[1], np.arange(14, 22) * 0.1)


def test_minmax_envelope_keeps_each_block_extreme():
    _, lower, upper = minmax_envelope(
        np.asarray([0, 9, 2, -8, 4], dtype=np.int16),
        sample_rate=1,
        max_points=3,
    )
    np.testing.assert_array_equal(lower, [0, -8, 4])
    np.testing.assert_array_equal(upper, [9, 2, 4])
