"""Create a readable full-recording overview of all 31 exported PN00-1 channels.

The recording is 43.75 minutes long at 512 Hz, so plotting every point would
hide short peaks beneath overplotting.  Each panel instead shows a consecutive
min/max envelope (about 0.5 seconds per bin by default), preserving the time
and amplitude of transient excursions while making the whole recording visible.

Run from the repository root:

    .\\.venv\\Scripts\\python.exe final_project/scripts/plot_pn00_1_channels.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHANNEL_DIR = ROOT / "data" / "processed" / "PN00-1_split_channels"
DEFAULT_OUTPUT = ROOT / "results" / "PN00-1_all_31_channel_overview.png"


def minmax_envelope(values: np.ndarray, sample_rate: float, max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a compact time/min/max representation without random subsampling."""
    block_size = int(np.ceil(values.size / max_points))
    block_count = int(np.ceil(values.size / block_size))
    padded_size = block_count * block_size
    padded = np.pad(
        values,
        (0, padded_size - values.size),
        mode="constant",
        constant_values=np.nan,
    ).reshape(block_count, block_size)
    midpoint_seconds = ((np.arange(block_count) * block_size) + (block_size - 1) / 2) / sample_rate
    return midpoint_seconds / 60, np.nanmin(padded, axis=1), np.nanmax(padded, axis=1)


def plot_channels(channel_dir: Path, output: Path, max_points: int) -> None:
    manifest_path = channel_dir / "channel_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != 31:
        raise ValueError(f"Expected 31 channels in {manifest_path}, found {len(manifest)}")

    fig, axes = plt.subplots(8, 4, figsize=(19, 22), sharex=True)
    axes = axes.ravel()
    for axis, row in zip(axes, manifest.itertuples(index=False)):
        digital = np.load(channel_dir / row.filename, mmap_mode="r")
        physical = digital.astype(np.float32) * row.scale_to_physical + row.offset_to_physical
        minutes, lower, upper = minmax_envelope(physical, row.sample_rate_hz, max_points)
        axis.fill_between(minutes, lower, upper, color="#2a6fbb", alpha=0.55, linewidth=0)
        axis.plot(minutes, (lower + upper) / 2, color="#123a63", linewidth=0.35)
        signal_type = "EEG" if row.is_eeg else "Auxiliary"
        axis.set_title(f"{int(row.export_channel):02d}. {row.channel_label} ({signal_type})", fontsize=9, loc="left", pad=3)
        axis.set_ylabel(row.physical_unit, fontsize=7)
        axis.tick_params(axis="both", labelsize=7, length=2)
        axis.grid(axis="x", color="0.90", linewidth=0.5)
    for axis in axes[len(manifest) :]:
        axis.set_visible(False)
    for axis in axes[-4:]:
        if axis.get_visible():
            axis.set_xlabel("Minutes from recording start", fontsize=8)
    fig.suptitle(
        "PN00-1: full-recording overview of 31 exported channels\n"
        "Each panel is a consecutive min/max envelope; no random samples were selected.",
        fontsize=15,
        y=0.995,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel-dir", type=Path, default=DEFAULT_CHANNEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-points-per-channel",
        type=int,
        default=5_000,
        help="Maximum min/max envelope bins per channel (default: 5000).",
    )
    args = parser.parse_args()
    if args.max_points_per_channel < 2:
        parser.error("--max-points-per-channel must be at least 2")
    plot_channels(args.channel_dir, args.output, args.max_points_per_channel)


if __name__ == "__main__":
    main()
