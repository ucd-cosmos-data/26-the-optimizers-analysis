"""Create a readable full-recording overview of the split PN00-1 EEG channels.

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
import pyedflib


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHANNEL_DIR = ROOT / "data" / "raw" / "splitdata" / "PN00" / "PN00-1"
DEFAULT_OUTPUT = DEFAULT_CHANNEL_DIR / "channel_overview.png"
DEFAULT_EDF = ROOT / "data" / "raw" / "PN00" / "PN00-1.edf"


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


def _signal_type(label: str) -> str:
    return "EEG" if str(label).upper().startswith("EEG") else "Auxiliary"


def plot_channels(
    channel_dir: Path,
    output: Path,
    max_points: int,
    *,
    edf_path: Path | None = None,
    signal_count: int = 31,
    landscape: bool = False,
) -> None:
    if edf_path is None:
        manifest_path = channel_dir / "channel_manifest.csv"
        manifest = pd.read_csv(manifest_path)
        if len(manifest) not in (29, 31):
            raise ValueError(
                f"Expected 29 or 31 EEG channels in {manifest_path}, "
                f"found {len(manifest)}."
            )
        channel_rows = list(manifest.itertuples(index=False))
    else:
        reader = pyedflib.EdfReader(str(edf_path))
        try:
            if signal_count > reader.signals_in_file:
                raise ValueError(
                    f"Requested {signal_count} signals, but {edf_path} contains "
                    f"only {reader.signals_in_file}."
                )
            channel_rows = [
                {
                    "channel_number": index + 1,
                    "channel_label": str(reader.getLabel(index)).strip(),
                    "physical_unit": str(reader.getPhysicalDimension(index)),
                    "sample_rate_hz": float(reader.getSampleFrequency(index)),
                    "physical": reader.readSignal(index),
                }
                for index in range(signal_count)
            ]
        finally:
            reader.close()

    rows, columns = (5, 7) if landscape else (8, 4)
    figsize = (22.5, 15) if landscape else (19, 22)
    if len(channel_rows) > rows * columns:
        raise ValueError(
            f"The {rows}-by-{columns} layout cannot hold "
            f"{len(channel_rows)} channels."
        )

    fig, axes_grid = plt.subplots(rows, columns, figsize=figsize, sharex=True)
    axes = axes_grid.ravel()
    plot_axes = list(axes[: len(channel_rows)])
    if landscape and len(channel_rows) == 31:
        # Four complete rows hold channels 1-28. Center the remaining three in
        # the final row so the wider 5-by-7 layout remains visually balanced.
        plot_axes = list(axes[:28]) + list(axes[30:33])
    for axis, row in zip(plot_axes, channel_rows):
        if isinstance(row, dict):
            physical = row["physical"]
            sample_rate = row["sample_rate_hz"]
            channel_number = row["channel_number"]
            channel_label = row["channel_label"]
            physical_unit = row["physical_unit"]
        else:
            digital = np.load(channel_dir / row.filename, mmap_mode="r")
            physical = (
                digital.astype(np.float32) * row.scale_to_physical
                + row.offset_to_physical
            )
            sample_rate = row.sample_rate_hz
            channel_number = getattr(
                row, "channel_number", getattr(row, "export_channel", 0)
            )
            channel_label = row.channel_label
            physical_unit = row.physical_unit
        minutes, lower, upper = minmax_envelope(
            physical, sample_rate, max_points
        )
        axis.fill_between(minutes, lower, upper, color="#2a6fbb", alpha=0.55, linewidth=0)
        axis.plot(minutes, (lower + upper) / 2, color="#123a63", linewidth=0.35)
        axis.set_title(
            f"{int(channel_number):02d}. {channel_label} "
            f"({_signal_type(channel_label)})",
            fontsize=8 if landscape else 9,
            loc="left",
            pad=3,
        )
        axis.set_ylabel(physical_unit, fontsize=7)
        axis.tick_params(axis="both", labelsize=7, length=2)
        axis.grid(axis="x", color="0.90", linewidth=0.5)
    used_axes = set(plot_axes)
    for axis in axes:
        if axis not in used_axes:
            axis.set_visible(False)

    # Label the bottom-most visible panel in every column. This keeps the last
    # occupied panel labeled when the 31-channel landscape grid has one blank.
    for column in range(columns):
        visible = [
            axes_grid[row, column]
            for row in range(rows)
            if axes_grid[row, column].get_visible()
        ]
        if visible:
            visible[-1].tick_params(labelbottom=True)
            visible[-1].set_xlabel("Minutes from recording start", fontsize=7)
    fig.suptitle(
        f"{channel_dir.name}: full-recording overview of "
        f"{len(channel_rows)} exported channels\n"
        "Each panel is a consecutive min/max envelope; no random samples were selected.",
        fontsize=14 if landscape else 15,
        y=0.985 if landscape else 0.995,
    )
    if landscape:
        fig.subplots_adjust(
            left=0.045,
            right=0.99,
            bottom=0.06,
            top=0.92,
            wspace=0.42,
            hspace=0.48,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches=None if landscape else "tight")
    plt.close(fig)
    print(f"Saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel-dir", type=Path, default=DEFAULT_CHANNEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--edf",
        type=Path,
        help="Read signals directly from an EDF instead of the split manifest.",
    )
    parser.add_argument(
        "--signal-count",
        type=int,
        default=31,
        help="Number of leading EDF signals to plot with --edf (default: 31).",
    )
    parser.add_argument(
        "--landscape",
        action="store_true",
        help="Use a roomier 5-by-7 grid on a 3:2 landscape canvas.",
    )
    parser.add_argument(
        "--max-points-per-channel",
        type=int,
        default=5_000,
        help="Maximum min/max envelope bins per channel (default: 5000).",
    )
    args = parser.parse_args()
    if args.max_points_per_channel < 2:
        parser.error("--max-points-per-channel must be at least 2")
    if args.signal_count < 1:
        parser.error("--signal-count must be at least 1")
    plot_channels(
        args.channel_dir,
        args.output,
        args.max_points_per_channel,
        edf_path=args.edf,
        signal_count=args.signal_count,
        landscape=args.landscape,
    )


if __name__ == "__main__":
    main()
