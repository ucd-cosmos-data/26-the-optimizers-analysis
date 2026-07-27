"""Create figures that directly address the preictal band-power question."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "results" / "figures"
BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]
COLORS = {
    "delta": "#4C78A8",
    "theta": "#72B7B2",
    "alpha": "#E0AC00",
    "beta": "#F58518",
    "gamma": "#E45756",
}


def save_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    boxes = [
        (0.2, "Raw EDF\nEEG", "#E8F1FA"),
        (2.8, "Align to\nseizure onset", "#EAF6F2"),
        (5.4, "Six 10-second\npreictal bins", "#FFF5D6"),
        (8.0, "Relative power\nin five bands", "#FCE8E6"),
        (10.6, "Compare with\ninterictal baseline", "#EEE7F8"),
    ]
    for x, label, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, 1.65),
                2.1,
                1.15,
                boxstyle="round,pad=0.05,rounding_size=0.08",
                linewidth=1.3,
                edgecolor="#36454F",
                facecolor=color,
            )
        )
        ax.text(x + 1.05, 2.225, label, ha="center", va="center", fontsize=10.5)
    for left, right in [(2.3, 2.8), (4.9, 5.4), (7.5, 8.0), (10.1, 10.6)]:
        ax.add_patch(
            FancyArrowPatch(
                (left, 2.225),
                (right, 2.225),
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=1.3,
                color="#36454F",
            )
        )
    ax.text(
        6.5,
        3.55,
        "Pre-seizure EEG band-power analysis",
        ha="center",
        fontsize=15,
        weight="bold",
    )
    ax.text(
        6.5,
        0.72,
        "Quality screen → 60 Hz notch → common-median reference → Welch spectrum → "
        "patient-aware summaries",
        ha="center",
        fontsize=9.5,
        color="#4A5568",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "01_analysis_pipeline.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_temporal_trajectories(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    axes = axes.ravel()
    for index, band in enumerate(BAND_ORDER):
        ax = axes[index]
        band_data = summary[summary["band"] == band].sort_values("time_bin")
        x = band_data["time_to_onset_end"].to_numpy()
        mean = band_data["patient_mean_log2_change"].to_numpy()
        low = band_data["ci95_low"].to_numpy()
        high = band_data["ci95_high"].to_numpy()
        ax.axhline(0, color="#667085", linewidth=1, linestyle="--")
        ax.fill_between(x, low, high, color=COLORS[band], alpha=0.2)
        ax.plot(x, mean, marker="o", linewidth=2.2, color=COLORS[band])
        ax.set_title(band.title(), weight="bold")
        ax.grid(alpha=0.2)
        ax.set_xticks([-50, -40, -30, -20, -10, 0])
    axes[5].axis("off")
    for ax in axes[[0, 3]]:
        ax.set_ylabel("Log2 change from\nmatched interictal power")
    for ax in axes[3:5]:
        ax.set_xlabel("Seconds before seizure onset")
    fig.suptitle(
        "Relative EEG power during the final 60 seconds before seizure onset",
        fontsize=15,
        weight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Zero means no change from interictal baseline; shaded regions are "
        "95% patient-bootstrap intervals.",
        ha="center",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(
        FIGURES / "02_preictal_temporal_trajectories.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_temporal_heatmap(summary: pd.DataFrame) -> None:
    matrix = (
        summary.pivot(
            index="band", columns="time_to_onset_end", values="patient_mean_log2_change"
        )
        .reindex(BAND_ORDER)
        .sort_index(axis=1)
    )
    limit = max(0.1, float(np.nanmax(np.abs(matrix.to_numpy()))))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(range(len(matrix.index)), [band.title() for band in matrix.index])
    ax.set_xticks(
        range(len(matrix.columns)),
        [f"{int(value - 10)} to {int(value)}" for value in matrix.columns],
    )
    ax.set_xlabel("Seconds before seizure onset")
    ax.set_title(
        "Temporal ordering and magnitude of preictal band-power changes",
        weight="bold",
        pad=12,
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            ax.text(
                column,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > limit * 0.55 else "#1F2937",
                fontsize=9,
            )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Log2 change from matched interictal power")
    fig.tight_layout()
    fig.savefig(
        FIGURES / "03_temporal_ordering_heatmap.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)


def save_consistency(summary: pd.DataFrame) -> None:
    data = summary.set_index("band").reindex(BAND_ORDER)
    x = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = [COLORS[band] for band in BAND_ORDER]
    axes[0].bar(
        x - 0.18,
        data["mean_within_patient_seizure_sd"],
        width=0.36,
        label="Between seizures within patients",
        color=colors,
    )
    axes[0].bar(
        x + 0.18,
        data["patient_sd"],
        width=0.36,
        label="Between patient averages",
        color=colors,
        alpha=0.5,
        edgecolor="#344054",
    )
    axes[0].set_xticks(x, [band.title() for band in BAND_ORDER])
    axes[0].set_ylabel("Standard deviation of late change")
    axes[0].set_title("Variability: lower is more consistent", weight="bold")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(
        x - 0.18,
        100 * data["patient_balanced_event_direction_agreement"],
        width=0.36,
        label="Seizures, balanced by patient",
        color=colors,
    )
    axes[1].bar(
        x + 0.18,
        100 * data["patient_direction_agreement"],
        width=0.36,
        label="Patient averages",
        color=colors,
        alpha=0.5,
        edgecolor="#344054",
    )
    axes[1].axhline(50, color="#667085", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, [band.title() for band in BAND_ORDER])
    axes[1].set_ylim(45, 105)
    axes[1].set_ylabel("Same-direction agreement (%)")
    axes[1].set_title("Direction agreement: higher is more consistent", weight="bold")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Consistency of band-power changes in the final 20 seconds",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES / "04_beta_gamma_consistency.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)


def main() -> None:
    required = {
        "temporal summary": RESULTS / "temporal_summary.csv",
        "consistency summary": RESULTS / "consistency_summary.csv",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run analyze_preictal_bandpower.py first. Missing:\n" + "\n".join(missing)
        )
    FIGURES.mkdir(parents=True, exist_ok=True)
    temporal = pd.read_csv(required["temporal summary"])
    consistency = pd.read_csv(required["consistency summary"])
    save_pipeline()
    save_temporal_trajectories(temporal)
    save_temporal_heatmap(temporal)
    save_consistency(consistency)
    print(f"Saved four research figures to {FIGURES}")


if __name__ == "__main__":
    main()
