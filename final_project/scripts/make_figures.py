"""Create report-ready diagrams from the extracted EEG analysis outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"
BANDS = [
    ("Delta", 0.5, 4.0, "#4C78A8"),
    ("Theta", 4.0, 8.0, "#72B7B2"),
    ("Alpha", 8.0, 13.0, "#F2CF5B"),
    ("Beta", 13.0, 30.0, "#F58518"),
    ("Gamma", 30.0, 100.0, "#E45756"),
]


def save_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    boxes = [
        (0.3, 1.7, 2.0, 1.1, "Raw EDF\nrecordings", "#E8F1FA"),
        (2.8, 1.7, 2.0, 1.1, "10-second\nwindows", "#EAF6F2"),
        (5.3, 1.7, 2.0, 1.1, "PSD / FFT\nfeatures", "#FFF5D6"),
        (7.8, 1.7, 2.0, 1.1, "Band powers\nδ θ α β γ", "#FCE8E6"),
        (10.3, 1.7, 2.0, 1.1, "Subject-held-out\nmodels", "#EEE7F8"),
    ]
    for x, y, width, height, label, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), width, height, boxstyle="round,pad=0.05,rounding_size=0.08",
                linewidth=1.4, edgecolor="#36454F", facecolor=color,
            )
        )
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=11)
    for left, right in [(2.3, 2.8), (4.8, 5.3), (7.3, 7.8), (9.8, 10.3)]:
        ax.add_patch(FancyArrowPatch((left, 2.25), (right, 2.25), arrowstyle="-|>", mutation_scale=16, linewidth=1.3, color="#36454F"))
    ax.text(6.5, 3.45, "EEG seizure-window classification workflow", ha="center", fontsize=15, weight="bold")
    ax.text(6.5, 0.75, "Detrend → 4-second Hann segments → integrate PSD over standard EEG bands → aggregate across available EEG channels", ha="center", fontsize=9.5, color="#4A5568")
    fig.tight_layout()
    fig.savefig(FIGURES / "01_analysis_pipeline.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_bands() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for index, (name, low, high, color) in enumerate(BANDS):
        ax.barh(index, high - low, left=low, height=0.58, color=color, edgecolor="white", linewidth=1.2)
        ax.text((low + high) / 2, index, f"{name}\n{low:g}–{high:g} Hz", ha="center", va="center", fontsize=10, weight="bold")
    ax.set_xscale("log")
    ax.set_yticks([])
    ax.set_xlabel("Frequency (Hz, logarithmic scale)")
    ax.set_title("EEG frequency bands used for power extraction", weight="bold", pad=12)
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(0.4, 130)
    fig.tight_layout()
    fig.savefig(FIGURES / "02_frequency_bands.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_model_results() -> None:
    metrics = pd.read_csv(ROOT / "results" / "model_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    order = metrics.sort_values("average_precision", ascending=True)
    y = np.arange(len(order))
    axes[0].barh(y - 0.15, order["roc_auc"], height=0.28, label="ROC-AUC", color="#4C78A8")
    axes[0].barh(y + 0.15, order["average_precision"], height=0.28, label="Average precision", color="#F58518")
    axes[0].set_yticks(y, [name.replace("_", " ").title() for name in order["model"]])
    axes[0].set_xlim(0, 0.75)
    axes[0].set_xlabel("Score")
    axes[0].set_title("Held-out model performance", weight="bold")
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].grid(axis="x", alpha=0.25)

    best = metrics.loc[metrics["roc_auc"].idxmax()]
    matrix = np.array(json.loads(best["confusion_matrix"]))
    image = axes[1].imshow(matrix, cmap="Blues")
    axes[1].set_xticks([0, 1], ["Predicted\nnon-seizure", "Predicted\nseizure"])
    axes[1].set_yticks([0, 1], ["Actual\nnon-seizure", "Actual\nseizure"])
    axes[1].set_title(f"Best ROC-AUC model: {best['model'].replace('_', ' ').title()}", weight="bold")
    for row in range(2):
        for column in range(2):
            axes[1].text(column, row, f"{matrix[row, column]:,}", ha="center", va="center", color="white" if matrix[row, column] > matrix.max() * 0.5 else "#1F2937", fontsize=12, weight="bold")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle("Seizure-window classification results", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "03_model_performance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    save_pipeline()
    save_bands()
    save_model_results()
    print(f"Saved diagrams to {FIGURES}")


if __name__ == "__main__":
    main()
