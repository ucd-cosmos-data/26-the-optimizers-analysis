"""Create all timing and consistency figures supported by seizure_band_timeline.csv.

The source table records one deviation time and one recovery time for each
band/seizure pair.  It supports temporal-ordering and cross-seizure
consistency figures, but it does not contain relative-power values, effect
sizes, or patient identifiers.  Accordingly, this script never implies that
it estimates power magnitude or consistency across patients.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TIMELINE = ROOT / "data" / "interim" / "seizure_band_timeline.csv"
FIGURES = ROOT / "results" / "figures"

BANDS = ["Alpha", "Beta", "Delta", "Theta", "Gamma"]
COLORS = {
    "Alpha": "#F2CF5B",
    "Beta": "#F58518",
    "Delta": "#4C78A8",
    "Theta": "#72B7B2",
    "Gamma": "#E45756",
}
HIGH_FREQUENCY = {"Beta", "Gamma"}
RNG = np.random.default_rng(20260727)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#1F2937",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "font.size": 10,
        }
    )


def load_timeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = pd.read_csv(TIMELINE)
    wide = wide.rename(columns={"Seizure number": "seizure_number"})
    wide["seizure_number"] = pd.to_numeric(wide["seizure_number"], errors="coerce")
    # The current export contains one trailing diagnostic string rather than
    # a seizure row. Keep only numbered seizure records from the CSV.
    wide = wide.dropna(subset=["seizure_number"]).copy()
    wide["seizure_number"] = wide["seizure_number"].astype(int)

    rows: list[pd.DataFrame] = []
    for band in BANDS:
        deviation_column = f"{band} deviates from interictal (seconds relative to seizure start)"
        recovery_column = f"{band} comes back (seconds relative to seizure start)"
        frame = wide[["seizure_number", deviation_column, recovery_column]].copy()
        frame = frame.rename(columns={deviation_column: "deviation_seconds", recovery_column: "recovery_seconds"})
        frame["band"] = band
        frame["deviation_seconds"] = pd.to_numeric(frame["deviation_seconds"], errors="coerce")
        frame["recovery_seconds"] = pd.to_numeric(frame["recovery_seconds"], errors="coerce")
        frame["deviation_detected"] = frame["deviation_seconds"].notna()
        frame["recovery_duration_seconds"] = frame["recovery_seconds"] - frame["deviation_seconds"]
        frame.loc[frame["recovery_duration_seconds"] < 0, "recovery_duration_seconds"] = np.nan
        rows.append(frame)
    long = pd.concat(rows, ignore_index=True)
    return wide, long


def summary_table(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for band in BANDS:
        values = long.loc[long["band"] == band]
        detected = values.loc[values["deviation_detected"], "deviation_seconds"]
        recovery = values.loc[values["recovery_duration_seconds"].notna(), "recovery_duration_seconds"]
        rows.append(
            {
                "Band": band,
                "Detected": int(len(detected)),
                "Detection %": len(detected) / len(values) * 100,
                "Median first deviation (s)": detected.median(),
                "Timing IQR (s)": detected.quantile(0.75) - detected.quantile(0.25),
                "Median recovery time (s)": values["recovery_seconds"].median(),
                "Median deviation-to-recovery (s)": recovery.median(),
            }
        )
    return pd.DataFrame(rows)


def save_04_deviation_heatmap(long: pd.DataFrame) -> None:
    matrix = long.pivot(index="seizure_number", columns="band", values="deviation_seconds").reindex(columns=BANDS)
    maximum = max(70, float(np.nanmax(np.abs(matrix.to_numpy()))))
    fig, ax = plt.subplots(figsize=(8.2, 12.5))
    image = ax.imshow(
        matrix,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum),
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(BANDS)), BANDS)
    ax.set_yticks(np.arange(len(matrix)), [f"{int(value)}" for value in matrix.index])
    ax.set_xlabel("Frequency band")
    ax.set_ylabel("Seizure number")
    ax.set_title("First detected band deviation for every seizure")
    for row in range(len(matrix)):
        for column, band in enumerate(BANDS):
            value = matrix.iloc[row, column]
            if np.isfinite(value):
                text_color = "white" if abs(value) > maximum * 0.55 else "#111827"
                ax.text(column, row, f"{value:+.0f}", ha="center", va="center", fontsize=7.2, color=text_color)
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Seconds relative to seizure onset (negative = preictal)")
    fig.text(0.5, 0.005, "Blank cells indicate no qualifying deviation recorded in the timeline table.", ha="center", fontsize=8.5, color="#4B5563")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(FIGURES / "04_timeline_deviation_heatmap.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_05_temporal_ordering(long: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), gridspec_kw={"width_ratios": [1.45, 1]})
    for position, band in enumerate(BANDS):
        values = long.loc[
            (long["band"] == band) & long["deviation_seconds"].notna(),
            "deviation_seconds",
        ].to_numpy()
        if len(values):
            box = axes[0].boxplot(
                values,
                positions=[position],
                widths=0.58,
                patch_artist=True,
                showfliers=False,
                boxprops={"facecolor": COLORS[band], "alpha": 0.45},
                medianprops={"color": "#111827", "linewidth": 1.7},
            )
            for element in ["whiskers", "caps"]:
                for artist in box[element]:
                    artist.set_color("#374151")
            axes[0].scatter(
                position + RNG.normal(0, 0.065, len(values)),
                values,
                color=COLORS[band],
                edgecolor="white",
                lw=0.35,
                s=26,
                alpha=0.88,
                zorder=3,
            )
    axes[0].axhline(0, color="#111827", lw=1.2, ls=":", label="Seizure onset")
    axes[0].set_xticks(np.arange(len(BANDS)), BANDS)
    axes[0].set_ylabel("First deviation (seconds relative to onset)")
    axes[0].set_title("Temporal ordering of first detected deviations")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, loc="lower right")

    detection = long.groupby("band")["deviation_detected"].agg(["mean", "sum", "count"]).reindex(BANDS)
    bars = axes[1].bar(np.arange(len(BANDS)), detection["mean"] * 100, color=[COLORS[band] for band in BANDS])
    for bar, row in zip(bars, detection.itertuples()):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{row.sum}/{row.count}", ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(np.arange(len(BANDS)), BANDS)
    axes[1].set_ylim(0, 110)
    axes[1].set_ylabel("Seizures with a detected deviation (%)")
    axes[1].set_title("Detection frequency")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Temporal ordering and detection of band changes", fontsize=15, weight="bold")
    fig.text(0.5, 0.012, "Each point is one seizure. Negative times occur before annotated scalp-EEG seizure onset.", ha="center", fontsize=8.5, color="#4B5563")
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(FIGURES / "05_timeline_temporal_ordering.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_06_seizure_event_plot(long: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 12.2))
    offsets = {band: offset for band, offset in zip(BANDS, np.linspace(-0.28, 0.28, len(BANDS)))}
    for band in BANDS:
        values = long.loc[long["band"] == band].dropna(subset=["deviation_seconds"])
        ax.scatter(
            values["deviation_seconds"],
            values["seizure_number"] + offsets[band],
            label=band,
            color=COLORS[band],
            s=30,
            alpha=0.85,
            edgecolor="white",
            lw=0.35,
        )
    ax.axvline(0, color="#111827", lw=1.3, ls=":", label="Seizure onset")
    ax.set_xlabel("First detected deviation (seconds relative to onset)")
    ax.set_ylabel("Seizure number")
    ax.set_yticks(sorted(long["seizure_number"].dropna().unique()))
    ax.set_title("Band-specific deviation times within each seizure")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.text(0.5, 0.01, "Markers sharing a row are bands from the same seizure; farther left indicates an earlier detected deviation.", ha="center", fontsize=8.5, color="#4B5563")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(FIGURES / "06_timeline_seizure_event_plot.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def rank_table(long: pd.DataFrame) -> pd.DataFrame:
    detected = long.loc[long["deviation_seconds"].notna()].copy()
    detected["within_seizure_rank"] = detected.groupby("seizure_number")["deviation_seconds"].rank(method="average", ascending=True)
    earliest = detected.groupby("seizure_number")["deviation_seconds"].transform("min")
    detected["is_first"] = detected["deviation_seconds"] == earliest
    return detected


def save_07_rank_patterns(ranks: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for position, band in enumerate(BANDS):
        values = ranks.loc[ranks["band"] == band, "within_seizure_rank"].to_numpy()
        if len(values):
            axes[0].boxplot(
                values,
                positions=[position],
                widths=0.58,
                patch_artist=True,
                showfliers=False,
                boxprops={"facecolor": COLORS[band], "alpha": 0.5},
                medianprops={"color": "#111827", "linewidth": 1.6},
            )
            axes[0].scatter(position + RNG.normal(0, 0.06, len(values)), values, color=COLORS[band], edgecolor="white", lw=0.35, s=22, alpha=0.85)
    axes[0].set_xticks(np.arange(len(BANDS)), BANDS)
    axes[0].set_ylabel("Within-seizure order rank (1 = earliest)")
    axes[0].set_title("Relative temporal order when multiple bands deviate")
    axes[0].grid(axis="y", alpha=0.2)

    first_frequency = ranks.groupby("band")["is_first"].mean().reindex(BANDS) * 100
    bars = axes[1].bar(np.arange(len(BANDS)), first_frequency, color=[COLORS[band] for band in BANDS])
    for bar, value in zip(bars, first_frequency):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{value:.0f}%", ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(np.arange(len(BANDS)), BANDS)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Detected seizures where band is earliest (%)")
    axes[1].set_title("How often each band is the first detected change")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Within-seizure temporal ordering patterns", fontsize=15, weight="bold")
    fig.text(0.5, 0.012, "Tied earliest deviations count for every band involved in the tie. Ranks use only bands with a detected deviation in that seizure.", ha="center", fontsize=8.5, color="#4B5563")
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(FIGURES / "07_timeline_within_seizure_ranks.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_08_recovery(long: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 5.2))
    for position, band in enumerate(BANDS):
        values = long.loc[(long["band"] == band) & long["recovery_seconds"].notna(), "recovery_seconds"].to_numpy()
        if len(values):
            axes[0].boxplot(values, positions=[position], widths=0.58, patch_artist=True, showfliers=False, boxprops={"facecolor": COLORS[band], "alpha": 0.45}, medianprops={"color": "#111827", "linewidth": 1.6})
            axes[0].scatter(position + RNG.normal(0, 0.06, len(values)), values, color=COLORS[band], edgecolor="white", lw=0.35, s=23, alpha=0.85)
    axes[0].axhline(0, color="#111827", lw=1.1, ls=":")
    axes[0].set_xticks(np.arange(len(BANDS)), BANDS)
    axes[0].set_ylabel("Return time (seconds relative to onset)")
    axes[0].set_title("When each band returns to baseline")
    axes[0].grid(axis="y", alpha=0.2)

    for position, band in enumerate(BANDS):
        values = long.loc[(long["band"] == band) & long["recovery_duration_seconds"].notna(), "recovery_duration_seconds"].to_numpy()
        if len(values):
            axes[1].boxplot(values, positions=[position], widths=0.58, patch_artist=True, showfliers=False, boxprops={"facecolor": COLORS[band], "alpha": 0.45}, medianprops={"color": "#111827", "linewidth": 1.6})
            axes[1].scatter(position + RNG.normal(0, 0.06, len(values)), values, color=COLORS[band], edgecolor="white", lw=0.35, s=23, alpha=0.85)
    axes[1].set_xticks(np.arange(len(BANDS)), BANDS)
    axes[1].set_ylabel("Seconds from deviation to recovery")
    axes[1].set_title("Duration of the recorded band deviation")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Recovery context for the detected preictal/ictal changes", fontsize=15, weight="bold")
    fig.text(0.5, 0.012, "Recovery graphs use only band–seizure pairs with both a deviation and a recorded return time.", ha="center", fontsize=8.5, color="#4B5563")
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(FIGURES / "08_timeline_recovery_patterns.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_09_consistency(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1))
    x = np.arange(len(BANDS))
    colors = [COLORS[band] for band in BANDS]
    bars = axes[0].bar(x, summary["Detection %"], color=colors)
    for bar, detected in zip(bars, summary["Detected"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.7, f"{detected}/47", ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(x, BANDS)
    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel("Seizures with a detected change (%)")
    axes[0].set_title("Detection consistency across seizures")
    axes[0].grid(axis="y", alpha=0.2)

    bars = axes[1].bar(x, summary["Timing IQR (s)"], color=colors)
    for bar, value in zip(bars, summary["Timing IQR (s)"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.0f}s", ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(x, BANDS)
    axes[1].set_ylabel("IQR of first-deviation time (seconds; lower is more consistent)")
    axes[1].set_title("Timing consistency across seizures")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Cross-seizure consistency: beta and gamma versus lower-frequency bands", fontsize=15, weight="bold")
    fig.text(0.5, 0.012, "The timeline table has no patient ID, so this figure evaluates consistency across seizures only. Beta and gamma are the high-frequency bands.", ha="center", fontsize=8.5, color="#4B5563")
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(FIGURES / "09_timeline_cross_seizure_consistency.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_10_summary_table(summary: pd.DataFrame) -> None:
    display = summary.copy()
    display["Detection %"] = display["Detection %"].map(lambda value: f"{value:.1f}%")
    for column in ["Median first deviation (s)", "Timing IQR (s)", "Median recovery time (s)", "Median deviation-to-recovery (s)"]:
        display[column] = display[column].map(lambda value: "N/A" if pd.isna(value) else f"{value:.0f}")
    display = display.rename(
        columns={
            "Detection %": "Detection\n%",
            "Median first deviation (s)": "Median first\ndeviation (s)",
            "Timing IQR (s)": "Timing\nIQR (s)",
            "Median recovery time (s)": "Median recovery\ntime (s)",
            "Median deviation-to-recovery (s)": "Median deviation-to-\nrecovery (s)",
        }
    )
    fig, ax = plt.subplots(figsize=(17, 3.8))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.11, 0.08, 0.10, 0.17, 0.11, 0.16, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        if row == 0:
            cell.set_facecolor("#1F2937")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif column == 0:
            band = display.iloc[row - 1, 0]
            cell.set_facecolor(COLORS[band])
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#F9FAFB" if row % 2 else "white")
    ax.set_title("Summary of timing and cross-seizure consistency from seizure_band_timeline.csv", fontsize=13, weight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(FIGURES / "10_timeline_summary_table.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def write_readme(summary: pd.DataFrame, wide: pd.DataFrame) -> None:
    lines = [
        "# Figures generated from `seizure_band_timeline.csv`",
        "",
        f"- Source rows (annotated seizures): {len(wide)}",
        "- Source fields: each band’s first deviation from interictal and its return time, in seconds relative to seizure onset.",
        "- Negative values are preictal; positive values occur after annotated onset.",
        "- The timeline has no raw relative-power measurements, effect sizes, or patient identifiers. It can therefore support timing and cross-seizure consistency figures, but not power-magnitude or cross-patient inference.",
        "",
        "## Band summary",
        "",
        summary.to_markdown(index=False, floatfmt=".2f"),
        "",
    ]
    (FIGURES / "README_timeline_figures.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_plotting()
    FIGURES.mkdir(parents=True, exist_ok=True)
    wide, long = load_timeline()
    summary = summary_table(long)
    ranks = rank_table(long)

    save_04_deviation_heatmap(long)
    save_05_temporal_ordering(long)
    save_06_seizure_event_plot(long)
    save_07_rank_patterns(ranks)
    save_08_recovery(long)
    save_09_consistency(summary)
    save_10_summary_table(summary)
    write_readme(summary, wide)
    print(f"Saved timeline figures to {FIGURES}")


if __name__ == "__main__":
    main()
