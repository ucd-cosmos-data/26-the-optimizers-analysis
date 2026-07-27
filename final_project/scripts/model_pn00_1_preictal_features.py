"""Extract robust PN00-1 EEG features and screen their preictal utility.

PN00-1 contains a single annotated seizure.  This script consequently does
not present a clinical prediction accuracy claim or a random-window classifier.
Instead, it fits a transparent *baseline-only* multivariate anomaly score on
clean interictal windows, then applies that fixed score to the ten 6-second
windows in the final minute before the annotated onset.  This avoids training
on seizure labels and makes the one-recording limitation explicit.

Features are calculated separately in each usable EEG channel and summarized
by the median (and IQR) across channels.  EKG and SpO2 are deliberately not
mixed with EEG spectral features.

Run from the repository root:

    .\\.venv\\Scripts\\python.exe final_project/scripts/model_pn00_1_preictal_features.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats


ROOT = Path(__file__).resolve().parents[1]
CHANNEL_DIR = ROOT / "data" / "processed" / "PN00-1_split_channels"
RESULTS_DIR = ROOT / "results" / "PN00-1_preictal_feature_model"

SAMPLE_RATE_HZ = 512.0
SEIZURE_ONSET_SECONDS = 1_143
SEIZURE_END_SECONDS = 1_213
WINDOW_SECONDS = 6
PREICTAL_SECONDS = 60
INTERICTAL_BUFFER_SECONDS = 15 * 60
MIN_USABLE_CHANNELS = 10
ARTIFACT_MAX_ABS_UV = 1_000.0
ARTIFACT_MIN_STD_UV = 0.5
WELCH_SECONDS = 4

# Delta (1-4 Hz) is excluded: PN00-1 declares a 1.591549-Hz high-pass filter.
# Gamma is excluded: PN00-1 declares a 30-Hz low-pass filter.
BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
MODEL_FEATURES = [
    "median_rms_uv",
    "median_line_length_uv_per_sample",
    "median_hjorth_mobility",
    "median_hjorth_complexity",
    "median_spectral_entropy",
    "median_theta_relative_power",
    "median_alpha_relative_power",
    "median_beta_relative_power",
    "spatial_iqr_line_length_uv_per_sample",
]


def _integrate(psd: np.ndarray, freqs: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = (freqs >= low) & (freqs <= high)
    return np.trapezoid(psd[:, mask], freqs[mask], axis=1)


def hjorth_features(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Hjorth mobility and complexity for rows representing channels."""
    variance = np.var(data, axis=1)
    first_difference = np.diff(data, axis=1)
    first_variance = np.var(first_difference, axis=1)
    mobility = np.sqrt(first_variance / np.maximum(variance, 1e-12))
    second_difference = np.diff(first_difference, axis=1)
    second_variance = np.var(second_difference, axis=1)
    derivative_mobility = np.sqrt(second_variance / np.maximum(first_variance, 1e-12))
    complexity = derivative_mobility / np.maximum(mobility, 1e-12)
    return mobility, complexity


def spectral_features(data: np.ndarray) -> dict[str, np.ndarray]:
    """Welch relative powers and entropy per channel from all window samples."""
    nperseg = min(int(WELCH_SECONDS * SAMPLE_RATE_HZ), data.shape[1])
    freqs, psd = signal.welch(
        data,
        fs=SAMPLE_RATE_HZ,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="linear",
        axis=1,
        scaling="density",
    )
    powers = {
        name: _integrate(psd, freqs, low, high) for name, (low, high) in BANDS.items()
    }
    total = np.sum(np.column_stack([powers[name] for name in BANDS]), axis=1)
    output = {f"{name}_relative_power": powers[name] / np.maximum(total, 1e-12) for name in BANDS}
    entropy_mask = (freqs >= 4) & (freqs <= 30)
    spectral_probability = psd[:, entropy_mask] / np.maximum(
        psd[:, entropy_mask].sum(axis=1, keepdims=True), 1e-12
    )
    entropy = -np.sum(
        spectral_probability * np.log(np.maximum(spectral_probability, 1e-12)), axis=1
    )
    output["spectral_entropy"] = entropy / np.log(spectral_probability.shape[1])
    return output


def channel_features(data_uv: np.ndarray) -> tuple[pd.DataFrame, int]:
    """Calculate all channel features in a 6-second window using robust QC."""
    detrended = signal.detrend(data_uv, axis=1, type="linear")
    referenced = detrended - np.median(detrended, axis=0, keepdims=True)
    std = np.std(referenced, axis=1)
    max_abs = np.max(np.abs(referenced), axis=1)
    usable = np.isfinite(referenced).all(axis=1) & (std >= ARTIFACT_MIN_STD_UV) & (max_abs <= ARTIFACT_MAX_ABS_UV)
    rows = pd.DataFrame({"usable": usable, "rms_uv": np.nan, "line_length_uv_per_sample": np.nan,
                         "zero_crossing_rate": np.nan, "hjorth_mobility": np.nan,
                         "hjorth_complexity": np.nan, "spectral_entropy": np.nan,
                         "theta_relative_power": np.nan, "alpha_relative_power": np.nan,
                         "beta_relative_power": np.nan})
    if not usable.any():
        return rows, 0
    clean = referenced[usable]
    notch_b, notch_a = signal.iirnotch(60.0, 30.0, fs=SAMPLE_RATE_HZ)
    clean = signal.filtfilt(notch_b, notch_a, clean, axis=1)
    mobility, complexity = hjorth_features(clean)
    spectrum = spectral_features(clean)
    rows.loc[usable, "rms_uv"] = np.sqrt(np.mean(clean**2, axis=1))
    rows.loc[usable, "line_length_uv_per_sample"] = np.mean(np.abs(np.diff(clean, axis=1)), axis=1)
    rows.loc[usable, "zero_crossing_rate"] = np.mean(clean[:, :-1] * clean[:, 1:] < 0, axis=1)
    rows.loc[usable, "hjorth_mobility"] = mobility
    rows.loc[usable, "hjorth_complexity"] = complexity
    for name, values in spectrum.items():
        rows.loc[usable, name] = values
    return rows, int(usable.sum())


def add_window_aggregates(features: pd.DataFrame) -> dict[str, float]:
    """Use medians/IQRs across usable channels to avoid noisy-channel dominance."""
    usable = features[features["usable"]]
    output: dict[str, float] = {}
    for name in [column for column in features.columns if column != "usable"]:
        values = usable[name].dropna()
        output[f"median_{name}"] = float(values.median()) if len(values) else np.nan
        output[f"spatial_iqr_{name}"] = float(values.quantile(0.75) - values.quantile(0.25)) if len(values) else np.nan
    return output


def window_definitions(duration_seconds: int) -> pd.DataFrame:
    """Use all nonoverlapping clean baseline windows and all ten preictal windows."""
    rows: list[dict[str, object]] = []
    for window_id, start in enumerate(range(SEIZURE_ONSET_SECONDS - PREICTAL_SECONDS, SEIZURE_ONSET_SECONDS, WINDOW_SECONDS), start=1):
        rows.append(
            {
                "window_id": f"preictal_{window_id:02d}",
                "condition": "preictal",
                "start_seconds": start,
                "end_seconds": start + WINDOW_SECONDS,
                "time_to_onset_midpoint_seconds": start + WINDOW_SECONDS / 2 - SEIZURE_ONSET_SECONDS,
            }
        )
    interictal_id = 0
    for start in range(0, duration_seconds - WINDOW_SECONDS + 1, WINDOW_SECONDS):
        end = start + WINDOW_SECONDS
        clean = end <= SEIZURE_ONSET_SECONDS - INTERICTAL_BUFFER_SECONDS or start >= SEIZURE_END_SECONDS + INTERICTAL_BUFFER_SECONDS
        if clean:
            interictal_id += 1
            rows.append(
                {
                    "window_id": f"interictal_{interictal_id:03d}",
                    "condition": "interictal",
                    "start_seconds": start,
                    "end_seconds": end,
                    "time_to_onset_midpoint_seconds": np.nan,
                }
            )
    return pd.DataFrame(rows)


def extract_feature_tables(channel_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return individual-channel, robust window, and window-definition tables."""
    manifest = pd.read_csv(channel_dir / "channel_manifest.csv")
    eeg = manifest[manifest["is_eeg"]].copy().reset_index(drop=True)
    if len(eeg) != 29:
        raise ValueError(f"Expected 29 EEG channels, found {len(eeg)}")
    if not np.allclose(eeg["sample_rate_hz"], SAMPLE_RATE_HZ):
        raise ValueError("Expected 512-Hz PN00-1 EEG channels")
    arrays = [np.load(channel_dir / row.filename, mmap_mode="r") for row in eeg.itertuples(index=False)]
    duration = int(eeg["duration_seconds"].iloc[0])
    windows = window_definitions(duration)
    channel_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    samples_per_window = int(WINDOW_SECONDS * SAMPLE_RATE_HZ)
    for number, window in enumerate(windows.itertuples(index=False), start=1):
        start_sample = int(window.start_seconds * SAMPLE_RATE_HZ)
        stop_sample = start_sample + samples_per_window
        data = np.vstack(
            [
                values[start_sample:stop_sample].astype(np.float64) * row.scale_to_physical + row.offset_to_physical
                for values, row in zip(arrays, eeg.itertuples(index=False))
            ]
        )
        per_channel, usable_count = channel_features(data)
        for row, channel in zip(per_channel.itertuples(index=False), eeg.itertuples(index=False)):
            channel_rows.append(
                {
                    "window_id": window.window_id,
                    "condition": window.condition,
                    "start_seconds": window.start_seconds,
                    "end_seconds": window.end_seconds,
                    "time_to_onset_midpoint_seconds": window.time_to_onset_midpoint_seconds,
                    "export_channel": channel.export_channel,
                    "channel_label": channel.channel_label,
                    **row._asdict(),
                }
            )
        window_rows.append(
            {
                "window_id": window.window_id,
                "condition": window.condition,
                "start_seconds": window.start_seconds,
                "end_seconds": window.end_seconds,
                "time_to_onset_midpoint_seconds": window.time_to_onset_midpoint_seconds,
                "eeg_channel_count": len(eeg),
                "usable_channel_count": usable_count,
                "qc_pass": usable_count >= MIN_USABLE_CHANNELS,
                **add_window_aggregates(per_channel),
            }
        )
        if number % 25 == 0 or number == len(windows):
            print(f"Features: {number}/{len(windows)} windows", flush=True)
    return pd.DataFrame(channel_rows), pd.DataFrame(window_rows), windows


def robust_baseline_model(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Fit a label-free robust anomaly model from clean interictal windows only."""
    valid = windows[windows["qc_pass"]].copy()
    baseline = valid[valid["condition"] == "interictal"].copy()
    preictal = valid[valid["condition"] == "preictal"].copy()
    medians = baseline[MODEL_FEATURES].median()
    mad = (baseline[MODEL_FEATURES] - medians).abs().median()
    scales = 1.4826 * mad
    fallback = baseline[MODEL_FEATURES].std(ddof=1)
    scales = scales.where(scales > 1e-12, fallback).replace(0, np.nan)
    standardized = (valid[MODEL_FEATURES] - medians) / scales
    valid["baseline_anomaly_score"] = np.sqrt(np.nanmean(standardized.to_numpy() ** 2, axis=1))
    baseline_scores = valid.loc[valid["condition"] == "interictal", "baseline_anomaly_score"]
    threshold = float(baseline_scores.quantile(0.95))
    valid["anomaly_threshold_95th_baseline"] = threshold
    valid["anomaly_flag"] = valid["baseline_anomaly_score"] > threshold
    feature_effect_rows: list[dict[str, object]] = []
    for feature in MODEL_FEATURES:
        baseline_values = baseline[feature].dropna()
        preictal_values = preictal[feature].dropna()
        standardized_shift = (preictal_values.median() - baseline_values.median()) / scales[feature]
        auc = stats.mannwhitneyu(preictal_values, baseline_values, alternative="two-sided").statistic / (len(preictal_values) * len(baseline_values))
        trend = stats.spearmanr(
            preictal["time_to_onset_midpoint_seconds"], preictal[feature], nan_policy="omit"
        )
        feature_effect_rows.append(
            {
                "feature": feature,
                "interictal_median": float(baseline_values.median()),
                "preictal_median": float(preictal_values.median()),
                "preictal_minus_interictal": float(preictal_values.median() - baseline_values.median()),
                "robust_standardized_shift": float(standardized_shift),
                "univariate_auc_preictal_vs_interictal": float(auc),
                "spearman_rho_time_to_onset": float(trend.statistic),
                "spearman_p_value": float(trend.pvalue),
            }
        )
    effects = pd.DataFrame(feature_effect_rows).sort_values("robust_standardized_shift", key=lambda values: values.abs(), ascending=False)
    summary = {
        "model": "Baseline-only robust multivariate anomaly score",
        "model_features": MODEL_FEATURES,
        "baseline_windows": len(baseline),
        "preictal_windows": len(preictal),
        "threshold_definition": "95th percentile of clean interictal anomaly scores",
        "threshold": threshold,
        "interictal_flag_rate_percent": float(100 * valid.loc[valid["condition"] == "interictal", "anomaly_flag"].mean()),
        "preictal_flag_rate_percent": float(100 * valid.loc[valid["condition"] == "preictal", "anomaly_flag"].mean()),
        "preictal_score_median": float(valid.loc[valid["condition"] == "preictal", "baseline_anomaly_score"].median()),
        "interictal_score_median": float(valid.loc[valid["condition"] == "interictal", "baseline_anomaly_score"].median()),
    }
    flagged = valid[(valid["condition"] == "preictal") & valid["anomaly_flag"]]
    summary["earliest_flagged_preictal_midpoint_seconds"] = (
        float(flagged["time_to_onset_midpoint_seconds"].min()) if len(flagged) else None
    )
    return valid, effects, summary


def plot_results(windows: pd.DataFrame, effects: pd.DataFrame, summary: dict[str, object], output: Path) -> None:
    """Plot anomaly separation and trajectories for the four most shifted features."""
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True, height_ratios=[1, 1.6])
    baseline = windows[windows["condition"] == "interictal"]
    preictal = windows[windows["condition"] == "preictal"].sort_values("time_to_onset_midpoint_seconds")
    axis = axes[0]
    axis.scatter(np.zeros(len(baseline)), baseline["baseline_anomaly_score"], color="0.55", alpha=0.65, s=28, label="Clean interictal windows")
    axis.scatter(np.ones(len(preictal)), preictal["baseline_anomaly_score"], color="#c83e4d", s=48, marker="D", label="Final-minute preictal windows")
    axis.axhline(summary["threshold"], color="#c83e4d", linestyle="--", label="95th-percentile interictal threshold")
    axis.set(xticks=[0, 1], xticklabels=["Interictal", "Preictal"], ylabel="Baseline-only anomaly score", title="PN00-1: fixed interictal model applied to the final minute before seizure onset")
    axis.legend(loc="best", fontsize=9)
    axis.grid(axis="y", color="0.9")

    top_features = effects.head(4)["feature"].tolist()
    axis = axes[1]
    colors = plt.get_cmap("tab10").colors
    for index, feature in enumerate(top_features):
        baseline_median = baseline[feature].median()
        baseline_mad_scale = 1.4826 * (baseline[feature] - baseline_median).abs().median()
        baseline_mad_scale = baseline_mad_scale if baseline_mad_scale > 1e-12 else baseline[feature].std(ddof=1)
        z = (preictal[feature] - baseline_median) / baseline_mad_scale
        axis.plot(preictal["time_to_onset_midpoint_seconds"], z, marker="o", linewidth=2, color=colors[index], label=feature.replace("median_", "").replace("_", " "))
    axis.axhline(0, color="black", linewidth=1)
    axis.axhline(2, color="0.4", linewidth=0.8, linestyle="--")
    axis.axhline(-2, color="0.4", linewidth=0.8, linestyle="--")
    axis.set(xlabel="Seconds before annotated seizure onset (6-s window midpoint)", ylabel="Robust feature shift from interictal median", xticks=np.arange(-57, 0, 6), title="Top four feature shifts across the preictal minute")
    axis.grid(axis="x", color="0.9")
    axis.legend(ncol=2, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def write_report(path: Path, effects: pd.DataFrame, summary: dict[str, object]) -> None:
    top = effects.head(6)
    lines = [
        "# PN00-1 preictal feature screen",
        "",
        "## What this model can and cannot show",
        "",
        "PN00-1 contains one annotated seizure. This is therefore an exploratory, within-recording screen, not an independent seizure-prediction validation. The model learns only the distribution of clean interictal EEG windows; it does not use preictal labels to set its score or threshold. A second patient or a second seizure is required for out-of-sample validation.",
        "",
        "## Feature set",
        "",
        "All values are first calculated in each quality-passing EEG lead, then summarized by the median across leads. The feature set includes RMS amplitude, line length, Hjorth mobility and complexity, spectral entropy, relative theta/alpha/beta power, and cross-channel line-length IQR. Delta is excluded because the PN00-1 high-pass filter is 1.591549 Hz; gamma is excluded because the low-pass filter is 30 Hz.",
        "",
        "## Baseline-only anomaly model",
        "",
        f"- Clean interictal windows: {summary['baseline_windows']}",
        f"- Final-minute preictal windows: {summary['preictal_windows']}",
        f"- Threshold: {summary['threshold']:.3f}, the 95th percentile of clean-interictal scores.",
        f"- Interictal flags: {summary['interictal_flag_rate_percent']:.1f}%.",
        f"- Preictal flags: {summary['preictal_flag_rate_percent']:.1f}%.",
        f"- Median score: {summary['interictal_score_median']:.3f} interictal vs {summary['preictal_score_median']:.3f} preictal.",
        f"- Earliest threshold-crossing preictal midpoint: {summary['earliest_flagged_preictal_midpoint_seconds']} seconds (null means no preictal window crossed).",
        "",
        "## Features with the largest robust preictal shifts",
        "",
        "| Feature | Interictal median | Preictal median | Robust shift | AUC | Spearman rho vs time-to-onset |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            f"| {row.feature} | {row.interictal_median:.4g} | {row.preictal_median:.4g} | {row.robust_standardized_shift:.2f} | {row.univariate_auc_preictal_vs_interictal:.2f} | {row.spearman_rho_time_to_onset:.2f} |"
        )
    lines.extend(
        [
            "",
            "AUC here is a descriptive ranking of the one seizure's ten final-minute windows against interictal windows. It must not be treated as a prospective or independent performance estimate; the windows are temporally correlated.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel-dir", type=Path, default=CHANNEL_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    channel_table, window_table, definitions = extract_feature_tables(args.channel_dir)
    modeled, effects, summary = robust_baseline_model(window_table)
    channel_table.to_csv(args.results_dir / "channel_window_features.csv", index=False)
    definitions.to_csv(args.results_dir / "window_definitions.csv", index=False)
    modeled.to_csv(args.results_dir / "robust_window_features_and_scores.csv", index=False)
    effects.to_csv(args.results_dir / "feature_effects_ranked.csv", index=False)
    (args.results_dir / "model_settings.json").write_text(
        json.dumps(
            {
                "recording": "PN00-1.edf",
                "seizure_onset_seconds": SEIZURE_ONSET_SECONDS,
                "seizure_end_seconds": SEIZURE_END_SECONDS,
                "window_seconds": WINDOW_SECONDS,
                "interictal_buffer_seconds": INTERICTAL_BUFFER_SECONDS,
                "bands_hz": BANDS,
                "model_features": MODEL_FEATURES,
                "summary": summary,
                "limitation": "One seizure only: no independent predictive validation is possible.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_results(modeled, effects, summary, args.results_dir / "PN00-1_preictal_feature_model.png")
    write_report(args.results_dir / "PN00-1_PREICTAL_FEATURE_REPORT.md", effects, summary)
    print(f"Saved feature screen to {args.results_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
