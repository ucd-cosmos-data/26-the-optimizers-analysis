"""Run the complete K-finder, K-suiter, P-versus-G analysis."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from src.reduced_sensor_pipeline import (  # noqa: E402
    ALLOWED_AUPRC_LOSS,
    TARGET_PATIENTS,
    build_clean_cohort_matrix,
    channel_set_summary,
    data_quality_summary,
    load_patient_cohort,
    paired_patient_bootstrap,
    run_k_finder,
    run_personalized_generalized,
    shared_channels,
)


def save_k_figure(curve: pd.DataFrame, k: int, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5.4))
    axis.plot(
        curve["k"],
        curve["worst_patient_auprc_loss_from_full"],
        color="#315c8c",
        marker="o",
        markersize=3.5,
        linewidth=1.8,
    )
    axis.axhline(
        ALLOWED_AUPRC_LOSS,
        color="#b64d35",
        linestyle="--",
        label="Maximum allowed loss = 0.03",
    )
    selected = curve.loc[curve["k"].eq(k)].iloc[0]
    axis.scatter(
        [k],
        [selected["worst_patient_auprc_loss_from_full"]],
        s=95,
        color="#d18b2c",
        edgecolor="white",
        zorder=4,
    )
    axis.annotate(
        f"K = {k}",
        (k, selected["worst_patient_auprc_loss_from_full"]),
        xytext=(10, 18),
        textcoords="offset points",
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "#555555"},
    )
    axis.set(
        title="K-finder: worst held-out-patient loss versus the full montage",
        xlabel="Number of EEG sensors",
        ylabel="Largest AUPRC decrease across 14 patients",
        xlim=(1, int(curve["k"].max())),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_comparison_figure(results: pd.DataFrame, output: Path) -> None:
    long = results.melt(
        id_vars=["patient_id"],
        value_vars=[
            "P_cross_entropy",
            "G_cross_entropy",
            "test_prevalence_oracle_cross_entropy",
        ],
        var_name="model",
        value_name="cross_entropy",
    )
    long["model"] = long["model"].map(
        {
            "P_cross_entropy": "Personalized (P)",
            "G_cross_entropy": "Generalized (G)",
            "test_prevalence_oracle_cross_entropy": (
                "Best constant (uses test prevalence)"
            ),
        }
    )
    fig, axis = plt.subplots(figsize=(9.4, 5.5))
    sns.barplot(
        data=long,
        x="patient_id",
        y="cross_entropy",
        hue="model",
        palette=["#2a9d8f", "#e76f51", "#9b9b9b"],
        ax=axis,
    )
    axis.set(
        title="Personalized versus unseen-patient generalized prediction",
        xlabel="Held-out patient",
        ylabel="Binary cross-entropy (lower is better)",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, title=None)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_channel_figure(
    frequencies: pd.DataFrame,
    output: Path,
) -> None:
    matrix = frequencies.pivot(
        index="model", columns="channel", values="patients_selected"
    )
    matrix = matrix.reindex(["P", "G"])
    fig, axis = plt.subplots(figsize=(13, 3.1))
    sns.heatmap(
        matrix,
        cmap="YlGnBu",
        vmin=0,
        vmax=5,
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        cbar_kws={"label": "Patients selected (of 5)"},
        ax=axis,
    )
    axis.set(
        title="How often each channel was selected in the five P and G models",
        xlabel="EEG channel",
        ylabel="Selection strategy",
    )
    axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_quality_figure(quality: pd.DataFrame, output: Path) -> None:
    plot = quality.copy()
    plot["rejected_percent"] = (
        100
        * plot["sensor_contexts_with_any_rejected_5s_window_fraction"]
    )
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    sns.barplot(
        data=plot,
        x="patient_id",
        y="rejected_percent",
        color="#6c8ebf",
        ax=axis,
    )
    axis.set(
        title="Artifact cleaning applied before model fitting",
        xlabel="Patient",
        ylabel="Sensor contexts containing ≥1 rejected 5-second window (%)",
    )
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete reduced-sensor EEG analysis."
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel sensor fits (-1 uses all CPU cores).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    cache_dir = (
        PROJECT
        / "data"
        / "processed"
        / "channel_features"
    )
    cohort_cache = (
        PROJECT
        / "data"
        / "processed"
        / "final"
        / "cohort_features_clean.npz"
    )
    output = PROJECT / "results" / "final"
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    print("Building one version-checked, leakage-cleaned cohort matrix...", flush=True)
    build_clean_cohort_matrix(cache_dir, cohort_cache)
    print("Running nested leave-one-patient-out K-finder...", flush=True)
    k_result = run_k_finder(
        cohort_cache,
        allowed_loss=ALLOWED_AUPRC_LOSS,
        n_jobs=args.n_jobs,
    )
    k = k_result.k
    k_result.subject_scores.to_csv(
        tables / "k_finder_patient_scores.csv", index=False
    )
    k_result.loss_curve.to_csv(
        tables / "k_finder_loss_curve.csv", index=False
    )
    k_result.channel_paths.to_csv(
        tables / "k_finder_channel_paths.csv", index=False
    )
    selected_row = k_result.loss_curve.loc[
        k_result.loss_curve["k"].eq(k)
    ].iloc[0]
    acceptable = k_result.loss_curve[
        "meets_0.03_worst_patient_limit"
    ].to_numpy(dtype=bool)
    connected_plateau_k = len(acceptable)
    for index in range(len(acceptable) - 1, -1, -1):
        if acceptable[index]:
            connected_plateau_k = index + 1
        else:
            break
    previous_row = k_result.loss_curve.loc[
        k_result.loss_curve["k"].eq(k - 1)
    ]
    selected_scores = k_result.subject_scores[k].to_numpy(dtype=float)
    full_scores = k_result.subject_scores[
        len(k_result.channels)
    ].to_numpy(dtype=float)
    prevalence_scores = k_result.subject_scores[
        "positive_fraction"
    ].to_numpy(dtype=float)
    k_summary = {
        "selected_k": k,
        "full_common_montage_sensors": len(k_result.channels),
        "patients": len(k_result.subject_scores),
        "metric": "held-out average precision (AUPRC)",
        "selection_rule": (
            "smallest K whose observed AUPRC decrease from the full "
            "montage is <= 0.03 for every held-out patient"
        ),
        "allowed_worst_patient_auprc_loss": ALLOWED_AUPRC_LOSS,
        "selected_k_worst_patient_auprc_loss": float(
            selected_row["worst_patient_auprc_loss_from_full"]
        ),
        "selected_k_mean_held_out_auprc": float(selected_scores.mean()),
        "full_montage_mean_held_out_auprc": float(full_scores.mean()),
        "mean_held_out_positive_fraction": float(prevalence_scores.mean()),
        "selected_k_patients_below_positive_fraction": int(
            np.sum(selected_scores < prevalence_scores)
        ),
        "full_montage_patients_below_positive_fraction": int(
            np.sum(full_scores < prevalence_scores)
        ),
        "connected_strict_plateau_k": connected_plateau_k,
        "fragility_warning": (
            "This is an observed preservation result selected on the same "
            "14 outer folds, not a clinical accuracy guarantee. The literal "
            "minimum can be an isolated pass because adding weak sensors can "
            "hurt an unweighted probability average. The connected plateau is "
            "the smallest K for which K and every larger count pass."
        ),
        "k_minus_1_worst_patient_auprc_loss": (
            float(
                previous_row.iloc[0][
                    "worst_patient_auprc_loss_from_full"
                ]
            )
            if not previous_row.empty
            else None
        ),
        "outer_validation": "leave one complete patient out",
        "inner_selection": "patient-grouped forward sensor selection",
        "random_seed": 42,
    }
    (tables / "k_finder_summary.json").write_text(
        json.dumps(k_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    save_k_figure(
        k_result.loss_curve, k, figures / "01_k_finder.png"
    )
    print(f"K-finder selected K={k}.", flush=True)

    print("Loading cleaned channel-level feature caches...", flush=True)
    patients = load_patient_cohort(cache_dir, TARGET_PATIENTS)
    common_channels = shared_channels(patients.values())
    quality = data_quality_summary(patients)
    quality.to_csv(tables / "data_quality_summary.csv", index=False)

    print("Training five personalized and five generalized models...", flush=True)
    results, rankings, predictions, event_losses = (
        run_personalized_generalized(
            patients,
            k=k,
            n_jobs=args.n_jobs,
        )
    )
    results.to_csv(tables / "patient_model_comparison.csv", index=False)
    main_table = results[
        [
            "patient_id",
            "n_train_seizures_P",
            "n_nominal_train_seizures_P",
            "n_test_seizures",
            "n_train_seizures_G",
            "train_events_excluded_for_recording_isolation",
            "personalized_channels",
            "generalized_channels",
            "test_positive_fraction",
            "test_prevalence_oracle_cross_entropy",
            "P_training_prevalence_baseline_cross_entropy",
            "G_training_prevalence_baseline_cross_entropy",
            "P_cross_entropy",
            "G_cross_entropy",
            "P_auprc",
            "G_auprc",
            "P_auroc",
            "G_auroc",
        ]
    ].copy()
    main_table["lower_cross_entropy_model"] = np.where(
        main_table["P_cross_entropy"] < main_table["G_cross_entropy"],
        "P",
        "G",
    )
    main_table.to_csv(tables / "main_results_table.csv", index=False)
    rankings.to_csv(tables / "channel_rankings.csv", index=False)
    predictions.to_csv(tables / "held_out_predictions.csv", index=False)
    event_losses.to_csv(tables / "held_out_event_losses.csv", index=False)

    frequencies, core = channel_set_summary(results, common_channels)
    frequencies.to_csv(tables / "channel_selection_frequency.csv", index=False)
    core.to_csv(tables / "core_channels.csv", index=False)
    bootstrap = paired_patient_bootstrap(results)
    bootstrap["selected_k"] = k
    bootstrap["interpretation"] = (
        "G minus P: positive values favor P because lower cross-entropy is better."
    )
    (tables / "paired_comparison_summary.json").write_text(
        json.dumps(bootstrap, indent=2) + "\n",
        encoding="utf-8",
    )

    save_comparison_figure(
        results, figures / "02_personalized_vs_generalized.png"
    )
    save_channel_figure(
        frequencies, figures / "03_channel_selection_frequency.png"
    )
    save_quality_figure(quality, figures / "04_artifact_cleaning.png")

    run_metadata = {
        "completed": True,
        "runtime_seconds": time.time() - started,
        "selected_k": k,
        "target_patients": list(TARGET_PATIENTS),
        "shared_channel_count": len(common_channels),
        "shared_channels": common_channels,
        "prediction_target": "seizure onset within the next five minutes",
        "landmark_spacing_seconds": 5,
        "past_context_seconds": 120,
        "test_rule": (
            "last max(ceil(0.20 * seizures), 2) events; their entire recording "
            "sessions are reserved for test"
        ),
        "features_per_channel": (
            "21: mean/latest/slope of delta, theta, alpha, beta, RMS, "
            "line length, and usable-window fraction"
        ),
        "base_model": (
            "median imputation + standardization + balanced L2 logistic "
            "regression (C=0.1) per channel; selected-channel probabilities averaged"
        ),
        "calibration": "positive-slope Platt fit on training-only OOF predictions",
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "git_worktree_was_dirty_at_run": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "code_sha256": {
            "run_analysis.py": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "src/reduced_sensor_pipeline.py": hashlib.sha256(
                (PROJECT / "src" / "reduced_sensor_pipeline.py").read_bytes()
            ).hexdigest(),
            "build_feature_cache.py": hashlib.sha256(
                (PROJECT / "build_feature_cache.py").read_bytes()
            ).hexdigest(),
            "src/channel_features.py": hashlib.sha256(
                (PROJECT / "src" / "channel_features.py").read_bytes()
            ).hexdigest(),
            "src/eeg_io.py": hashlib.sha256(
                (PROJECT / "src" / "eeg_io.py").read_bytes()
            ).hexdigest(),
        },
        "episode_manifest_sha256": hashlib.sha256(
            (PROJECT / "metadata" / "episode_manifest.csv").read_bytes()
        ).hexdigest(),
        "clean_cohort_metadata_sha256": hashlib.sha256(
            cohort_cache.with_suffix(".json").read_bytes()
        ).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            **{
                package: importlib.metadata.version(package)
                for package in (
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "joblib",
                    "matplotlib",
                    "seaborn",
                    "pyedflib",
                )
            },
        },
        "research_only": True,
    }
    (tables / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Finished in {run_metadata['runtime_seconds']:.1f}s. "
        f"Results: {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
