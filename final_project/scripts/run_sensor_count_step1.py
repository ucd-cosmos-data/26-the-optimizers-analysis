"""Build the 29-channel cohort features and print only the Step-1 value K."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

import personalized_channels_workflow as pc
import rolling_seizure_forecasting as rsf
import split_eeg_channels as split_eeg
from seizure_sensor_selection import CohortSensorCountSelector, CountCurvePoint


EXPECTED_SUBJECTS = 14
EXPECTED_COMMON_CHANNELS = 29


def save_plateau_artifacts(
    curve: list[CountCurvePoint],
    k: int,
    margin: float,
    results_dir: Path,
) -> None:
    """Save count-only curve data and a report-ready plateau graph."""

    results_dir.mkdir(parents=True, exist_ok=True)
    curve_path = results_dir / "sensor_count_step1_curve.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(curve[0].__dataclass_fields__),
        )
        writer.writeheader()
        writer.writerows(
            {
                field: getattr(point, field)
                for field in point.__dataclass_fields__
            }
            for point in curve
        )

    counts = np.asarray([point.k for point in curve])
    means = np.asarray([point.mean_score for point in curve])
    lower = np.asarray([point.score_ci_lower for point in curve])
    upper = np.asarray([point.score_ci_upper for point in curve])
    baseline = means[-1]

    fig, axis = plt.subplots(figsize=(10, 6))
    axis.axvspan(
        k - 0.5,
        counts[-1] + 0.5,
        color="#c05a20",
        alpha=0.05,
        label=f"Accepted plateau: {k}–29 sensors",
    )
    axis.axhspan(
        baseline - margin,
        baseline,
        color="#2f855a",
        alpha=0.12,
        label=f"29-sensor accuracy − {margin:.02f} margin",
    )
    axis.fill_between(
        counts,
        lower,
        upper,
        color="#3568a8",
        alpha=0.16,
        label="95% subject-level confidence interval",
    )
    axis.plot(
        counts,
        means,
        color="#24578f",
        marker="o",
        markersize=4,
        linewidth=2,
        label="Mean held-out representative accuracy",
    )
    axis.axhline(
        baseline,
        color="#2f855a",
        linestyle="--",
        linewidth=1.5,
        label=f"Full 29-sensor baseline ({baseline:.3f})",
    )
    axis.axvline(k, color="#c05a20", linestyle="--", linewidth=2)
    axis.scatter(
        [k],
        [means[k - 1]],
        s=90,
        color="#c05a20",
        edgecolor="white",
        linewidth=1,
        zorder=5,
    )
    axis.annotate(
        f"K = {k}\nplateau edge",
        xy=(k, means[k - 1]),
        xytext=(-68, 32),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#c05a20"},
        color="#7d3817",
        fontweight="bold",
    )
    axis.set(
        title="Cohort sensor-count performance plateau",
        xlabel="Number of sensors",
        ylabel="Representative accuracy (macro average precision)",
        xlim=(1, counts[-1]),
    )
    axis.set_xticks(sorted(set(range(1, counts[-1] + 1, 2)) | {k, 29}))
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
    )
    fig.tight_layout()
    fig.savefig(
        results_dir / "sensor_count_step1_plateau.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def pooled_cohort_features(
    *,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return channel-aligned features, targets, and subject IDs."""

    paths = pc.personalized_paths(Path.cwd())
    forecast_config = rsf.ForecastConfig()
    manifest = pc.load_manifest(paths, forecast_config)
    missing_split = [
        split_eeg.split_recording_dir(edf_path) / "channel_manifest.csv"
        for edf_path in manifest["edf_path"].drop_duplicates()
        if not (
            split_eeg.split_recording_dir(edf_path) / "channel_manifest.csv"
        ).exists()
    ]
    if missing_split:
        raise FileNotFoundError(
            f"{len(missing_split)} recording(s) have not been split by channel. "
            "Run `python final_project/scripts/split_eeg_channels.py` first."
        )
    patient_data = [
        pc.build_patient_feature_data(
            patient_manifest,
            paths["feature_cache"],
            forecast_config,
            force=force,
            verbose=False,
        )
        for _, patient_manifest in manifest.groupby("patient_id", sort=True)
    ]
    if len(patient_data) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_SUBJECTS} subjects, found {len(patient_data)}."
        )
    common_channels = sorted(
        set.intersection(
            *(set(patient.channel_names) for patient in patient_data)
        ),
        key=pc._natural_recording_key,
    )
    if len(common_channels) != EXPECTED_COMMON_CHANNELS:
        raise ValueError(
            f"Expected {EXPECTED_COMMON_CHANNELS} cohort-common channels, "
            f"found {len(common_channels)}."
        )

    feature_blocks = []
    targets = []
    subject_ids = []
    for patient in patient_data:
        channel_map = pc.compact_channel_column_map(
            patient.channel_feature_columns
        )
        feature_blocks.append(
            np.stack(
                [
                    patient.frame[channel_map[channel]].to_numpy(np.float32)
                    for channel in common_channels
                ],
                axis=1,
            )
        )
        targets.append(
            patient.frame["has_event_in_5m"].to_numpy(np.uint8)
        )
        subject_ids.append(
            np.repeat(patient.patient_id, len(patient.frame))
        )

    X = np.concatenate(feature_blocks)
    y = np.concatenate(targets)
    subjects = np.concatenate(subject_ids)
    cache_path = (
        paths["processed"] / "cohort_sensor_count_29_features.npz"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X=X,
        y=y,
        subjects=subjects,
        channels=np.asarray(common_channels),
    )
    return X, y, subjects


def run_step1(
    *,
    margin: float = 0.02,
    confidence: float = 0.95,
    inner_splits: int = 4,
    n_jobs: int = -1,
    force_features: bool = False,
) -> int:
    """Run the complete cohort calculation, save its settings, and return K."""

    started = time.time()
    X, y, subjects = pooled_cohort_features(force=force_features)
    selector = CohortSensorCountSelector(
        scoring="average_precision",
        noninferiority_margin=margin,
        confidence=confidence,
        inner_splits=inner_splits,
        outer_splits=None,
        n_jobs=n_jobs,
        random_state=42,
    )
    k = selector.select_k(X, y, subjects=subjects)

    result_path = Path(__file__).resolve().parents[1] / "results"
    save_plateau_artifacts(
        selector.count_curve_,
        k,
        margin,
        result_path,
    )
    (result_path / "sensor_count_step1.json").write_text(
        json.dumps(
            {
                "K": k,
                "full_sensor_count": X.shape[1],
                "n_subjects": int(np.unique(subjects).size),
                "n_rows": len(y),
                "features_per_sensor": X.shape[2],
                "signal_source": "data/raw/splitdata channel arrays",
                "evaluation": "greedy sensor probability ensemble",
                "scoring": "average_precision",
                "noninferiority_margin": margin,
                "confidence": confidence,
                "inner_splits": inner_splits,
                "outer_validation": "leave_one_subject_out",
                "random_state": 42,
                "runtime_seconds": time.time() - started,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return k


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the cohort-level EEG sensor count K."
    )
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()

    k = run_step1(
        margin=args.margin,
        confidence=args.confidence,
        inner_splits=args.inner_splits,
        n_jobs=args.n_jobs,
        force_features=args.force_features,
    )
    print(k)


if __name__ == "__main__":
    main()
