"""The K-Suiter: rank the top K EEG channels for one patient.

The selector is intentionally patient-specific.  It reserves the latest
seizure/control pair for a final untouched evaluation, then uses repeated
expanding chronological validation on earlier episodes. The ranking utility
matches the forecast alarm trade-off:

    value = 2 * sensitivity - 0.10 * false alarms/hour
            - 0.25 * time in warning + 0.20 * AUROC

This is exploratory research code, not a clinical decision system.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import personalized_channels_workflow as pc
import rolling_seizure_forecasting as rsf


K_FINDER_RESULT_NAME = "sensor_count_selected.json"
STABILITY_BONUS = 0.05


@dataclass(frozen=True)
class KFinderResult:
    """Validated cohort-level K selected by the K-Finder workflow."""

    k: int
    source_path: Path
    metadata: dict[str, Any]


def load_k_finder_result(project_dir: str | Path) -> KFinderResult:
    """Load the authoritative count written by K-Finder.

    K-Finder deliberately selects a cohort-level *count*, while K-Suiter
    selects patient-specific identities. The authoritative handoff uses the
    strict worst-held-out-patient rule. Reading this one versioned artifact
    prevents copying a displayed notebook value into a second notebook.
    """

    path = Path(project_dir) / "results" / K_FINDER_RESULT_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"K-Finder result is missing: {path}. Run k-finder.ipynb first."
        )
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"K-Finder result is not valid JSON: {path}.") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"K-Finder result must contain a JSON object: {path}.")
    k = metadata.get("K")
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError(f"K-Finder result has invalid K={k!r}: {path}.")
    return KFinderResult(int(k), path.resolve(), metadata)


@dataclass(frozen=True)
class ChannelRecommendation:
    """One channel's position and value in the greedy patient-specific ranking."""

    rank: int
    channel: str
    value: float
    marginal_value: float | None
    validation_auprc: float
    validation_brier: float


def _ordered_event_ids(frame: pd.DataFrame) -> list[str]:
    """Return chronological seizure IDs, retaining their matched controls."""

    events = frame.loc[frame["episode_type"].eq("preictal")].copy()
    sort_columns = [column for column in ("recording", "landmark_seconds") if column in events]
    if sort_columns:
        events = events.sort_values(sort_columns)
    return events["source_event_id"].astype(str).drop_duplicates().tolist()


def _alarm_utility(
    frame: pd.DataFrame,
    risk: np.ndarray,
    threshold: float,
) -> tuple[float, float, float, float, float]:
    """Return final-forecast-aligned utility and its component metrics."""

    warning = rsf._warning_summary(frame, risk, threshold)
    y = frame["has_event_in_5m"].to_numpy(dtype=int)
    auroc = float(roc_auc_score(y, risk)) if np.unique(y).size == 2 else 0.5
    utility = (
        2.0 * warning["sensitivity"]
        - 0.10 * warning["false_alarms_per_hour"]
        - 0.25 * warning["time_in_warning"]
        + 0.20 * auroc
    )
    return (
        float(utility),
        float(warning["sensitivity"]),
        float(warning["false_alarms_per_hour"]),
        float(warning["time_in_warning"]),
        auroc,
    )


def _selector_probabilities(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    config: pc.PersonalizedConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one fold without deriving imputation values from its validation set."""

    x_train = train[columns].to_numpy(dtype=float)
    x_test = test[columns].to_numpy(dtype=float)
    fill = np.nanmedian(x_train, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, fill)
    x_test = np.where(np.isfinite(x_test), x_test, fill)
    model = pc._selector_estimator(config)
    model.fit(x_train, train["has_event_in_5m"].to_numpy(dtype=int))
    return model.predict_proba(x_train)[:, 1], model.predict_proba(x_test)[:, 1]


def _alarm_selection_score(
    frame: pd.DataFrame,
    columns: list[str],
    config: pc.PersonalizedConfig,
) -> tuple[float, float, float, float, float, float, float]:
    """Expanding-validation score matching the forecast alarm trade-off."""

    splits, method = pc._chronological_validation_splits(frame)
    if method != "expanding_chronological_validation":
        raise ValueError("At least two development seizures are required.")
    values: list[tuple[float, float, float, float, float]] = []
    probabilities: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    for fit_index, valid_index in splits:
        fit = frame.iloc[fit_index]
        valid = frame.iloc[valid_index]
        fit_risk, valid_risk = _selector_probabilities(fit, valid, columns, config)
        threshold, _ = rsf.select_warning_threshold(
            fit, fit_risk, target_time_in_warning=0.25
        )
        values.append(_alarm_utility(valid, valid_risk, threshold))
        probabilities.append(valid_risk)
        truths.append(valid["has_event_in_5m"].to_numpy(dtype=int))
    utility, sensitivity, false_alarms, warning_time, auroc = np.mean(values, axis=0)
    probability = np.concatenate(probabilities)
    truth = np.concatenate(truths)
    return (
        float(utility),
        float(sensitivity),
        float(false_alarms),
        float(warning_time),
        float(auroc),
        float(average_precision_score(truth, probability)),
        float(brier_score_loss(truth, probability)),
    )


def _select_stable_alarm_channels(
    frame: pd.DataFrame,
    channel_map: dict[str, list[str]],
    config: pc.PersonalizedConfig,
    stability_frequency: dict[str, float],
) -> tuple[list[str], pd.DataFrame]:
    """Greedily select a stable montage using alarm-policy validation utility."""

    chosen: list[str] = []
    remaining = set(channel_map)
    cache: dict[tuple[str, ...], tuple[float, float, float, float, float, float, float]] = {}
    rows: list[dict[str, Any]] = []
    for rank in range(1, config.k + 1):
        candidates = []
        for channel in sorted(remaining, key=pc._natural_recording_key):
            subset = [*chosen, channel]
            key = tuple(sorted(subset, key=pc._natural_recording_key))
            if key not in cache:
                cache[key] = _alarm_selection_score(
                    frame, pc._subset_columns(channel_map, key), config
                )
            utility, sensitivity, false_alarms, warning_time, auroc, ap, brier = cache[key]
            stability = stability_frequency.get(channel, 0.0)
            candidates.append(
                (utility + STABILITY_BONUS * stability, channel, utility,
                 stability, sensitivity, false_alarms, warning_time, auroc, ap, brier)
            )
        candidate = max(candidates, key=lambda value: (value[0], value[1]))
        _, channel, utility, stability, sensitivity, false_alarms, warning_time, auroc, ap, brier = candidate
        chosen.append(channel)
        remaining.remove(channel)
        rows.append(
            {
                "step": rank,
                "channel": channel,
                "channels": ", ".join(chosen),
                "selection_score": utility + STABILITY_BONUS * stability,
                "validation_alarm_utility": utility,
                "stability_frequency": stability,
                "validation_sensitivity": sensitivity,
                "validation_false_alarms_per_hour": false_alarms,
                "validation_time_in_warning": warning_time,
                "validation_auroc": auroc,
                "validation_auprc": ap,
                "validation_brier": brier,
            }
        )
    return chosen, pd.DataFrame(rows)


class KSuiter:
    """Rank channels that best suit a patient's seizure-forecasting history.

    Parameters
    ----------
    config:
        Reuses the personalized workflow's model and reproducibility settings.
        ``k`` and ``swap_refinement`` are overridden by :meth:`rank`; a ranking
        must be nested, so one-for-one swaps are deliberately not used.
    """

    def __init__(self, config: pc.PersonalizedConfig | None = None) -> None:
        self.config = pc.PersonalizedConfig() if config is None else config

    @staticmethod
    def _validate_patient(
        patient: pc.PatientFeatureData,
        k: int,
    ) -> dict[str, list[str]]:
        if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
            raise ValueError("k must be a positive integer.")
        if not isinstance(patient, pc.PatientFeatureData):
            raise TypeError("patient must be a PatientFeatureData instance.")
        frame = patient.frame
        required = {"patient_id", "source_event_id", "has_event_in_5m"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"patient.frame is missing columns: {missing}.")
        ids = frame["patient_id"].astype(str).unique()
        if len(ids) != 1 or ids[0] != str(patient.patient_id):
            raise ValueError(
                "patient.frame must contain exactly the declared patient_id."
            )
        if frame["source_event_id"].astype(str).nunique() < 2:
            raise ValueError(
                "At least two historical seizure events are required for "
                "chronological validation."
            )
        if set(frame["has_event_in_5m"].astype(int).unique()) != {0, 1}:
            raise ValueError("The patient history must contain both target classes.")

        # The full project cache contains a wide, regular feature map; use its
        # prespecified compact subset to control patient-level overfit.  A
        # caller may also supply an already-compact or custom feature map.
        expected_compact_width = (
            len(pc.COMPACT_FEATURE_NAMES) * len(pc.COMPACT_AGGREGATIONS)
        )
        try:
            compact_map = pc.compact_channel_column_map(
                patient.channel_feature_columns
            )
        except (AssertionError, ValueError):
            compact_map = {}
        channel_map = (
            compact_map
            if compact_map
            and all(
                len(columns) == expected_compact_width
                for columns in compact_map.values()
            )
            else {
                channel: list(columns)
                for channel, columns in patient.channel_feature_columns.items()
            }
        )
        if not channel_map or any(not columns for columns in channel_map.values()):
            raise ValueError("Every available channel needs at least one feature.")
        if k > len(channel_map):
            raise ValueError(
                f"Requested k={k}, but patient {patient.patient_id} has only "
                f"{len(channel_map)} available channels."
            )
        absent = sorted(
            {
                column
                for columns in channel_map.values()
                for column in columns
                if column not in frame.columns
            }
        )
        if absent:
            raise ValueError(
                f"patient.frame lacks {len(absent)} mapped feature columns."
            )
        return channel_map

    def rank(
        self,
        patient: pc.PatientFeatureData,
        k: int,
    ) -> list[ChannelRecommendation]:
        """Return the patient's top K channels in greedy conditional-value order.

        Rank 1 is the best channel alone.  Each later rank is the channel that
        adds the most validation value to the channels already selected.
        """

        channel_map = self._validate_patient(patient, k)
        config = replace(self.config, k=int(k), swap_refinement=False)
        config.validate()
        selected, trace, validation_method = pc.select_fixed_k_channels(
            patient.frame,
            channel_map,
            config,
        )
        if validation_method != "expanding_chronological_validation":
            raise RuntimeError(
                "K-Suiter requires at least two events and must not use "
                "training resubstitution."
            )

        values = trace["selection_score"].to_numpy(dtype=float)
        marginal = np.diff(values, prepend=np.nan)
        recommendations = [
            ChannelRecommendation(
                rank=position,
                channel=channel,
                value=float(values[position - 1]),
                marginal_value=(
                    None if position == 1 else float(marginal[position - 1])
                ),
                validation_auprc=float(
                    trace.iloc[position - 1]["validation_auprc"]
                ),
                validation_brier=float(
                    trace.iloc[position - 1]["validation_brier"]
                ),
            )
            for position, channel in enumerate(selected, start=1)
        ]
        self.patient_id_ = str(patient.patient_id)
        self.k_ = int(k)
        self.recommendations_ = recommendations
        self.ranking_ = pd.DataFrame(
            [recommendation.__dict__ for recommendation in recommendations]
        )
        return recommendations

    def recommend(
        self,
        patient: pc.PatientFeatureData,
        k: int,
    ) -> list[str]:
        """Return only the ordered channel names for ``patient`` and ``k``."""

        return [
            recommendation.channel
            for recommendation in self.rank(patient, k)
        ]

    def recommend_from_k_finder(
        self,
        patient: pc.PatientFeatureData,
        project_dir: str | Path,
    ) -> list[str]:
        """Select stable channels and evaluate only on the final held-out event."""
        result = load_k_finder_result(project_dir)
        channel_map = self._validate_patient(patient, result.k)
        config = replace(self.config, k=result.k, swap_refinement=False)
        config.validate()
        event_ids = _ordered_event_ids(patient.frame)
        if len(event_ids) < 3:
            raise ValueError(
                "Stable K-Suiter selection needs at least three seizure events: "
                "two for development and one final held-out event."
            )
        final_event = event_ids[-1]
        development_events = event_ids[:-1]
        event_values = patient.frame["source_event_id"].astype(str)
        development = patient.frame.loc[event_values.isin(development_events)].copy()
        held_out = patient.frame.loc[event_values.eq(final_event)].copy()

        # Repeat the same chronological selection over every usable historical
        # prefix.  The final fit receives a small bonus for channels that recur.
        stability_rows: list[dict[str, Any]] = []
        selection_counts = {channel: 0 for channel in channel_map}
        prefixes = range(2, len(development_events) + 1)
        for prefix_size in prefixes:
            prefix_ids = development_events[:prefix_size]
            prefix = development.loc[
                development["source_event_id"].astype(str).isin(prefix_ids)
            ].copy()
            selected, _ = _select_stable_alarm_channels(
                prefix, channel_map, config, {}
            )
            for channel in selected:
                selection_counts[channel] += 1
            stability_rows.extend(
                {
                    "development_events": prefix_size,
                    "channel": channel,
                    "selected": True,
                }
                for channel in selected
            )
        repeat_count = len(development_events) - 1
        stability_frequency = {
            channel: count / repeat_count
            for channel, count in selection_counts.items()
        }
        selected, trace = _select_stable_alarm_channels(
            development, channel_map, config, stability_frequency
        )
        columns = pc._subset_columns(channel_map, selected)
        development_risk, held_out_risk = _selector_probabilities(
            development, held_out, columns, config
        )
        threshold, _ = rsf.select_warning_threshold(
            development, development_risk, target_time_in_warning=0.25
        )
        utility, sensitivity, false_alarms, warning_time, auroc = _alarm_utility(
            held_out, held_out_risk, threshold
        )
        y = held_out["has_event_in_5m"].to_numpy(dtype=int)
        self.final_holdout_metrics_ = {
            "held_out_source_event_id": final_event,
            "alarm_utility": utility,
            "seizure_sensitivity": sensitivity,
            "false_alarms_per_hour": false_alarms,
            "time_in_warning": warning_time,
            "auroc": auroc,
            "auprc": float(average_precision_score(y, held_out_risk)),
            "binary_brier": float(brier_score_loss(y, held_out_risk)),
            "warning_threshold": float(threshold),
        }
        self.stability_ = pd.DataFrame(stability_rows)
        self.stability_frequency_ = stability_frequency
        self.development_event_ids_ = development_events
        self.held_out_event_id_ = final_event
        self.patient_id_ = str(patient.patient_id)
        self.k_ = int(result.k)
        self.recommendations_ = [
            ChannelRecommendation(
                rank=index,
                channel=channel,
                value=float(trace.iloc[index - 1]["validation_alarm_utility"]),
                marginal_value=(
                    None
                    if index == 1
                    else float(
                        trace.iloc[index - 1]["validation_alarm_utility"]
                        - trace.iloc[index - 2]["validation_alarm_utility"]
                    )
                ),
                validation_auprc=float(trace.iloc[index - 1]["validation_auprc"]),
                validation_brier=float(trace.iloc[index - 1]["validation_brier"]),
            )
            for index, channel in enumerate(selected, start=1)
        ]
        self.ranking_ = trace.copy()
        self.ranking_.insert(0, "rank", np.arange(1, len(trace) + 1))
        self.k_finder_result_ = result
        return selected

    def save_recommendation(
        self,
        output_dir: str | Path,
    ) -> tuple[Path, Path]:
        """Write canonical channels and full ranking for a completed selection.

        The JSON is a direct handoff artifact: its ``included_eeg_channels``
        value can be passed to ``ForecastConfig`` without editing labels.  The
        CSV preserves the validation audit behind that short list.
        """

        if not hasattr(self, "ranking_") or not hasattr(self, "patient_id_"):
            raise RuntimeError("Run recommend() before saving a recommendation.")
        if not hasattr(self, "k_finder_result_"):
            raise RuntimeError(
                "Use recommend_from_k_finder() so saved output records its K source."
            )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"k_suiter_{self.patient_id_}_k{self.k_}"
        csv_path = output_dir / f"{stem}_ranking.csv"
        json_path = output_dir / f"{stem}_recommendation.json"
        self.ranking_.to_csv(csv_path, index=False)
        stability_path = output_dir / f"{stem}_stability.csv"
        self.stability_.to_csv(stability_path, index=False)
        payload = {
            "patient_id": self.patient_id_,
            "k": self.k_,
            "included_eeg_channels": self.ranking_["channel"].tolist(),
            "selection_method": "patient-specific expanding chronological validation",
            "value_function": "validation AUPRC - 0.25 * validation Brier score",
            "k_finder_result_path": str(self.k_finder_result_.source_path),
            "k_finder_metadata": self.k_finder_result_.metadata,
            "ranking_csv": csv_path.name,
            "stability_csv": stability_path.name,
            "stability_frequency": self.stability_frequency_,
            "development_source_event_ids": self.development_event_ids_,
            "final_held_out_metrics": self.final_holdout_metrics_,
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return json_path, csv_path


def recommend_channels(
    patient: pc.PatientFeatureData,
    k: int,
    *,
    config: pc.PersonalizedConfig | None = None,
) -> list[str]:
    """Convenience interface: ``patient + k -> ordered channel names``."""

    return KSuiter(config=config).recommend(patient, k)


def recommend_and_save_from_k_finder(
    patient: pc.PatientFeatureData,
    *,
    project_dir: str | Path,
    output_dir: str | Path | None = None,
    config: pc.PersonalizedConfig | None = None,
) -> tuple[list[str], Path, Path]:
    """Use K-Finder's saved K and persist K-Suiter's channel handoff files."""

    suiter = KSuiter(config=config)
    channels = suiter.recommend_from_k_finder(patient, project_dir)
    destination = (
        Path(project_dir) / "results" / "k_suiter"
        if output_dir is None
        else Path(output_dir)
    )
    json_path, csv_path = suiter.save_recommendation(destination)
    return channels, json_path, csv_path
