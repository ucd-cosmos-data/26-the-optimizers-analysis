"""Personalized, leakage-safe EEG sensor selection for seizure prediction.

The selector expects features extracted independently for every EEG channel:

    X.shape == (n_windows, n_sensors, n_features_per_sensor)

Run one selector per patient for personalized selection. ``groups`` should identify
recordings/sessions (or seizure episodes) so that correlated windows cannot appear
in both train and validation folds.

The search is greedy forward selection with swap refinement.  It evaluates at
most O(p^3) candidates for p sensors, rather than all 2**p subsets.  The final
reported performance is from nested cross-validation; the sensor subset shown to
the user is refit on all available data after that unbiased evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence
import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True)
class AlarmMetrics:
    """Event-level seizure-prediction metrics."""

    sensitivity: float
    false_alarms_per_hour: float
    mean_warning_time_minutes: float
    brier_score: float
    n_seizures: int
    n_detected_seizures: int
    n_false_alarms: int
    interictal_hours: float


@dataclass(frozen=True)
class CurvePoint:
    k: int
    sensors: tuple[int, ...]
    utility_mean: float
    utility_se: float
    metrics: AlarmMetrics


@dataclass
class SelectionResult:
    """Returned by :meth:`PersonalizedSensorSelector.fit_select`."""

    selected_sensor_indices: list[int]
    selected_sensor_names: list[str]
    elbow_k: int
    nested_cv_metrics: AlarmMetrics
    nested_cv_fold_metrics: list[AlarmMetrics]
    selection_curve: list[CurvePoint]
    outer_fold_sensor_indices: list[list[int]]
    figure: plt.Figure | None = field(default=None, repr=False)

    def summary(self) -> str:
        m = self.nested_cv_metrics
        return (
            f"k={self.elbow_k}; sensors={self.selected_sensor_indices}; "
            f"sensitivity={m.sensitivity:.3f}; "
            f"false alarms/hour={m.false_alarms_per_hour:.3f}; "
            f"mean warning={m.mean_warning_time_minutes:.1f} min; "
            f"Brier={m.brier_score:.3f}"
        )


def default_estimator() -> BaseEstimator:
    """A fast, probability-producing baseline; replace with your own estimator."""

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            solver="liblinear",
            random_state=0,
        ),
    )


def _safe_float_mean(values: Iterable[float]) -> float:
    a = np.asarray(list(values), dtype=float)
    return float(np.nanmean(a)) if np.any(np.isfinite(a)) else float("nan")


def compute_alarm_metrics(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    *,
    duration_hours: ArrayLike,
    seizure_ids: ArrayLike,
    time_to_seizure_minutes: ArrayLike,
    threshold: float = 0.5,
    refractory_minutes: float = 30.0,
    window_start_minutes: ArrayLike | None = None,
    recording_ids: ArrayLike | None = None,
) -> AlarmMetrics:
    """Compute event sensitivity, false alarms/hour, warning time, and Brier.

    Parameters
    ----------
    y_true:
        One for preictal windows and zero for interictal windows.
    duration_hours:
        Duration represented by each window. Overlapping windows should contribute
        only their *stride* duration, not their full width.
    seizure_ids:
        Event identifier for preictal windows. Use -1 (or any value not used by a
        seizure) for interictal windows.
    time_to_seizure_minutes:
        Minutes from the window/alarm to seizure onset for preictal windows.
        Values for interictal windows may be NaN.
    window_start_minutes:
        Absolute recording time. If supplied, adjacent positive interictal windows
        within ``refractory_minutes`` count as one false-alarm episode. If omitted,
        each positive interictal window counts as an alarm (a conservative rate).
    recording_ids:
        Recording/session identifier. When recording times restart at zero, this
        prevents alarms from separate recordings being merged into one episode.
    """

    y = np.asarray(y_true, dtype=int)
    prob = np.asarray(probabilities, dtype=float)
    duration = np.asarray(duration_hours, dtype=float)
    event = np.asarray(seizure_ids)
    tts = np.asarray(time_to_seizure_minutes, dtype=float)
    n = len(y)
    if not all(len(a) == n for a in (prob, duration, event, tts)):
        raise ValueError("All metric inputs must have the same number of windows.")
    if np.any((prob < 0) | (prob > 1) | ~np.isfinite(prob)):
        raise ValueError("probabilities must be finite and lie in [0, 1].")
    if np.any(duration < 0):
        raise ValueError("duration_hours cannot be negative.")

    alarms = prob >= threshold
    preictal = y == 1
    event_values = np.unique(event[preictal])
    detected = 0
    warnings_: list[float] = []
    for event_id in event_values:
        event_alarm = preictal & (event == event_id) & alarms
        if np.any(event_alarm):
            detected += 1
            # Largest time-to-seizure is the earliest correct warning.
            warnings_.append(float(np.nanmax(tts[event_alarm])))

    interictal = ~preictal
    false_mask = interictal & alarms
    if window_start_minutes is None:
        n_false = int(false_mask.sum())
    else:
        starts = np.asarray(window_start_minutes, dtype=float)
        if len(starts) != n:
            raise ValueError("window_start_minutes must match y_true.")
        recording = (
            np.zeros(n, dtype=int)
            if recording_ids is None
            else np.asarray(recording_ids)
        )
        if len(recording) != n:
            raise ValueError("recording_ids must match y_true.")
        n_false = 0
        for recording_id in np.unique(recording[false_mask]):
            alarm_times = np.sort(
                starts[false_mask & (recording == recording_id)]
            )
            if alarm_times.size:
                n_false += 1 + int(
                    np.sum(np.diff(alarm_times) > refractory_minutes)
                )

    interictal_hours = float(duration[interictal].sum())
    far = n_false / interictal_hours if interictal_hours > 0 else float("nan")
    sensitivity = detected / len(event_values) if len(event_values) else float("nan")
    return AlarmMetrics(
        sensitivity=float(sensitivity),
        false_alarms_per_hour=float(far),
        mean_warning_time_minutes=_safe_float_mean(warnings_),
        brier_score=float(brier_score_loss(y, prob)),
        n_seizures=int(len(event_values)),
        n_detected_seizures=int(detected),
        n_false_alarms=n_false,
        interictal_hours=interictal_hours,
    )


def default_utility(
    metrics: AlarmMetrics,
    *,
    false_alarm_weight: float = 0.15,
    brier_weight: float = 0.25,
    warning_cap_minutes: float = 60.0,
    warning_weight: float = 0.05,
) -> float:
    """Scalarize the four outcomes for subset search.

    Sensitivity remains dominant. FAR has a log penalty so a few difficult
    recordings cannot overwhelm the search, Brier rewards calibration, and early
    correct warnings receive a small capped reward. Change the weights to match
    the clinical operating point, or pass a custom ``utility_fn``.
    """

    sensitivity = np.nan_to_num(metrics.sensitivity, nan=0.0)
    far = np.nan_to_num(metrics.false_alarms_per_hour, nan=1e6)
    warning = np.nan_to_num(metrics.mean_warning_time_minutes, nan=0.0)
    return float(
        sensitivity
        - false_alarm_weight * np.log1p(far)
        - brier_weight * metrics.brier_score
        + warning_weight * min(warning / warning_cap_minutes, 1.0)
    )


class PersonalizedSensorSelector:
    """Select a compact patient-specific EEG montage with nested CV."""

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        *,
        max_sensors: int = 31,
        inner_splits: int = 4,
        outer_splits: int = 5,
        threshold: float = 0.5,
        refractory_minutes: float = 30.0,
        utility_fn: Callable[[AlarmMetrics], float] = default_utility,
        elbow_tolerance: float = 0.02,
        swap_refinement: bool = True,
        random_state: int = 42,
    ) -> None:
        self.estimator = estimator if estimator is not None else default_estimator()
        self.max_sensors = max_sensors
        self.inner_splits = inner_splits
        self.outer_splits = outer_splits
        self.threshold = threshold
        self.refractory_minutes = refractory_minutes
        self.utility_fn = utility_fn
        self.elbow_tolerance = elbow_tolerance
        self.swap_refinement = swap_refinement
        self.random_state = random_state

    @staticmethod
    def _features(X: FloatArray, sensors: Sequence[int]) -> FloatArray:
        return X[:, sensors, :].reshape(len(X), -1)

    def _splitter(self, y: IntArray, groups: ArrayLike | None, n_splits: int):
        if groups is None:
            warnings.warn(
                "No groups supplied: correlated windows may leak across folds. "
                "Pass recording/session IDs for publishable estimates.",
                stacklevel=3,
            )
            splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=self.random_state
            )
            return list(splitter.split(np.zeros(len(y)), y))
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        return list(splitter.split(np.zeros(len(y)), y, np.asarray(groups)))

    def _cross_validated_predictions(
        self,
        X: FloatArray,
        y: IntArray,
        sensors: Sequence[int],
        splits: Sequence[tuple[IntArray, IntArray]],
    ) -> FloatArray:
        pred = np.full(len(y), np.nan, dtype=float)
        features = self._features(X, sensors)
        for train, valid in splits:
            if np.unique(y[train]).size < 2:
                raise ValueError("Every training fold must contain both classes.")
            model = clone(self.estimator)
            model.fit(features[train], y[train])
            pred[valid] = model.predict_proba(features[valid])[:, 1]
        if np.any(~np.isfinite(pred)):
            raise RuntimeError("Cross-validation did not predict every window.")
        return pred

    def _metric_kwargs(
        self,
        metadata: dict[str, ArrayLike],
        indices: ArrayLike | None = None,
    ) -> dict[str, ArrayLike | float | None]:
        idx = slice(None) if indices is None else np.asarray(indices)
        return {
            "duration_hours": np.asarray(metadata["duration_hours"])[idx],
            "seizure_ids": np.asarray(metadata["seizure_ids"])[idx],
            "time_to_seizure_minutes": np.asarray(
                metadata["time_to_seizure_minutes"]
            )[idx],
            "window_start_minutes": (
                None
                if metadata.get("window_start_minutes") is None
                else np.asarray(metadata["window_start_minutes"])[idx]
            ),
            "recording_ids": (
                None
                if metadata.get("recording_ids") is None
                else np.asarray(metadata["recording_ids"])[idx]
            ),
            "threshold": self.threshold,
            "refractory_minutes": self.refractory_minutes,
        }

    def _evaluate_subset(
        self,
        X: FloatArray,
        y: IntArray,
        sensors: Sequence[int],
        splits: Sequence[tuple[IntArray, IntArray]],
        metadata: dict[str, ArrayLike],
    ) -> tuple[float, float, AlarmMetrics]:
        pred = self._cross_validated_predictions(X, y, sensors, splits)
        fold_utilities: list[float] = []
        for _, valid in splits:
            m = compute_alarm_metrics(
                y[valid], pred[valid], **self._metric_kwargs(metadata, valid)
            )
            fold_utilities.append(self.utility_fn(m))
        pooled = compute_alarm_metrics(y, pred, **self._metric_kwargs(metadata))
        utility = float(np.mean(fold_utilities))
        se = (
            float(np.std(fold_utilities, ddof=1) / np.sqrt(len(fold_utilities)))
            if len(fold_utilities) > 1
            else 0.0
        )
        return utility, se, pooled

    def _search_curve(
        self,
        X: FloatArray,
        y: IntArray,
        groups: ArrayLike | None,
        metadata: dict[str, ArrayLike],
    ) -> list[CurvePoint]:
        splits = self._splitter(y, groups, self.inner_splits)
        p = X.shape[1]
        limit = min(self.max_sensors, p)
        chosen: list[int] = []
        remaining = set(range(p))
        curve: list[CurvePoint] = []
        cache: dict[tuple[int, ...], tuple[float, float, AlarmMetrics]] = {}

        def evaluate(subset: Iterable[int]):
            key = tuple(sorted(subset))
            if key not in cache:
                cache[key] = self._evaluate_subset(X, y, key, splits, metadata)
            return cache[key]

        for k in range(1, limit + 1):
            candidates = []
            for sensor in sorted(remaining):
                subset = tuple(sorted((*chosen, sensor)))
                candidates.append((evaluate(subset)[0], sensor, subset))
            _, added, best_subset = max(candidates, key=lambda x: (x[0], -x[1]))
            chosen = list(best_subset)
            remaining.remove(added)

            # One-for-one exchanges repair common mistakes made by pure forward
            # selection while retaining a tractable search.
            if self.swap_refinement and k > 1 and remaining:
                improved = True
                while improved:
                    improved = False
                    current_score = evaluate(chosen)[0]
                    swaps = []
                    for old in chosen:
                        for new in sorted(remaining):
                            subset = tuple(sorted((set(chosen) - {old}) | {new}))
                            swaps.append((evaluate(subset)[0], old, new, subset))
                    score, old, new, subset = max(
                        swaps, key=lambda x: (x[0], -x[1], -x[2])
                    )
                    if score > current_score + 1e-12:
                        chosen = list(subset)
                        remaining.remove(new)
                        remaining.add(old)
                        improved = True

            utility, se, metrics = evaluate(chosen)
            curve.append(
                CurvePoint(
                    k=k,
                    sensors=tuple(sorted(chosen)),
                    utility_mean=utility,
                    utility_se=se,
                    metrics=metrics,
                )
            )
        return curve

    def _choose_elbow(self, curve: Sequence[CurvePoint]) -> int:
        """Smallest k within tolerance of the best mean CV utility.

        This is a stable elbow/plateau rule: extra sensors are retained only when
        they improve utility by more than ``elbow_tolerance``. It is preferable to
        geometric knee rules when CV curves are noisy or non-monotonic.
        """

        utility = np.asarray([p.utility_mean for p in curve])
        best = float(np.nanmax(utility))
        eligible = np.flatnonzero(utility >= best - self.elbow_tolerance)
        return int(curve[int(eligible[0])].k)

    @staticmethod
    def _validate(
        X: ArrayLike,
        y: ArrayLike,
        metadata: dict[str, ArrayLike],
        sensor_names: Sequence[str] | None,
    ) -> tuple[FloatArray, IntArray, list[str]]:
        features = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=int)
        if features.ndim != 3:
            raise ValueError("X must have shape (windows, sensors, features).")
        if len(target) != len(features) or set(np.unique(target)) - {0, 1}:
            raise ValueError("y must be a binary vector aligned with X.")
        if np.any(~np.isfinite(features)):
            raise ValueError("X contains NaN or infinite values.")
        required = {
            "duration_hours",
            "seizure_ids",
            "time_to_seizure_minutes",
        }
        missing = required - metadata.keys()
        if missing:
            raise ValueError(f"metadata is missing: {sorted(missing)}")
        for key in required:
            if len(np.asarray(metadata[key])) != len(target):
                raise ValueError(f"metadata[{key!r}] is not aligned with X.")
        names = (
            [str(i) for i in range(features.shape[1])]
            if sensor_names is None
            else list(sensor_names)
        )
        if len(names) != features.shape[1]:
            raise ValueError("sensor_names length must equal X.shape[1].")
        return features, target, names

    def fit_select(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        metadata: dict[str, ArrayLike],
        groups: ArrayLike | None,
        sensor_names: Sequence[str] | None = None,
        plot_path: str | Path | None = None,
    ) -> SelectionResult:
        """Select sensors and estimate achieved performance with nested CV."""

        Xv, yv, names = self._validate(X, y, metadata, sensor_names)
        groups_array = None if groups is None else np.asarray(groups)
        if groups_array is not None and len(groups_array) != len(yv):
            raise ValueError("groups must be aligned with X.")
        working_metadata = dict(metadata)
        if groups_array is not None:
            working_metadata.setdefault("recording_ids", groups_array)

        outer = self._splitter(yv, groups_array, self.outer_splits)
        oof = np.full(len(yv), np.nan)
        fold_metrics: list[AlarmMetrics] = []
        fold_sensors: list[list[int]] = []

        for train, test in outer:
            train_metadata = {
                key: np.asarray(value)[train]
                for key, value in working_metadata.items()
                if value is not None
            }
            curve = self._search_curve(
                Xv[train],
                yv[train],
                None if groups_array is None else groups_array[train],
                train_metadata,
            )
            k = self._choose_elbow(curve)
            sensors = list(curve[k - 1].sensors)
            fold_sensors.append(sensors)
            model = clone(self.estimator)
            model.fit(self._features(Xv[train], sensors), yv[train])
            oof[test] = model.predict_proba(self._features(Xv[test], sensors))[:, 1]
            fold_metrics.append(
                compute_alarm_metrics(
                    yv[test],
                    oof[test],
                    **self._metric_kwargs(working_metadata, test),
                )
            )

        nested_metrics = compute_alarm_metrics(
            yv, oof, **self._metric_kwargs(working_metadata)
        )

        # Refit the selection on all data only after unbiased nested evaluation.
        final_curve = self._search_curve(
            Xv, yv, groups_array, working_metadata
        )
        elbow_k = self._choose_elbow(final_curve)
        selected = list(final_curve[elbow_k - 1].sensors)
        figure = self.plot_curve(final_curve, elbow_k)
        if plot_path is not None:
            path = Path(plot_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=180, bbox_inches="tight")

        self.selected_sensor_indices_ = selected
        self.selected_sensor_names_ = [names[i] for i in selected]
        self.elbow_k_ = elbow_k
        self.selection_curve_ = final_curve
        self.nested_cv_metrics_ = nested_metrics
        self.model_ = clone(self.estimator).fit(
            self._features(Xv, selected), yv
        )

        return SelectionResult(
            selected_sensor_indices=selected,
            selected_sensor_names=self.selected_sensor_names_,
            elbow_k=elbow_k,
            nested_cv_metrics=nested_metrics,
            nested_cv_fold_metrics=fold_metrics,
            selection_curve=final_curve,
            outer_fold_sensor_indices=fold_sensors,
            figure=figure,
        )

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        """Predict after ``fit_select`` using only the selected sensors."""

        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit_select before predict_proba.")
        features = np.asarray(X, dtype=float)
        return self.model_.predict_proba(
            self._features(features, self.selected_sensor_indices_)
        )[:, 1]

    def plot_curve(
        self, curve: Sequence[CurvePoint], elbow_k: int
    ) -> plt.Figure:
        """Plot utility and each clinical metric against sensor count."""

        k = np.asarray([point.k for point in curve])
        utility = np.asarray([point.utility_mean for point in curve])
        utility_se = np.asarray([point.utility_se for point in curve])
        sensitivity = np.asarray([point.metrics.sensitivity for point in curve])
        far = np.asarray(
            [point.metrics.false_alarms_per_hour for point in curve]
        )
        warning = np.asarray(
            [point.metrics.mean_warning_time_minutes for point in curve]
        )
        brier = np.asarray([point.metrics.brier_score for point in curve])

        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
        panels = [
            (axes[0, 0], utility, "Selection utility", True),
            (axes[0, 1], sensitivity, "Sensitivity", True),
            (axes[0, 2], far, "False alarms / hour", False),
            (axes[1, 0], warning, "Mean warning time (min)", True),
            (axes[1, 1], brier, "Brier score", False),
        ]
        for ax, values, label, higher_is_better in panels:
            ax.plot(k, values, marker="o", linewidth=1.6)
            ax.axvline(elbow_k, color="tab:red", linestyle="--", alpha=0.8)
            ax.set_ylabel(label)
            ax.grid(alpha=0.25)
            direction = "↑" if higher_is_better else "↓"
            ax.set_title(f"{label} ({direction})")
        axes[0, 0].fill_between(
            k, utility - utility_se, utility + utility_se, alpha=0.2
        )
        axes[1, 0].set_xlabel("Number of sensors (k)")
        axes[1, 1].set_xlabel("Number of sensors (k)")
        axes[1, 2].axis("off")
        axes[1, 2].text(
            0.05,
            0.75,
            f"Selected elbow: k = {elbow_k}\n"
            "Dashed line = smallest montage\n"
            "within utility tolerance of best.",
            fontsize=11,
            va="top",
        )
        fig.suptitle("Personalized EEG sensor-selection curve")
        fig.tight_layout()
        return fig


def compare_personalized_with_global(
    personalized_results: dict[str, SelectionResult],
    global_result: SelectionResult,
) -> dict[str, float]:
    """Summarize the research comparison without pooling patient windows.

    Returns macro-averaged personalized metrics next to the global montage's
    nested-CV metrics. The global selector must itself have been evaluated with
    patient/recording groups held out.
    """

    if not personalized_results:
        raise ValueError("At least one personalized result is required.")
    metrics = [r.nested_cv_metrics for r in personalized_results.values()]
    global_m = global_result.nested_cv_metrics
    return {
        "personalized_mean_k": float(
            np.mean([r.elbow_k for r in personalized_results.values()])
        ),
        "personalized_sensitivity": _safe_float_mean(
            m.sensitivity for m in metrics
        ),
        "personalized_false_alarms_per_hour": _safe_float_mean(
            m.false_alarms_per_hour for m in metrics
        ),
        "personalized_mean_warning_time_minutes": _safe_float_mean(
            m.mean_warning_time_minutes for m in metrics
        ),
        "personalized_brier_score": _safe_float_mean(
            m.brier_score for m in metrics
        ),
        "global_k": float(global_result.elbow_k),
        "global_sensitivity": global_m.sensitivity,
        "global_false_alarms_per_hour": global_m.false_alarms_per_hour,
        "global_mean_warning_time_minutes": global_m.mean_warning_time_minutes,
        "global_brier_score": global_m.brier_score,
    }
