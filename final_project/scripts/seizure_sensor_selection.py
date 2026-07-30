"""Step 1: estimate one cohort-level EEG sensor count, K.

This module deliberately does not select a deployment montage.  It answers only:

    What is the smallest number of sensors whose held-out, subject-level
    performance is non-inferior to using every available sensor?

Sensor identities are nuisance variables in this step. Each sensor gets an
independent probability model, and greedy forward search combines held-out
probabilities inside each outer training fold. This avoids repeatedly fitting
high-dimensional models for every candidate combination. No selected sensor
names or indices are returned or retained.

The search costs O(p**2) candidate evaluations for p sensors instead of the
O(2**p) evaluations required by exhaustive subset enumeration. For the 29
cohort-common sensors this means 435 candidate ensembles per outer fold, but
only O(p) model fits per fold.

This is exploratory research software, not a clinical seizure detector.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray
from scipy.stats import t
from sklearn.base import BaseEstimator, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedGroupKFold,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_NONINFERIORITY_MARGIN = 0.02


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]
IndexSplit = tuple[NDArray[np.int_], NDArray[np.int_]]
ProbabilityScorer = Callable[[IntArray, FloatArray], float]


@dataclass(frozen=True)
class CountCurvePoint:
    """Count-level held-out performance; contains no sensor identities."""

    k: int
    mean_score: float
    score_ci_lower: float
    score_ci_upper: float
    mean_difference_from_full: float
    noninferiority_lower_bound: float
    on_full_sensor_plateau: bool


def default_estimator() -> BaseEstimator:
    """Return a small, non-neural baseline model with probabilities."""

    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            solver="liblinear",
            random_state=0,
        ),
    )


class CohortSensorCountSelector:
    """Estimate K across the whole cohort without returning sensor identities.

    Parameters
    ----------
    estimator:
        Any scikit-learn binary estimator compatible with ``scoring``.
    scoring:
        ``"average_precision"``, ``"roc_auc"``, ``"neg_brier_score"``, or a
        callable with signature ``scorer(y_true, probability)``. Higher values
        must be better. Average precision is the default because seizure targets
        are usually imbalanced.
    noninferiority_margin:
        Largest acceptable score loss relative to all sensors. For example,
        0.02 permits an absolute average-precision loss of at most two points.
        This value should be chosen before inspecting results.
    confidence:
        Confidence level for the one-sided paired non-inferiority bound.
    inner_splits:
        Subject-grouped folds used only to search sensor combinations inside
        each outer training set.
    outer_splits:
        ``None`` performs leave-one-subject-out assessment (14 folds for 14
        subjects). An integer uses that many stratified subject-grouped folds.
    n_jobs:
        Number of per-sensor models evaluated concurrently.
    random_state:
        Seed for grouped split reproducibility.
    """

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        *,
        scoring: str | ProbabilityScorer = "average_precision",
        noninferiority_margin: float = DEFAULT_NONINFERIORITY_MARGIN,
        confidence: float = 0.95,
        inner_splits: int = 4,
        outer_splits: int | None = None,
        n_jobs: int = 1,
        random_state: int = 42,
    ) -> None:
        self.estimator = default_estimator() if estimator is None else estimator
        self.scoring = scoring
        self.noninferiority_margin = noninferiority_margin
        self.confidence = confidence
        self.inner_splits = inner_splits
        self.outer_splits = outer_splits
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _scorer(self) -> ProbabilityScorer:
        if self.scoring == "average_precision":
            return lambda truth, probability: float(
                average_precision_score(truth, probability)
            )
        if self.scoring == "roc_auc":
            return lambda truth, probability: float(
                roc_auc_score(truth, probability)
            )
        if self.scoring == "neg_brier_score":
            return lambda truth, probability: -float(
                brier_score_loss(truth, probability)
            )
        if not callable(self.scoring):
            raise ValueError(
                "scoring must be average_precision, roc_auc, "
                "neg_brier_score, or a callable."
            )
        return self.scoring

    @staticmethod
    def _subject_macro_score(
        y: IntArray,
        probabilities: FloatArray,
        subjects: NDArray,
        scorer: ProbabilityScorer,
    ) -> float:
        """Score every subject separately, then give subjects equal weight."""

        scores = []
        for subject in np.unique(subjects):
            mask = subjects == subject
            scores.append(float(scorer(y[mask], probabilities[mask])))
        values = np.asarray(scores, dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError(
                "scoring produced a non-finite value for at least one subject."
            )
        return float(np.mean(values))

    def _inner_splits(
        self,
        y: IntArray,
        subjects: NDArray,
    ) -> list[IndexSplit]:
        n_subjects = np.unique(subjects).size
        n_splits = min(self.inner_splits, n_subjects)
        if n_splits < 2:
            raise ValueError("Each outer training set needs at least two subjects.")
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        return list(splitter.split(np.zeros(len(y)), y, subjects))

    def _outer_splits(
        self,
        y: IntArray,
        subjects: NDArray,
    ) -> list[IndexSplit]:
        n_subjects = np.unique(subjects).size
        if self.outer_splits is None:
            return list(
                LeaveOneGroupOut().split(np.zeros(len(y)), y, subjects)
            )
        n_splits = min(self.outer_splits, n_subjects)
        if n_splits < 2:
            raise ValueError("outer_splits must create at least two folds.")
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state + 1,
        )
        return list(splitter.split(np.zeros(len(y)), y, subjects))

    def _sensor_oof_probabilities(
        self,
        X: FloatArray,
        y: IntArray,
        subjects: NDArray,
        splits: Sequence[IndexSplit],
    ) -> FloatArray:
        """Fit each sensor once per fold and return its held-out probabilities."""

        def predict_sensor(sensor: int) -> FloatArray:
            probability = np.full(len(y), np.nan, dtype=float)
            for train, valid in splits:
                if np.unique(y[train]).size != 2:
                    raise ValueError(
                        "Every inner training fold must contain both classes."
                    )
                model = clone(self.estimator).fit(X[train, sensor, :], y[train])
                probability[valid] = model.predict_proba(
                    X[valid, sensor, :]
                )[:, 1]
            return probability

        columns = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(predict_sensor)(sensor) for sensor in range(X.shape[1])
        )
        probabilities = np.column_stack(columns)
        if np.any(~np.isfinite(probabilities)):
            raise RuntimeError("Inner cross-validation did not predict every row.")
        return probabilities

    def _forward_path(
        self,
        y: IntArray,
        sensor_probabilities: FloatArray,
        subjects: NDArray,
        scorer: ProbabilityScorer,
    ) -> list[tuple[int, ...]]:
        """Greedily combine held-out sensor probabilities without refitting."""

        p = sensor_probabilities.shape[1]
        chosen: tuple[int, ...] = ()
        remaining = set(range(p))
        path: list[tuple[int, ...]] = []
        probability_sum = np.zeros(len(y), dtype=float)

        while remaining:
            additions = sorted(remaining)
            k = len(chosen) + 1
            scores = [
                self._subject_macro_score(
                    y,
                    (probability_sum + sensor_probabilities[:, sensor]) / k,
                    subjects,
                    scorer,
                )
                for sensor in additions
            ]
            # Deterministic ties make repeated runs exactly reproducible.
            best_position = max(
                range(len(additions)),
                key=lambda i: (scores[i], -additions[i]),
            )
            added = additions[best_position]
            chosen = tuple((*chosen, added))
            probability_sum += sensor_probabilities[:, added]
            remaining.remove(added)
            path.append(chosen)
        return path

    def _outer_sensor_probabilities(
        self,
        X_train: FloatArray,
        y_train: IntArray,
        X_test: FloatArray,
    ) -> FloatArray:
        """Fit one model per sensor and predict the untouched outer fold."""

        def predict_sensor(sensor: int) -> FloatArray:
            model = clone(self.estimator).fit(
                X_train[:, sensor, :],
                y_train,
            )
            return model.predict_proba(X_test[:, sensor, :])[:, 1]

        columns = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(predict_sensor)(sensor) for sensor in range(X_train.shape[1])
        )
        return np.column_stack(columns)

    @staticmethod
    def _validate(
        X: ArrayLike,
        y: ArrayLike,
        subjects: ArrayLike,
    ) -> tuple[FloatArray, IntArray, NDArray]:
        features = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=int)
        subject_ids = np.asarray(subjects)

        if features.ndim != 3:
            raise ValueError("X must have shape (windows, sensors, features).")
        if features.shape[1] < 1 or features.shape[2] < 1:
            raise ValueError("X must contain at least one sensor and one feature.")
        if np.any(np.isinf(features)):
            raise ValueError("X contains infinite values.")
        if target.ndim != 1 or len(target) != len(features):
            raise ValueError("y must be a vector aligned with X.")
        if set(np.unique(target)) != {0, 1}:
            raise ValueError("y must contain both binary classes 0 and 1.")
        if subject_ids.ndim != 1 or len(subject_ids) != len(features):
            raise ValueError("subjects must be a vector aligned with X.")
        unique_subjects = np.unique(subject_ids)
        if unique_subjects.size < 3:
            raise ValueError("At least three subjects are required.")
        for subject in unique_subjects:
            if np.unique(target[subject_ids == subject]).size != 2:
                raise ValueError(
                    "Every subject must contain both classes for subject-level "
                    "performance comparison."
                )
        return features, target, subject_ids

    def _validate_settings(self) -> None:
        if self.noninferiority_margin < 0:
            raise ValueError("noninferiority_margin cannot be negative.")
        if not 0.5 < self.confidence < 1:
            raise ValueError("confidence must lie in (0.5, 1).")
        if not isinstance(self.inner_splits, Integral) or self.inner_splits < 2:
            raise ValueError("inner_splits must be an integer of at least 2.")
        if self.outer_splits is not None and (
            not isinstance(self.outer_splits, Integral)
            or self.outer_splits < 2
        ):
            raise ValueError("outer_splits must be None or an integer of at least 2.")
        if not isinstance(self.n_jobs, Integral) or self.n_jobs == 0:
            raise ValueError("n_jobs must be a nonzero integer.")

    def _smallest_noninferior_k(self, subject_scores: FloatArray) -> int:
        """Find the contiguous plateau connected to the all-sensor baseline.

        Tests proceed from p-1 down to 1 and stop at the first failure. This
        fixed-sequence rule prevents an isolated noisy point farther left on the
        curve from being mistaken for the full-sensor plateau.
        """

        n_subjects, p = subject_scores.shape
        baseline = subject_scores[:, -1]
        selected_k = p
        critical_value = float(t.ppf(self.confidence, n_subjects - 1))

        for k in range(p - 1, 0, -1):
            differences = subject_scores[:, k - 1] - baseline
            mean_difference = float(np.mean(differences))
            standard_error = float(np.std(differences, ddof=1) / np.sqrt(n_subjects))
            lower_bound = mean_difference - critical_value * standard_error
            if lower_bound >= -self.noninferiority_margin:
                selected_k = k
            else:
                break
        return selected_k

    def _count_curve(
        self,
        subject_scores: FloatArray,
        selected_k: int,
    ) -> list[CountCurvePoint]:
        """Aggregate held-out subject scores by count for plotting."""

        n_subjects, p = subject_scores.shape
        baseline = subject_scores[:, -1]
        score_critical = float(
            t.ppf((1 + self.confidence) / 2, n_subjects - 1)
        )
        difference_critical = float(t.ppf(self.confidence, n_subjects - 1))
        curve = []
        for column in range(p):
            values = subject_scores[:, column]
            score_se = float(np.std(values, ddof=1) / np.sqrt(n_subjects))
            differences = values - baseline
            difference_se = float(
                np.std(differences, ddof=1) / np.sqrt(n_subjects)
            )
            mean_score = float(np.mean(values))
            mean_difference = float(np.mean(differences))
            curve.append(
                CountCurvePoint(
                    k=column + 1,
                    mean_score=mean_score,
                    score_ci_lower=mean_score - score_critical * score_se,
                    score_ci_upper=mean_score + score_critical * score_se,
                    mean_difference_from_full=mean_difference,
                    noninferiority_lower_bound=(
                        mean_difference
                        - difference_critical * difference_se
                    ),
                    on_full_sensor_plateau=column + 1 >= selected_k,
                )
            )
        return curve

    def select_k(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        subjects: ArrayLike,
    ) -> int:
        """Return only the cohort-level sensor count K.

        ``X`` must have shape ``(windows, sensors, features_per_sensor)`` and all
        subjects must share the same ordered sensor axis. The outer assessment
        holds out complete subjects. Inside each outer training set, forward
        search builds candidate combinations for k=1..p. Held-out subjects are
        never used to choose those combinations.
        """

        self._validate_settings()
        Xv, yv, subject_ids = self._validate(X, y, subjects)
        scorer = self._scorer()
        unique_subjects = np.unique(subject_ids)
        subject_position = {
            subject: position
            for position, subject in enumerate(unique_subjects)
        }
        p = Xv.shape[1]
        scores = np.full((len(unique_subjects), p), np.nan, dtype=float)

        for train, test in self._outer_splits(yv, subject_ids):
            inner_splits = self._inner_splits(yv[train], subject_ids[train])
            inner_probabilities = self._sensor_oof_probabilities(
                Xv[train],
                yv[train],
                subject_ids[train],
                inner_splits,
            )
            path = self._forward_path(
                yv[train],
                inner_probabilities,
                subject_ids[train],
                scorer,
            )
            outer_probabilities = self._outer_sensor_probabilities(
                Xv[train],
                yv[train],
                Xv[test],
            )
            for k, sensors in enumerate(path, start=1):
                ensemble_probability = np.mean(
                    outer_probabilities[:, sensors],
                    axis=1,
                )
                for subject in np.unique(subject_ids[test]):
                    subject_mask = subject_ids[test] == subject
                    scores[subject_position[subject], k - 1] = float(
                        scorer(
                            yv[test][subject_mask],
                            ensemble_probability[subject_mask],
                        )
                    )

        if np.any(~np.isfinite(scores)):
            raise RuntimeError("Outer cross-validation did not score every subject.")
        k = int(self._smallest_noninferior_k(scores))

        # Retain count-level diagnostics only. Sensor identities belong to later
        # personalized/generalized selection steps.  Per-subject scores support
        # a stricter, worst-subject channel-count audit without retaining names.
        self.k_ = k
        self.subject_scores_ = scores.copy()
        self.count_curve_ = self._count_curve(scores, k)
        return k

    def smallest_k_with_max_loss(
        self, max_loss: float = DEFAULT_NONINFERIORITY_MARGIN
    ) -> int:
        """Return the smallest count whose observed loss is bounded for every subject.

        Call :meth:`select_k` first.  Unlike the default non-inferiority rule,
        this is a descriptive worst-subject criterion: every held-out subject's
        score at ``k`` must be no more than ``max_loss`` below its score using
        all sensors.  It is not a population-level statistical guarantee.
        """

        if max_loss < 0:
            raise ValueError("max_loss cannot be negative.")
        if not hasattr(self, "subject_scores_"):
            raise RuntimeError("Call select_k before auditing the maximum loss.")
        full_scores = self.subject_scores_[:, -1]
        losses = full_scores[:, None] - self.subject_scores_
        acceptable = np.max(losses, axis=0) <= max_loss
        candidates = np.flatnonzero(acceptable)
        if candidates.size == 0:
            raise RuntimeError("The full-sensor model must satisfy max_loss=0.")
        self.max_loss_curve_ = np.max(losses, axis=0)
        return int(candidates[0] + 1)


def select_sensor_count(
    X: ArrayLike,
    y: ArrayLike,
    *,
    subjects: ArrayLike,
    estimator: BaseEstimator | None = None,
    scoring: str | ProbabilityScorer = "average_precision",
    noninferiority_margin: float = DEFAULT_NONINFERIORITY_MARGIN,
    confidence: float = 0.95,
    inner_splits: int = 4,
    outer_splits: int | None = None,
    n_jobs: int = 1,
    random_state: int = 42,
) -> int:
    """Convenience function that returns only K."""

    selector = CohortSensorCountSelector(
        estimator=estimator,
        scoring=scoring,
        noninferiority_margin=noninferiority_margin,
        confidence=confidence,
        inner_splits=inner_splits,
        outer_splits=outer_splits,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    return selector.select_k(X, y, subjects=subjects)
