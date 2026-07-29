import numpy as np
import pytest

from seizure_sensor_selection import (
    CohortSensorCountSelector,
    select_sensor_count,
)


def _cohort(seed: int = 7):
    """Synthetic cohort where one sensor contains nearly all useful signal."""

    rng = np.random.default_rng(seed)
    n_subjects, windows_per_subject, p = 6, 40, 5
    subjects = np.repeat(np.arange(n_subjects), windows_per_subject)
    y = np.tile(np.repeat([0, 1], windows_per_subject // 2), n_subjects)
    X = rng.normal(scale=0.25, size=(len(y), p, 1))
    X[:, 3, 0] += 3.0 * y
    return X, y, subjects


def test_returns_only_cohort_level_integer_k():
    X, y, subjects = _cohort()
    selector = CohortSensorCountSelector(
        noninferiority_margin=0.02,
        confidence=0.90,
        inner_splits=3,
        outer_splits=3,
        random_state=2,
    )
    result = selector.select_k(X, y, subjects=subjects)

    assert type(result) is int
    assert result == 1
    assert selector.k_ == 1
    assert len(selector.count_curve_) == X.shape[1]
    assert selector.count_curve_[-1].mean_difference_from_full == 0
    assert not hasattr(selector, "selected_sensor_indices_")
    assert not hasattr(selector, "selected_sensor_names_")


def test_convenience_function_returns_k():
    X, y, subjects = _cohort()
    X[::17, 0, 0] = np.nan
    k = select_sensor_count(
        X,
        y,
        subjects=subjects,
        confidence=0.90,
        inner_splits=3,
        outer_splits=3,
    )
    assert type(k) is int
    assert 1 <= k <= X.shape[1]


def test_requires_subject_level_nonpersonalized_input():
    X, y, subjects = _cohort()
    subjects[subjects == 5] = 4
    y[subjects == 4] = 1
    selector = CohortSensorCountSelector(outer_splits=3)
    with pytest.raises(ValueError, match="Every subject"):
        selector.select_k(X, y, subjects=subjects)


def test_plateau_rule_does_not_jump_over_a_failed_count():
    selector = CohortSensorCountSelector(
        noninferiority_margin=0.02,
        confidence=0.95,
    )
    # k=4 and k=3 match the five-sensor baseline, k=2 fails. Even though k=1
    # happens to score well, it is disconnected from the baseline plateau.
    scores = np.array(
        [
            [0.91, 0.70, 0.89, 0.90, 0.90],
            [0.89, 0.71, 0.90, 0.91, 0.90],
            [0.90, 0.69, 0.90, 0.90, 0.90],
            [0.91, 0.70, 0.91, 0.90, 0.90],
        ]
    )
    assert selector._smallest_noninferior_k(scores) == 3
