import matplotlib
import numpy as np

matplotlib.use("Agg")

from seizure_sensor_selection import (  # noqa: E402
    PersonalizedSensorSelector,
    compute_alarm_metrics,
)


def test_alarm_metrics_counts_events_and_debounces_false_alarms():
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.8, 0.9, 0.8, 0.1, 0.8, 0.2, 0.9])
    metrics = compute_alarm_metrics(
        y,
        p,
        duration_hours=np.repeat(0.5, len(y)),
        seizure_ids=np.array([-1, -1, -1, 10, 10, 11, 11]),
        time_to_seizure_minutes=np.array(
            [np.nan, np.nan, np.nan, 30, 20, 25, 5]
        ),
        window_start_minutes=np.array([0, 5, 50, 100, 110, 200, 210]),
        refractory_minutes=30,
    )
    assert metrics.sensitivity == 1.0
    assert metrics.n_false_alarms == 2
    assert metrics.false_alarms_per_hour == 2 / 1.5
    assert metrics.mean_warning_time_minutes == 12.5


def test_end_to_end_returns_indices_metrics_and_plot(tmp_path):
    rng = np.random.default_rng(4)
    n_groups, windows_per_group, p, f = 12, 10, 5, 2
    n = n_groups * windows_per_group
    groups = np.repeat(np.arange(n_groups), windows_per_group)
    y = np.tile(np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), n_groups)
    X = rng.normal(size=(n, p, f))
    X[:, 2, 0] += 2.0 * y
    seizure_ids = np.where(y == 1, groups, -1)
    metadata = {
        "duration_hours": np.repeat(1 / 60, n),
        "seizure_ids": seizure_ids,
        "time_to_seizure_minutes": np.where(
            y == 1, np.tile([np.nan] * 5 + [25, 20, 15, 10, 5], n_groups), np.nan
        ),
        "window_start_minutes": np.arange(n, dtype=float),
    }
    path = tmp_path / "curve.png"
    selector = PersonalizedSensorSelector(
        max_sensors=3,
        inner_splits=3,
        outer_splits=3,
        swap_refinement=False,
        random_state=1,
    )
    result = selector.fit_select(
        X,
        y,
        metadata=metadata,
        groups=groups,
        sensor_names=[f"EEG-{i}" for i in range(p)],
        plot_path=path,
    )
    assert 2 in result.selected_sensor_indices
    assert 1 <= result.elbow_k <= 3
    assert 0 <= result.nested_cv_metrics.sensitivity <= 1
    assert result.figure is not None
    assert path.exists()
    assert selector.predict_proba(X[:3]).shape == (3,)
