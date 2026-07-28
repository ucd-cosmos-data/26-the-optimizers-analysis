# Personalized EEG sensor selection

`seizure_sensor_selection.py` selects a compact montage without enumerating all
`2^31` sensor subsets. It uses greedy forward additions plus one-for-one swap
refinement, then chooses the smallest `k` on the cross-validated utility plateau.

The reported performance is from **nested** grouped cross-validation:

- The inner folds select the subset and elbow `k`.
- The outer folds measure sensitivity, false alarms/hour, mean warning time, and
  Brier score on data that played no role in selection.
- After evaluation, the selector runs once on all available data to return the
  final deployment montage.

## Expected inputs

```python
X.shape == (n_windows, 31, n_features_per_channel)
y.shape == (n_windows,)  # 1=preictal, 0=interictal
```

Features must remain separated by channel. For example, if you compute bandpower,
entropy, and line length per channel, the final axis contains those features.

`groups` should identify independent recordings or sessions. If a recording
contains one seizure, its seizure ID is also suitable. Never randomly split
overlapping windows.

## Integration

```python
from seizure_sensor_selection import PersonalizedSensorSelector

metadata = {
    # Use the window stride, not window width, for overlapping windows.
    "duration_hours": np.full(len(y_patient), window_stride_seconds / 3600),
    # Unique event ID on preictal windows; -1 on interictal windows.
    "seizure_ids": seizure_id_per_window,
    # Minutes until onset on preictal windows; NaN on interictal windows.
    "time_to_seizure_minutes": minutes_to_onset,
    # Optional but recommended: time within each recording.
    "window_start_minutes": window_start_minutes,
}

selector = PersonalizedSensorSelector(
    estimator=your_probability_model,  # must implement fit + predict_proba
    max_sensors=31,
    inner_splits=4,
    outer_splits=5,
    threshold=0.5,
    refractory_minutes=30,
    elbow_tolerance=0.02,
)

result = selector.fit_select(
    X_patient,
    y_patient,
    metadata=metadata,
    groups=recording_id_per_window,
    sensor_names=channel_names,
    plot_path="patient_07_sensor_curve.png",
)

print(result.selected_sensor_indices)
print(result.selected_sensor_names)
print(result.summary())
result.figure.show()
```

To personalize, run this independently for each patient. For the one-size-fits-all
comparison, concatenate patients and run a global selector while grouping by a
compound patient/recording ID. Report macro-averaged patient metrics, not a
window-weighted average. `compare_personalized_with_global` produces a compact
summary after both evaluations.

## Metric definitions

- **Sensitivity:** fraction of seizure events with at least one alarm in their
  preictal window.
- **False alarms/hour:** distinct interictal alarm episodes divided by interictal
  monitoring hours. Positives within the refractory interval count once.
- **Mean warning time:** mean time from the earliest correct alarm to onset,
  across detected seizures.
- **Brier score:** mean squared probability error over all held-out windows.

The default scalar utility keeps sensitivity dominant while penalizing false
alarms and Brier score and lightly rewarding earlier warnings. Its weights are
explicit in `default_utility`; pre-register them or replace `utility_fn` with a
clinically chosen objective before analyzing test results.
