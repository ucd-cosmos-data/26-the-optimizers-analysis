# Step 1: choose the number of sensors

`seizure_sensor_selection.py` performs only the first step of the project:
estimate a single cohort-level sensor count, `K`. It does not produce a
personalized model, a generalized model, or a sensor combination.

## Definition of K

`K` is the smallest count on the performance plateau connected to the
all-sensor result. A count belongs to that plateau when its held-out,
subject-level score is statistically non-inferior to the score from all 29
sensors by the pre-specified margin.

Accuracy versus sensor count should normally be interpreted as a rising curve
that plateaus, not as a symmetric bell curve. Extra noisy features can sometimes
reduce held-out accuracy, but the algorithm does not assume that they will.

## How the model works

The model starts with the 29 EEG channels shared by all 14 subjects. At each
five-second landmark, every channel is represented independently by 24 compact
spectral and signal-summary features from the preceding two minutes. The target
is whether a seizure occurs within the next five minutes.

The current cohort contains four interictal control episodes per seizure:
47 preictal episodes and 188 controls, producing 14,100 landmark rows. A
class-balanced logistic-regression model is fitted separately for each sensor.
Candidate K-sensor models average the individual sensor probabilities; no
neural network is used.

All combination search happens inside the training subjects. Performance is
then measured on a completely held-out subject using average precision, with
each subject contributing equal weight. The algorithm returns only the smallest
count on the statistically non-inferior plateau and discards the temporary
sensor identities.

The control ratio and non-inferiority margin are part of the model definition
and must be fixed before analysis. Average precision depends on class
prevalence, so changing from one to four controls per seizure changes the
performance curve and can change K even when the EEG signals are unchanged.

## Algorithm

For each outer subject-grouped fold:

1. Hold out complete subjects.
2. Fit one probability model per sensor on the remaining subjects only.
3. Build one candidate probability ensemble for every count from 1 through 29
   using greedy forward search.
4. Evaluate every count on the held-out subjects.
5. Give every subject equal weight, regardless of its number of windows.

It then compares each count with the 29-sensor score using paired, one-sided
non-inferiority confidence bounds. Starting at 28, it walks downward and stops
at the first count that is meaningfully worse. This yields the left edge of the
full-sensor plateau.

The forward search examines at most `29 + 28 + ... + 1 = 435` ensembles per
outer fold, rather than all `2^29` subsets. It fits only one model per sensor per
fold; combination evaluation averages held-out probabilities and requires no
additional model fitting. Sensor identities are temporary nuisance variables
inside training folds and are neither returned nor retained.

## Input

The integrated pipeline reads channel-separated arrays from
`final_project/data/raw/splitdata`, not directly from EDF during feature
extraction. Build that mirrored dataset once:

```bash
python final_project/scripts/split_eeg_channels.py
```

Each recording directory contains one lossless `.npy` file per available EEG
channel, `channel_manifest.csv`, and `channel_overview.png`. The raw recordings
have 29 or 31 EEG channels; the cohort intersection used below remains 29.

```python
X.shape == (n_windows, 29, n_features_per_sensor)
y.shape == (n_windows,)          # binary target
subjects.shape == (n_windows,)   # one of the 14 subject IDs
```

All subjects must share the same ordered sensor axis. Features must remain
separated by sensor. Each subject must contain both target classes so that the
subject-level score is defined.

## Run

```python
from seizure_sensor_selection import select_sensor_count

K = select_sensor_count(
    X_all_subjects,
    y_all_subjects,
    subjects=subject_id_per_window,
    scoring="average_precision",
    noninferiority_margin=0.02,
    confidence=0.95,
    inner_splits=4,
    outer_splits=None,  # leave one subject out: 14 folds for 14 subjects
    n_jobs=-1,
    random_state=42,
)

print(K)
```

For the downloaded Siena data, the end-to-end reproducible command is:

```bash
python final_project/scripts/split_eeg_channels.py
python final_project/scripts/run_sensor_count_step1.py
```

This prints one integer and nothing about which sensors were used. The later
personalized and generalized steps should accept this integer as their fixed
target count.

Choose `noninferiority_margin` before running the analysis. A value of `0.02`
means that a reduced count may lose no more than two absolute
average-precision points relative to all sensors. A clinically meaningful
margin should replace this example value when one is available.
