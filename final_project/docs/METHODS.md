# Exact methods

This document specifies the final reduced-sensor experiment. It describes the
code in `src/` and supersedes every legacy notebook.

## 1. Study data

- Dataset: Siena Scalp EEG Database 1.0.0
- Patients: 14
- EDF recordings: 41
- Annotated seizures: 47
- Source sample rate: 512 Hz
- Common EEG montage: 29 channels
- P/G cohort: PN00, PN06, PN10, PN12, PN14

The episode manifest contains one five-minute preictal episode and four
candidate interictal control episodes per seizure: 47 positive and 188
negative episodes.

## 2. Prediction target

Landmarks are spaced five seconds apart. For a landmark at time `t`, the model
uses only EEG from `[t - 120 seconds, t]`.

The binary target is:

```text
y(t) = 1 when annotated seizure onset is within (t, t + 300 seconds]
y(t) = 0 for sampled interictal controls
```

Each five-minute episode produces 60 landmarks. This is sampled-episode
classification, not continuous alarm evaluation.

## 3. Signal quality

Each channel is linearly detrended within every five-second micro-window. A
window is unusable when it contains non-finite values or satisfies any of:

```text
MAD < 0.05 µV
MAD > 250 µV
max absolute centered amplitude > 1,500 µV
```

Physiological features from an unusable window become missing. The separate
usability value remains zero. Missing values are imputed with medians learned
only from each model's fitting data.

## 4. Channel features

Signals are resampled to 64 Hz. Periodogram power is computed within:

```text
delta  0.5–4 Hz
theta    4–8 Hz
alpha   8–13 Hz
beta   13–30 Hz
```

Relative band power uses total analyzed power from 0.5 Hz through the 32 Hz
Nyquist frequency as its denominator. A legacy 30–45 Hz “low gamma” value is
invalid after 64 Hz resampling and is not used.

Seven micro-features are retained per channel:

1. relative delta power;
2. relative theta power;
3. relative alpha power;
4. relative beta power;
5. log RMS;
6. log line length; and
7. usable-window fraction.

For the 24 five-second micro-windows in the preceding 120 seconds, the mean,
latest value, and linear slope are calculated. The final width is therefore
`7 × 3 = 21` features per channel.

No excluded channel contributes to another channel's features. In particular,
the feature builder does not apply an across-channel common reference.

## 5. Leakage cleaning

An episode represents raw time from 120 seconds before its first landmark
through its last landmark. Before fitting:

1. a control overlapping positive preictal raw time is removed;
2. overlapping controls are joined into one component and assigned to one
   event group;
3. identical `(recording, landmark_seconds)` rows are deduplicated;
4. raw intervals are asserted not to cross event groups; and
5. hashes of the complete compact feature vector are asserted not to cross
   event groups.

PN00 required 15 controls to be reassigned and 564 repeated raw landmark rows
to be removed. No positive-overlap control was found.

## 6. Shared base model

K-finder, P, and G use the same sensor-level model:

```text
SimpleImputer(strategy="median")
StandardScaler()
LogisticRegression(
    C=0.1,
    class_weight="balanced",
    penalty="l2",
    solver="liblinear",
    random_state=42
)
```

One model is fitted for each sensor. For a selected set `S`, raw ensemble risk
is the unweighted mean:

```text
p_S(x) = mean over sensor s in S of p_s(x)
```

For P/G evaluation, a one-dimensional Platt transformation with a constrained
positive slope is fitted to training-only out-of-fold ensemble predictions.
The positive slope preserves ranking. K-finder uses AUPRC, so applying this
monotonic calibration would not change its sensor-count result.

The final `C=0.1` setting was chosen after a provisional comparison with
`C=1.0` and `C=0.01`: it reduced extreme overconfidence without the stronger
prediction collapse seen at `C=0.01`. That sensitivity check used this project
data rather than a separate nested tuning loop. Final held-out metrics are
therefore exploratory and require confirmation on an untouched cohort.

## 7. K-finder

K-finder uses nested patient-level validation across all 14 patients.

For each outer leave-one-patient-out fold:

1. The remaining 13 patients form the development set.
2. Four-fold stratified group validation keeps every inner patient intact.
3. Out-of-fold predictions are produced independently for all 29 sensors.
4. Greedy forward selection adds the sensor that maximizes macro-patient
   AUPRC of the current probability average.
5. Sensor models are refitted to the 13 outer-training patients.
6. Every prefix from K=1 through K=29 is scored on the unseen outer patient.

Let `AP_i(k)` be held-out AUPRC for patient `i` at sensor count `k`. The
reported rule is:

```text
loss_i(k) = AP_i(29) - AP_i(k)
K = min { k : max_i loss_i(k) ≤ 0.03 }
```

This gives K=16 on the final run. K=15 has worst-patient loss 0.0427; K=16 has
loss 0.0167. The sensor identity path is learned separately inside each outer
fold because K-finder estimates a count, not one global montage.

The result is post-selection descriptive evidence. It does not include a
second untouched cohort for uncertainty around K.

## 8. Personalized split

For each P/G target, seizure events are ordered by natural recording filename
and onset location. The last:

```text
max(ceil(0.20 × number of seizures), 2)
```

events are test events. Every recording containing a test seizure is reserved
for test. Earlier seizures in those recordings are excluded from P training.
Controls are retained only when their recording belongs to that partition and
are assigned to a seizure group in the same recording.

This gives:

| Patient | P train | Test | Excluded |
|---|---:|---:|---|
| PN00 | 3 | 2 | 0 |
| PN06 | 3 | 2 | 0 |
| PN10 | 6 | 2 | 2 |
| PN12 | 2 | 2 | 0 |
| PN14 | 2 | 2 | 0 |

PN10 S07 and S08 are excluded because they share the S09 test recording.
Train and test recording sets, raw landmarks, and compact-feature hashes are
disjoint for every target.

## 9. K-suiter, P, and G

For each target patient:

### Personalized P

- Training population: only that patient's retained P training sessions.
- Channel-selection predictions: leave one complete seizure/control group out.
- Test population: the two reserved seizure sessions.

### Generalized G

- Training population: all data from the other four target-cohort patients.
- Target patient's rows and labels are absent.
- Channel-selection predictions: leave one training patient out.
- Test population: the exact same rows used for P.

For both P and G, greedy K-suiter adds the channel that minimizes binary
cross-entropy of the current uncalibrated out-of-fold probability average.
The first 16 channels are fitted with the shared base model and calibrated
from the same training-only OOF predictions. A 29-channel version is also
fitted as a sensitivity comparison.

P and G differ in training population and grouping, not model architecture.
G has much more training data, so the comparison does not isolate
personalization from sample size.

## 10. Metrics

Primary P/G comparison:

- binary cross-entropy at five-second landmarks; lower is better.

Secondary descriptive metrics:

- landmark AUPRC;
- landmark AUROC;
- Brier score;
- the same four metrics after averaging probabilities within episode; and
- mean cross-entropy within each held-out seizure/control group.

For each model, a deployable constant comparator uses that model's training
positive rate. The test-prevalence entropy is also saved as an explicitly
labelled oracle best constant; it is not a deployable baseline.

The paired patient statistic is `G loss - P loss`; negative values favor G.
Uncertainty is described with a 10,000-repeat patient bootstrap and the exact
two-sided sign-flip test over all `2^5` sign assignments. With five patients,
the exact test is emphasized.

## 11. Core channels

Three strict intersections are reported:

- intersection of the five P channel sets;
- intersection of the five G channel sets; and
- intersection of all ten P and G sets.

No frequency threshold is used to redefine “core.” Frequency counts are saved
separately.

## 12. Reproducibility

The final random seed is 42. The run records:

- the Git commit present at execution;
- whether the worktree was dirty;
- SHA-256 hashes of the analysis code; and
- Python and analysis-package versions.

Patient feature-cache signatures hash the consumed manifest content and source
recording metadata. `run_analysis.py` rebuilds the cleaned cohort and K-finder
result on every run; it cannot silently reuse a stale K.
