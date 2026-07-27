# PN00-1 preictal feature screen

## What this model can and cannot show

PN00-1 contains one annotated seizure. This is therefore an exploratory, within-recording screen, not an independent seizure-prediction validation. The model learns only the distribution of clean interictal EEG windows; it does not use preictal labels to set its score or threshold. A second patient or a second seizure is required for out-of-sample validation.

## Feature set

All values are first calculated in each quality-passing EEG lead, then summarized by the median across leads. The feature set includes RMS amplitude, line length, Hjorth mobility and complexity, spectral entropy, relative theta/alpha/beta power, and cross-channel line-length IQR. Delta is excluded because the PN00-1 high-pass filter is 1.591549 Hz; gamma is excluded because the low-pass filter is 30 Hz.

## Baseline-only anomaly model

- Clean interictal windows: 124
- Final-minute preictal windows: 10
- Threshold: 4.195, the 95th percentile of clean-interictal scores.
- Interictal flags: 5.6%.
- Preictal flags: 20.0%.
- Median score: 0.893 interictal vs 1.213 preictal.
- Earliest threshold-crossing preictal midpoint: -57.0 seconds (null means no preictal window crossed).

## Features with the largest robust preictal shifts

| Feature | Interictal median | Preictal median | Robust shift | AUC | Spearman rho vs time-to-onset |
|---|---:|---:|---:|---:|---:|
| median_line_length_uv_per_sample | 2.016 | 2.371 | 1.14 | 0.72 | -0.90 |
| median_rms_uv | 7.549 | 9.234 | 1.13 | 0.72 | -0.59 |
| median_theta_relative_power | 0.3599 | 0.3099 | -0.98 | 0.24 | 0.56 |
| median_beta_relative_power | 0.4546 | 0.4992 | 0.85 | 0.78 | -0.45 |
| median_spectral_entropy | 0.9068 | 0.9183 | 0.80 | 0.84 | -0.52 |
| spatial_iqr_line_length_uv_per_sample | 1.643 | 1.858 | 0.78 | 0.54 | -0.95 |

AUC here is a descriptive ranking of the one seizure's ten final-minute windows against interictal windows. It must not be treated as a prospective or independent performance estimate; the windows are temporally correlated.
