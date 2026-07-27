# Final-60-second EEG frequency-band analysis

## Design and verification

- **Analyzed seizures:** 47 from 14 patients.
- **Preictal interval:** ten consecutive 6-second windows from -60 to 0 seconds; plotted at midpoints -57 to -3 seconds.
- **Signal calculation:** every sample in every window was detrended, robustly common-median referenced, quality screened, notched at 60 Hz when possible, and analyzed with 4-second Hann Welch segments with 50% overlap.
- **Band powers:** calculated per usable EEG channel, transformed to channel-relative power, then aggregated by the median across channels.
- **Interictal controls:** clean, non-overlapping 60-second blocks at least 15 minutes from every annotated seizure; their 6-second windows create one patient-and-band baseline.
- **Filter validity:** an EDF channel was analyzed for a band only if its declared high-pass cutoff was at or below the band lower edge and its low-pass cutoff was at or above the band upper edge. Delta is therefore only reported when 1-4 Hz was preserved.
- **Dependence warning:** neighboring windows from the same seizure and seizures from the same patient are correlated. The total number of window rows is not the statistical sample size; correlations are calculated within seizure and patient trajectories are summarized with equal patient weight.

## Recording checks

All 41 EDF recordings were audited for EEG sampling rate, units, and acquisition-filter passband. Sampling-rate and filter details are in `recording_metadata_audit.csv`; resolved onset annotations and any exclusions are in `seizure_annotation_audit.csv`.

## Band-level findings

| Band | Valid seizure paths | Seizure Pearson r, median [IQR] | Patient Pearson r, median [IQR] | Max |patient-median z| | Sustained seizures | Median first sustained time | Patients with sustained change |
|---|---:|---|---|---:|---:|---|---:|
| Delta | 9 | -0.31 [-0.59, 0.34] | -0.31 [-0.36, 0.39] | 0.66 | 1/9 (11.1%) | -9.0 s | 1/6 (16.7%) |
| Theta | 43 | 0.03 [-0.27, 0.43] | 0.25 [-0.15, 0.51] | 0.30 | 2/43 (4.7%) | -57.0 s | 2/13 (15.4%) |
| Alpha | 47 | -0.11 [-0.42, 0.18] | -0.18 [-0.44, 0.11] | 0.27 | 3/47 (6.4%) | -15.0 s | 3/14 (21.4%) |
| Beta | 36 | 0.22 [-0.27, 0.49] | 0.14 [-0.01, 0.55] | 0.39 | 3/36 (8.3%) | -57.0 s | 3/12 (25.0%) |
| Gamma | 7 | -0.04 [-0.38, 0.07] | -0.43 [-0.43, -0.43] | 0.30 | 0/7 (0.0%) | not estimable s | 0/1 (0.0%) |

Negative correlations mean standardized relative power tended to fall as the annotated onset approached (time moves from -57 to -3 s); positive correlations mean it tended to rise. Spearman results are reported alongside Pearson results in `correlation_summary.csv`.

## Patient-equal median standardized power by 6-second window

Each entry is the median of the patient-level median values at that window, so each patient has equal weight. The accompanying patient IQR is in `trajectory_summary.csv` and shown in the plots.

| Band | -57 | -51 | -45 | -39 | -33 | -27 | -21 | -15 | -9 | -3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Delta | 0.26 | -0.18 | -0.66 | -0.16 | 0.12 | -0.00 | -0.26 | -0.63 | -0.40 | -0.28 |
| Theta | -0.12 | -0.12 | -0.05 | 0.14 | 0.04 | -0.30 | -0.21 | -0.13 | -0.03 | -0.02 |
| Alpha | -0.12 | 0.05 | -0.05 | -0.27 | -0.05 | 0.10 | -0.05 | -0.06 | 0.03 | 0.00 |
| Beta | -0.01 | 0.13 | 0.31 | 0.28 | 0.18 | 0.12 | 0.36 | 0.30 | 0.37 | 0.39 |
| Gamma | -0.05 | 0.18 | 0.14 | -0.02 | -0.09 | -0.07 | -0.23 | -0.30 | 0.06 | -0.03 |

## Outputs

- `preictal_window_features.csv`: one band row per seizure-window, with patient ID, seizure ID, QC, source recording, patient baseline, and z score.
- `patient_band_baselines.csv`: patient-specific interictal means, SDs, and clean-window counts.
- `seizure_correlations.csv`, `patient_correlations.csv`, and `correlation_summary.csv`: repeated-measures-aware temporal correlations.
- `trajectory_summary.csv` and `sustained_change_summary.csv`: requested median/IQR and abnormal-run measures.
- `figures/preictal_<band>_trajectory.png`: one trajectory graph for each band.
