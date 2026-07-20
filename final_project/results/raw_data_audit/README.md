# Raw EEG data audit

Generated from the files currently present in `data/raw`.

## Summary

- **Participants in metadata:** 14
- **Expected EDF recordings (RECORDS):** 41
- **Present EDF recordings:** 41 (100.0%)
- **Missing expected EDF recordings:** 0
- **Available recording time:** 8,461.2 minutes
- **EDF files with an exact header/file-size match:** 41/41
- **EDF read/header errors:** 0
- **Signal screen:** 40 seconds per channel per EDF (20 evenly spaced 2-second samples), totaling 1,733 channel-level checks.

## Issues found

1. **Archive completeness: all 41 EDF files listed in `RECORDS` are present.**
2. **Recording-duration discrepancy:** EDF headers total 8,461.2 minutes, versus 7,704 minutes in `subject_info.csv` (+757.2 minutes; +9.8%). Reconcile this before using `rec_time_minutes` as a precise duration field.
3. **No structural truncation was detected** among readable EDFs: a valid EDF header was read and each present file's exact byte count agrees with its declared number of data records.
4. **No sampled scalp-EEG lead was near-flat or materially clipped** (0 flat and 0 clipped `EEG ...` channel/file combinations). EDF does not encode NaNs; this does not rule out brief artifacts outside the 40-second-per-channel screen.
5. **Amplitude outlier screen:** 197/1255 sampled EEG channel/file combinations exceed the IQR high-RMS threshold of 102.7 uV. Most are concentrated in PN10 (50), PN12 (36), PN07 (31), PN05 (14), PN06 (13), PN16 (12), PN14 (11), PN13 (10), PN03 (9), PN11 (5), PN00 (3), PN17 (2), PN01 (1). This may reflect seizure activity, movement/electrode artifact, or DC offset; review these signals before using a global amplitude rejection rule. The largest sampled RMS values occur in PN12, especially P9/P10.
6. **Channel montages vary, including capitalization/labels for some leads.** Use a common-channel intersection and normalize names (for example `Fp2` vs `FP2`) before pooling recordings. Auxiliary EKG, pulse-oximeter, heart-rate, marker, and unnamed signals are present in the EDFs but excluded from the EEG amplitude screen; many of those auxiliary signals are constant.
7. **Metadata formatting:** the `subject_info.csv` header and fields contain leading spaces after commas. This is harmless when parsed with whitespace trimming, but can break code that refers to columns by their untrimmed names.

## Interpretation notes

“Outliers” in scalp EEG can be genuine seizures, movement, eye, muscle, or electrode artifacts. The amplitude plot is intended to prioritize manual review, rather than automatically remove high-amplitude windows.

## Deliverables

- `edf_file_inventory.csv`: header-derived recording inventory and integrity check.
- `subject_coverage.csv` and `missing_edf_files.csv`: dataset coverage against `RECORDS`.
- `sampled_channel_metrics.csv`: sampled amplitude, flatness, and clipping metrics.
- PNG charts `01`-`04`: archive overview and representative raw signal excerpts.
