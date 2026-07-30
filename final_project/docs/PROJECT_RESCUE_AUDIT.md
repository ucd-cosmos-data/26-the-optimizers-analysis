# Project rescue audit

This audit explains why the original project could not produce one defensible
answer, what replaced it, and what happened to every legacy code and result
category. It is a historical record, not a second methods document. The
supported methods are in `docs/METHODS.md`; the supported entry point is
`run_analysis.py`.

## Final replacement pipeline

The rescue replaced the disconnected Gen1, K-Finder, K-Suiter, personalized
(P), and generalized (G) implementations with one shared architecture:

1. Build causal channel features from the preceding 120 seconds at five-second
   landmarks.
2. For every channel, use training-only median imputation, standardization, and
   balanced L2 logistic regression (`C=0.1`).
3. Average the selected channels' probabilities.
4. Fit a positive-slope Platt calibrator using training-only out-of-fold
   predictions.
5. Use the same estimator, features, target, and calibration for K-Finder, P,
   and G.

The target is seizure onset within the next five minutes. K-Finder uses nested
leave-one-patient-out evaluation across all 14 patients and the 29 EEG channels
shared by everyone.

The final observed result is **K = 16**:

- Full 29-channel mean held-out AUPRC: **0.2378**
- K=16 mean held-out AUPRC: **0.2434**
- Worst held-out-patient AUPRC decrease at K=16: **0.0167**
- Allowed worst-patient decrease: **0.03**
- K=15 worst-patient decrease: **0.0427**, so K=15 fails
- Connected strict plateau: **K=25**
- Six of 14 patients are below their positive fraction with both K=16 and the
  full montage

Therefore K=16 is the smallest *observed* pass under the stated preservation
rule. It does not establish accurate or clinically useful seizure prediction.
The isolated pass and K=25 connected plateau are both reported because adding
weak channels to an unweighted probability average can make the curve
non-monotonic.

For P and G, only PN00, PN06, PN10, PN12, and PN14 are used. P trains on the
target patient's training seizures. G trains on the other four eligible
patients. Both are tested on exactly the same last
`max(ceil(0.20 × seizures), 2)` seizure events, with their complete recording
sessions reserved from training.

## Eleven root causes

| # | Root cause found | Consequence | Rescue action |
|---:|---|---|---|
| 1 | Gen1, K-Finder, P, and G used different estimators and feature definitions: ExtraTrees/SGD, per-sensor logistic models, HistGradientBoosting hazard models, and neural-network prototypes all coexisted. | Their scores and selected channels did not describe one system. | Replaced them with the single per-channel logistic ensemble above. |
| 2 | The scripts mixed prediction targets and evaluations: discrete hazards, alarm dashboards, binary risks, AUPRC, warning curves, and cross-entropy were treated as interchangeable. | P, G, and K results could not be compared fairly. | Fixed one five-minute binary target; K uses held-out AUPRC and P-versus-G uses held-out binary cross-entropy with AUPRC/AUROC as secondary metrics. |
| 3 | K was not tied to the final model. Old files supported mutually inconsistent stories including K=2, K=4, K=12, and a connected K=22 plateau. | Downstream K-Suiter and model results inherited an arbitrary sensor count. | Recomputed K with the final architecture and nested patient-held-out selection; the final observed value is K=16, with connected plateau K=25. |
| 4 | “31 channels” was used as if it meant 31 common EEG sensors. PN00's 31-channel export included 29 EEG channels plus EKG and SpO2, and some recordings have only 29 EEG leads. | The stated full montage and some figures were mislabeled. | Defined the full deployable montage as the 29 EEG channels shared by all 14 patients. |
| 5 | The personalized split used `max(1, ceil(0.20 × seizures))`. | Four of the five target patients had only one held-out seizure, violating the requested minimum of two. | Enforced the exact minimum-two test rule. |
| 6 | The old generalized workflow used the wrong cohort structure, including an eight-person development scheme in one implementation. | It was not the requested four-patients-train/one-patient-test experiment. | For each target, G now trains only on the other four eligible patients. |
| 7 | The old G implementation reused channels selected by the target patient's personalized model. | Target labels leaked into the supposedly generalized sensor choice. | G channel ranking now uses only out-of-fold predictions from its four training patients. |
| 8 | Old P and G scores were calculated on different target examples: G could see all target events while P used only its held-out subset. | The reported P-versus-G difference was not paired. | Both models now score identical held-out rows and event groups. |
| 9 | Control windows were not isolated by the raw time they represented. PN00 contained overlapping control groups and 564 duplicated raw landmark rows across groups; recording-sharing could also cross a nominal event split. | Exact or overlapping EEG could enter training and test. | Reassigned overlapping controls to one group, removed exact duplicate raw landmarks, asserted zero cross-group interval/feature-hash overlap, and reserved complete test recording sessions. |
| 10 | High-voltage and unusable-window handling lived in a standalone notebook and was not part of model feature construction. | Abnormal voltages could silently affect model inputs, while the cleaning claims did not match the trained data. | Integrated per-window quality rejection into feature extraction, retained usability as a feature, converted rejected physiological features to missing values, and imputed from training data only. |
| 11 | The directory contained stale executed notebook output, conflicting reports, hard-coded Windows paths, compiled bytecode, duplicated plots, and almost no usable README explanation. One Gen1 path also depended on removed `np.trapz` behavior. | Results could appear successful even when their source code no longer ran, and another team member could not identify the source of truth. | Added one entry point, version/hash metadata, canonical final tables and figures, portable metadata paths, tests, clear documentation, and the disposition below. |

## Legacy source disposition

### Replaced raw-to-feature code

The useful preprocessing logic was extracted from three large legacy files
into small, single-purpose modules. A real direct-EDF episode reproduced every
retained legacy feature exactly (`max absolute difference = 0.0`) before the
old files were removed.

| Legacy file | Useful portion recovered | Final replacement | Disposition |
|---|---|---|---|
| `scripts/split_eeg_channels.py` | Calibrated EEG reads and optional split-cache support | `src/eeg_io.py` | Replaced; delete |
| `scripts/personalized_channels_workflow.py` | Per-channel feature extraction, quality mask, and cache construction | `src/channel_features.py` | Replaced; delete |
| `scripts/rolling_seizure_forecasting.py` | Small configuration and path helpers | `src/channel_features.py` and `build_feature_cache.py` | Obsolete modeling removed; delete |

The supported code is now `build_feature_cache.py`, `run_analysis.py`, and the
three focused modules under `src/`.

### Replaced modeling scripts

| Legacy file | Exact problem | Disposition |
|---|---|---|
| `scripts/generalized_personalized_comparison.py` | Wrong cohort logic, target-derived channels for G, and unequal P/G test examples | Replaced by the paired P/G implementation in `src/reduced_sensor_pipeline.py`; delete |
| `scripts/k_suiter.py` | Old patient-specific ranking was tied to the obsolete hazard workflow and K12 handoff | Replaced by training-only cross-entropy forward ranking; delete |
| `scripts/run_sensor_count_step1.py` | Produced an old K result with a different model/plateau rule | Replaced by the nested K-Finder; delete |
| `scripts/seizure_sensor_selection.py` | Generic sensor-count estimator was not the final P/G architecture and allowed incompatible scoring modes | Replaced by the nested K-Finder; delete |
| `scripts/neural_network_pipeline.py` | Separate unvalidated architecture with no comparable final result | Out of scope for the common architecture; delete |

### Replaced or unrelated analysis scripts

| Legacy file | Exact problem | Disposition |
|---|---|---|
| `scripts/model_pn00_1_preictal_features.py` | One-recording, one-seizure exploratory anomaly screen; not patient-held-out prediction | Delete |
| `scripts/plot_pn00_1_channels.py` | PN00-only preview generator, duplicated by 41 recording overview images | Delete |
| `scripts/split_pn00_1_channels.py` | Created the misleading “31 channel” set by adding EKG and SpO2 to 29 EEG channels | Delete |
| `scripts/sample_preictal_bandpower.py` | Separate band-power research question and output, not the sensor-reduction pipeline | Delete |
| `scripts/make_figures.py` | Plotted the abandoned band-power analysis | Delete |
| `scripts/make_timeline_figures.py` | Plotted the abandoned band-power timeline analysis | Delete |

### Legacy tests and usage notes

| Files | Exact problem | Disposition |
|---|---|---|
| `scripts/test_generalized_personalized_comparison.py` | Encoded the obsolete eight-plus-six/generalized structure | Replace with paired-row and train-patient-isolation tests; delete legacy file |
| `scripts/test_k_suiter.py` | Encoded the obsolete K handoff and single-final-event assumptions | Replace with greedy-ranking and leakage tests; delete legacy file |
| `scripts/test_personalized_channels_workflow.py` | Tested the old split and HistGradientBoosting hazard model | Replace with the current split/calibration tests; delete legacy file |
| `scripts/test_rolling_seizure_forecasting.py` | Mixed useful cache/path tests with obsolete hazard, alarm, and model-bundle tests | Migrate only path rebasing/cache-signature checks to preprocessing tests; delete legacy file |
| `scripts/test_seizure_sensor_selection.py` | Tested the old K estimator and plateau behavior | Replace with final K invariants; delete legacy file |
| `scripts/test_split_eeg_channels.py` | Useful only for the retained raw splitter | Migrate its calibration/envelope checks to preprocessing tests; delete legacy file |
| `scripts/K_SUITER_USAGE.md` | Instructions for the obsolete K12/K-Suiter workflow | Replace with root documentation; delete |
| `scripts/SENSOR_SELECTION_USAGE.md` | Instructions for the obsolete sensor-selection workflow | Replace with root documentation; delete |

## Legacy notebook disposition

| Legacy notebook | Exact problem | Disposition |
|---|---|---|
| `notebooks/eeg_outlier_cleaning.ipynb` | Standalone cleaning experiment; its decisions were not used by the trained models | Replaced by integrated quality masking; delete |
| `scripts/gen1-model.ipynb` | Different estimator/features, stale output after a failing `np.trapz` code path | Delete |
| `scripts/generalized_vs_personalized.ipynb` | Notebook wrapper around the invalid old P/G comparison | Delete |
| `scripts/k-finder.ipynb` | Old K result did not use the final model or strict patient-held-out rule | Delete |
| `scripts/k-suiter-shreyas.ipynb` | Alternate, incomplete K-Suiter experiment | Delete |
| `scripts/k_suiter.executed.ipynb` | Frozen stale output from the obsolete K12 workflow | Delete |
| `scripts/k_suiter.ipynb` | Duplicate notebook interface for the obsolete K-Suiter | Delete |
| `scripts/neural_network.ipynb` | Separate unvalidated architecture | Delete |
| `scripts/personalized_channels.ipynb` | Old P model and invalid one-seizure minimum test split | Delete |
| `scripts/personalized_channels_quick.ipynb` | Truncated quick run, not a full experiment | Delete |
| `scripts/step5_channel_intersection.ipynb` | Calculated intersections from stale channel selections | Replaced by `core_channels.csv`; delete |
| `scripts/k_suiter_leave_one_patient_out.ipynb` | Partial LOPO experiment with team edits; not the final implementation | **Preserved intact as `archive/legacy_k_suiter_leave_one_patient_out.ipynb`** |

The LOPO notebook was already modified before the rescue. Its exact working
copy was moved, not discarded or overwritten.

## Legacy result disposition

Only `results/final/` is a supported model-result directory. The retained
dataset audit tables live in `metadata/raw_audit/`; the episode definition
lives in `metadata/`.

| Legacy result category | Files covered | Exact problem | Disposition |
|---|---|---|---|
| Misnamed PN00 overview | `results/PN00-1_all_31_channel_overview.png` | One recording; “31” included EKG and SpO2 | Delete |
| Old K-Suiter outputs | All 5 files in `results/k_suiter/` | PN00-only K12/Gen1 results from the incompatible architecture | Delete |
| Old personalized outputs | All 4 files in `results/personalized_channels/` | Obsolete K4/HistGradientBoosting model and invalid test allocation | Delete |
| Quick personalized outputs | All 4 files in `results/personalized_channels_quick/` | Partial diagnostic run, not the full cohort experiment | Delete |
| Old sensor-count outputs | `sensor_count_selected.json`, `sensor_count_step1.json`, `sensor_count_step1_curve.csv`, `sensor_count_step1_plateau.png`, `sensor_count_subject_scores.csv`, `sensor_count_worst_patient.json`, `sensor_count_worst_patient_loss_curve.csv` | Conflicting old K definitions/results | Delete |
| Old rolling-forecast outputs | All 31 non-manifest files in `results/rolling_forecast/`, including hazard models, predictions, warning curves, alarm dashboards, interpretability tables, and five PNGs | Obsolete target, splits, controls, architecture, and alarm claims | Delete |
| Episode manifest | `episode_manifest.csv`, `episode_manifest_metadata.json` | Valid episode source, but stored among stale results with absolute machine paths | Move to `metadata/`, normalize paths, retain |
| Raw-data audit | `edf_file_inventory.csv`, `missing_edf_files.csv`, `sampled_channel_metrics.csv`, `subject_coverage.csv` | Valid provenance mixed with model outputs | Move to `metadata/raw_audit/`, retain |
| Dataset coverage figure | `results/raw_data_audit/01_metadata_and_coverage.png` | Valid descriptive figure in a stale directory | Move to `results/final/figures/00_dataset_coverage.png`, retain |
| Other raw-audit material | `README.md`, `read_errors.csv`, `02_recording_durations.png`, `03_signal_quality_screen.png`, `04_representative_traces.png` | Redundant figures; final cleaning used different definitions; `read_errors.csv` was a malformed one-byte empty file | Delete |
| Tracked Gen1 processed data | All 52 files in `processed/k_suiter_gen1/` | PN00-only K12 landmark and episode caches | Delete |
| Ignored old LOPO cache | All 282 files in `data/processed/rolling_forecast/k_suiter_leave_one_patient_out/` | Stale K12 four-train/one-test cache | Delete |
| Ignored old sensor matrix | `data/processed/rolling_forecast/cohort_sensor_count_29_features.npz` | Superseded by leakage-cleaned final cohort matrix | Delete |
| Ignored band-power table | `data/processed/preictal_sampled_bandpower.csv` | Abandoned side analysis | Delete |
| Recording previews | All 41 `data/raw/splitdata/*/*/channel_overview.png` files | Redundant generated previews, not analysis evidence | Delete; retain the underlying arrays and manifests |
| Compiled caches | Every `__pycache__/` and `*.pyc` file | Machine-generated clutter that can hide stale imports | Delete and ignore |

The 31 removed rolling-forecast files were:

`ARTIFACT_INVENTORY.csv`, `all_seizure_peak_bin_matrix.csv`,
`all_seizure_peak_readings.csv`, `all_seizure_reading_probabilities.csv`,
`binary_test_landmark_probabilities.csv`, `binary_test_metrics.csv`,
`development_oof_landmark_probabilities.csv`, `development_oof_metrics.csv`,
`development_oof_patient_metrics.csv`, `development_oof_warning_curve.csv`,
`development_warning_curve.csv`, `forecast_evaluation.png`,
`global_permutation_importance.csv`, `holdout_landmark_probabilities.csv`,
`holdout_metrics.csv`, `holdout_patient_metrics.csv`,
`local_counterfactual_explanation.csv`, `model_interpretability.png`,
`moving_horizon_rolling_example.csv`, `moving_horizon_rolling_example.png`,
`rolling_discrete_hazard_model.joblib`, `seizure_split_allocation.csv`,
`study_summary.csv`, `test_alarm_dashboard.png`,
`test_alarm_dashboard_metrics.csv`, `test_confidence_intervals.csv`,
`test_landmark_probabilities.csv`, `test_metrics.csv`,
`test_patient_metrics.csv`, `test_peak_bin_matrix.csv`, and
`test_reading_probability_heatmaps.png`.

## Reproducibility guardrail

Cleanup must not remove:

- The 41 source EDF files, `RECORDS`, `SHA256SUMS.txt`, `LICENSE.txt`, and
  `subject_info.csv`
- The 1,255 split EEG arrays and their 41 channel manifests and metadata files
- The 14 current patient cache `.joblib` files and 14 matching signature JSON
  files under `data/processed/channel_features/`
- `data/processed/final/cohort_features_clean.npz` and its JSON audit
- `metadata/episode_manifest.csv` and
  `metadata/episode_manifest_metadata.json`
- `download_eeg_data.sh` and `docs/DATA_DOWNLOAD.md`

The source EDF files and generated feature caches are intentionally not model
results. They remain available so the final analysis can be rerun and audited.
