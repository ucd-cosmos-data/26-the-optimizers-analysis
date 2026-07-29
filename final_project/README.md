# Pre-seizure EEG band-power project

## Research question

How do delta, theta, alpha, beta, and gamma **relative power** change during
the final 60 seconds before scalp-EEG seizure onset compared with interictal
(between-seizure) activity? Are beta and gamma changes more consistent across
seizures and patients than the lower-frequency changes?

This is an exploratory analysis of the Siena Scalp EEG Database: 41 recordings,
47 annotated seizures, and 14 patients. It is a research project, not a medical
seizure detector.

## The idea in plain language

EEG electrodes measure tiny voltage changes at the scalp. A frequency band is a
range of signal speeds: delta is slow; gamma is fast. **Relative power** is the
fraction of measured EEG power that falls in one band.

For each seizure, the analysis:

1. Finds the annotated seizure-onset second.
2. Takes the final 60 seconds before onset.
3. Divides that minute into six 10-second bins.
4. Calculates relative power for all five bands.
5. Compares each bin with quiet periods from the same patient.
6. Summarizes patients equally, so a patient with many seizures does not
   dominate the result.
7. Measures whether each band changes in a similar direction and by a similar
   amount across seizures and patients.

## Folder map

```text
final_project/
├── README.md                         # Start here
├── METHODS.md                        # Exact scientific definitions and cautions
├── EEG_download_instructions.md      # Where the raw data comes from
├── download_eeg_data.sh              # Downloads and verifies PhysioNet files
├── data/
│   ├── raw/                          # Original EDF recordings; never edited
│   │   └── splitdata/                # Mirrored per-recording EEG arrays + plots
│   └── processed/                    # Generated feature and sampled-point tables
├── scripts/
│   ├── analyze_preictal_bandpower.py # Raw EEG → features → summaries
│   ├── sample_preictal_bandpower.py  # 100 sampled points/seizure → regressions
│   ├── split_eeg_channels.py         # Every EDF → per-channel arrays + overview
│   ├── seizure_sensor_selection.py   # Step 1: cohort-level sensor count K only
│   ├── run_sensor_count_step1.py     # Split-data integration; prints K
│   └── make_figures.py               # Summaries → four research figures
└── results/
    ├── raw_data_audit/               # Existing completeness/quality audit
    ├── event_inventory.csv           # Included and excluded seizures
    ├── interictal_control_blocks.csv # Exactly which controls were selected
    ├── temporal_summary.csv          # Magnitude by band and pre-onset bin
    ├── consistency_summary.csv       # Variability and direction agreement
    ├── consistency_comparisons.csv   # Beta/gamma vs lower-band comparisons
    ├── patient_interictal_bandpower_ranges.csv
    ├── sampled_bandpower_regression.csv
    ├── band_filter_validation.csv
    ├── analysis_settings.json        # Reproducibility settings
    └── figures/                      # Report-ready plots
```

`data/raw` and `data/processed` are ignored by Git because they are large,
except for the per-recording `splitdata/**/channel_overview.png` graphs. Small
result tables and final figures are also kept so collaborators can inspect the
findings without downloading all EEG files.

## Run the pipeline

From the repository root (`26-the-optimizers-analysis`):

```powershell
python final_project/scripts/analyze_preictal_bandpower.py
python final_project/scripts/make_figures.py
```

Create the channel-separated source used by the sensor-count workflow:

```powershell
python final_project/scripts/split_eeg_channels.py
```

This mirrors all 41 EDF recordings below `data/raw/splitdata`, stores one
lossless digital array per available EEG channel, and puts all channels for
each recording on one overview page. Siena recordings contain either 29 or 31
EEG channels. To overwrite the graphs without deleting them first:

```powershell
python final_project/scripts/split_eeg_channels.py --plots-only --force-plots
```

## Cohort sensor-count selection (Step 1)

This step estimates only the number of sensors, `K`; it does not choose the
final sensor identities or train personalized/generalized models. It uses the
29 EEG channels shared by all 14 subjects. Each sensor is represented by 24
compact features at five-second landmarks, with the target defined as seizure
onset within the next five minutes.

The current cohort uses four interictal controls per seizure: 47 preictal
episodes and 188 controls, or 14,100 landmark rows. A class-balanced logistic
model is fitted separately for each sensor. Greedy search forms candidate
K-sensor ensembles by averaging sensor probabilities, and leave-one-subject-out
validation evaluates them using average precision. The output is the smallest
count statistically non-inferior to the full 29-sensor baseline. No neural
network is used, and temporary sensor identities are discarded.

Run the current calculation with:

```powershell
python final_project/scripts/run_sensor_count_step1.py
```

With the current `0.02` non-inferiority margin, the executed notebook returns
`K = 4`. Both the control ratio and margin must be fixed before analysis:
average precision depends on class prevalence, so changing either setting can
change the plateau and K even when the underlying EEG signals are unchanged.
See [SENSOR_SELECTION_USAGE.md](scripts/SENSOR_SELECTION_USAGE.md) for the
complete algorithm and input specification.

The first script performs the scientific analysis. The second only turns its
summary tables into figures, so it must run second.

The separate sampled-point analysis uses the validated event and control
inventories, takes 100 stratified time points per seizure (10 from each of ten
6-second bins), and creates one regression plot per frequency band:

```powershell
python final_project/scripts/sample_preictal_bandpower.py
```

Its long-format point table is
`data/processed/preictal_sampled_bandpower.csv`. Each band contains 4,700
preictal points across the 47 seizures. Points are colored by whether their
relative power falls inside that patient's empirical central 95% interictal
range.

The default operational definition of an interictal control is a 60-second
block from the same patient, at least 15 minutes from every annotated seizure.
This relatively short buffer is a limitation of the available recordings.
Repeat the analysis with a stricter buffer as a sensitivity check:

```powershell
python final_project/scripts/analyze_preictal_bandpower.py --interictal-buffer-minutes 30
python final_project/scripts/make_figures.py
```

## How to read the main result

The analysis uses:

```text
log2(preictal relative power / matched interictal relative power)
```

- `0`: no change from the matched interictal baseline
- `+1`: twice the interictal relative power
- `-1`: half the interictal relative power

The four generated figures show:

1. The complete analysis pipeline.
2. The six-bin trajectory for every frequency band.
3. A heatmap for temporal ordering and magnitude.
4. Variability and direction agreement for the beta/gamma consistency question.

Lower variability and higher same-direction agreement mean a band is more
consistent. The hypothesis should only be called supported if beta and gamma
perform better on both ideas and the uncertainty supports that conclusion.

## Current exploratory result

The validated default run includes all 47 seizures and all 14 patients, uses
141 matched control blocks, and analyzes 1,128 EEG windows. No window failed the
minimum-channel quality rule.

Descriptively, theta and alpha first have patient-bootstrap intervals below
zero in the `-30 to -20` second bin. Beta does so only in the final 10 seconds.
Delta and gamma intervals include zero in every bin. The current consistency
measures do **not** support the hypothesis that beta and gamma are generally
more consistent than the lower-frequency bands; gamma is among the most
variable bands. These are exploratory summaries, not corrected confirmatory
tests.

## Important limitation

There are 47 seizures but only 14 independent patients. Multiple seizures from
one person are related observations. The summaries therefore average within
each patient before estimating the group time course. Findings should be
described as exploratory and should not be generalized to clinical seizure
prediction.

## Rolling seizure-forecasting notebook

`scripts/rolling_seizure_forecasting.ipynb` is a separate, EDF-based
time-to-event forecasting experiment. It uses a fixed two-minute causal EEG
context and, at each five-second landmark, produces a probability distribution
over configurable future onset bins plus an explicit no-onset outcome. Its
features are time-domain signal descriptors only; it does not use the
band-power features described above.

The notebook uses the requested seizure allocation (35 train, 4 validation,
8 test) and four seizure-free interictal episodes per seizure. It fits a
two-stage model: an Extra Trees nonlinear horizon-risk classifier and a
regularized logistic discrete-time survival model for onset-bin timing. The
time-domain features retain robust across-channel dispersion and synchrony;
they still contain no frequency-band powers or FFT features. The operational
alarm is deliberately more conservative than a raw threshold: it turns on
only after two consecutive high-risk readings and stays on until 13
consecutive low-risk five-second readings (65 seconds) occur.

Run it from `final_project/scripts` with the project virtual environment:

```powershell
..\..\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace rolling_seizure_forecasting.ipynb
```

At the bottom of the executed notebook, the final held-out dashboard displays
sensitivity, continuous-risk AUROC, false alarms per hour, and a clearly
labeled landmark-level confusion matrix. The same panel is saved as
`results/rolling_forecast/test_alarm_dashboard.png`; its numerical summary is
saved as `test_alarm_dashboard_metrics.csv`.

The operating threshold is selected on development data to prioritize at
least 90% seizure sensitivity, then minimize false alarms among qualifying
thresholds. This is a tradeoff, not a guaranteed clinical performance level.
With only 47 seizures and 14 patients, AUROC and false-alarm estimates are
high-variance; the test set must not be repeatedly tuned against and a new
locked external cohort is required for an unbiased final claim.
