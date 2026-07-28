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
│   └── processed/                    # Generated feature and sampled-point tables
├── scripts/
│   ├── analyze_preictal_bandpower.py # Raw EEG → features → summaries
│   ├── sample_preictal_bandpower.py  # 100 sampled points/seizure → regressions
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

`data/raw` and `data/processed` are ignored by Git because they are large.
Small result tables and final figures are kept so collaborators can inspect the
findings without downloading all EEG files.

## Run the pipeline

From the repository root (`26-the-optimizers-analysis`):

```powershell
python final_project/scripts/analyze_preictal_bandpower.py
python final_project/scripts/make_figures.py
```

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
