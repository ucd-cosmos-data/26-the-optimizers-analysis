# EEG band-power extraction and seizure models

This project extracts spectral EEG features from the EDF recordings in
`data/raw` and trains baseline seizure-window classifiers.

## What was extracted

Each recording is processed as non-overlapping 10-second windows. For every
window, the script finds channels whose EDF label begins with `EEG`, detrends
4-second Hann-tapered segments with 50% overlap, and integrates the PSD in
these bands:

| Band | Frequency range |
| --- | ---: |
| delta | 0.5–4 Hz |
| theta | 4–8 Hz |
| alpha | 8–13 Hz |
| beta | 13–30 Hz |
| gamma | 30–100 Hz |

`data/processed/eeg_bandpowers.csv` contains the mean, median, standard
deviation, relative power, and log10 power for each band across the available
EEG channels in each window. Power is integrated PSD in the EDF physical units
(µV² for this dataset).

Windows are labeled `seizure=1` when they overlap a seizure interval in the
subject seizure-list files. The provided labels contain 490 positive windows
and 50,259 negative windows.

## Models and outputs

The script trains logistic regression, random forest, and histogram gradient
boosting models. The train/test split is by subject, not by window, to avoid
leaking adjacent recordings from the same person into both sets.

- `models/*.joblib`: trained models plus their feature column lists
- `results/model_metrics.csv`: held-out metrics, including ROC-AUC and average precision
- `results/subject_split.json`: exact subject split and model feature list

The current held-out results are summarized below. Average precision is the
most useful headline metric here because seizure windows are rare.

Report-ready diagrams are saved in `results/figures/`:

- `01_analysis_pipeline.png`: processing and modeling workflow
- `02_frequency_bands.png`: delta, theta, alpha, beta, and gamma ranges
- `03_model_performance.png`: model scores and the best model's confusion matrix

| Model | Balanced accuracy | F1 | ROC-AUC | Average precision |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.533 | 0.053 | 0.572 | 0.063 |
| Histogram gradient boosting | 0.580 | 0.084 | 0.681 | 0.050 |
| Random forest | 0.574 | 0.116 | 0.650 | 0.049 |

## Reproduce

From the repository root:

```powershell
python final_project/scripts/extract_bandpowers_and_train.py
```

To reuse the existing feature CSV and retrain only the models:

```powershell
python final_project/scripts/extract_bandpowers_and_train.py --reuse-features
```

To regenerate the diagrams after updating the metrics:

```powershell
python final_project/scripts/make_figures.py
```
