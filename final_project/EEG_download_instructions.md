# Raw EEG download

## Source

- Dataset: Siena Scalp EEG Database, version 1.0.0
- Host: PhysioNet
- Dataset page: https://physionet.org/content/siena-scalp-eeg/1.0.0/
- DOI: https://doi.org/10.13026/5d4a-j060
- License: Creative Commons Attribution 4.0

This project uses the Siena EEG data, not the Iris dataset.

## Download

From the repository root:

```bash
bash final_project/download_eeg_data.sh
```

The downloader uses eight parallel transfers. To use fewer:

```bash
EEG_PARALLEL_DOWNLOADS=4 bash final_project/download_eeg_data.sh
```

The script downloads directly from PhysioNet's official S3 mirror, resumes
partial files, checks every SHA-256 checksum, and verifies that all EDF files
listed in `RECORDS` are present.

## Expected raw folder

```text
final_project/data/raw/
├── PN00/
├── PN01/
├── ... 14 patient folders total ...
├── PN17/
├── LICENSE.txt
├── RECORDS
├── SHA256SUMS.txt
└── subject_info.csv
```

Each patient folder contains:

- `*.edf`: multichannel scalp-EEG recordings
- `Seizures-list-PNxx.txt`: recording times and annotated seizure intervals

The raw files are never edited. All derived data goes to `data/processed` or
`results`.
