# EEG Final Project: Raw-data download

This project uses the **Siena Scalp EEG Database**, not the Iris dataset. The
raw data comes from PhysioNet and remains unchanged.

## 1. Source

- **Repository:** PhysioNet
- **Dataset:** Siena Scalp EEG Database, version 1.0.0
- **Dataset page:** https://physionet.org/content/siena-scalp-eeg/1.0.0/
- **ZIP download:** https://physionet.org/content/siena-scalp-eeg/get-zip/1.0.0/
- **DOI:** https://doi.org/10.13026/5d4a-j060
- **Creator:** Paolo Detti
- **License:** Creative Commons Attribution 4.0 International

## 2. Folder structure

The downloaded subject folders and filenames are preserved under:

```text
\Users\shrey\OneDrive\Desktop\Cosmos\26-the-optimizers-analysis\final_project\data\raw>
├── PN00/
├── PN01/
├── ...
├── PN17/
├── LICENSE.txt
├── RECORDS
├── SHA256SUMS.txt
└── subject_info.csv
```

There are 14 subject folders. Subject identifiers are not consecutive.

## 3. Download command

From the repository root, run:

```bash
bash final_project/download_eeg_data.sh
```

The script uses eight concurrent downloads by default. To choose another safe
level of concurrency, set `EEG_PARALLEL_DOWNLOADS`, for example:

```bash
EEG_PARALLEL_DOWNLOADS=4 bash final_project/download_eeg_data.sh
```

## 4. What each command does

- `mkdir -p` creates only the required raw-data and subject directories.
- `curl --continue-at -` downloads each original file and resumes an incomplete
  download rather than restarting it.
- `xargs -P` downloads several independent files concurrently.
- `shasum -a 256` checks every file against PhysioNet's published
  `SHA256SUMS.txt` manifest.
- A file that is already present and has the expected checksum is not downloaded
  again.

The official per-file S3 mirror is used instead of retaining the 13 GB ZIP.
The resulting raw files are the same files listed in PhysioNet's checksum
manifest, without an extra archive consuming disk space.

## 5. Raw files

| Raw file | Meaning |
|---|---|
| `PNxx/*.edf` | Original EEG/EKG recordings in European Data Format |
| `PNxx/Seizures-list-PNxx.txt` | Sampling rate, usable channels, recording times, and seizure intervals for one subject |
| `subject_info.csv` | Subject age/sex, seizure classification, channel and seizure counts, and recording duration |
| `RECORDS` | PhysioNet index of the EDF recordings |
| `SHA256SUMS.txt` | Published SHA-256 hashes used to verify file integrity |
| `LICENSE.txt` | Dataset license terms |
| `siena-scalp-eeg-database-1.0.0.zip` | Optional distribution archive; not retained by the per-file downloader |

## 6. Important rule

This step downloads raw data only. Do not rename columns, clean data, remove
rows, create train/test files, train a model, calculate accuracy, or create
visualizations.

## 7. First five observations in Python

To display the first five rows of the subject-level metadata without modifying
the raw data:

```python
from pathlib import Path

import pandas as pd

raw_dir = Path("final_project/data/raw")
subject_info = pd.read_csv(raw_dir / "subject_info.csv")
print(subject_info.head(5))
```

The EDF recordings are multichannel time series rather than CSV rows. To inspect
the first five time samples from one recording without loading the whole file:

```python
from pathlib import Path

import mne
import pandas as pd

edf_path = Path("final_project/data/raw/PN00/PN00-1.edf")
raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
first_five = pd.DataFrame(
    raw.get_data(start=0, stop=5).T,
    columns=raw.ch_names,
    index=raw.times[:5],
)
first_five.index.name = "time_seconds"
print(first_five)
```
