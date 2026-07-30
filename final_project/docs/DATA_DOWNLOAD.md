# Raw EEG data

## Source

- Dataset: [Siena Scalp EEG Database 1.0.0](https://physionet.org/content/siena-scalp-eeg/1.0.0/)
- DOI: [10.13026/5d4a-j060](https://doi.org/10.13026/5d4a-j060)
- License: Creative Commons Attribution 4.0

The final analysis expects 41 EDF recordings for 14 patients.

## Download and verify

From the `26-the-optimizers-analysis` directory:

```bash
bash final_project/download_eeg_data.sh
```

The downloader:

- resumes partial files;
- uses eight parallel transfers by default;
- checks the supplied SHA-256 sums; and
- verifies every EDF listed in `RECORDS`.

To use fewer parallel transfers:

```bash
EEG_PARALLEL_DOWNLOADS=4 bash final_project/download_eeg_data.sh
```

The expected layout is:

```text
final_project/data/raw/
├── PN00/
├── PN01/
├── ...
├── PN17/
├── LICENSE.txt
├── RECORDS
├── SHA256SUMS.txt
└── subject_info.csv
```

The raw files are never edited.

## Build model features

From `final_project`:

```bash
python build_feature_cache.py --force
```

This reads `metadata/episode_manifest.csv`, validates the EDF paths, applies
the five-second artifact rules, and writes one cache plus one content
signature for each of the 14 patients under:

```text
data/processed/channel_features/
```

Omit `--force` to reuse a cache only when its manifest and source signature
match.

The existing `data/raw/splitdata/` tree is an optional speed cache. It contains
lossless per-channel arrays derived from the EDF recordings. Direct EDF and
split-cache reads were checked on the same segment and produced identical
physical sample values. Splitdata can therefore be rebuilt or omitted without
changing the scientific input.

## Integrity result for this run

- EDF files present: 41 of 41
- Supplied SHA-256 checks: all matched
- Exported channel-recording arrays checked: 1,255
- Shape/read errors: 0

The descriptive inventory is in `metadata/raw_audit/`. Its sampled voltage
screen is provenance only; the model's actual artifact rejection occurs
during feature extraction and is summarized in
`results/final/tables/data_quality_summary.csv`.
