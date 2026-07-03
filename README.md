# Data Analysis Repo

Data analysis workspace, organized as a collection of projects.

## Structure

Each project (`proj1/`, `proj2/`, ...) follows the same layout:

```
proj*/
├── data/
│   ├── raw/            # Original, immutable data
│   ├── interim/        # Intermediate, partially processed data
│   └── processed/      # Final, analysis-ready data
├── notebooks/          # Exploratory and analysis notebooks
├── models/             # Trained model artifacts
├── results/
│   └── figures/        # Generated plots and figures
└── README.md           # Project-specific notes
```

> By default, `data/` and `results/figures/` directories are git-ignored.

## Adding a new project

1. Copy the structure above into a new `proj*/` directory.
2. Document the project's purpose, data sources, and key findings in its `README.md`.