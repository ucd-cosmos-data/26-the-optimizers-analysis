from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_DIR / "data" / "raw" / "Group 2_ Optimizers Survey.csv"
INTERIM_PATH = PROJECT_DIR / "data" / "interim" / "optimizers_survey_clean.csv"


COLUMN_MAP = {
    "Timestamp": "submitted_at",
    "What cluster is your roommate in?": "roommate_cluster",
    "Are you and your roommate in the same grade?": "same_grade",
    "Do you and your roommate share similar sleeping schedules?": "similar_sleep_schedule",
    "On a scale of 1-10 how much do you like your roommate?": "roommate_rating",
    "What interests do you share the most strongly with your roommate?": "shared_interest",
}


def clean_survey(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_path, keep_default_na=False)
    df.columns = df.columns.str.strip()

    form_metadata_cols = [
        col
        for col in df.columns
        if col == "Total score" or col.endswith("[Score]") or col.endswith("[Feedback]")
    ]
    df = df.drop(columns=form_metadata_cols).rename(columns=COLUMN_MAP)

    missing_cols = sorted(set(COLUMN_MAP.values()) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing expected cleaned columns: {missing_cols}")

    df = df[list(COLUMN_MAP.values())].copy()

    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()

    df["submitted_at"] = pd.to_datetime(
        df["submitted_at"].str.removesuffix(" MDT"),
        format="%Y/%m/%d %I:%M:%S %p",
    ).dt.tz_localize("America/Denver")

    df["roommate_cluster"] = (
        df["roommate_cluster"].str.extract(r"^\s*(\d+)", expand=False).astype("Int64")
    )
    df["same_grade"] = df["same_grade"].map({"Yes": True, "No": False})
    df["similar_sleep_schedule"] = df["similar_sleep_schedule"].map(
        {"Yes": True, "No": False}
    )
    df["roommate_rating"] = pd.to_numeric(df["roommate_rating"], errors="coerce").astype(
        "Int64"
    )
    df["shared_interest"] = df["shared_interest"].replace(
        {"None": "No shared interest"}
    )
    df["shared_interest"] = df["shared_interest"].astype("category")

    return df


def main() -> None:
    cleaned = clean_survey()
    INTERIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(INTERIM_PATH, index=False)
    print(f"Wrote {len(cleaned)} rows to {INTERIM_PATH}")


if __name__ == "__main__":
    main()
