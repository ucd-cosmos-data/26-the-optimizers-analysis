"""Build the compact per-patient channel feature caches from raw EEG."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.channel_features import (
    FeatureConfig,
    build_feature_caches,
    load_episode_manifest,
)


PROJECT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT / "metadata" / "episode_manifest.csv"
DEFAULT_CACHE_DIR = (
    PROJECT
    / "data"
    / "processed"
    / "channel_features"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the 21-feature-per-channel caches used by run_analysis.py."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Canonical episode manifest CSV.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Destination for patient joblib/JSON cache pairs.",
    )
    parser.add_argument(
        "--patient",
        action="append",
        dest="patients",
        help="Build one patient ID; repeat for multiple patients.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when the content signature matches.",
    )
    parser.add_argument(
        "--use-split-cache",
        action="store_true",
        help=(
            "Read optional data/raw/splitdata arrays instead of canonical EDFs."
        ),
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=PROJECT / "data" / "raw" / "splitdata",
        help="Location of the optional channel-separated signal cache.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-episode progress messages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_episode_manifest(
        args.manifest, project_root=PROJECT
    )
    built = build_feature_caches(
        manifest,
        args.cache_dir,
        project_root=PROJECT,
        patient_ids=args.patients,
        config=FeatureConfig(),
        force=args.force,
        verbose=not args.quiet,
        use_split_cache=args.use_split_cache,
        split_root=args.split_root,
    )
    total_rows = sum(len(patient.frame) for patient in built)
    print(
        f"Ready: {len(built)} patient cache(s), {total_rows:,} landmark rows, "
        f"{len(built[0].channel_names) if built else 0}+ channels per patient."
    )


if __name__ == "__main__":
    main()
