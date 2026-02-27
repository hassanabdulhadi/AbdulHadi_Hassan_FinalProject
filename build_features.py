from pathlib import Path

import pandas as pd  # Import pandas for table loading and feature engineering.

ROOT = Path(__file__).resolve().parent  # Resolve files relative to this script for portability.
SOURCE_FILE = ROOT / "champions_league_table.csv"  # Input table used to derive model-ready features.
OUTPUT_FILE = ROOT / "ucl_features.csv"  # Output file where engineered features are saved.


def build_features() -> None:  # Define one entry function to keep this script clean and reusable.
    df = pd.read_csv(SOURCE_FILE, encoding="latin1")  # Read the source table into a DataFrame.

    df["PPM"] = df["Pts"] / df["MP"]  # Points per match feature.
    df["GF_per_match"] = df["GF"] / df["MP"]  # Goals scored per match feature.
    df["GA_per_match"] = df["GA"] / df["MP"]  # Goals conceded per match feature.
    df["GD_per_match"] = df["GD"] / df["MP"]  # Goal difference per match feature.

    df["Strength"] = (  # Build one weighted strength score from the core per-match stats.
        df["PPM"] * 0.4
        + df["GD_per_match"] * 0.3
        + df["GF_per_match"] * 0.2
        - df["GA_per_match"] * 0.1
    )

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")  # Persist engineered features to CSV.
    print(f"{OUTPUT_FILE} created successfully")  # Confirm success for quick CLI feedback.


if __name__ == "__main__":  # Run feature building only when executed directly.
    build_features()  # Execute the feature generation workflow.
