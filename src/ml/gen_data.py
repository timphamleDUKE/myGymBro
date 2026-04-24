from __future__ import annotations

from pathlib import Path

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
RAW_WORKOUTS_CSV = PROJECT_ROOT / "data" / "raw" / "weightlifting_721_workouts.csv"
USER_WORKOUTS_CSV = PROJECT_ROOT / "data" / "user" / "workouts.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REAL_OUTPUT_CSV = OUTPUT_DIR / "real_workouts_like_app.csv"
SYNTHETIC_OUTPUT_CSV = OUTPUT_DIR / "synthetic_workouts_like_app.csv"
TRAIN_OUTPUT_CSV = OUTPUT_DIR / "train_workouts.csv"
TEST_OUTPUT_CSV = OUTPUT_DIR / "test_workouts.csv"

APP_COLUMNS = [
    "exercise",
    "sets",
    "reps",
    "weight",
    "date",
    "rpe",
    "rir",
    "logged_at",
]

TRAIN_SPLIT_RATIO = 0.8
RANDOM_SEED = 2026


def load_app_schema(path: Path) -> list[str]:
    df = pd.read_csv(path, nrows=0)
    return df.columns.tolist()


def load_source_workouts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {
        "Date",
        "Exercise Name",
        "Set Order",
        "Weight",
        "Reps",
        "Notes",
        "Workout Notes",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in source CSV: {sorted(missing)}")
    return df


def _coerce_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def preprocess_source_workouts(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="coerce")
    cleaned["Exercise Name"] = _coerce_text(cleaned["Exercise Name"])
    cleaned["Notes"] = _coerce_text(cleaned["Notes"])
    cleaned["Workout Notes"] = _coerce_text(cleaned["Workout Notes"])
    cleaned["Weight"] = pd.to_numeric(cleaned["Weight"], errors="coerce").fillna(0.0)
    cleaned["Reps"] = pd.to_numeric(cleaned["Reps"], errors="coerce").fillna(0)
    cleaned["Set Order"] = pd.to_numeric(cleaned["Set Order"], errors="coerce").fillna(0).astype(int)

    # Remove malformed rows and cardio-style rows with no meaningful load/reps.
    cleaned = cleaned.dropna(subset=["Date"])
    cleaned = cleaned[cleaned["Exercise Name"] != ""]
    cleaned = cleaned[(cleaned["Weight"] > 0) | (cleaned["Reps"] > 0)]

    return cleaned.sort_values(["Date", "Exercise Name", "Set Order"]).reset_index(drop=True)


def _select_working_weight(group: pd.DataFrame) -> float:
    weighted_sets = group[group["Weight"] > 0].copy()
    if weighted_sets.empty:
        return 0.0

    weight_counts = weighted_sets["Weight"].value_counts()
    max_count = int(weight_counts.iloc[0])
    candidate_weights = weight_counts[weight_counts == max_count].index.tolist()
    return float(max(candidate_weights))


def collapse_to_app_workout_rows(df: pd.DataFrame) -> pd.DataFrame:
    grouped_rows: list[dict] = []

    for (logged_at, exercise), group in df.groupby(["Date", "Exercise Name"], sort=True):
        working_weight = _select_working_weight(group)
        working_sets = group[group["Weight"] == working_weight].copy()
        if working_sets.empty:
            working_sets = group.copy()

        sets = int(len(working_sets))
        reps = int(round(working_sets["Reps"].median()))
        weight = float(round(working_weight, 2))

        grouped_rows.append(
            {
                "exercise": exercise,
                "sets": max(1, sets),
                "reps": max(0, reps),
                "weight": weight,
                "date": logged_at.date().isoformat(),
                "rpe": "",
                "rir": "",
                "logged_at": logged_at.isoformat(timespec="seconds"),
            }
        )

    app_like = pd.DataFrame(grouped_rows, columns=APP_COLUMNS)
    return app_like.sort_values(["date", "logged_at", "exercise"]).reset_index(drop=True)


def build_metadata(df: pd.DataFrame) -> SingleTableMetadata:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    metadata.update_column("exercise", sdtype="categorical")
    metadata.update_column("sets", sdtype="numerical")
    metadata.update_column("reps", sdtype="numerical")
    metadata.update_column("weight", sdtype="numerical")
    metadata.update_column("date", sdtype="datetime", datetime_format="%Y-%m-%d")
    metadata.update_column("logged_at", sdtype="datetime", datetime_format="%Y-%m-%dT%H:%M:%S")
    metadata.update_column("rpe", sdtype="categorical")
    metadata.update_column("rir", sdtype="categorical")

    return metadata


def fit_synthesizer(df: pd.DataFrame) -> GaussianCopulaSynthesizer:
    metadata = build_metadata(df)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(df)
    return synthesizer


def generate_synthetic_workouts(
    synthesizer: GaussianCopulaSynthesizer, num_rows: int
) -> pd.DataFrame:
    synthetic = synthesizer.sample(num_rows=num_rows)
    synthetic = synthetic[APP_COLUMNS].copy()

    synthetic["exercise"] = _coerce_text(synthetic["exercise"])
    synthetic["sets"] = pd.to_numeric(synthetic["sets"], errors="coerce").fillna(1).round().clip(lower=1).astype(int)
    synthetic["reps"] = pd.to_numeric(synthetic["reps"], errors="coerce").fillna(0).round().clip(lower=0).astype(int)
    synthetic["weight"] = (
        pd.to_numeric(synthetic["weight"], errors="coerce").fillna(0.0).round(2).clip(lower=0.0)
    )
    synthetic["date"] = pd.to_datetime(synthetic["date"], errors="coerce").dt.date.astype(str)
    synthetic["logged_at"] = pd.to_datetime(synthetic["logged_at"], errors="coerce").dt.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    synthetic["rpe"] = ""
    synthetic["rir"] = ""

    synthetic = synthetic.dropna(subset=["date", "logged_at"])
    synthetic = synthetic[synthetic["exercise"] != ""]
    return synthetic.reset_index(drop=True)


def split_train_test(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    train_size = int(len(shuffled) * train_ratio)
    train_df = shuffled.iloc[:train_size].reset_index(drop=True)
    test_df = shuffled.iloc[train_size:].reset_index(drop=True)
    return train_df, test_df


def save_outputs(
    real_df: pd.DataFrame, synthetic_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    real_df.to_csv(REAL_OUTPUT_CSV, index=False)
    synthetic_df.to_csv(SYNTHETIC_OUTPUT_CSV, index=False)
    train_df.to_csv(TRAIN_OUTPUT_CSV, index=False)
    test_df.to_csv(TEST_OUTPUT_CSV, index=False)


def main() -> None:
    app_schema = load_app_schema(USER_WORKOUTS_CSV)
    if app_schema != APP_COLUMNS:
        raise ValueError(
            "workouts.csv schema does not match expected app columns. "
            f"Found {app_schema}, expected {APP_COLUMNS}."
        )

    source_df = load_source_workouts(RAW_WORKOUTS_CSV)
    real_df = collapse_to_app_workout_rows(preprocess_source_workouts(source_df))

    if real_df.empty:
        raise ValueError("No usable workout rows were produced from the source CSV.")

    # Match the synthetic schema to the app's workout logger schema exactly.
    real_df = real_df[APP_COLUMNS].copy()

    synthesizer = fit_synthesizer(real_df)
    synthetic_df = generate_synthetic_workouts(synthesizer, num_rows=len(real_df))
    train_df, test_df = split_train_test(synthetic_df, TRAIN_SPLIT_RATIO)

    save_outputs(real_df, synthetic_df, train_df, test_df)

    print(f"[DONE] Saved app-shaped real dataset to {REAL_OUTPUT_CSV}")
    print(f"[DONE] Saved synthetic dataset to {SYNTHETIC_OUTPUT_CSV}")
    print(f"[DONE] Saved train split ({len(train_df)} rows) to {TRAIN_OUTPUT_CSV}")
    print(f"[DONE] Saved test split ({len(test_df)} rows) to {TEST_OUTPUT_CSV}")
    print(f"[INFO] Source schema aligned to workout.csv columns: {', '.join(APP_COLUMNS)}")


if __name__ == "__main__":
    main()
