from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
RAW_WORKOUTS_CSV = PROJECT_ROOT / "data" / "raw" / "weightlifting_721_workouts.csv"
USER_WORKOUTS_CSV = PROJECT_ROOT / "data" / "user" / "user_workouts.csv"
WORKOUTS_CSV = PROJECT_ROOT / "data" / "processed" / "workouts.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUTPUT_CSV = PROCESSED_DIR / "train_workouts.csv"
TEST_OUTPUT_CSV = PROCESSED_DIR / "test_workouts.csv"

TRAIN_SPLIT_RATIO = 0.8

BASE_COLUMNS = [
    "exercise",
    "sets",
    "reps",
    "weight",
    "date",
    "rpe",
    "rir",
    "logged_at",
]
TARGET_COLUMN = "actual_next_weight"


def load_existing_schema(path: Path) -> list[str]:
    if not path.exists():
        return BASE_COLUMNS + [TARGET_COLUMN]
    return pd.read_csv(path, nrows=0).columns.tolist()


def load_source_workouts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {
        "Date",
        "Exercise Name",
        "Set Order",
        "Weight",
        "Reps",
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
    cleaned["Weight"] = pd.to_numeric(cleaned["Weight"], errors="coerce").fillna(0.0)
    cleaned["Reps"] = pd.to_numeric(cleaned["Reps"], errors="coerce").fillna(0)
    cleaned["Set Order"] = (
        pd.to_numeric(cleaned["Set Order"], errors="coerce").fillna(0).astype(int)
    )

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


def collapse_to_workout_rows(df: pd.DataFrame) -> pd.DataFrame:
    grouped_rows: list[dict[str, object]] = []

    for (logged_at, exercise), group in df.groupby(["Date", "Exercise Name"], sort=True):
        working_weight = _select_working_weight(group)
        working_sets = group[group["Weight"] == working_weight].copy()
        if working_sets.empty:
            working_sets = group.copy()

        grouped_rows.append(
            {
                "exercise": exercise,
                "sets": max(1, int(len(working_sets))),
                "reps": max(0, int(round(working_sets["Reps"].median()))),
                "weight": float(round(working_weight, 2)),
                "date": logged_at.date().isoformat(),
                "rpe": "",
                "rir": "",
                "logged_at": logged_at.isoformat(timespec="seconds"),
            }
        )

    workouts = pd.DataFrame(grouped_rows, columns=BASE_COLUMNS)
    return workouts.sort_values(["date", "logged_at", "exercise"]).reset_index(drop=True)


def add_actual_next_weight(df: pd.DataFrame) -> pd.DataFrame:
    supervised = df.copy()
    supervised["date"] = pd.to_datetime(supervised["date"], errors="coerce")
    supervised["logged_at"] = pd.to_datetime(supervised["logged_at"], errors="coerce")
    supervised = supervised.sort_values(["exercise", "date", "logged_at"]).reset_index(drop=True)

    supervised[TARGET_COLUMN] = supervised.groupby("exercise")["weight"].shift(-1)
    supervised = supervised.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    supervised[TARGET_COLUMN] = (
        pd.to_numeric(supervised[TARGET_COLUMN], errors="coerce").round(2)
    )

    supervised["date"] = supervised["date"].dt.date.astype(str)
    supervised["logged_at"] = supervised["logged_at"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return supervised.sort_values(["logged_at", "exercise"]).reset_index(drop=True)


def load_user_workouts(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"User workouts CSV was not found at {path}")

    df = pd.read_csv(path)
    missing = [column for column in BASE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"user_workouts.csv is missing expected columns: {missing}")

    # Keep the user-owned file free of supervised training targets.
    return df[BASE_COLUMNS].copy()


def build_user_workouts_from_raw() -> pd.DataFrame:
    cleaned_user_workouts = collapse_to_workout_rows(
        preprocess_source_workouts(load_source_workouts(RAW_WORKOUTS_CSV))
    )
    if cleaned_user_workouts.empty:
        raise ValueError("No usable workout rows were produced from the source CSV.")
    return cleaned_user_workouts


def split_train_test(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.copy()
    ordered["logged_at"] = pd.to_datetime(ordered["logged_at"], errors="coerce")
    ordered["date"] = pd.to_datetime(ordered["date"], errors="coerce")
    ordered = ordered.sort_values(["logged_at", "date", "exercise"]).reset_index(drop=True)

    train_size = int(len(ordered) * train_ratio)
    train_df = ordered.iloc[:train_size].copy()
    test_df = ordered.iloc[train_size:].copy()

    train_df["date"] = train_df["date"].dt.date.astype(str)
    test_df["date"] = test_df["date"].dt.date.astype(str)
    train_df["logged_at"] = train_df["logged_at"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    test_df["logged_at"] = test_df["logged_at"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_outputs(workouts_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    workouts_df.to_csv(WORKOUTS_CSV, index=False)
    train_df.to_csv(TRAIN_OUTPUT_CSV, index=False)
    test_df.to_csv(TEST_OUTPUT_CSV, index=False)


def main() -> None:
    if not USER_WORKOUTS_CSV.exists():
        cleaned_user_workouts = build_user_workouts_from_raw()
        cleaned_user_workouts.to_csv(USER_WORKOUTS_CSV, index=False)
        print(f"[DONE] Saved cleaned user workouts dataset to {USER_WORKOUTS_CSV}")

    user_workouts_df = load_user_workouts(USER_WORKOUTS_CSV)
    if user_workouts_df.empty:
        user_workouts_df = build_user_workouts_from_raw()
        user_workouts_df.to_csv(USER_WORKOUTS_CSV, index=False)
        print(f"[DONE] Rebuilt empty user workouts dataset at {USER_WORKOUTS_CSV}")

    expected_columns = load_existing_schema(WORKOUTS_CSV)
    workouts_df = add_actual_next_weight(user_workouts_df)
    workouts_df = workouts_df[expected_columns].copy()

    train_df, test_df = split_train_test(workouts_df, TRAIN_SPLIT_RATIO)
    save_outputs(workouts_df, train_df, test_df)

    print(f"[DONE] Saved derived supervised workouts dataset to {WORKOUTS_CSV}")
    print(f"[DONE] Saved train split ({len(train_df)} rows) to {TRAIN_OUTPUT_CSV}")
    print(f"[DONE] Saved test split ({len(test_df)} rows) to {TEST_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
