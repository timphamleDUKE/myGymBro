from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pandas as pd
import xgboost as xgb


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DEFAULT_TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "train_workouts.csv"
DEFAULT_TEST_CSV = PROJECT_ROOT / "data" / "processed" / "test_workouts.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "src" / "test" / "xgboost-plus" / "xgboost_plus_predictions.csv"
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "xgboost-plus"
DEFAULT_MODEL_PREFIX = "xgboost_plus_next_weight"

MODEL_GROUPS = (
    "global",
    "upper_compound",
    "lower_compound",
    "isolation",
    "bodyweight",
)

FEATURE_COLUMNS = ["exercise", "sets", "reps", "weight", "rpe", "rir"]
TARGET_COLUMN = "actual_next_weight"
HISTORY_COLUMNS = ["sets", "reps", "weight", "rpe", "rir"]
HISTORY_WINDOW = 3
MIN_TRAIN_ROWS = 10
_LOADED_MODEL_ARTIFACTS: dict[tuple[str, str], tuple[xgb.XGBRegressor, list[str], dict]] = {}


UPPER_COMPOUND_KEYWORDS = (
    "bench",
    "press",
    "row",
    "pull up",
    "pull-up",
    "chin up",
    "chin-up",
    "lat pulldown",
    "pulldown",
)
LOWER_COMPOUND_KEYWORDS = (
    "squat",
    "deadlift",
    "rdl",
    "lunge",
    "leg press",
    "hip thrust",
)
ISOLATION_KEYWORDS = (
    "curl",
    "extension",
    "raise",
    "fly",
    "pushdown",
    "pressdown",
    "shrug",
    "calf",
)
BODYWEIGHT_KEYWORDS = (
    "plank",
    "crunch",
    "sit up",
    "sit-up",
    "push up",
    "push-up",
    "chin up",
    "chin-up",
    "pull up",
    "pull-up",
    "dip",
)


@dataclass
class XGBoostPlusPrediction:
    exercise: str
    current_weight: float
    predicted_next_weight: float
    recommended_increment: float
    model_group: str
    top_factors: list[str]
    reason: str
    model_type: str = "xgboost_plus_regressor"


def model_path_for_group(model_group: str) -> Path:
    return DEFAULT_MODEL_DIR / f"{DEFAULT_MODEL_PREFIX}_{model_group}.json"


def features_path_for_group(model_group: str) -> Path:
    return DEFAULT_MODEL_DIR / f"{DEFAULT_MODEL_PREFIX}_{model_group}_features.json"


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def classify_exercise(exercise: str) -> str:
    name = str(exercise).strip().lower()

    if _contains_any(name, BODYWEIGHT_KEYWORDS):
        return "bodyweight"
    if _contains_any(name, LOWER_COMPOUND_KEYWORDS):
        return "lower_compound"
    if _contains_any(name, UPPER_COMPOUND_KEYWORDS):
        return "upper_compound"
    if _contains_any(name, ISOLATION_KEYWORDS):
        return "isolation"
    return "global"


def add_model_group(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.copy()
    grouped["model_group"] = grouped["exercise"].astype(str).apply(classify_exercise)
    return grouped


def load_workouts(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = set(FEATURE_COLUMNS + ["date", "logged_at", TARGET_COLUMN])
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return add_model_group(df)


def _add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    history_df = df.copy()
    history_df["_original_order"] = range(len(history_df))
    history_df["_date_sort"] = pd.to_datetime(history_df["date"], errors="coerce")
    history_df["_logged_at_sort"] = pd.to_datetime(history_df["logged_at"], errors="coerce")
    history_df["exercise"] = history_df["exercise"].fillna("").astype(str)
    history_df["model_group"] = history_df.get("model_group", "global")

    for column in HISTORY_COLUMNS:
        history_df[column] = pd.to_numeric(history_df[column], errors="coerce")

    history_df = history_df.sort_values(
        ["exercise", "_logged_at_sort", "_date_sort", "_original_order"],
        na_position="last",
    )
    exercise_groups = history_df.groupby("exercise", sort=False)

    for lag in range(1, HISTORY_WINDOW + 1):
        for column in HISTORY_COLUMNS:
            history_df[f"previous_{lag}_{column}"] = exercise_groups[column].shift(lag)

    history_df["previous_3_weight_avg"] = exercise_groups["weight"].transform(
        lambda series: series.shift(1).rolling(HISTORY_WINDOW, min_periods=1).mean()
    )
    history_df["previous_3_reps_avg"] = exercise_groups["reps"].transform(
        lambda series: series.shift(1).rolling(HISTORY_WINDOW, min_periods=1).mean()
    )
    history_df["previous_weight_change"] = (
        history_df["previous_1_weight"] - history_df["previous_2_weight"]
    )
    history_df["history_count"] = exercise_groups.cumcount().clip(upper=HISTORY_WINDOW)

    return history_df.sort_values("_original_order").drop(
        columns=["_original_order", "_date_sort", "_logged_at_sort"]
    )


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    features = _add_history_features(add_model_group(df))
    features["exercise"] = features["exercise"].fillna("").astype(str)
    features["model_group"] = features["model_group"].fillna("global").astype(str)

    for column in features.columns:
        if column in {"exercise", "model_group", "date", "logged_at", TARGET_COLUMN, "_dataset"}:
            continue
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)

    features = features.drop(columns=["date", "logged_at", TARGET_COLUMN], errors="ignore")
    return pd.get_dummies(features, columns=["exercise", "model_group"], dtype=float)


def _filter_for_group(df: pd.DataFrame, model_group: str) -> pd.DataFrame:
    if model_group == "global":
        return df.copy()
    return df[df["model_group"] == model_group].copy()


def build_train_test_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_group: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    train_group = _filter_for_group(add_model_group(train_df), model_group).reset_index(drop=True)
    test_group = _filter_for_group(add_model_group(test_df), model_group).reset_index(drop=True)

    train_with_marker = train_group.copy()
    test_with_marker = test_group.copy()
    train_with_marker["_dataset"] = "train"
    test_with_marker["_dataset"] = "test"

    combined_df = pd.concat([train_with_marker, test_with_marker], ignore_index=True)
    combined_features = _prepare_features(combined_df)
    x_train = combined_features[combined_features["_dataset"] == "train"].drop(columns=["_dataset"])
    x_test = combined_features[combined_features["_dataset"] == "test"].drop(columns=["_dataset"])
    y_train = pd.to_numeric(train_group[TARGET_COLUMN], errors="coerce").reset_index(drop=True)
    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    return x_train, x_test, y_train


def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
    valid_rows = y_train.notna()
    if valid_rows.sum() < MIN_TRAIN_ROWS:
        raise ValueError(f"Training data must contain at least {MIN_TRAIN_ROWS} usable rows.")

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(x_train.loc[valid_rows], y_train.loc[valid_rows])
    return model


def save_model_artifacts(
    model: xgb.XGBRegressor,
    feature_columns: list[str],
    model_group: str,
    train_rows: int,
    model_path: Path | str | None = None,
    features_path: Path | str | None = None,
) -> None:
    model_output_path = Path(model_path) if model_path is not None else model_path_for_group(model_group)
    features_output_path = (
        Path(features_path) if features_path is not None else features_path_for_group(model_group)
    )
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    features_output_path.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(str(model_output_path))
    features_output_path.write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "model_group": model_group,
                "train_rows": train_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_model_artifacts(
    model_group: str,
    model_path: Path | str | None = None,
    features_path: Path | str | None = None,
) -> tuple[xgb.XGBRegressor, list[str], dict]:
    resolved_model_path = Path(model_path) if model_path is not None else model_path_for_group(model_group)
    resolved_features_path = (
        Path(features_path) if features_path is not None else features_path_for_group(model_group)
    )
    cache_key = (str(resolved_model_path.resolve()), str(resolved_features_path.resolve()))
    if cache_key in _LOADED_MODEL_ARTIFACTS:
        return _LOADED_MODEL_ARTIFACTS[cache_key]

    if not resolved_model_path.exists():
        raise FileNotFoundError(
            f"XGBoost plus model artifact not found at {resolved_model_path}. "
            "Run `python -m src.ml.train_xgboost_plus` first."
        )
    if not resolved_features_path.exists():
        raise FileNotFoundError(
            f"XGBoost plus feature metadata not found at {resolved_features_path}. "
            "Run `python -m src.ml.train_xgboost_plus` first."
        )

    model = xgb.XGBRegressor()
    model.load_model(str(resolved_model_path))

    metadata = json.loads(resolved_features_path.read_text(encoding="utf-8"))
    artifacts = (model, metadata["feature_columns"], metadata)
    _LOADED_MODEL_ARTIFACTS[cache_key] = artifacts
    return artifacts


def _available_model_group(preferred_group: str) -> str:
    if model_path_for_group(preferred_group).exists() and features_path_for_group(preferred_group).exists():
        return preferred_group
    return "global"


def predict_next_weight_from_history(
    workout_history: pd.DataFrame,
    model_group: str | None = None,
) -> XGBoostPlusPrediction:
    if workout_history.empty:
        raise ValueError("workout_history must include at least the current workout row.")

    current_row = workout_history.iloc[-1]
    preferred_group = model_group or classify_exercise(str(current_row.get("exercise", "")))
    selected_group = _available_model_group(preferred_group)
    model, feature_columns, metadata = load_model_artifacts(selected_group)

    history = add_model_group(workout_history)
    features = _prepare_features(history).reindex(columns=feature_columns, fill_value=0.0)
    current_weight = float(
        pd.to_numeric(pd.Series([current_row.get("weight")]), errors="coerce")
        .fillna(0.0)
        .iloc[0]
    )
    predicted_weight = round(float(model.predict(features.tail(1))[0]), 2)
    return XGBoostPlusPrediction(
        exercise=str(current_row.get("exercise", "")),
        current_weight=current_weight,
        predicted_next_weight=predicted_weight,
        recommended_increment=round(predicted_weight - current_weight, 2),
        model_group=selected_group,
        top_factors=[
            "current workout row",
            "previous 3 same-exercise performances",
            f"{metadata.get('model_group', selected_group)} XGBoost plus regressor",
        ],
        reason=(
            "Predicted from the category-specific XGBoost plus model using current row "
            "features and available same-exercise history."
        ),
    )


def _build_prediction_history(
    row: pd.Series,
    workout_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    current_row = row.to_dict()
    for column in FEATURE_COLUMNS + ["date", "logged_at"]:
        current_row.setdefault(column, "")

    current_df = pd.DataFrame([current_row])
    if workout_history is None or workout_history.empty:
        return current_df

    history = workout_history.copy()
    if "exercise" in history.columns:
        history = history[history["exercise"].astype(str) == str(current_row["exercise"])]

    if "logged_at" in history.columns and current_row.get("logged_at"):
        history_logged_at = pd.to_datetime(history["logged_at"], errors="coerce")
        current_logged_at = pd.to_datetime(current_row["logged_at"], errors="coerce")
        if pd.notna(current_logged_at):
            history = history[history_logged_at < current_logged_at]

    return pd.concat([history, current_df], ignore_index=True, sort=False)


def predict_from_row_xgboost_plus(
    row: pd.Series,
    workout_history: pd.DataFrame | None = None,
) -> XGBoostPlusPrediction:
    prediction_history = _build_prediction_history(row, workout_history)
    return predict_next_weight_from_history(prediction_history)


def train_group_models(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, list[str]]:
    trained_feature_columns: dict[str, list[str]] = {}

    for model_group in MODEL_GROUPS:
        x_train, _, y_train = build_train_test_features(train_df, test_df, model_group)
        model = train_model(x_train, y_train)
        save_model_artifacts(
            model=model,
            feature_columns=x_train.columns.tolist(),
            model_group=model_group,
            train_rows=int(y_train.notna().sum()),
        )
        trained_feature_columns[model_group] = x_train.columns.tolist()

    return trained_feature_columns


def xgboost_plus_predictions_dataframe(
    train_csv: Path | str = DEFAULT_TRAIN_CSV,
    test_csv: Path | str = DEFAULT_TEST_CSV,
) -> pd.DataFrame:
    train_df = load_workouts(train_csv)
    test_df = load_workouts(test_csv)
    train_group_models(train_df, test_df)
    full_history = pd.concat([train_df, test_df], ignore_index=True, sort=False)

    predictions = [
        asdict(predict_from_row_xgboost_plus(row, full_history))
        for _, row in test_df.iterrows()
    ]
    output_df = test_df.drop(columns=["model_group"], errors="ignore").reset_index(drop=True)
    return pd.concat([output_df, pd.DataFrame(predictions)], axis=1)


def save_xgboost_plus_predictions(
    train_csv: Path | str = DEFAULT_TRAIN_CSV,
    test_csv: Path | str = DEFAULT_TEST_CSV,
    output_csv: Path | str = DEFAULT_OUTPUT_CSV,
) -> Path:
    merged = xgboost_plus_predictions_dataframe(train_csv, test_csv)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    output_path = save_xgboost_plus_predictions(DEFAULT_TRAIN_CSV, DEFAULT_TEST_CSV, DEFAULT_OUTPUT_CSV)
    print(f"[DONE] Saved XGBoost plus predictions to {output_path}")


if __name__ == "__main__":
    main()
