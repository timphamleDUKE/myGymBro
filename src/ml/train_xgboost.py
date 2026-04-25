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
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "src" / "test" / "xgboost" / "xgboost_predictions.csv"
DEFAULT_MODEL_DIR = BASE_DIR / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "xgboost_next_weight.json"
DEFAULT_FEATURES_PATH = DEFAULT_MODEL_DIR / "xgboost_next_weight_features.json"

FEATURE_COLUMNS = ["exercise", "sets", "reps", "weight", "rpe", "rir"]
TARGET_COLUMN = "actual_next_weight"
HISTORY_COLUMNS = ["sets", "reps", "weight", "rpe", "rir"]
HISTORY_WINDOW = 3
_LOADED_MODEL_ARTIFACTS: dict[tuple[str, str], tuple[xgb.XGBRegressor, list[str]]] = {}


@dataclass
class XGBoostPrediction:
    exercise: str
    current_weight: float
    predicted_next_weight: float
    recommended_increment: float
    top_factors: list[str]
    reason: str
    model_type: str = "xgboost_regressor"


def load_workouts(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = set(FEATURE_COLUMNS + ["date", "logged_at", TARGET_COLUMN])
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def _add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    history_df = df.copy()
    history_df["_original_order"] = range(len(history_df))
    history_df["_date_sort"] = pd.to_datetime(history_df["date"], errors="coerce")
    history_df["_logged_at_sort"] = pd.to_datetime(history_df["logged_at"], errors="coerce")
    history_df["exercise"] = history_df["exercise"].fillna("").astype(str)

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
    features = _add_history_features(df)
    features["exercise"] = features["exercise"].fillna("").astype(str)

    for column in features.columns:
        if column in {"exercise", "date", "logged_at", TARGET_COLUMN, "_dataset"}:
            continue
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)

    features = features.drop(columns=["date", "logged_at", TARGET_COLUMN], errors="ignore")
    return pd.get_dummies(features, columns=["exercise"], dtype=float)


def build_train_test_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    train_with_marker = train_df.copy()
    test_with_marker = test_df.copy()
    train_with_marker["_dataset"] = "train"
    test_with_marker["_dataset"] = "test"

    combined_df = pd.concat([train_with_marker, test_with_marker], ignore_index=True)
    combined_features = _prepare_features(combined_df)
    x_train = combined_features[combined_features["_dataset"] == "train"].drop(columns=["_dataset"])
    x_test = combined_features[combined_features["_dataset"] == "test"].drop(columns=["_dataset"])
    y_train = pd.to_numeric(train_df[TARGET_COLUMN], errors="coerce")
    return x_train, x_test, y_train


def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
    valid_rows = y_train.notna()
    if not valid_rows.any():
        raise ValueError("Training data does not contain any usable target values.")

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
    model_path: Path | str = DEFAULT_MODEL_PATH,
    features_path: Path | str = DEFAULT_FEATURES_PATH,
) -> None:
    model_output_path = Path(model_path)
    features_output_path = Path(features_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    features_output_path.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(str(model_output_path))
    features_output_path.write_text(
        json.dumps({"feature_columns": feature_columns}, indent=2),
        encoding="utf-8",
    )


def load_model_artifacts(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    features_path: Path | str = DEFAULT_FEATURES_PATH,
) -> tuple[xgb.XGBRegressor, list[str]]:
    resolved_model_path = Path(model_path)
    resolved_features_path = Path(features_path)
    cache_key = (str(resolved_model_path.resolve()), str(resolved_features_path.resolve()))
    if cache_key in _LOADED_MODEL_ARTIFACTS:
        return _LOADED_MODEL_ARTIFACTS[cache_key]

    if not resolved_model_path.exists():
        raise FileNotFoundError(
            f"XGBoost model artifact not found at {resolved_model_path}. "
            "Run `python -m src.ml.train_xgboost` first."
        )
    if not resolved_features_path.exists():
        raise FileNotFoundError(
            f"XGBoost feature metadata not found at {resolved_features_path}. "
            "Run `python -m src.ml.train_xgboost` first."
        )

    model = xgb.XGBRegressor()
    model.load_model(str(resolved_model_path))

    metadata = json.loads(resolved_features_path.read_text(encoding="utf-8"))
    artifacts = (model, metadata["feature_columns"])
    _LOADED_MODEL_ARTIFACTS[cache_key] = artifacts
    return artifacts


def predict_next_weight_from_history(
    workout_history: pd.DataFrame,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    features_path: Path | str = DEFAULT_FEATURES_PATH,
) -> XGBoostPrediction:
    if workout_history.empty:
        raise ValueError("workout_history must include at least the current workout row.")

    model, feature_columns = load_model_artifacts(model_path, features_path)
    features = _prepare_features(workout_history).reindex(columns=feature_columns, fill_value=0.0)
    current_row = workout_history.iloc[-1]
    current_weight = float(
        pd.to_numeric(pd.Series([current_row.get("weight")]), errors="coerce")
        .fillna(0.0)
        .iloc[0]
    )
    predicted_weight = round(float(model.predict(features.tail(1))[0]), 2)
    return XGBoostPrediction(
        exercise=str(current_row.get("exercise", "")),
        current_weight=current_weight,
        predicted_next_weight=predicted_weight,
        recommended_increment=round(predicted_weight - current_weight, 2),
        top_factors=[
            "current workout row",
            "previous 3 same-exercise performances",
            "trained XGBoost regressor",
        ],
        reason="Predicted from the trained XGBoost model using current row features and available exercise history.",
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


def predict_from_row_xgboost(
    row: pd.Series,
    workout_history: pd.DataFrame | None = None,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    features_path: Path | str = DEFAULT_FEATURES_PATH,
) -> XGBoostPrediction:
    prediction_history = _build_prediction_history(row, workout_history)
    return predict_next_weight_from_history(prediction_history, model_path, features_path)


def xgboost_predictions_dataframe(
    train_csv: Path | str = DEFAULT_TRAIN_CSV,
    test_csv: Path | str = DEFAULT_TEST_CSV,
) -> pd.DataFrame:
    train_df = load_workouts(train_csv)
    test_df = load_workouts(test_csv)

    x_train, x_test, y_train = build_train_test_features(train_df, test_df)
    model = train_model(x_train, y_train)
    save_model_artifacts(model, x_train.columns.tolist())
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0.0)

    predictions = [
        asdict(
            XGBoostPrediction(
                exercise=str(row["exercise"]),
                current_weight=float(row["weight"]),
                predicted_next_weight=round(float(value), 2),
                recommended_increment=round(float(value) - float(row["weight"]), 2),
                top_factors=[
                    "current workout row",
                    "previous 3 same-exercise performances",
                    "trained XGBoost regressor",
                ],
                reason="Predicted from the trained XGBoost model using current row features and available exercise history.",
            )
        )
        for value, (_, row) in zip(model.predict(x_test), test_df.iterrows())
    ]
    return pd.concat([test_df.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)


def save_xgboost_predictions(
    train_csv: Path | str = DEFAULT_TRAIN_CSV,
    test_csv: Path | str = DEFAULT_TEST_CSV,
    output_csv: Path | str = DEFAULT_OUTPUT_CSV,
) -> Path:
    merged = xgboost_predictions_dataframe(train_csv, test_csv)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    output_path = save_xgboost_predictions(DEFAULT_TRAIN_CSV, DEFAULT_TEST_CSV, DEFAULT_OUTPUT_CSV)
    print(f"[DONE] Saved XGBoost predictions to {output_path}")


if __name__ == "__main__":
    main()
