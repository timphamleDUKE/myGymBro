from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DEFAULT_TEST_CSV = PROJECT_ROOT / "data" / "processed" / "test_workouts.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "src" / "test" / "baseline" / "baseline_predictions.csv"


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
class BaselinePrediction:
    exercise: str
    current_weight: float
    predicted_next_weight: float
    recommended_increment: float
    top_factors: list[str]
    reason: str
    model_type: str = "baseline_progressive_overload"


def load_workouts(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {"exercise", "sets", "reps", "weight"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def _normalize_exercise_name(exercise: str) -> str:
    return str(exercise).strip().lower()


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def infer_base_increment(exercise: str, current_weight: float) -> float:
    name = _normalize_exercise_name(exercise)

    if _contains_any(name, LOWER_COMPOUND_KEYWORDS):
        return 10.0 if current_weight >= 225 else 5.0

    if _contains_any(name, UPPER_COMPOUND_KEYWORDS):
        return 5.0 if current_weight >= 135 else 2.5

    if _contains_any(name, ISOLATION_KEYWORDS):
        return 2.5

    if _contains_any(name, BODYWEIGHT_KEYWORDS):
        return 2.5 if current_weight > 0 else 0.0

    return 2.5


def _parse_optional_float(value: object) -> float | None:
    if pd.isna(value) or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def predict_next_weight(
    exercise: str,
    current_weight: float,
    reps: int,
    sets: int,
    rpe: float | None = None,
    rir: float | None = None,
) -> BaselinePrediction:
    base_increment = infer_base_increment(exercise, current_weight)
    top_factors: list[str] = []

    # If the user logs bodyweight-only work with no external load, keep it unchanged.
    if current_weight <= 0:
        return BaselinePrediction(
            exercise=exercise,
            current_weight=float(current_weight),
            predicted_next_weight=float(current_weight),
            recommended_increment=0.0,
            top_factors=[
                "no external load logged",
                "bodyweight or unloaded movement",
            ],
            reason="No external load logged, so the baseline keeps weight unchanged.",
        )

    multiplier = 1.0
    reason = "Completed the session, so apply a standard progressive overload increase."
    top_factors.append(f"base increment inferred as {base_increment}")

    if rir is not None:
        top_factors.append(f"rir={rir}")
        if rir <= 0:
            multiplier = 0.0
            top_factors.append("no reps in reserve")
            reason = "No reps left in reserve, so hold the weight steady next session."
        elif rir == 1:
            multiplier = 0.5
            top_factors.append("one rep in reserve")
            reason = "Only one rep in reserve, so use a smaller increase next session."
        elif rir >= 3:
            multiplier = 1.25
            top_factors.append("high reps in reserve")
            reason = "Plenty of reps in reserve, so take a slightly larger increase."
    elif rpe is not None:
        top_factors.append(f"rpe={rpe}")
        if rpe >= 9.5:
            multiplier = 0.0
            top_factors.append("very high effort")
            reason = "Very high effort set, so hold the weight steady next session."
        elif rpe >= 8.5:
            multiplier = 0.5
            top_factors.append("hard effort")
            reason = "Hard effort set, so use a smaller increase next session."
        elif rpe <= 7:
            multiplier = 1.25
            top_factors.append("lower effort")
            reason = "Lower effort set, so take a slightly larger increase."
    else:
        # Fall back to rep-based heuristics when perceived effort is unavailable.
        top_factors.append("missing RPE/RIR, using reps/sets fallback")
        if reps <= 3 and sets <= 2:
            multiplier = 0.5
            top_factors.append("low-rep low-set session")
            reason = "Low-rep work usually progresses more conservatively, so use a smaller increase."
        elif reps >= 10:
            multiplier = 1.0
            top_factors.append("higher-rep session completed")
            reason = "Higher-rep work completed successfully, so apply a standard increase."
        else:
            top_factors.append("standard completed session")

    increment = round(base_increment * multiplier, 2)
    predicted = round(float(current_weight) + increment, 2)
    top_factors.append(f"increment applied={increment}")

    return BaselinePrediction(
        exercise=exercise,
        current_weight=float(current_weight),
        predicted_next_weight=predicted,
        recommended_increment=increment,
        top_factors=top_factors,
        reason=reason,
    )

# Used during inference on test set and for generating predictions for the Coach Chat page
def predict_from_row_baseline(row: pd.Series) -> BaselinePrediction:
    return predict_next_weight(
        exercise=str(row["exercise"]),
        current_weight=float(row["weight"]),
        reps=int(row["reps"]),
        sets=int(row["sets"]),
        rpe=_parse_optional_float(row.get("rpe")),
        rir=_parse_optional_float(row.get("rir")),
    )


def batch_predict(df: pd.DataFrame) -> pd.DataFrame:
    predictions = [asdict(predict_from_row_baseline(row)) for _, row in df.iterrows()]
    predictions_df = pd.DataFrame(predictions)
    export_columns = [
        "predicted_next_weight",
        "recommended_increment",
        "top_factors",
        "reason",
        "model_type",
    ]
    return predictions_df[export_columns].copy()


def next_weight_baseline(input_csv: Path | str = DEFAULT_TEST_CSV) -> list[dict[str, Any]]:
    df = load_workouts(input_csv)
    predictions = [asdict(predict_from_row_baseline(row)) for _, row in df.iterrows()]
    return predictions


def baseline_predictions_dataframe(input_csv: Path | str = DEFAULT_TEST_CSV) -> pd.DataFrame:
    df = load_workouts(input_csv)
    predictions = batch_predict(df)
    return pd.concat([df.reset_index(drop=True), predictions], axis=1)


def save_baseline_predictions(
    input_csv: Path | str = DEFAULT_TEST_CSV,
    output_csv: Path | str = DEFAULT_OUTPUT_CSV,
) -> Path:
    merged = baseline_predictions_dataframe(input_csv)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    output_path = save_baseline_predictions(DEFAULT_TEST_CSV, DEFAULT_OUTPUT_CSV)
    print(f"[DONE] Saved baseline predictions to {output_path}")


if __name__ == "__main__":
    main()
