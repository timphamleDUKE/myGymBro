from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DEFAULT_TEST_CSV = PROJECT_ROOT / "data" / "processed" / "test_workouts.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "test" / "baseline_predictions.csv"


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

    # If the user logs bodyweight-only work with no external load, keep it unchanged.
    if current_weight <= 0:
        return BaselinePrediction(
            exercise=exercise,
            current_weight=float(current_weight),
            predicted_next_weight=float(current_weight),
            recommended_increment=0.0,
            reason="No external load logged, so the baseline keeps weight unchanged.",
        )

    multiplier = 1.0
    reason = "Completed the session, so apply a standard progressive overload increase."

    if rir is not None:
        if rir <= 0:
            multiplier = 0.0
            reason = "No reps left in reserve, so hold the weight steady next session."
        elif rir == 1:
            multiplier = 0.5
            reason = "Only one rep in reserve, so use a smaller increase next session."
        elif rir >= 3:
            multiplier = 1.25
            reason = "Plenty of reps in reserve, so take a slightly larger increase."
    elif rpe is not None:
        if rpe >= 9.5:
            multiplier = 0.0
            reason = "Very high effort set, so hold the weight steady next session."
        elif rpe >= 8.5:
            multiplier = 0.5
            reason = "Hard effort set, so use a smaller increase next session."
        elif rpe <= 7:
            multiplier = 1.25
            reason = "Lower effort set, so take a slightly larger increase."
    else:
        # Fall back to rep-based heuristics when perceived effort is unavailable.
        if reps <= 3 and sets <= 2:
            multiplier = 0.5
            reason = "Low-rep work usually progresses more conservatively, so use a smaller increase."
        elif reps >= 10:
            multiplier = 1.0
            reason = "Higher-rep work completed successfully, so apply a standard increase."

    increment = round(base_increment * multiplier, 2)
    predicted = round(float(current_weight) + increment, 2)

    return BaselinePrediction(
        exercise=exercise,
        current_weight=float(current_weight),
        predicted_next_weight=predicted,
        recommended_increment=increment,
        reason=reason,
    )


def predict_from_row(row: pd.Series) -> BaselinePrediction:
    return predict_next_weight(
        exercise=str(row["exercise"]),
        current_weight=float(row["weight"]),
        reps=int(row["reps"]),
        sets=int(row["sets"]),
        rpe=_parse_optional_float(row.get("rpe")),
        rir=_parse_optional_float(row.get("rir")),
    )


def batch_predict(df: pd.DataFrame) -> pd.DataFrame:
    predictions = [asdict(predict_from_row(row)) for _, row in df.iterrows()]
    return pd.DataFrame(predictions)


def main() -> None:
    df = load_workouts(DEFAULT_TEST_CSV)
    predictions = batch_predict(df)
    merged = pd.concat([df.reset_index(drop=True), predictions], axis=1)
    merged.to_csv(DEFAULT_OUTPUT_CSV, index=False)
    print(f"[DONE] Saved baseline predictions to {DEFAULT_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
