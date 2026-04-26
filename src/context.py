import re

import pandas as pd

from src.init import load_profile, load_workout_logs
from src.ml.baseline import predict_from_row_baseline
from src.ml.train_xgboost import predict_from_row_xgboost
from src.rag.retrieve_context import retrieve_rag_context

TOP_K_RAG_CONTEXT = 3
XGBOOST_MIN_WORKOUT_ROWS = 500


def _is_prediction_question(user_message: str) -> bool:
    lowered = user_message.lower()
    prediction_markers = (
        "what should i",
        "what weight should i",
        "next weight",
        "next session",
        "next workout",
        "what should i bench",
        "what should i squat",
        "what should i deadlift",
    )
    return any(marker in lowered for marker in prediction_markers)


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _lift_aliases() -> dict[str, tuple[str, ...]]:
    return {
        "bench": ("bench", "bench press", "paused bench press", "close grip bench press", "spoto press"),
        "squat": ("squat", "back squat", "front squat", "box squat", "pause squat", "hack squat"),
        "deadlift": ("deadlift", "romanian deadlift", "rdl", "sumo deadlift", "conventional deadlift", "rack pull"),
        "press": ("overhead press", "strict press", "push press", "shoulder press"),
    }


def _load_user_workouts() -> pd.DataFrame:
    df = load_workout_logs()

    if df.empty:
        return df

    if "logged_at" in df.columns:
        df["logged_at"] = pd.to_datetime(df["logged_at"], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def _load_user_profile() -> dict:
    return load_profile()


def _find_target_workout_row(user_message: str, workouts: pd.DataFrame) -> pd.Series | None:
    if workouts.empty or "exercise" not in workouts.columns:
        return None

    normalized_query = _normalize_text(user_message)
    query_tokens = set(normalized_query.split())
    scored_rows: list[tuple[int, int, pd.Series]] = []

    for _, row in workouts.iterrows():
        exercise = str(row.get("exercise", "")).strip()
        if not exercise:
            continue
        normalized_exercise = _normalize_text(exercise)
        exercise_tokens = set(normalized_exercise.split())
        overlap = len(exercise_tokens & query_tokens)

        if normalized_exercise and normalized_exercise in normalized_query:
            overlap += 3
        elif exercise_tokens and not overlap:
            continue

        timestamp = row.get("logged_at")
        sortable_timestamp = 0
        if pd.notna(timestamp):
            sortable_timestamp = int(timestamp.timestamp())

        scored_rows.append((overlap, sortable_timestamp, row))

    if scored_rows:
        scored_rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored_rows[0][2]

    # Second pass: use common lift aliases if plain token overlap failed.
    for intent_token, aliases in _lift_aliases().items():
        if intent_token not in query_tokens:
            continue

        alias_matches = workouts[
            workouts["exercise"].astype(str).str.lower().apply(
                lambda exercise_name: any(alias in exercise_name for alias in aliases)
            )
        ]
        if not alias_matches.empty:
            alias_matches = alias_matches.sort_values(["logged_at", "date"], ascending=False, na_position="last")
            return alias_matches.iloc[0]

    # Final fallback: use the most recent logged workout row.
    latest = workouts.sort_values(["logged_at", "date"], ascending=False, na_position="last")
    return latest.iloc[0] if not latest.empty else None


def _format_prediction_block(prediction: dict, model_name: str) -> str:
    exercise = prediction.get("exercise", "exercise")
    next_weight = prediction.get("predicted_next_weight")
    reason = prediction.get("reason", "")

    return (
        f"- {model_name} predicts next {exercise}: {next_weight} lb\n"
        f"- Reason: {reason}"
    )


def _format_date(value: object) -> str:
    if pd.isna(value) or value == "":
        return "unknown date"
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%Y-%m-%d")


def _format_number(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "unknown"
    if float(number).is_integer():
        return str(int(number))
    return f"{float(number):.1f}".rstrip("0").rstrip(".")


def _format_rag_contexts(rag_contexts: list[dict]) -> str:
    lines = ["Relevant training context:"]

    for rank, ctx in enumerate(rag_contexts, start=1):
        title = ctx.get("title") or ctx.get("source_name") or "Untitled"
        source = ctx.get("website") or ctx.get("source_file") or "unknown"
        url = ctx.get("url") or "unknown"
        excerpt = re.sub(r"\s+", " ", ctx.get("text", "")).strip()[:350]

        lines.append(
            f"[{rank}] Title: {title}\n"
            f"Source: {source}\n"
            f"URL: {url}\n"
            f"Excerpt: {excerpt}"
        )

    return "\n\n".join(lines)


def _most_recent_matching_workout(workouts: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series | None:
    if workouts.empty or "exercise" not in workouts.columns:
        return None

    matches = workouts[
        workouts["exercise"].astype(str).str.lower().apply(
            lambda exercise_name: any(alias in exercise_name for alias in aliases)
        )
    ]
    if matches.empty:
        return None

    return matches.sort_values(["logged_at", "date"], ascending=False, na_position="last").iloc[0]


def _format_user_workout_context(workouts: pd.DataFrame) -> str:
    lines = ["User workout context:"]

    if workouts.empty:
        lines.append("- No logged workout history found.")
        return "\n".join(lines)

    seen_exercises: set[str] = set()
    for _, aliases in _lift_aliases().items():
        row = _most_recent_matching_workout(workouts, aliases)
        if row is None:
            continue

        exercise = str(row.get("exercise", "exercise")).strip()
        normalized_exercise = _normalize_text(exercise)
        if normalized_exercise in seen_exercises:
            continue
        seen_exercises.add(normalized_exercise)

        weight = _format_number(row.get("weight"))
        reps = _format_number(row.get("reps"))
        date = _format_date(row.get("date") or row.get("logged_at"))
        lines.append(f"- Most recent {exercise}: {weight} lb x {reps} reps on {date}")

    if len(lines) == 1:
        latest = workouts.sort_values(["logged_at", "date"], ascending=False, na_position="last").head(3)
        for _, row in latest.iterrows():
            exercise = str(row.get("exercise", "exercise")).strip()
            weight = _format_number(row.get("weight"))
            reps = _format_number(row.get("reps"))
            date = _format_date(row.get("date") or row.get("logged_at"))
            lines.append(f"- Recent {exercise}: {weight} lb x {reps} reps on {date}")

    return "\n".join(lines)


def _format_user_profile_context(profile: dict) -> str:
    lines = ["User profile context:"]

    if not profile:
        lines.append("- No user profile found.")
        return "\n".join(lines)

    labels = {
        "goal": "Goal",
        "experience_level": "Experience level",
        "equipment_access": "Equipment access",
        "training_frequency": "Training frequency",
    }

    for key, label in labels.items():
        value = profile.get(key)
        if value in (None, "", []):
            continue
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        elif key == "training_frequency":
            value = f"{value} days per week"
        lines.append(f"- {label}: {value}")

    if len(lines) == 1:
        lines.append("- No usable user profile fields found.")

    return "\n".join(lines)


def _build_prompt_context(user_message: str) -> str:
    try:
        rag_contexts = retrieve_rag_context(user_message, top_k=TOP_K_RAG_CONTEXT)
    except Exception as exc:
        return f"RAG index unavailable ({exc})."

    prompt_sections = []
    if rag_contexts:
        prompt_sections.append(_format_rag_contexts(rag_contexts))
    else:
        prompt_sections.append("Relevant training context:\nNo relevant training context yet.")

    workouts = _load_user_workouts()
    prompt_sections.append(_format_user_workout_context(workouts))

    profile = _load_user_profile()
    prompt_sections.append(_format_user_profile_context(profile))

    if _is_prediction_question(user_message):
        target_row = _find_target_workout_row(user_message, workouts)
        if target_row is not None:
            if len(workouts) > XGBOOST_MIN_WORKOUT_ROWS:
                try:
                    prediction = predict_from_row_xgboost(target_row, workouts)
                    prediction_context = _format_prediction_block(prediction.__dict__, "XGBoost")
                except Exception as exc:
                    prediction = predict_from_row_baseline(target_row)
                    prediction_context = _format_prediction_block(prediction.__dict__, "Baseline")
                    prediction_context += f"\n- XGBoost prediction unavailable: {exc}"
            else:
                prediction = predict_from_row_baseline(target_row)
                prediction_context = _format_prediction_block(prediction.__dict__, "Baseline")
        else:
            prediction_context = "- I could not find a matching logged exercise to estimate the next weight."
        prompt_sections.append(f"Prediction context:\n{prediction_context}")
    else:
        prompt_sections.append("Prediction context:\n- No next-weight prediction requested.")
    return "\n\n".join(prompt_sections)


def prompt_context(user_message: str) -> str:
    return _build_prompt_context(user_message)
    
