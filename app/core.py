from datetime import datetime
import json
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
WORKOUTS_CSV = DATA_DIR / "workouts.csv"
PROFILE_JSON = DATA_DIR / "profile.json"
CHAT_JSON = DATA_DIR / "chat_history.json"

WORKOUT_COLUMNS = [
    "exercise",
    "sets",
    "reps",
    "weight",
    "date",
    "rpe",
    "rir",
    "logged_at",
]


def ensure_data_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not WORKOUTS_CSV.exists():
        pd.DataFrame(columns=WORKOUT_COLUMNS).to_csv(WORKOUTS_CSV, index=False)

    if not PROFILE_JSON.exists():
        PROFILE_JSON.write_text(json.dumps({}), encoding="utf-8")

    if not CHAT_JSON.exists():
        CHAT_JSON.write_text(json.dumps([]), encoding="utf-8")


def init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()


def load_profile() -> dict:
    if not PROFILE_JSON.exists():
        return {}
    try:
        return json.loads(PROFILE_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_profile(profile: dict) -> None:
    PROFILE_JSON.write_text(json.dumps(profile, indent=2), encoding="utf-8")


def load_chat_history() -> list[dict]:
    if not CHAT_JSON.exists():
        return []
    try:
        data = json.loads(CHAT_JSON.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def save_chat_history(history: list[dict]) -> None:
    CHAT_JSON.write_text(json.dumps(history, indent=2), encoding="utf-8")


def append_workout_log(entry: dict) -> None:
    if WORKOUTS_CSV.exists():
        df = pd.read_csv(WORKOUTS_CSV)
    else:
        df = pd.DataFrame(columns=WORKOUT_COLUMNS)
    updated = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    updated.to_csv(WORKOUTS_CSV, index=False)


def load_workout_logs() -> pd.DataFrame:
    if not WORKOUTS_CSV.exists():
        return pd.DataFrame(columns=WORKOUT_COLUMNS)
    return pd.read_csv(WORKOUTS_CSV)


def placeholder_coach_reply(user_message: str) -> str:
    timestamp = datetime.now().strftime("%H:%M")
    return (
        f"[{timestamp}] Coach placeholder: got your message '{user_message}'. "
        "Recommendation logic will be added in a future milestone."
    )
