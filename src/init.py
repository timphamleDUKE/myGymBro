from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
SEED_WORKOUTS_CSV = PROJECT_ROOT / "data" / "user" / "user_workouts.csv"

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


def load_seed_workouts() -> pd.DataFrame:
    if not SEED_WORKOUTS_CSV.exists():
        return pd.DataFrame(columns=WORKOUT_COLUMNS)

    try:
        workouts = pd.read_csv(SEED_WORKOUTS_CSV)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=WORKOUT_COLUMNS)

    for column in WORKOUT_COLUMNS:
        if column not in workouts.columns:
            workouts[column] = ""

    return workouts[WORKOUT_COLUMNS].copy()


def ensure_data_files() -> None:
    """Initialize session-only demo state.

    User-entered data is intentionally not written to disk so deployed users do
    not share profile, workout, or chat data with each other.
    """
    init_state()


def init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "profile" not in st.session_state:
        st.session_state.profile = {}
    if "workout_logs" not in st.session_state:
        st.session_state.workout_logs = load_seed_workouts()


def load_profile() -> dict:
    init_state()
    return dict(st.session_state.profile)


def save_profile(profile: dict) -> None:
    init_state()
    st.session_state.profile = dict(profile)


def load_chat_history() -> list[dict]:
    init_state()
    return list(st.session_state.chat_history)


def save_chat_history(history: list[dict]) -> None:
    init_state()
    st.session_state.chat_history = list(history)


def append_workout_log(entry: dict) -> None:
    init_state()
    updated = pd.concat(
        [st.session_state.workout_logs, pd.DataFrame([entry])],
        ignore_index=True,
    )
    st.session_state.workout_logs = updated[WORKOUT_COLUMNS].copy()


def load_workout_logs() -> pd.DataFrame:
    init_state()
    return st.session_state.workout_logs.copy()
