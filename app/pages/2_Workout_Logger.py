from datetime import date, datetime

import streamlit as st

from core import append_workout_log, load_workout_logs


st.title("Workout Logger")
st.write("Log exercise, volume, and intensity details.")

with st.form("workout_log_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    exercise = col1.text_input("Exercise", placeholder="e.g., Back Squat")
    sets = col1.number_input("Sets", min_value=1, step=1, value=3)
    reps = col1.number_input("Reps", min_value=1, step=1, value=8)
    weight = col1.number_input("Weight", min_value=0.0, step=2.5, value=0.0)

    workout_date = col2.date_input("Date", value=date.today())
    rpe = col2.number_input(
        "RPE (optional)", min_value=0.0, max_value=10.0, step=0.5, value=0.0
    )
    rir = col2.number_input("RIR (optional)", min_value=0, max_value=10, step=1, value=0)

    submitted = st.form_submit_button("Save Workout")

if submitted:
    if not exercise.strip():
        st.error("Exercise is required.")
    else:
        entry = {
            "exercise": exercise.strip(),
            "sets": int(sets),
            "reps": int(reps),
            "weight": float(weight),
            "date": str(workout_date),
            "rpe": float(rpe) if rpe > 0 else "",
            "rir": int(rir) if rir > 0 else "",
            "logged_at": datetime.now().isoformat(timespec="seconds"),
        }
        append_workout_log(entry)
        st.success("Workout saved.")

st.markdown("### Saved Workouts")
logs = load_workout_logs()
if logs.empty:
    st.info("No workouts logged yet.")
else:
    st.dataframe(logs.sort_values(by="date", ascending=False), use_container_width=True)
