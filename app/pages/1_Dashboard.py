import streamlit as st
import pandas as pd

from src.init import DATA_DIR, load_profile, load_workout_logs


st.title("Dashboard")
st.write("Quick overview of your fitness data and coach chat history.")

profile = load_profile()
workouts = load_workout_logs()
messages = st.session_state.chat_history

col1, col2, col3 = st.columns(3)
col1.metric("Workout logs", len(workouts))
col2.metric("Chat messages", len(messages))
col3.metric("Profile saved", "Yes" if profile else "No")

st.markdown("### Latest Workout Logs")
if workouts.empty:
    st.info("No workouts saved yet. Log your first session in Workout Logger.")
else:
    sorted_workouts = workouts.copy()
    sorted_workouts["_sort_date"] = pd.to_datetime(sorted_workouts["date"], errors="coerce")
    sorted_workouts = sorted_workouts.sort_values("_sort_date", ascending=False).drop(columns="_sort_date")
    st.dataframe(sorted_workouts.head(5), use_container_width=True, hide_index=True)

st.markdown("### Current Saved Profile")

if profile:
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Primary Goal", profile.get("goal", "Not set") or "Not set")
        st.metric("Experience Level", profile.get("experience_level", "Not set"))

    with col2:
        st.metric(
            "Training Frequency",
            f"{profile.get('training_frequency', 'Not set')} days/week"
            if profile.get("training_frequency")
            else "Not set",
        )
        st.metric(
            "Last Updated",
            profile.get("updated_at", "Not available")
        )

    st.markdown("#### Equipment Access")
    equipment = profile.get("equipment_access", [])

    if equipment:
        cols = st.columns(len(equipment))
        for i, item in enumerate(equipment):
            cols[i].success(item)
    else:
        st.info("No equipment selected.")
else:
    st.info("No profile saved yet.")
