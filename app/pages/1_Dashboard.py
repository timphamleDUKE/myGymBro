import streamlit as st

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
    st.dataframe(workouts.tail(5), use_container_width=True, hide_index=True)

st.markdown("### Saved Goals Snapshot")
if profile:
    st.json(profile)
else:
    st.info("No profile/goals saved yet.")
