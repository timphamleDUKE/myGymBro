from datetime import datetime

import streamlit as st

from core import load_profile, save_profile


st.title("Profile & Goals")
st.write("Set your training context so coaching stays personalized.")

current = load_profile()

with st.form("profile_form"):
    goal = st.text_input("Primary Goal", value=current.get("goal", ""))

    levels = ["Beginner", "Intermediate", "Advanced"]
    selected_level = current.get("experience_level", "Beginner")
    experience_level = st.selectbox(
        "Experience Level",
        levels,
        index=levels.index(selected_level) if selected_level in levels else 0,
    )

    equipment_access = st.multiselect(
        "Equipment Access",
        ["Bodyweight only", "Dumbbells", "Barbell", "Machines", "Bands"],
        default=current.get("equipment_access", []),
    )

    frequency_options = [1, 2, 3, 4, 5, 6, 7]
    saved_frequency = int(current.get("training_frequency", 3))
    training_frequency = st.selectbox(
        "Training Frequency (days/week)",
        frequency_options,
        index=max(0, min(6, saved_frequency - 1)),
    )

    submitted = st.form_submit_button("Save Profile")

if submitted:
    profile = {
        "goal": goal.strip(),
        "experience_level": experience_level,
        "equipment_access": equipment_access,
        "training_frequency": training_frequency,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_profile(profile)
    st.success("Profile/goals saved.")

st.markdown("### Current Saved Profile")
latest = load_profile()
if latest:
    st.json(latest)
else:
    st.info("No profile saved yet.")
