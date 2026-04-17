from datetime import datetime

import streamlit as st

from core import load_profile, save_profile


st.title("Profile & Goals")
st.write("Set your training context so coaching stays personalized.")

st.markdown("### Current Saved Profile")
latest = load_profile()
if latest:
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Primary Goal", latest.get("goal", "Not set") or "Not set")
        st.metric("Experience Level", latest.get("experience_level", "Not set"))

    with col2:
        st.metric(
            "Training Frequency",
            f"{latest.get('training_frequency', 'Not set')} days/week"
            if latest.get("training_frequency")
            else "Not set",
        )
        st.metric(
            "Last Updated",
            latest.get("updated_at", "Not available")
        )

    st.markdown("#### Equipment Access")
    equipment = latest.get("equipment_access", [])

    if equipment:
        cols = st.columns(len(equipment))
        for i, item in enumerate(equipment):
            cols[i].success(item)
    else:
        st.info("No equipment selected.")
else:
    st.info("No profile saved yet.")

st.divider()

current = load_profile()

with st.form("profile_form"):

    st.subheader("Save Profile")
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
    save_profile({
        "goal": goal.strip(),
        "experience_level": experience_level,
        "equipment_access": equipment_access,
        "training_frequency": training_frequency,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    })

    st.session_state["profile_saved"] = True
    st.rerun()

# show message AFTER rerun
if st.session_state.get("profile_saved"):
    st.success("Profile/goals saved.")
    del st.session_state["profile_saved"]