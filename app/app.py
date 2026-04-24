import streamlit as st
from pathlib import Path
from src.init import ensure_data_files, init_state

st.set_page_config(page_title="myGymBro", page_icon=":material/exercise:", layout="wide")

st.logo(":material/exercise:", size="large")

dashboard_page = st.Page("pages/1_Dashboard.py", title="Dashboard")
workout_logger_page = st.Page("pages/2_Workout_Logger.py", title="Workout Logger")
profile_goals_page = st.Page("pages/3_Profile_Goals.py", title="Profile & Goals")
coach_chat_page = st.Page("pages/4_Coach_Chat.py", title="Coach Chat")

ensure_data_files()
init_state()

navigation = st.navigation(
    [dashboard_page, workout_logger_page, profile_goals_page, coach_chat_page]
)
navigation.run()