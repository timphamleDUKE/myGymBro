import streamlit as st
from pathlib import Path
from core import ensure_data_files, init_state

dashboard_page = st.Page("pages/1_Dashboard.py", title="Dashboard")
workout_logger_page = st.Page("pages/2_Workout_Logger.py", title="Workout Logger")
profile_goals_page = st.Page("pages/3_Profile_Goals.py", title="Profile & Goals")
coach_chat_page = st.Page("pages/4_Coach_Chat.py", title="Coach Chat")

st.set_page_config(page_title="myGymBro", page_icon="💪", layout="wide")
ensure_data_files()
init_state()

logo_path = Path(__file__).resolve().parent / "assets" / "logo.png"
if logo_path.exists():
    st.logo(str(logo_path), size="large")

navigation = st.navigation(
    [dashboard_page, workout_logger_page, profile_goals_page, coach_chat_page]
)
navigation.run()