import json
from datetime import datetime, time, timedelta

import streamlit as st

from src.context import prompt_context
from src.init import DATA_DIR, init_state, save_chat_history
from src.reply import stream_ai_response


init_state()

DAILY_PROMPT_LIMIT = 30
PROMPT_USAGE_JSON = DATA_DIR / "prompt_usage.json"


def today_label() -> str:
    return datetime.now().date().isoformat()


def load_prompt_usage() -> dict:
    if not PROMPT_USAGE_JSON.exists():
        return {"date": today_label(), "used": 0}

    try:
        usage = json.loads(PROMPT_USAGE_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"date": today_label(), "used": 0}

    if usage.get("date") != today_label():
        return {"date": today_label(), "used": 0}

    return {
        "date": today_label(),
        "used": max(0, int(usage.get("used", 0))),
    }


def save_prompt_usage(usage: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROMPT_USAGE_JSON.write_text(json.dumps(usage, indent=2), encoding="utf-8")


def can_send_prompt() -> tuple[bool, int]:
    usage = load_prompt_usage()
    remaining = DAILY_PROMPT_LIMIT - usage["used"]
    return remaining > 0, remaining


def record_prompt() -> None:
    usage = load_prompt_usage()
    usage["used"] += 1
    save_prompt_usage(usage)


def reset_time_label() -> str:
    tomorrow = datetime.now().date() + timedelta(days=1)
    next_midnight = datetime.combine(tomorrow, time.min)
    return next_midnight.strftime("%I:%M %p").lstrip("0")


st.title("Coach Chat")
st.write("Chat session is persisted locally for your next run.")

left, right = st.columns([3, 1])

with right:
    allowed, remaining = can_send_prompt()

    st.subheader("Session")
    st.write(f"Messages: {len(st.session_state.chat_history)}")
    st.metric("Prompts left today", f"{max(remaining, 0)} / {DAILY_PROMPT_LIMIT}")
    st.caption(f"Resets at {reset_time_label()}")

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        save_chat_history([])
        st.success("History cleared.")

with left:
    messages_container = st.container()

    with messages_container:
        for message in st.session_state.chat_history:
            role = message.get("role", "assistant")
            avatar = ":material/exercise:" if role == "assistant" else ":material/person:"
            with st.chat_message(role, avatar=avatar):
                st.write(message.get("content", ""))

    user_input = st.chat_input("Ask GymBro...")

    if user_input:
        allowed, remaining = can_send_prompt()

        if not allowed:
            st.warning(f"Daily prompt limit reached. Resets at {reset_time_label()}.")
            st.stop()

        record_prompt()

        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with messages_container:
            with st.chat_message("user", avatar=":material/person:"):
                st.write(user_input)

            timestamp = datetime.now().strftime("%H:%M")

            with st.spinner("GymBro is thinking..."):
                user_input_prompt_context = prompt_context(user_input)
                chat_history_before_reply = st.session_state.chat_history[:-1]

            with st.chat_message("assistant", avatar=":material/exercise:"):
                response_placeholder = st.empty()
                full_text = f"[{timestamp}]: "
                response_placeholder.markdown(full_text)

                for chunk in stream_ai_response(
                    user_prompt=user_input,
                    context=user_input_prompt_context,
                    chat_history=chat_history_before_reply,
                ):
                    full_text += chunk
                    response_placeholder.markdown(full_text)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": full_text.strip()}
        )
        save_chat_history(st.session_state.chat_history)
        st.rerun()
