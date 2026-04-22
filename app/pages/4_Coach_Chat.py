import streamlit as st
import time
from core import init_state, placeholder_coach_reply, save_chat_history


init_state()


st.title("Coach Chat")
st.write("Chat session is persisted locally for your next run.")

left, right = st.columns([3, 1])

with right:
    st.subheader("Session")
    st.write(f"Messages: {len(st.session_state.chat_history)}")
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        save_chat_history([])
        st.success("History cleared.")

with left:
    messages_container = st.container()

    with messages_container:
        for message in st.session_state.chat_history:
            role = message.get("role", "assistant")
            avatar = ":material/exercise:" if role == "assistant" else "👤"
            with st.chat_message(role, avatar=avatar):
                st.write(message.get("content", ""))

    user_input = st.chat_input("Ask your coach...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with messages_container:
            with st.chat_message("user", avatar="👤"):
                st.write(user_input)

            reply = placeholder_coach_reply(user_input)

            with st.chat_message("assistant", avatar=":material/exercise:"):
                response_placeholder = st.empty()
                full_text = ""

                for word in reply.split():
                    full_text += word + " "
                    response_placeholder.write(full_text.strip())
                    time.sleep(0.03)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": full_text.strip()}
        )
        save_chat_history(st.session_state.chat_history)
        st.rerun()