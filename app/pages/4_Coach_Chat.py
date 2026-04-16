import streamlit as st

from core import placeholder_coach_reply, save_chat_history


st.title("Coach Chat")
st.write("Chat session is persisted locally for your next run.")

left, right = st.columns([2, 1])

with right:
    st.subheader("Session")
    st.write(f"Messages: {len(st.session_state.chat_history)}")
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        save_chat_history([])
        st.success("History cleared.")

with left:
    for message in st.session_state.chat_history:
        role = message.get("role", "assistant")
        with st.chat_message(role):
            st.write(message.get("content", ""))

    user_input = st.chat_input("Ask your coach...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        reply = placeholder_coach_reply(user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        save_chat_history(st.session_state.chat_history)
        st.rerun()
