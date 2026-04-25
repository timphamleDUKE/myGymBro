import os
import streamlit as st
from openai import OpenAI

MODEL_NAME = "gpt-5-nano"
BACKUP_MODEL_NAME = "Mistral on-site"
LITELLM_BASE_URL = "https://litellm.oit.duke.edu/v1"


def _get_api_key() -> str:
    api_key = os.getenv("LITELLM_TOKEN") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get("LITELLM_TOKEN") or st.secrets.get("OPENAI_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError(
            "Missing API key. Create .streamlit/secrets.toml with "
            'LITELLM_TOKEN = "your-real-token", or set the LITELLM_TOKEN environment variable.'
        )

    return api_key


def _get_client() -> OpenAI:
    return OpenAI(
        api_key=_get_api_key(),
        base_url=LITELLM_BASE_URL,
    )


def get_ai_response(user_prompt, context="", chat_history=None):
    chat_history = chat_history or []

    messages = [
        {
            "role": "system",
            "content": (
                "You are GymBro, a helpful fitness assistant. Give practical, safe workout advice. "
                "Use the provided context to inform your answers, but do not fabricate information. "
                "If the context does not contain relevant information, answer based on your general knowledge. "
                "Format responses with short paragraphs and bullet points when useful. "
                "Use line breaks between sections. Keep answers concise and easy to scan."
            )
        }
    ]

    for message in chat_history[-8:]:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})

    messages.append(
        {
            "role": "user",
            "content": f"""
Context:
{context}

User question:
{user_prompt}
"""
        }
    )

    response = _get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )

    return response.choices[0].message.content
