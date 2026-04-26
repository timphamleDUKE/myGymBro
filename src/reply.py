import os
import streamlit as st
from openai import OpenAI, OpenAIError

MODEL_NAME = "gpt-5-nano"
LITELLM_BASE_URL = "https://litellm.oit.duke.edu/v1"
MAX_COMPLETION_TOKENS = 2000
FALLBACK_MAX_TOKENS = 800


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


def _is_internal_error_message(content: str) -> bool:
    error_markers = (
        "I reached the chat model, but it returned an empty response",
        "I could not get a non-empty chat response",
        "I could not reach the chat model",
        "I ran into an error while generating a reply",
    )
    return any(marker in content for marker in error_markers)


def build_messages(user_prompt, context="", chat_history=None) -> list[dict]:
    chat_history = chat_history or []

    messages = [
        {
            "role": "system",
            "content": (
                "You are GymBro, a fitness assistant in a workout tracking app. "
                "Give practical, safe, and personalized advice using:\n"
                "1) Retrieved knowledge (RAG)\n"
                "2) User workout history\n"
                "3) Model predictions\n"
                "4) User profiles and goals\n\n"

                "Priority: context > user data > predictions > general knowledge.\n\n"

                "Rules:\n"
                "- Do not fabricate information\n"
                "- Say when context is missing\n"
                "- Prioritize safety and realistic progression\n\n"

                "Behavior:\n"
                "- Give specific, actionable advice\n"
                "- Use user data when available\n"
                "- Explain weight recommendations briefly\n"
                "- Sanity-check model predictions\n\n"

                "Style:\n"
                "- Concise, structured, bullet points when helpful\n"
                "- Supportive tone\n"
                "- Do not say the user provided context, data, workout history, or predictions. "
                "Treat app context as background knowledge from the product."
            )
        }
    ]

    if context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Private app context for this response. Use it to personalize the answer, "
                    "but do not mention that this context was provided and do not quote the "
                    "section labels unless the user asks about the app internals.\n\n"
                    f"{context}"
                ),
            }
        )

    for message in chat_history[-8:]:
        role = message.get("role")
        content = message.get("content")
        if (
            role in {"user", "assistant"}
            and isinstance(content, str)
            and not _is_internal_error_message(content)
        ):
            messages.append({"role": role, "content": content})

    messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    return messages


def _stream_chat_completion(messages: list[dict]):
    request_variants = [
        {
            "model": MODEL_NAME,
            "messages": messages,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "reasoning_effort": "minimal",
            "stream": True,
        },
        {
            "model": MODEL_NAME,
            "messages": messages,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "stream": True,
        },
        {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": FALLBACK_MAX_TOKENS,
            "stream": True,
        },
    ]

    last_error = None
    for request in request_variants:
        try:
            return _get_client().chat.completions.create(**request)
        except (TypeError, OpenAIError) as exc:
            last_error = exc

    raise RuntimeError(f"I could not start a streaming chat response: {last_error}")


def _chunk_text(chunk) -> str:
    if not chunk.choices:
        return ""

    delta = chunk.choices[0].delta
    content = getattr(delta, "content", None)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    return ""


def stream_ai_response(user_prompt, context="", chat_history=None):
    messages = build_messages(user_prompt, context, chat_history)

    try:
        stream = _stream_chat_completion(messages)
    except Exception as exc:
        yield str(exc)
        return

    streamed_any_text = False
    try:
        for chunk in stream:
            text = _chunk_text(chunk)
            if text:
                streamed_any_text = True
                yield text
    except OpenAIError as exc:
        yield f"\n\nI lost the streaming connection to the chat model: {exc}"
        return

    if not streamed_any_text:
        yield "I reached the chat model, but it returned an empty streamed response."
