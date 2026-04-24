from datetime import datetime
from src.rag.retrieve_context import build_grounded_advice, retrieve_rag_context

def placeholder_coach_reply(user_message: str) -> str:
    timestamp = datetime.now().strftime("%H:%M")
    try:
        contexts = retrieve_rag_context(user_message, top_k=5)
    except Exception as exc:
        return f"[{timestamp}] Coach: RAG index unavailable ({exc})."

    if not contexts:
        return f"[{timestamp}] Coach: I could not find relevant training context yet."

    advice = build_grounded_advice(contexts)
    if not advice:
        advice = contexts[0]["text"][:420].strip()

    sources = ", ".join(
        url
        for url in dict.fromkeys(ctx.get("url", "unknown") for ctx in contexts)
    )
    return f"[{timestamp}] Coach: {advice}\n\nSources: {sources}"