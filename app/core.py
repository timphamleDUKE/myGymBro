from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
WORKOUTS_CSV = DATA_DIR / "workouts.csv"
PROFILE_JSON = DATA_DIR / "profile.json"
CHAT_JSON = DATA_DIR / "chat_history.json"

RAG_INDEX_DIR = DATA_DIR / "rag" / "index"
RAG_EMBEDDINGS_FILE = RAG_INDEX_DIR / "embeddings.npy"
RAG_CHUNKS_FILE = RAG_INDEX_DIR / "chunks.json"
RAG_CONFIG_FILE = RAG_INDEX_DIR / "index_config.json"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_QUERY_LENGTH = 512

WORKOUT_COLUMNS = [
    "exercise",
    "sets",
    "reps",
    "weight",
    "date",
    "rpe",
    "rir",
    "logged_at",
]

_RAG_CACHE: dict = {
    "loaded": False,
    "embeddings": None,
    "chunks": None,
    "tokenizer": None,
    "model": None,
    "device": None,
}


def ensure_data_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not WORKOUTS_CSV.exists():
        pd.DataFrame(columns=WORKOUT_COLUMNS).to_csv(WORKOUTS_CSV, index=False)

    if not PROFILE_JSON.exists():
        PROFILE_JSON.write_text(json.dumps({}), encoding="utf-8")

    if not CHAT_JSON.exists():
        CHAT_JSON.write_text(json.dumps([]), encoding="utf-8")


def init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()


def load_profile() -> dict:
    if not PROFILE_JSON.exists():
        return {}
    try:
        return json.loads(PROFILE_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_profile(profile: dict) -> None:
    PROFILE_JSON.write_text(json.dumps(profile, indent=2), encoding="utf-8")


def load_chat_history() -> list[dict]:
    if not CHAT_JSON.exists():
        return []
    try:
        data = json.loads(CHAT_JSON.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def save_chat_history(history: list[dict]) -> None:
    CHAT_JSON.write_text(json.dumps(history, indent=2), encoding="utf-8")


def append_workout_log(entry: dict) -> None:
    if WORKOUTS_CSV.exists():
        df = pd.read_csv(WORKOUTS_CSV)
    else:
        df = pd.DataFrame(columns=WORKOUT_COLUMNS)
    updated = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    updated.to_csv(WORKOUTS_CSV, index=False)


def load_workout_logs() -> pd.DataFrame:
    if not WORKOUTS_CSV.exists():
        return pd.DataFrame(columns=WORKOUT_COLUMNS)
    return pd.read_csv(WORKOUTS_CSV)


def _load_rag_assets() -> None:
    if _RAG_CACHE["loaded"]:
        return

    if not RAG_EMBEDDINGS_FILE.exists() or not RAG_CHUNKS_FILE.exists():
        raise FileNotFoundError("RAG index missing. Run app/data/rag/build_embeddings.py first.")

    model_name = DEFAULT_EMBED_MODEL
    if RAG_CONFIG_FILE.exists():
        cfg = json.loads(RAG_CONFIG_FILE.read_text(encoding="utf-8"))
        model_name = cfg.get("embed_model", DEFAULT_EMBED_MODEL)

    embeddings = np.load(RAG_EMBEDDINGS_FILE).astype("float32")
    chunks = json.loads(RAG_CHUNKS_FILE.read_text(encoding="utf-8"))

    if len(embeddings) != len(chunks):
        raise ValueError("Mismatch between embeddings.npy and chunks.json lengths.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    _RAG_CACHE.update(
        {
            "loaded": True,
            "embeddings": embeddings,
            "chunks": chunks,
            "tokenizer": tokenizer,
            "model": model,
            "device": device,
        }
    )


def _embed_query(query: str) -> np.ndarray:
    tokenizer = _RAG_CACHE["tokenizer"]
    model = _RAG_CACHE["model"]
    device = _RAG_CACHE["device"]

    inputs = tokenizer(
        query,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_QUERY_LENGTH,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden_states.size()).float()
        masked_hidden = hidden_states * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)
        mask_sum = attention_mask.sum(dim=1).clamp(min=1e-9)
        embedding = sum_hidden / mask_sum
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

    return embedding.cpu().numpy()[0].astype("float32")


def retrieve_rag_context(query: str, top_k: int = 3) -> list[dict]:
    _load_rag_assets()
    embeddings = _RAG_CACHE["embeddings"]
    chunks = _RAG_CACHE["chunks"]
    q = _embed_query(query)

    scores = embeddings @ q
    top_indices = np.argsort(scores)[-top_k:][::-1]
    results = []

    for idx in top_indices:
        chunk = chunks[int(idx)]
        results.append(
            {
                "score": float(scores[idx]),
                "source_name": chunk.get("source_name"),
                "source_file": chunk.get("source_file"),
                "chunk_index": chunk.get("chunk_index"),
                "text": chunk.get("text", ""),
            }
        )
    return results


def placeholder_coach_reply(user_message: str) -> str:
    timestamp = datetime.now().strftime("%H:%M")
    try:
        contexts = retrieve_rag_context(user_message, top_k=3)
    except Exception as exc:
        return f"[{timestamp}] Coach: RAG index unavailable ({exc})."

    if not contexts:
        return f"[{timestamp}] Coach: I could not find relevant training context yet."

    advice = contexts[0]["text"][:420].strip()
    sources = ", ".join(sorted({ctx.get("source_name", "unknown") for ctx in contexts}))
    return f"[{timestamp}] Coach: {advice}\n\nSources: {sources}"