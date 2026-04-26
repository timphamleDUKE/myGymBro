import json
from pathlib import Path
import re

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
RAG_INDEX_DIR = PROJECT_ROOT / "data" / "processed"
RAG_EMBEDDINGS_FILE = RAG_INDEX_DIR / "embeddings.npy"
RAG_CHUNKS_FILE = RAG_INDEX_DIR / "chunks.json"
RAG_CONFIG_FILE = RAG_INDEX_DIR / "index_config.json"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_QUERY_LENGTH = 512
INITIAL_RETRIEVAL_MULTIPLIER = 5
MAX_INITIAL_CANDIDATES = 20

_RAG_CACHE: dict = {
    "loaded": False,
    "embeddings": None,
    "chunks": None,
    "tokenizer": None,
    "model": None,
    "device": None,
}

LOW_VALUE_TEXT_MARKERS = (
    "in this video",
    "today we're",
    "today we are",
    "i'm going to",
    "i'll be",
    "tier list",
    "top 10",
    "top five",
    "make sure",
    "thanks for watching",
    "welcome back",
    "welcome to the video",
    "feature image",
    "shutterstock",
    "link in the description",
)


def _tokenize_for_overlap(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _query_prefers_explanatory_sources(query: str) -> bool:
    lowered = query.lower()
    return any(
        phrase in lowered
        for phrase in (
            "what is",
            "what are",
            "best",
            "worst",
            "how to",
            "why",
            "guide",
            "explain",
            "difference",
        )
    )


def _contains_low_value_text(text: str, window: int | None = None) -> bool:
    lowered = text.lower()
    if window is not None:
        lowered = lowered[:window]
    return any(marker in lowered for marker in LOW_VALUE_TEXT_MARKERS)


def _metadata_rerank_score(query: str, chunk: dict, base_score: float) -> float:
    score = base_score
    query_tokens = _tokenize_for_overlap(query)
    title = chunk.get("title", "")
    title_tokens = _tokenize_for_overlap(title)
    text = chunk.get("text", "")
    source_type = (chunk.get("type", "") or "").lower()
    website = (chunk.get("website", "") or "").lower()

    if query_tokens and title_tokens:
        title_overlap = len(query_tokens & title_tokens) / max(1, len(query_tokens))
        score += 0.18 * title_overlap

    if _query_prefers_explanatory_sources(query) and source_type == "article":
        score += 0.05

    trusted_domains = (
        "barbellmedicine.com",
        "strongerbyscience.com",
        "powerliftingtechnique.com",
        "pubmed.ncbi.nlm.nih.gov",
        "pmc.ncbi.nlm.nih.gov",
    )
    if any(domain in website for domain in trusted_domains):
        score += 0.03

    if _contains_low_value_text(text, window=300):
        score -= 0.08

    return score


def _load_rag_assets() -> None:
    if _RAG_CACHE["loaded"]:
        return

    if not RAG_EMBEDDINGS_FILE.exists() or not RAG_CHUNKS_FILE.exists():
        raise FileNotFoundError("RAG index missing. Run src/rag/build_embeddings.py first.")

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
    initial_k = min(len(scores), max(top_k * INITIAL_RETRIEVAL_MULTIPLIER, MAX_INITIAL_CANDIDATES))
    candidate_indices = np.argsort(scores)[-initial_k:][::-1]
    reranked_candidates: list[dict] = []

    for idx in candidate_indices:
        chunk = chunks[int(idx)]
        base_score = float(scores[idx])
        reranked_score = _metadata_rerank_score(query, chunk, base_score)
        reranked_candidates.append(
            {
                "idx": int(idx),
                "base_score": base_score,
                "reranked_score": reranked_score,
                "chunk": chunk,
            }
        )

    reranked_candidates.sort(key=lambda item: item["reranked_score"], reverse=True)

    results = []
    for rank, candidate in enumerate(reranked_candidates[:top_k], start=1):
        chunk = candidate["chunk"]
        results.append(
            {
                "rank": rank,
                "score": candidate["reranked_score"],
                "embedding_score": candidate["base_score"],
                "source_file": chunk.get("source_file"),
                "source_name": chunk.get("source_name"),
                "chunk_index": chunk.get("chunk_index"),
                "title": chunk.get("title", ""),
                "author": chunk.get("author", ""),
                "website": chunk.get("website", ""),
                "url": chunk.get("url", ""),
                "type": chunk.get("type", ""),
                "text": chunk.get("text", ""),
            }
        )

    return results
