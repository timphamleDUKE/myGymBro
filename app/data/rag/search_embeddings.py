import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "index"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
CHUNKS_FILE = INDEX_DIR / "chunks.json"
CONFIG_FILE = INDEX_DIR / "index_config.json"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH = 512


def load_index() -> tuple[np.ndarray, list[dict], str]:
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(f"Missing embeddings file: {EMBEDDINGS_FILE}")
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Missing chunks file: {CHUNKS_FILE}")

    embeddings = np.load(EMBEDDINGS_FILE).astype("float32")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    model_name = DEFAULT_MODEL
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        model_name = cfg.get("embed_model", DEFAULT_MODEL)

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(embeddings)} embeddings but {len(chunks)} chunk records"
        )
    return embeddings, chunks, model_name


def compute_query_embedding(
    query: str, model: AutoModel, tokenizer: AutoTokenizer, device: str
) -> np.ndarray:
    inputs = tokenizer(
        query,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
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


def search(query: str, top_k: int = 5) -> list[dict]:
    embeddings, chunks, model_name = load_index()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    query_embedding = compute_query_embedding(query, model, tokenizer, device)

    # Dot product works as cosine since vectors are normalized.
    scores = embeddings @ query_embedding
    top_indices = np.argsort(scores)[-top_k:][::-1]

    results: list[dict] = []
    for rank, idx in enumerate(top_indices, start=1):
        chunk = chunks[int(idx)]
        results.append(
            {
                "rank": rank,
                "score": float(scores[idx]),
                "source_file": chunk.get("source_file"),
                "source_name": chunk.get("source_name"),
                "chunk_index": chunk.get("chunk_index"),
                "text": chunk.get("text", ""),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Search saved RAG embeddings")
    parser.add_argument("query", type=str, help="User query text")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top chunks to return")
    args = parser.parse_args()

    results = search(args.query, top_k=args.top_k)

    print(f"\nQuery: {args.query}\n")
    for item in results:
        preview = item["text"][:350].replace("\n", " ")
        print(
            f"[{item['rank']}] score={item['score']:.4f} "
            f"source={item['source_name']} chunk={item['chunk_index']}"
        )
        print(f"    {preview}")
        print()


if __name__ == "__main__":
    main()
