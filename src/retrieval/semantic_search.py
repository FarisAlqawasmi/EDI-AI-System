from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

# FAISS import name differs depending on install
try:
    import faiss  # type: ignore
except ImportError as e:
    raise SystemExit(
        "FAISS is not installed. Try: pip install faiss-cpu\n"
        "On Apple Silicon you may need: pip install faiss-cpu --no-cache-dir"
    ) from e

from dotenv import load_dotenv
from openai import OpenAI

# ----------------------------
# Load .env robustly
# ----------------------------
# File is: <project>/src/retrieval/semantic_search.py
# So project root is parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)

# ----------------------------
# Paths (make them robust too)
# ----------------------------
INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.jsonl"

# Must match what you used in indexing
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
TOP_K = int(os.getenv("TOP_K", "5"))


@dataclass
class MetaRecord:
    # Optional field (some pipelines store this)
    faiss_id: int | None = None

    chunk_id: str = ""
    doc_id: str = ""
    title: str = ""
    source_path: str = ""
    chunk_index: int = 0
    char_start: int = 0
    char_end: int = 0
    text: str = ""


def parse_meta_record(obj: dict) -> MetaRecord:
    """Create a MetaRecord from a dict, ignoring unknown keys."""
    allowed = set(MetaRecord.__dataclass_fields__.keys())
    filtered = {k: v for k, v in obj.items() if k in allowed}
    return MetaRecord(**filtered)


def load_metadata(path: Path) -> List[MetaRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata.jsonl at: {path}")

    records: List[MetaRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(parse_meta_record(obj))
            except Exception as e:
                raise ValueError(f"Bad JSON on line {line_no} in {path}: {e}") from e

    if not records:
        raise ValueError(f"No metadata records found in {path}")
    return records


def load_faiss_index(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing FAISS index at: {path}")
    return faiss.read_index(str(path))


def embed_query(client: OpenAI, query: str) -> np.ndarray:
    """Create a single embedding vector for the query."""
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=query,
    )
    return np.array(resp.data[0].embedding, dtype="float32")


def search(
    index,
    metadata: List[MetaRecord],
    query_vec: np.ndarray,
    top_k: int = 5,
) -> List[Tuple[int, float, MetaRecord]]:
    """
    Returns list of (rank, score, record)

    If your index is cosine/IP-based, we normalize the query.
    (This should match your indexing script behaviour.)
    """
    q = query_vec.reshape(1, -1)
    faiss.normalize_L2(q)

    scores, ids = index.search(q, top_k)
    scores = scores[0]
    ids = ids[0]

    results: List[Tuple[int, float, MetaRecord]] = []
    for rank, (idx, score) in enumerate(zip(ids, scores), start=1):
        if idx == -1:
            continue
        if idx < 0 or idx >= len(metadata):
            continue
        results.append((rank, float(score), metadata[idx]))

    return results


def print_results(results: List[Tuple[int, float, MetaRecord]], show_text_chars: int = 600) -> None:
    if not results:
        print("No results found.")
        return

    for rank, score, rec in results:
        print("=" * 80)
        print(f"Rank: {rank} | Score: {score:.4f}")
        print(f"chunk_id: {rec.chunk_id} | doc_id: {rec.doc_id} | chunk_index: {rec.chunk_index}")
        print(f"title: {rec.title}")
        print(f"source_path: {rec.source_path}")
        print(f"char_range: [{rec.char_start}, {rec.char_end}]")

        text = rec.text.strip().replace("\n", " ")
        if len(text) > show_text_chars:
            text = text[:show_text_chars].rstrip() + " ..."
        print(f"text: {text}")
    print("=" * 80)


def main() -> None:
    # --- API key check (prevents confusing errors later) ---
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY not found in environment.\n"
            f"Tried loading .env from: {DOTENV_PATH}\n"
            "Make sure your .env contains a line like:\n"
            "OPENAI_API_KEY=sk-...\n"
        )

    client = OpenAI()

    print(f"Loading FAISS index from: {FAISS_INDEX_PATH}")
    index = load_faiss_index(FAISS_INDEX_PATH)

    print(f"Loading metadata from: {METADATA_PATH}")
    metadata = load_metadata(METADATA_PATH)

    # --- Step 3 correctness checks (important before Step 4 RAG) ---
    print(f"Embedding model: {EMBED_MODEL}")
    print(f"Metadata records: {len(metadata)}")
    print(f"FAISS index ntotal: {index.ntotal}")
    print(f"FAISS index dim (d): {index.d}")

    if index.ntotal != len(metadata):
        raise SystemExit(
            "Mismatch: FAISS index rows (ntotal) != metadata records.\n"
            "This usually means you rebuilt one without the other.\n"
            "Fix: re-run your FAISS indexing step to regenerate BOTH files together."
        )

    print()

    while True:
        query = input("Enter a query (or 'exit'): ")
        query = (query or "").strip()

        # ✅ exit fix
        if query.lower() in {"exit", "quit"}:
            print("Exiting search.")
            break

        if query == "":
            continue

        qvec = embed_query(client, query)

        # Dim check (prevents using wrong index/embedding model combo)
        if qvec.shape[0] != index.d:
            raise SystemExit(f"Dim mismatch: query dim {qvec.shape[0]} vs index dim {index.d}")

        results = search(index, metadata, qvec, top_k=TOP_K)
        print_results(results)


if __name__ == "__main__":
    main()