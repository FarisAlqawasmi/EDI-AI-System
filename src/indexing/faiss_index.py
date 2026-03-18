from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# FAISS import (package name is faiss-cpu or faiss-gpu, module is faiss)
import faiss


# -------------------------
# Paths / defaults
# -------------------------
DEFAULT_CHUNKS_PATH = Path("data/processed/chunks/chunks.jsonl")
DEFAULT_OUT_DIR = Path("data/processed/index")

DEFAULT_EMBED_MODEL = "text-embedding-3-small"

# A practical default: keep batches moderate to avoid request-size + rate-limit issues
DEFAULT_BATCH_SIZE = 64

# Basic retry/backoff for transient errors
MAX_RETRIES = 6
BASE_BACKOFF_SECONDS = 1.0


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    title: str
    source_path: str
    chunk_index: int
    char_start: int
    char_end: int
    text: str


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e
    return items


def to_chunk_record(d: Dict[str, Any]) -> ChunkRecord:
    # minimal validation so failures are obvious
    required = ["chunk_id", "doc_id", "title", "source_path", "chunk_index", "char_start", "char_end", "text"]
    missing = [k for k in required if k not in d]
    if missing:
        raise ValueError(f"Chunk missing required fields: {missing}")
    return ChunkRecord(
        chunk_id=str(d["chunk_id"]),
        doc_id=str(d["doc_id"]),
        title=str(d["title"]),
        source_path=str(d["source_path"]),
        chunk_index=int(d["chunk_index"]),
        char_start=int(d["char_start"]),
        char_end=int(d["char_end"]),
        text=str(d["text"]),
    )


def batched(seq: List[Any], batch_size: int) -> List[List[Any]]:
    return [seq[i : i + batch_size] for i in range(0, len(seq), batch_size)]


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, eps)


def embed_texts_with_retry(
    client: OpenAI,
    model: str,
    texts: List[str],
) -> np.ndarray:
    """
    Returns embeddings as float32 numpy array shape (n, d).
    Retries with exponential backoff for transient failures (rate limits, timeouts).
    """
    attempt = 0
    while True:
        try:
            resp = client.embeddings.create(model=model, input=texts)
            # Ensure order matches input
            data_sorted = sorted(resp.data, key=lambda x: x.index)
            vecs = np.array([d.embedding for d in data_sorted], dtype=np.float32)
            return vecs
        except Exception as e:
            attempt += 1
            if attempt > MAX_RETRIES:
                raise RuntimeError(f"Embedding failed after {MAX_RETRIES} retries. Last error: {e}") from e
            sleep_s = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            # cap the sleep a bit
            sleep_s = min(sleep_s, 30.0)
            print(f"[warn] Embedding call failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            print(f"[warn] Backing off for {sleep_s:.1f}s...")
            time.sleep(sleep_s)


def build_faiss_index_cosine(embeddings: np.ndarray) -> faiss.Index:
    """
    Cosine similarity = inner product on L2-normalised vectors.
    We'll store normalised vectors and use IndexFlatIP.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")

    embeddings = l2_normalize(embeddings).astype(np.float32)
    d = embeddings.shape[1]

    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index


def write_metadata_jsonl(out_path: Path, meta_rows: List[Dict[str, Any]]) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS index from chunks.jsonl using OpenAI embeddings.")
    parser.add_argument("--chunks", type=str, default=str(DEFAULT_CHUNKS_PATH), help="Path to chunks.jsonl")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory for index+metadata")
    parser.add_argument("--model", type=str, default=DEFAULT_EMBED_MODEL, help="OpenAI embedding model")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size")
    parser.add_argument("--max_chunks", type=int, default=0, help="Debug: cap number of chunks (0 = no cap)")
    args = parser.parse_args()

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not found. Put it in your .env file (OPENAI_API_KEY=...).")

    client = OpenAI(api_key=api_key)

    chunks_path = Path(args.chunks)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not chunks_path.exists():
        raise SystemExit(f"Chunks file not found: {chunks_path}")

    raw_items = read_jsonl(chunks_path)
    if args.max_chunks and args.max_chunks > 0:
        raw_items = raw_items[: args.max_chunks]

    records = [to_chunk_record(x) for x in raw_items]
    if not records:
        raise SystemExit("No chunks found in JSONL.")

    print(f"Loaded chunks: {len(records)} from {chunks_path}")
    print(f"Embedding model: {args.model}")
    print(f"Batch size: {args.batch_size}")

    texts = [r.text for r in records]

    all_vecs: List[np.ndarray] = []
    batches = batched(texts, args.batch_size)

    for bi, batch in enumerate(batches, start=1):
        print(f"Embedding batch {bi}/{len(batches)} (n={len(batch)})...")
        vecs = embed_texts_with_retry(client, args.model, batch)
        all_vecs.append(vecs)

    embeddings = np.vstack(all_vecs).astype(np.float32)
    print(f"Embeddings shape: {embeddings.shape}")

    # Build FAISS index (cosine via normalised + inner product)
    index = build_faiss_index_cosine(embeddings)

    # Save index
    index_path = out_dir / "faiss.index"
    faiss.write_index(index, str(index_path))

    # Save metadata aligned with FAISS ids (0..N-1)
    # Keep it minimal but useful for retrieval + UI
    meta_rows: List[Dict[str, Any]] = []
    for faiss_id, r in enumerate(records):
        meta_rows.append(
            {
                "faiss_id": faiss_id,
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "title": r.title,
                "source_path": r.source_path,
                "chunk_index": r.chunk_index,
                "char_start": r.char_start,
                "char_end": r.char_end,
                # You can omit text here if you want smaller metadata.
                # Keeping it is convenient for quick prototypes.
                "text": r.text,
            }
        )

    meta_path = out_dir / "metadata.jsonl"
    write_metadata_jsonl(meta_path, meta_rows)

    print("\n✅ FAISS indexing complete.")
    print(f"Index saved to:     {index_path}")
    print(f"Metadata saved to:  {meta_path}")
    print(f"Vectors stored:     {index.ntotal}")


if __name__ == "__main__":
    main()