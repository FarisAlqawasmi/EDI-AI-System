
from __future__ import annotations

import sys
from pathlib import Path

# Ensure `src/` is on sys.path so we can import `personalisation` when running this file directly.
_SRC_DIR = Path(__file__).resolve().parents[1]  # .../src
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from personalisation.personalisation import parse_user_profile_answer, parse_personalisation_consent
import numpy as np

# FAISS import name differs depending on install
try:
    import faiss  # type: ignore
except ImportError as e:
    raise SystemExit(
        "FAISS is not installed. Try: pip install faiss-cpu\n"
        "On Apple Silicon you may need: pip install faiss-cpu --no-cache-dir"
    ) from e

from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime, timezone


# ----------------------------
# Load .env reliably
# Project root is two levels above this file: src/rag/rag_answer.py
# ----------------------------
DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)

#----------------------------
# Logging RAG runs
#----------------------------

LOGS_DIR = Path("data/logs")
RAG_RUNS_PATH = LOGS_DIR / "rag_runs.jsonl"


def ensure_logs_dir() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def append_rag_run_log(payload: dict) -> None:
    """
    Append one JSON object per line (JSONL).
    Safe for later analysis and submission evidence.
    """
    ensure_logs_dir()
    with RAG_RUNS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

# ----------------------------
# Paths
# ----------------------------
INDEX_DIR = Path("data/processed/index")
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.jsonl"


# ----------------------------
# Config (env override)
# ----------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")

# Store model responses in OpenAI so they appear in the Dashboard Logs.
# (Dashboard retention rules apply; see OpenAI docs.)
STORE_RESPONSES = os.getenv("STORE_RESPONSES", "1").strip().lower() not in {"0", "false"}

# Conversation mode: persist multi-turn state on OpenAI side.
# If enabled, we create one conversation per session and attach each response to it.
USE_CONVERSATION = os.getenv("USE_CONVERSATION", "1").strip().lower() not in {"0", "false"}

# Router model (can be same as CHAT_MODEL, but configurable)
ROUTER_MODEL = os.getenv("ROUTER_MODEL", CHAT_MODEL)

# If True, store router calls too (usually not needed)
STORE_ROUTER = os.getenv("STORE_ROUTER", "0").strip().lower() not in {"0", "false"}

# Print router decisions to the terminal for debugging
DEBUG_ROUTER = os.getenv("DEBUG_ROUTER", "1").strip().lower() not in {"0", "false"}

TOP_K = int(os.getenv("TOP_K", "5"))

# Relevance gate: if the best retrieved score is below this, treat as out-of-domain / no-good-evidence.
# For cosine-similarity (normalized vectors + IndexFlatIP), scores are roughly in [-1, 1].
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.35"))

# Optional: if the best score is only barely above the rest, you can still proceed; keep off for now.
# (Reserved for later tuning.)

# Show a snippet of each retrieved chunk in the terminal (not needed for the LLM prompt)
SHOW_TEXT_CHARS = int(os.getenv("SHOW_TEXT_CHARS", "260"))

# Short preview stored in logs for future UI citation popups
CITE_PREVIEW_CHARS = int(os.getenv("CITE_PREVIEW_CHARS", "220"))

# If your index was built with cosine similarity (IndexFlatIP + normalized vectors),
# keep normalization on. If you built L2 index, set NORMALIZE_QUERY=0 in env.
NORMALIZE_QUERY = os.getenv("NORMALIZE_QUERY", "1").strip().lower() not in {"0", "false"}


@dataclass
class MetaRecord:
    # Optional field in case your metadata includes it
    faiss_id: Optional[int] = None

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


def embed_text(client: OpenAI, text: str) -> np.ndarray:
    """Embed a single text into a float32 vector."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    vec = np.array(resp.data[0].embedding, dtype="float32")
    return vec


def retrieve_top_k(
    index,
    metadata: List[MetaRecord],
    query_vec: np.ndarray,
    top_k: int,
) -> List[Tuple[int, float, MetaRecord]]:
    """
    Returns list of (rank, score, record)

    Assumes IndexFlatIP+cosine style if NORMALIZE_QUERY is on (recommended).
    """
    q = query_vec.reshape(1, -1)

    if NORMALIZE_QUERY:
        faiss.normalize_L2(q)

    scores, ids = index.search(q, top_k)
    scores = scores[0]
    ids = ids[0]

    results: List[Tuple[int, float, MetaRecord]] = []
    for rank, (idx, score) in enumerate(zip(ids, scores), start=1):
        if idx == -1:
            continue
        if 0 <= idx < len(metadata):
            results.append((rank, float(score), metadata[idx]))
    return results


def build_context_block(results: List[Tuple[int, float, MetaRecord]]) -> str:
    """
    Build the retrieval context we’ll send to the LLM.

    Key change:
    - Each chunk has a stable citation token: [CITE:chunk_id]
      The model MUST cite using that token, e.g. [CITE:100:9]
    """
    parts: List[str] = []
    for rank, score, r in results:
        # keep the raw text, but wrap metadata in a structured way
        parts.append(
            "\n".join(
                [
                    f"=== SOURCE {rank} ===",
                    f"CITATION: [CITE:{r.chunk_id}]",
                    f"chunk_id: {r.chunk_id}",
                    f"doc_id: {r.doc_id}",
                    f"title: {r.title}",
                    f"source_path: {r.source_path}",
                    f"char_range: [{r.char_start}, {r.char_end}]",
                    "CONTENT:",
                    r.text.strip(),
                ]
            )
        )
    return "\n\n".join(parts)


def format_sources_list(results: List[Tuple[int, float, MetaRecord]]) -> str:
    """
    A clean, human-readable sources list to print under the answer.
    """
    lines: List[str] = []
    for rank, score, r in results:
        lines.append(
            f"- [CITE:{r.chunk_id}] {r.title} | {r.source_path} | chars {r.char_start}-{r.char_end} | score {score:.4f}"
        )
    return "\n".join(lines)

def make_preview_text(text: str, max_chars: int) -> str:
    """
    Create a compact, single-line preview snippet for citation popups.

    Heuristics:
    - collapse whitespace
    - require a minimum amount of alphabetic content
    - avoid ultra-short / header-only fragments
    """
    if not text:
        return ""

    # collapse whitespace
    s = " ".join(text.strip().split())

    # quality checks to avoid junk previews
    alpha_chars = sum(c.isalpha() for c in s)
    if len(s) < 40 or alpha_chars < 20:
        return "[Short excerpt — open source to view full context]"

    if len(s) <= max_chars:
        return s

    return s[:max_chars].rstrip() + " ..."

def build_citations_map(results: List[Tuple[int, float, MetaRecord]]) -> dict:
    """Build a machine-readable mapping from citation token -> source metadata.

    This is what a future web UI can use to turn [CITE:doc:chunk] into clickable popups.
    Keys match exactly what the LLM outputs in the answer.
    """
    cite_map: dict = {}
    for rank, score, r in results:
        cite_key = f"CITE:{r.chunk_id}"
        cite_map[cite_key] = {
            "cite": cite_key,
            "rank": rank,
            "score": float(score),
            "chunk_id": r.chunk_id,
            "doc_id": r.doc_id,
            "title": r.title,
            "source_path": r.source_path,
            "char_start": int(r.char_start),
            "char_end": int(r.char_end),
            "preview_text": make_preview_text(r.text, CITE_PREVIEW_CHARS),
        }
    return cite_map


def route_user_turn(
    client: OpenAI,
    user_text: str,
    *,
    last_user_question: Optional[str],
    last_answer: Optional[str],
    last_results: Optional[List[Tuple[int, float, MetaRecord]]],
) -> Dict[str, Any]:
    """AI-based router.

    Decides how to handle the next user turn given the previous turn context.

    Returns a dict with keys:
      - action: one of {"FOLLOWUP_EXPLAIN", "FOLLOWUP_REFINE", "NEW_QUERY"}
      - standalone_question: string to retrieve against (only for FOLLOWUP_REFINE or NEW_QUERY)
      - reason: short explanation (debug)
    """

    # If no prior context, this must be a new query
    if not last_user_question or not last_answer:
        return {
            "action": "NEW_QUERY",
            "standalone_question": user_text.strip(),
            "reason": "no_prior_context",
        }

    system_msg = (
        "You are a routing assistant for a RAG system. "
        "Given the previous user question, the previous assistant answer, and the new user message, "
        "decide what the user intends.\n\n"
        "Actions:\n"
        "- FOLLOWUP_EXPLAIN: user is asking about the previous answer, confidence, or citations (e.g., 'are you sure?', 'why those sources?', 'show evidence', 'how did you pick top 5', 'which chunk supports claim X').\n"
        "  In this case, DO NOT request a new retrieval; the system should answer using the previous retrieved sources.\n"
        "- FOLLOWUP_REFINE: user is asking a follow-up that depends on the previous topic (e.g., 'what about ethnic minority researchers?', 'explain more', 'what were the outcomes?'), "
        "  and needs a new retrieval using a rewritten standalone question that includes the missing context.\n"
        "- NEW_QUERY: user is switching to a different topic or starting a fresh question unrelated to the previous one.\n\n"
        "Output MUST be valid JSON with exactly these keys: action, standalone_question, reason.\n"
        "- action must be one of: FOLLOWUP_EXPLAIN, FOLLOWUP_REFINE, NEW_QUERY\n"
        "- standalone_question must be a fully self-contained question suitable for retrieval.\n"
        "  If action is FOLLOWUP_EXPLAIN, set standalone_question to an empty string.\n"
        "- reason should be a short phrase.\n"
        "Do not include any extra keys."
    )

    def _summarize_prev_sources(
        results: Optional[List[Tuple[int, float, MetaRecord]]],
        *,
        max_sources: int = 6,
        max_preview_chars: int = 140,
    ) -> str:
        if not results:
            return "None"
        lines: List[str] = []
        for rank, score, r in results[:max_sources]:
            preview = " ".join((r.text or "").strip().split())
            if len(preview) > max_preview_chars:
                preview = preview[:max_preview_chars].rstrip() + " ..."
            lines.append(
                f"- rank={rank} score={score:.4f} cite=[CITE:{r.chunk_id}] title=\"{r.title}\" preview=\"{preview}\""
            )
        return "\n".join(lines)

    prev_sources_summary = _summarize_prev_sources(last_results)

    user_payload = (
        "PREVIOUS_USER_QUESTION:\n"
        f"{last_user_question}\n\n"
        "PREVIOUS_ASSISTANT_ANSWER:\n"
        f"{last_answer}\n\n"
        "PREVIOUS_RETRIEVED_SOURCES:\n"
        f"{prev_sources_summary}\n\n"
        "NEW_USER_MESSAGE:\n"
        f"{user_text}\n"
    )

    def _extract_json_obj(s: str) -> Optional[dict]:
        """Best-effort extraction of a JSON object from a string.

        Handles cases where the model outputs extra text around the JSON.
        """
        if not s:
            return None
        s = s.strip()
        # Fast path
        try:
            return json.loads(s)
        except Exception:
            pass
        # Best-effort: take substring from first '{' to last '}'
        i = s.find("{")
        j = s.rfind("}")
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(s[i : j + 1])
            except Exception:
                return None
        return None

    def _router_retry_plain_json() -> Optional[dict]:
        """Retry router with json_object response_format (less strict than schema).

        Some models occasionally fail json_schema formatting; json_object often succeeds.
        """
        try:
            r2 = client.responses.create(
                model=ROUTER_MODEL,
                input=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_payload},
                ],
                temperature=0,
                store=STORE_ROUTER,
                response_format={"type": "json_object"},
            )
            raw2 = (getattr(r2, "output_text", "") or "").strip()
            return _extract_json_obj(raw2)
        except Exception:
            return None

    # Try structured JSON output first (newer SDKs/models support it)
    try:
        resp = client.responses.create(
            model=ROUTER_MODEL,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_payload},
            ],
            temperature=0,
            store=STORE_ROUTER,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "rag_router",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["FOLLOWUP_EXPLAIN", "FOLLOWUP_REFINE", "NEW_QUERY"],
                            },
                            "standalone_question": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["action", "standalone_question", "reason"],
                    },
                },
            },
        )
        raw = (getattr(resp, "output_text", "") or "").strip()
        obj = _extract_json_obj(raw)
        if obj is None:
            # fallback: pull from content blocks
            pieces: List[str] = []
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", "") == "message":
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", "") in {"output_text", "text"}:
                            pieces.append(getattr(c, "text", "") or "")
            obj = _extract_json_obj("".join(pieces).strip())
        if obj is None:
            raise ValueError("router_json_parse_failed")
    except Exception:
        # Fallback: plain text JSON
        resp = client.responses.create(
            model=ROUTER_MODEL,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_payload},
            ],
            temperature=0,
            store=STORE_ROUTER,
        )
        raw = (getattr(resp, "output_text", "") or "").strip()
        obj = _extract_json_obj(raw)

        # If we still couldn't parse, do one more retry with json_object formatting.
        if obj is None:
            obj = _router_retry_plain_json()

        # Final fallback: default to FOLLOWUP_EXPLAIN when prior context exists,
        # because it avoids garbage retrieval on short follow-ups.
        if obj is None:
            obj = {
                "action": "FOLLOWUP_EXPLAIN",
                "standalone_question": "",
                "reason": "router_parse_failed_default_followup_explain",
            }

    # Validate + normalize
    action = str(obj.get("action", "NEW_QUERY")).strip()
    if action not in {"FOLLOWUP_EXPLAIN", "FOLLOWUP_REFINE", "NEW_QUERY"}:
        action = "NEW_QUERY"

    standalone = str(obj.get("standalone_question", "")).strip()
    reason = str(obj.get("reason", "")).strip() or "router"

    if action == "FOLLOWUP_EXPLAIN":
        standalone = ""
    elif not standalone:
        standalone = user_text.strip()

    return {"action": action, "standalone_question": standalone, "reason": reason}



# New: LLM-powered followup explain and preview logging helpers
def call_llm_followup_explain(
    client: OpenAI,
    followup_question: str,
    *,
    prev_user_question: str,
    prev_assistant_answer: str,
    prev_results: List[Tuple[int, float, MetaRecord]],
    conversation_id: Optional[str] = None,
    previous_response_id: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Answer a follow-up that is ABOUT the previous turn without doing new retrieval.

    Uses ONLY the previously retrieved SOURCES (prev_results).
    """

    if not prev_results:
        return (
            "I don't have any previously retrieved sources in this conversation yet. Ask an EDI question first.",
            None,
        )

    context = build_context_block(prev_results)

    system_msg = (
        "You are a retrieval-augmented assistant for an EDI document collection.\n"
        "The user is asking a follow-up about the previous question/answer.\n\n"
        "CORE RULES:\n"
        "- If the user asks about EDI content (facts, recommendations, limitations, evidence), answer using ONLY the provided SOURCES.\n"
        "- For any factual claim about EDI content, EVERY sentence must end with at least one citation token that appears in SOURCES.\n"
        "- If the SOURCES do not contain enough to answer, say so clearly.\n\n"
        "REFORMAT / SUMMARY RULES (very important):\n"
        "- If the user is asking to REPHRASE, REFORMAT, or SUMMARISE the previous assistant answer (e.g., 'just the names', 'headings only', 'no explanations', 'bullet list only'), DO NOT introduce new measures or details.\n"
        "- In that case, output ONLY the high-level headings/categories that already appear in PREVIOUS_ASSISTANT_ANSWER.\n"
        "- Do not add sub-bullets or extra items. Do not add any extra commentary.\n"
        "- When you are only repeating headings from PREVIOUS_ASSISTANT_ANSWER (no new EDI facts), citations are NOT required.\n\n"
        "SYSTEM EXPLANATIONS:\n"
        "- If you are explaining how sources were selected/ranked by the system (e.g., similarity search), you may do so WITHOUT citations.\n\n"
        "STYLE:\n"
        "- Be concise and directly answer the follow-up question.\n"
        "- Put each numbered item and bullet on its own line; use blank lines between sections.\n"
    )

    user_msg = (
        "FOLLOWUP_QUESTION:\n"
        f"{followup_question}\n\n"
        "PREVIOUS_USER_QUESTION:\n"
        f"{prev_user_question}\n\n"
        "PREVIOUS_ASSISTANT_ANSWER (use this for reformatting; verify against SOURCES for factual content):\n"
        f"{prev_assistant_answer}\n\n"
        "SOURCES (reuse; do not retrieve new ones):\n"
        f"{context}"
    )

    input_messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    kwargs: dict = {
        "model": CHAT_MODEL,
        "input": input_messages,
        "temperature": 0.2,
        "store": STORE_RESPONSES,
    }

    if conversation_id:
        kwargs["conversation"] = conversation_id
    elif previous_response_id:
        kwargs["previous_response_id"] = previous_response_id

    resp = client.responses.create(**kwargs)

    answer_text = (getattr(resp, "output_text", None) or "").strip()
    if not answer_text:
        try:
            pieces: List[str] = []
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", "") == "message":
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", "") in {"output_text", "text"}:
                            pieces.append(getattr(c, "text", "") or "")
            answer_text = "".join(pieces).strip()
        except Exception:
            answer_text = ""

    return answer_text, getattr(resp, "id", None)


def results_to_log_rows_with_preview(
    results: List[Tuple[int, float, MetaRecord]],
    *,
    preview_chars: int = 200,
) -> List[dict]:
    """Like results_to_log_rows, but also stores a compact preview to help debug follow-ups."""
    rows: List[dict] = []
    for rank, score, r in results:
        preview = " ".join((r.text or "").strip().split())
        if len(preview) > preview_chars:
            preview = preview[:preview_chars].rstrip() + " ..."
        rows.append(
            {
                "rank": rank,
                "score": score,
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "title": r.title,
                "source_path": r.source_path,
                "char_start": r.char_start,
                "char_end": r.char_end,
                "preview": preview,
            }
        )
    return rows


# ------------------------
# Router debug helpers
# ------------------------


def debug_print_router(route: Dict[str, Any], *, enabled: bool) -> None:
    if not enabled:
        return
    action = str(route.get("action", ""))
    reason = str(route.get("reason", ""))
    standalone = str(route.get("standalone_question", ""))
    if standalone and len(standalone) > 140:
        standalone = standalone[:140].rstrip() + " ..."
    print(f"[router] action={action} reason={reason} standalone={standalone}")


def call_llm_answer(
    client: OpenAI,
    question: str,
    context: str,
    *,
    conversation_id: Optional[str] = None,
    previous_response_id: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Generate an answer using the Responses API.

    Returns (answer_text, response_id).

    We use the Responses API so you can:
    - attach to a Conversation for multi-turn chat
    - see stored calls in the OpenAI Dashboard Logs (when store=True)
    """

    system_msg = (
        "You are a retrieval-augmented assistant.\n"
        "Answer the user's question using ONLY the provided SOURCES.\n"
        "If the sources do not contain enough information, say so clearly.\n\n"
        "CITATION RULE (strict):\n"
        "- Put citations INLINE, not only at the end.\n"
        "- Every sentence that contains a factual claim MUST end with at least one citation token.\n"
        "- Citation tokens look exactly like: [CITE:100:9]\n"
        "- Use ONLY citation tokens that appear in the SOURCES.\n"
        "- If a sentence uses multiple sources, include multiple tokens at the end of that sentence.\n\n"
        "Do not use outside knowledge. Do not invent facts.\n"
        "Keep the answer concise and structured.\n"
        "Format your answer in Markdown: use blank lines between sections, put each numbered item (1. 2. 3.) and each bullet (- item) on its own line."
    )

    user_msg = (
        "QUESTION:\n"
        f"{question}\n\n"
        "SOURCES:\n"
        f"{context}"
    )

    # Build a chat-style input array for the Responses API
    input_messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    kwargs: dict = {
        "model": CHAT_MODEL,
        "input": input_messages,
        "temperature": 0.2,
        "store": STORE_RESPONSES,
    }

    # Conversation takes priority if provided.
    if conversation_id:
        kwargs["conversation"] = conversation_id
    elif previous_response_id:
        kwargs["previous_response_id"] = previous_response_id

    resp = client.responses.create(**kwargs)

    # `output_text` is the easiest way to get the final assistant text.
    answer_text = (getattr(resp, "output_text", None) or "").strip()

    # Fallback for older SDK shapes
    if not answer_text:
        try:
            # If output_text is empty, try to stitch from output blocks
            pieces: List[str] = []
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", "") == "message":
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", "") in {"output_text", "text"}:
                            pieces.append(getattr(c, "text", "") or "")
            answer_text = "".join(pieces).strip()
        except Exception:
            answer_text = ""

    return answer_text, getattr(resp, "id", None)


def print_retrieval(results: List[Tuple[int, float, MetaRecord]]) -> None:
    if not results:
        print("No retrieved chunks.")
        return

    print("=" * 90)
    print(f"Top-{len(results)} retrieved chunks:")
    for rank, score, r in results:
        snippet = r.text.strip().replace("\n", " ")
        if len(snippet) > SHOW_TEXT_CHARS:
            snippet = snippet[:SHOW_TEXT_CHARS].rstrip() + " ..."
        print("-" * 90)
        print(f"Rank {rank} | Score: {score:.4f} | {r.chunk_id} | {r.title}")
        print(f"Path: {r.source_path}")
        print(f"Char range: [{r.char_start}, {r.char_end}]")
        print(f"Snippet: {snippet}")
    print("=" * 90)

def results_to_log_rows(results: List[Tuple[int, float, MetaRecord]]) -> List[dict]:
    rows: List[dict] = []
    for rank, score, r in results:
        rows.append(
            {
                "rank": rank,
                "score": score,
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "title": r.title,
                "source_path": r.source_path,
                "char_start": r.char_start,
                "char_end": r.char_end,
            }
        )
    return rows

def main() -> None:
    # key check
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY not found in environment.\n"
            f"Tried loading .env from: {DOTENV_PATH}\n"
            "Make sure your .env contains: OPENAI_API_KEY=...\n"
        )

    client = OpenAI()

    print(f"Loading FAISS index from: {FAISS_INDEX_PATH.resolve()}")
    index = load_faiss_index(FAISS_INDEX_PATH)

    print(f"Loading metadata from: {METADATA_PATH.resolve()}")
    metadata = load_metadata(METADATA_PATH)

    # sanity prints
    d = getattr(index, "d", None)
    ntotal = getattr(index, "ntotal", None)
    print(f"Embedding model: {EMBED_MODEL}")
    print(f"Chat model:      {CHAT_MODEL}")
    print(f"Metadata records: {len(metadata)}")
    if ntotal is not None:
        print(f"FAISS index ntotal: {ntotal}")
    if d is not None:
        print(f"FAISS index dim (d): {d}")
    print(f"TOP_K: {TOP_K}")
    print()

    conversation_id: Optional[str] = None
    previous_response_id: Optional[str] = None
    last_user_question: Optional[str] = None
    last_answer: Optional[str] = None
    last_results: List[Tuple[int, float, MetaRecord]] = []

    user_profile: Dict[str, Optional[str]] = {"role": None, "edi_familiarity": None}
    personalisation_enabled: bool = False
    awaiting_consent: bool = False
    awaiting_role: bool = False
    awaiting_familiarity: bool = False

    if USE_CONVERSATION:
        # Create one conversation per CLI session.
        conv = client.conversations.create()
        conversation_id = getattr(conv, "id", None)
        print(f"Conversation enabled. conversation_id={conversation_id}")
        print("Type /new to start a fresh conversation, or /exit to quit.")
        print()

    # --- Startup onboarding conversation (once per CLI session) ---
    awaiting_consent = True
    print("Welcome to the EDI assistant. I answer questions grounded in the EDI document collection.\n")
    print("Before we dive in, can I ask you a few quick questions to tailor my answers? (yes/no)\n")

    while True:
        prompt = "Enter a question (or /new, /exit): "
        if awaiting_consent:
            prompt = "Reply yes/no (or /exit): "
        elif awaiting_role:
            prompt = "Your role (student/teacher/generic user) (or /exit): "
        elif awaiting_familiarity:
            prompt = "EDI familiarity (low/medium/high) (or /exit): "
        q = input(prompt).strip()
        if not q:
            continue
        q_lower = q.lower()
        if q_lower in {"quit", "/exit"}:
            break
        if q_lower in {"/new"}:
            # Start a fresh conversation (and clear chained state)
            previous_response_id = None
            last_user_question = None
            last_answer = None
            last_results = []
            if USE_CONVERSATION:
                conv = client.conversations.create()
                conversation_id = getattr(conv, "id", None)
                print(f"Started new conversation. conversation_id={conversation_id}")
            else:
                print("Started new session (no conversation persistence).")
            continue

        # --- Onboarding conversation: consent -> role -> familiarity ---
        if awaiting_consent:
            decision = parse_personalisation_consent(
                client,
                q,
                router_model=ROUTER_MODEL,
                store_router=STORE_ROUTER,
            )

            if decision == "yes":
                personalisation_enabled = True
                awaiting_consent = False
                awaiting_role = True
                print("Great. First question: what is your role? (student / teacher / generic user)\n")
                continue

            if decision == "no":
                personalisation_enabled = False
                awaiting_consent = False
                awaiting_role = False
                awaiting_familiarity = False
                print("No problem — I’ll answer generically.\n")
                continue

            print("Sorry — please reply with 'yes' or 'no'.")
            continue

        if personalisation_enabled and awaiting_role:
            upd = parse_user_profile_answer(
                client,
                q,
                router_model=ROUTER_MODEL,
                store_router=STORE_ROUTER,
                current_profile=user_profile,
            )
            if upd.get("role"):
                user_profile["role"] = upd["role"]
                awaiting_role = False
                awaiting_familiarity = True
                print("Thanks. Second question: how familiar are you with EDI? (low / medium / high)\n")
                continue

            print("Please reply with one of: student, teacher, generic user")
            continue

        if personalisation_enabled and awaiting_familiarity:
            upd = parse_user_profile_answer(
                client,
                q,
                router_model=ROUTER_MODEL,
                store_router=STORE_ROUTER,
                current_profile=user_profile,
            )
            if upd.get("edi_familiarity"):
                user_profile["edi_familiarity"] = upd["edi_familiarity"]
                awaiting_familiarity = False
                print(
                    f"Got it — role={user_profile['role']}, EDI familiarity={user_profile['edi_familiarity']}.\n"
                    "You can now ask your question.\n"
                )
                continue

            print("Please reply with one of: low, medium, high")
            continue

        # --- AI router decides how to handle this turn ---
        route = route_user_turn(
            client,
            q,
            last_user_question=last_user_question,
            last_answer=last_answer,
            last_results=last_results,
        )
        debug_print_router(route, enabled=DEBUG_ROUTER)
        # If user is asking about the previous answer/citations, reuse last retrieval (no new FAISS search)
        if route["action"] == "FOLLOWUP_EXPLAIN" and last_user_question and last_answer and last_results:
            answer, response_id = call_llm_followup_explain(
                client,
                q,
                prev_user_question=last_user_question,
                prev_assistant_answer=last_answer,
                prev_results=last_results,
                conversation_id=conversation_id,
                previous_response_id=previous_response_id,
            )

            # If we're chaining (no Conversation API), keep thread state.
            if not USE_CONVERSATION:
                previous_response_id = response_id

            # Advance conversation memory while keeping the same retrieved sources
            last_user_question = q
            last_answer = answer
            # last_results stays the same

            log_payload = {
                "openai_response_id": response_id,
                "conversation_id": conversation_id,
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "question": q,
                "router_action": route["action"],
                "router_reason": route["reason"],
                "embed_model": EMBED_MODEL,
                "chat_model": CHAT_MODEL,
                "top_k": TOP_K,
                "normalize_query": NORMALIZE_QUERY,
                "min_relevance_score": MIN_RELEVANCE_SCORE,
                "best_score": last_results[0][1] if last_results else None,
                "retrieved": results_to_log_rows_with_preview(last_results),
                "citations_map": build_citations_map(last_results),
                "answer": answer,
                "user_profile": user_profile,
                "personalisation_enabled": personalisation_enabled,
            }
            append_rag_run_log(log_payload)

            print("\nFINAL ANSWER")
            print("=" * 90)
            print(answer)
            print("=" * 90)

            print("\nSOURCES USED")
            print("-" * 90)
            print(format_sources_list(last_results))
            print("-" * 90)
            print()
            continue

        # For FOLLOWUP_REFINE or NEW_QUERY, we retrieve against a standalone question
        standalone_question = route.get("standalone_question", "") or q

        # 1) embed question
        qvec = embed_text(client, standalone_question)

        # 2) retrieve
        results = retrieve_top_k(index, metadata, qvec, top_k=TOP_K)

        # 3) show retrieved chunks (debug/inspection)
        print_retrieval(results)

        # 3.5) relevance gate (avoid answering out-of-domain questions)
        best_score = results[0][1] if results else float("-inf")
        if not results or best_score < MIN_RELEVANCE_SCORE:
            answer = (
                "I couldn't find sufficiently relevant evidence in the EDI document collection to answer that. "
                f"(best similarity score={best_score:.4f}, threshold={MIN_RELEVANCE_SCORE:.2f})\n\n"
                "Try rephrasing the question using EDI-related terms, or ask something that should be covered by the resources."
            )

            log_payload = {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "question": q,
                "standalone_question": standalone_question,
                "router_action": route["action"],
                "router_reason": route["reason"],
                "embed_model": EMBED_MODEL,
                "chat_model": CHAT_MODEL,
                "top_k": TOP_K,
                "normalize_query": NORMALIZE_QUERY,
                "min_relevance_score": MIN_RELEVANCE_SCORE,
                "best_score": best_score,
                "retrieved": results_to_log_rows(results),
                "citations_map": build_citations_map(results),
                "openai_response_id": None,
                "conversation_id": conversation_id,
                "answer": answer,
                "blocked_reason": "below_min_relevance" if results else "no_results",
                "user_profile": user_profile,
                "personalisation_enabled": personalisation_enabled,
            }
            append_rag_run_log(log_payload)

            print("\nFINAL ANSWER")
            print("=" * 90)
            print(answer)
            print("=" * 90)

            print("\nSOURCES USED")
            print("-" * 90)
            print(format_sources_list(results))
            print("-" * 90)
            print()
            continue

        # 4) build context + generate answer (we have enough evidence)
        context = build_context_block(results)
        answer, response_id = call_llm_answer(
            client,
            q,  # keep the user's original wording for the chat answer
            context,
            conversation_id=conversation_id,
            previous_response_id=previous_response_id,
        )

        # If we're chaining (no Conversation API), keep thread state.
        if not USE_CONVERSATION:
            previous_response_id = response_id

        # Update conversation-local memory for AI routing
        last_user_question = q
        last_answer = answer
        last_results = results

        log_payload = {
            "openai_response_id": response_id,
            "conversation_id": conversation_id,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "question": q,
            "standalone_question": standalone_question,
            "router_action": route["action"],
            "router_reason": route["reason"],
            "embed_model": EMBED_MODEL,
            "chat_model": CHAT_MODEL,
            "top_k": TOP_K,
            "normalize_query": NORMALIZE_QUERY,
            "min_relevance_score": MIN_RELEVANCE_SCORE,
            "best_score": results[0][1] if results else None,
            "retrieved": results_to_log_rows(results),
            "citations_map": build_citations_map(results),
            "answer": answer,
            "user_profile": user_profile,
            "personalisation_enabled": personalisation_enabled,
        }
        append_rag_run_log(log_payload)

        print("\nFINAL ANSWER")
        print("=" * 90)
        print(answer)
        print("=" * 90)

        # 5) print sources list under the answer (clean + auditable)
        print("\nSOURCES USED")
        print("-" * 90)
        print(format_sources_list(results))
        print("-" * 90)
        print()


if __name__ == "__main__":
    main()