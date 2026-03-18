from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dotenv import load_dotenv
from openai import OpenAI


# =========================
# Paths
# =========================
MANIFEST_PATH = Path("data/processed/manifest.csv")
DB_PATH = Path("data/processed/summaries.sqlite")
JSONL_PATH = Path("data/processed/summaries.jsonl")

DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# =========================
# Model & chunking
# =========================
DEFAULT_MODEL = "gpt-4.1-mini"

# Char-based chunking (no tokenizer dependency)
CHUNK_SIZE_CHARS = 12000
CHUNK_OVERLAP_CHARS = 500

MIN_TEXT_CHARS = 200  # if below this, likely useless


# =========================
# Prompt (EXACT as requested)
# =========================
SUMMARY_PROMPT = (
    "Write a clear, stakeholder-style summary of this resource in 4–5 sentences.\n"
    "Must include:\n"
    "1) What the resource is about.\n"
    "2) Who it is for.\n"
    "3) The key points (group related points; avoid long lists).\n"
    "If the text reads like a report (e.g., executive summary, findings, results, conclusions): "
    "prioritise the key findings and recommendations (if present).\n"
    "If the text reads like guidance (instructional/advisory): "
    "prioritise practical steps and who should use them.\n"
    "Use ONLY the provided text. Do not invent details. "
    "If the text does not specify something important (e.g., audience, scope), say so briefly."
)


# =========================
# Exclusion rules (metadata-driven)
# =========================
EXCLUDE_TYPE_KEYWORDS = {
    "podcast", "audio", "video", "webinar", "recording", "livestream",
    "episode", "talk", "lecture", "panel", "interview",
}

EXCLUDE_FORMAT_KEYWORDS = {
    "podcast", "audio", "video", "webinar", "recording",
}

EXCLUDE_URL_HINTS = {
    "podcast", "/podcasts", "episode", "/episode",
    "soundcloud", "spotify", "youtube", "youtu.be", "vimeo",
}

# statuses we allow for summarisation attempts
ALLOWED_STATUSES_FOR_SUMMARY = {"OK", "OK_BUT_SHORT"}

# author columns we’ll try to read (if present in manifest.csv)
AUTHOR_COLUMN_CANDIDATES = [
    "author", "Author", "AUTHOR",
    "publisher", "Publisher", "PUBLISHER",
    "organisation", "Organisation", "ORGANISATION",
    "organization", "Organization", "ORGANIZATION",
]


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _contains_any(haystack: str, needles: set[str]) -> bool:
    h = _norm(haystack)
    return any(n in h for n in needles)


def is_non_text_resource(resource_type: str, resource_format: str, url: str) -> bool:
    if _contains_any(resource_type, EXCLUDE_TYPE_KEYWORDS):
        return True
    if _contains_any(resource_format, EXCLUDE_FORMAT_KEYWORDS):
        return True
    if _contains_any(url, EXCLUDE_URL_HINTS):
        return True
    return False


def pick_author_from_row(r: dict) -> str:
    for col in AUTHOR_COLUMN_CANDIDATES:
        v = (r.get(col) or "").strip()
        if v:
            return v
    return ""


# =========================
# Data model
# =========================
@dataclass
class ResourceRow:
    rid: str
    title: str
    author: str
    included_excluded: str
    status: str
    kind: str
    clean_url: str
    final_url: str
    text_path: str
    text_len: int

    # Optional extra metadata (may be blank if manifest doesn't contain)
    resource_type: str = ""
    resource_format: str = ""


# =========================
# SQLite
# =========================
def init_db(db_path: Path) -> None:
    """
    Creates table if missing.
    Also adds the new `author` column if you already have an older DB.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
                id TEXT PRIMARY KEY,
                title TEXT,
                author TEXT,
                included_excluded TEXT,
                status TEXT,
                kind TEXT,
                clean_url TEXT,
                final_url TEXT,
                text_path TEXT,
                text_len INTEGER,

                resource_type TEXT,
                resource_format TEXT,

                model TEXT,
                summary TEXT,
                created_at_utc REAL,

                error TEXT
            );
            """
        )

        # Backwards-compatible: add column if DB existed without it
        cols = {row[1] for row in conn.execute("PRAGMA table_info(summaries);").fetchall()}
        if "author" not in cols:
            conn.execute("ALTER TABLE summaries ADD COLUMN author TEXT;")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_summaries_title ON summaries(title);")
        conn.commit()


def get_existing_row(db_path: Path, rid: str) -> Optional[dict]:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT id, title, author, model, summary, error, created_at_utc FROM summaries WHERE id = ? LIMIT 1;",
            (rid,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "title": row[1],
            "author": row[2] or "",
            "model": row[3] or "",
            "summary": row[4] or "",
            "error": row[5] or "",
            "created_at_utc": row[6] or 0.0,
        }


def upsert_summary(
    db_path: Path,
    row: ResourceRow,
    *,
    model: str,
    summary: Optional[str],
    error: str = "",
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO summaries (
                id, title, author, included_excluded, status, kind, clean_url, final_url, text_path, text_len,
                resource_type, resource_format,
                model, summary, created_at_utc, error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                author=excluded.author,
                included_excluded=excluded.included_excluded,
                status=excluded.status,
                kind=excluded.kind,
                clean_url=excluded.clean_url,
                final_url=excluded.final_url,
                text_path=excluded.text_path,
                text_len=excluded.text_len,
                resource_type=excluded.resource_type,
                resource_format=excluded.resource_format,
                model=excluded.model,
                summary=excluded.summary,
                created_at_utc=excluded.created_at_utc,
                error=excluded.error
            ;
            """,
            (
                row.rid,
                row.title,
                row.author,
                row.included_excluded,
                row.status,
                row.kind,
                row.clean_url,
                row.final_url,
                row.text_path,
                row.text_len,
                row.resource_type,
                row.resource_format,
                model,
                summary or "",
                time.time(),
                error,
            ),
        )
        conn.commit()


# =========================
# JSONL output (fresh each run)
# =========================
def reset_jsonl() -> None:
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if JSONL_PATH.exists():
        JSONL_PATH.unlink()


def append_jsonl(obj: dict) -> None:
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JSONL_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# =========================
# Manifest loading
# =========================
def load_manifest_rows(manifest_path: Path) -> List[ResourceRow]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    out: List[ResourceRow] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rid = str((r.get("id") or "")).strip()
            title = (r.get("title") or "").strip()
            author = pick_author_from_row(r)

            included_excluded = (r.get("included_excluded") or "").strip()
            status = (r.get("status") or "").strip()
            kind = (r.get("kind") or "").strip()
            clean_url = (r.get("clean_url") or "").strip()
            final_url = (r.get("final_url") or "").strip()
            text_path = (r.get("text_path") or "").strip()

            resource_type = (r.get("resource_type") or "").strip()
            resource_format = (r.get("resource_format") or "").strip()

            try:
                text_len = int(r.get("text_len") or 0)
            except Exception:
                text_len = 0

            if not rid:
                rid = f"NO_ID::{title}" if title else ""

            out.append(
                ResourceRow(
                    rid=rid,
                    title=title,
                    author=author,
                    included_excluded=included_excluded,
                    status=status,
                    kind=kind,
                    clean_url=clean_url,
                    final_url=final_url,
                    text_path=text_path,
                    text_len=text_len,
                    resource_type=resource_type,
                    resource_format=resource_format,
                )
            )
    return out


def filter_only_included(rows: List[ResourceRow]) -> List[ResourceRow]:
    """Keep ONLY Included resources (you said your manifest has only 74 Included anyway, but keep safe)."""
    return [r for r in rows if _norm(r.included_excluded) == "included"]


# =========================
# Chunking (char-based)
# =========================
def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)

    return chunks


# =========================
# OpenAI summarisation
# =========================
def _get_output_text(resp) -> str:
    return (getattr(resp, "output_text", "") or "").strip()


def summarise_text(
    client: OpenAI,
    *,
    model: str,
    title: str,
    text: str,
) -> str:
    system_msg = (
        "You summarise documents strictly from provided text.\n"
        "Rules:\n"
        "- Use ONLY the provided text.\n"
        "- Do NOT invent details.\n"
        "- Output must be 4-5 sentences.\n"
    )

    user_msg = (
        f"TITLE: {title}\n\n"
        f"INSTRUCTION: {SUMMARY_PROMPT}\n\n"
        "DOCUMENT TEXT:\n"
        f"{text}"
    )

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    )
    return _get_output_text(resp)


def summarise_resource_map_reduce(
    client: OpenAI,
    *,
    model: str,
    title: str,
    full_text: str,
) -> str:
    chunks = split_text(full_text, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
    if not chunks:
        return ""

    if len(chunks) == 1:
        return summarise_text(client, model=model, title=title, text=chunks[0])

    chunk_summaries: List[str] = []
    for i, ch in enumerate(chunks, start=1):
        s = summarise_text(client, model=model, title=title, text=ch)
        chunk_summaries.append(f"[Chunk {i}/{len(chunks)}] {s}")

    combined = "\n".join(chunk_summaries)
    return summarise_text(client, model=model, title=title, text=combined)


# =========================
# Eligibility logic
# =========================
def summarisation_decision(row: ResourceRow) -> Tuple[bool, str]:
    """
    Returns: (should_call_openai, reason_if_not)
    """
    k = _norm(row.kind)
    st = (row.status or "").strip()

    # Only summarise textual kinds
    if k not in {"pdf", "html"}:
        return False, f"SKIPPED_KIND_{row.kind or 'UNKNOWN'}"

    # Only allowed extraction statuses
    if st not in ALLOWED_STATUSES_FOR_SUMMARY:
        return False, f"SKIPPED_STATUS_{st or 'UNKNOWN'}"

    # Skip anything explicitly marked as skipped
    if st.upper().startswith("SKIPPED"):
        return False, f"SKIPPED_STATUS_{st}"

    # Metadata-driven skip (podcasts/videos/etc)
    url = row.final_url or row.clean_url
    if is_non_text_resource(row.resource_type, row.resource_format, url):
        return False, "SKIPPED_NON_TEXT_RESOURCE"

    # Need a text path
    if not row.text_path:
        return False, "SKIPPED_NO_TEXT_PATH"

    text_file = Path(row.text_path)
    if not text_file.exists():
        return False, "SKIPPED_MISSING_TEXT_FILE"

    # Quick length check using manifest text_len (cheap)
    if int(row.text_len or 0) < MIN_TEXT_CHARS:
        return False, f"SKIPPED_TOO_SHORT_len={row.text_len}"

    return True, ""


# =========================
# Main runner
# =========================
def run(
    *,
    model: str,
    limit: Optional[int],
    force: bool,
    sleep_s: float,
) -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env or export it.")

    client = OpenAI(api_key=api_key)

    init_db(DB_PATH)
    reset_jsonl()

    rows = filter_only_included(load_manifest_rows(MANIFEST_PATH))

    if limit is not None:
        rows = rows[:limit]

    print(f"Processing {len(rows)} Included resources (will summarise eligible ones, store placeholders for others)")
    print(f"DB:    {DB_PATH}")
    print(f"JSONL: {JSONL_PATH}")
    print(f"Model: {model}")

    summarised = 0
    placeholders = 0

    for idx, r in enumerate(rows, start=1):
        should_summarise, skip_reason = summarisation_decision(r)

        existing = get_existing_row(DB_PATH, r.rid)
        existing_summary = (existing or {}).get("summary", "").strip()

        # If not eligible, ensure a row exists in DB with summary="" and error reason
        if not should_summarise:
            # Only write placeholder if it doesn't exist OR we want to update metadata/error
            upsert_summary(
                DB_PATH,
                r,
                model=(existing or {}).get("model", "") or model,
                summary=(existing_summary if existing_summary else ""),
                error=(skip_reason if not existing_summary else (existing or {}).get("error", skip_reason)),
            )
            placeholders += 1

            # Read back (so JSONL reflects DB state)
            cur = get_existing_row(DB_PATH, r.rid) or {}
            append_jsonl(
                {
                    "id": r.rid,
                    "title": r.title,
                    "author": r.author,
                    "included_excluded": r.included_excluded,
                    "status": r.status,
                    "kind": r.kind,
                    "resource_type": r.resource_type,
                    "resource_format": r.resource_format,
                    "final_url": r.final_url,
                    "text_len": int(r.text_len or 0),
                    "summary_available": bool((cur.get("summary") or "").strip()),
                    "summary": (cur.get("summary") or ""),
                    "error": (cur.get("error") or skip_reason),
                    "model": (cur.get("model") or ""),
                }
            )
            print(f"[{idx}/{len(rows)}] PLACEHOLDER: {r.rid} | {r.title} | {skip_reason}")
            continue

        # Eligible: summarise only if needed
        if (not force) and existing_summary:
            print(f"[{idx}/{len(rows)}] SKIP (exists): {r.rid} | {r.title}")
            # still emit JSONL entry
            cur = get_existing_row(DB_PATH, r.rid) or {}
            append_jsonl(
                {
                    "id": r.rid,
                    "title": r.title,
                    "author": r.author,
                    "included_excluded": r.included_excluded,
                    "status": r.status,
                    "kind": r.kind,
                    "resource_type": r.resource_type,
                    "resource_format": r.resource_format,
                    "final_url": r.final_url,
                    "text_len": int(r.text_len or 0),
                    "summary_available": True,
                    "summary": (cur.get("summary") or ""),
                    "error": (cur.get("error") or ""),
                    "model": (cur.get("model") or ""),
                }
            )
            continue

        # Actually call OpenAI
        text_file = Path(r.text_path)
        full_text = text_file.read_text(encoding="utf-8", errors="ignore").strip()

        # Double-check length on real text
        if len(full_text) < MIN_TEXT_CHARS:
            reason = f"SKIPPED_TOO_SHORT_len={len(full_text)}"
            print(f"[{idx}/{len(rows)}] PLACEHOLDER: {r.rid} | {r.title} | {reason}")
            upsert_summary(DB_PATH, r, model=model, summary="", error=reason)
            placeholders += 1
            cur = get_existing_row(DB_PATH, r.rid) or {}
            append_jsonl(
                {
                    "id": r.rid,
                    "title": r.title,
                    "author": r.author,
                    "included_excluded": r.included_excluded,
                    "status": r.status,
                    "kind": r.kind,
                    "resource_type": r.resource_type,
                    "resource_format": r.resource_format,
                    "final_url": r.final_url,
                    "text_len": int(r.text_len or 0),
                    "summary_available": False,
                    "summary": "",
                    "error": reason,
                    "model": (cur.get("model") or model),
                }
            )
            continue

        print(f"[{idx}/{len(rows)}] SUMMARISE: {r.rid} | {r.title} | len={len(full_text)}")
        try:
            summary = summarise_resource_map_reduce(client, model=model, title=r.title, full_text=full_text)
            upsert_summary(DB_PATH, r, model=model, summary=summary, error="")
            summarised += 1

            cur = get_existing_row(DB_PATH, r.rid) or {}
            append_jsonl(
                {
                    "id": r.rid,
                    "title": r.title,
                    "author": r.author,
                    "included_excluded": r.included_excluded,
                    "status": r.status,
                    "kind": r.kind,
                    "resource_type": r.resource_type,
                    "resource_format": r.resource_format,
                    "final_url": r.final_url,
                    "text_len": len(full_text),
                    "summary_available": True,
                    "summary": summary,
                    "error": "",
                    "model": model,
                }
            )

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"  -> ERROR: {err}")
            upsert_summary(DB_PATH, r, model=model, summary="", error=err)
            placeholders += 1

            cur = get_existing_row(DB_PATH, r.rid) or {}
            append_jsonl(
                {
                    "id": r.rid,
                    "title": r.title,
                    "author": r.author,
                    "included_excluded": r.included_excluded,
                    "status": r.status,
                    "kind": r.kind,
                    "resource_type": r.resource_type,
                    "resource_format": r.resource_format,
                    "final_url": r.final_url,
                    "text_len": int(r.text_len or 0),
                    "summary_available": False,
                    "summary": "",
                    "error": (cur.get("error") or err),
                    "model": (cur.get("model") or model),
                }
            )

        if sleep_s > 0:
            time.sleep(sleep_s)

    print("Done.")
    print(f"Summarised:   {summarised}")
    print(f"Placeholders: {placeholders}")
    print(f"Total rows:   {len(rows)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarise Included resources into SQLite + JSONL (placeholders for non-summarised).")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default {DEFAULT_MODEL})")
    p.add_argument("--limit", type=int, default=None, help="Limit number of resources (debugging).")
    p.add_argument("--force", action="store_true", help="Re-summarise even if summary exists.")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(model=args.model, limit=args.limit, force=args.force, sleep_s=args.sleep)