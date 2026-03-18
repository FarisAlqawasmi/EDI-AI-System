"""
Database helper module for EDI-AI-System web app.
Provides clean DB access functions.
Uses only existing DB/manifest fields; no new detection or network checks.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Tuple, Any

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "summaries.sqlite"

# Error substrings that indicate fetch/extraction failure (case-insensitive)
ERROR_FAILURE_SUBSTRINGS = (
    "fetch failed",
    "invalid url",
    "timeout",
    "timed out",
    "404",
    "403",
    "500",
    "connection",
    "name resolution",
)


def should_hide_resource(row: dict[str, Any]) -> bool:
    """
    Decide if a resource should be hidden from the website based only on existing DB fields.
    No file existence or network checks.
    """
    kind = (row.get("kind") or "").strip().lower()
    status = (row.get("status") or "").strip().upper()
    error = (row.get("error") or "").strip().lower()
    final_url = (row.get("final_url") or "").strip()
    clean_url = (row.get("clean_url") or "").strip()

    # Media-only: hide
    if kind == "media":
        return True
    # Status: FAILED or SKIPPED_MEDIA (or starts with FAILED)
    if status == "SKIPPED_MEDIA" or status.startswith("FAILED"):
        return True
    # Error indicates fetch/extraction failure
    if error and any(sub in error for sub in ERROR_FAILURE_SUBSTRINGS):
        return True
    # No real destination
    if not final_url and not clean_url:
        return True
    return False


def get_db_connection():
    """Get a database connection."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    return sqlite3.connect(DB_PATH)


def find_raw_path(resource_id: str, kind: str) -> Optional[str]:
    """
    Try to find the raw_path for a resource by searching the downloads directory.
    Returns the path relative to project root if found, None otherwise.
    """
    if kind != "pdf":
        return None
    
    RAW_DIR = PROJECT_ROOT / "data" / "raw" / "downloads"
    if not RAW_DIR.exists():
        return None
    
    # Try to find PDF files that might match this resource ID
    # Common patterns: {id}_*.pdf, {id}.pdf, etc.
    patterns = [
        f"{resource_id}_*.pdf",
        f"{resource_id}.pdf",
        f"*_{resource_id}.pdf",
    ]
    
    for pattern in patterns:
        matches = list(RAW_DIR.glob(pattern))
        if matches:
            # Return relative path from project root
            return str(matches[0].relative_to(PROJECT_ROOT))
    
    # Also try searching all PDFs and matching by ID in filename
    for pdf_file in RAW_DIR.glob("*.pdf"):
        if resource_id in pdf_file.stem:
            return str(pdf_file.relative_to(PROJECT_ROOT))
    
    return None


def get_resources(
    search_query: Optional[str] = None,
    page: int = 1,
    page_size: int = 15,
) -> Tuple[list[dict], int]:
    """
    Get paginated Included resources, optionally filtered by search query.
    Returns (resources_list, total_count).
    """
    with get_db_connection() as conn:
        conn.row_factory = sqlite3.Row
        
        # Fetch all Included rows (with status, clean_url for should_hide_resource)
        base_query = """
            SELECT
                id, title, author, final_url, clean_url, kind, status, text_path,
                summary, error
            FROM summaries
            WHERE LOWER(included_excluded) = 'included'
        """
        params = []
        if search_query:
            base_query += " AND (title LIKE ? OR author LIKE ?)"
            search_pattern = f"%{search_query}%"
            params = [search_pattern, search_pattern]

        order_sql = """
            ORDER BY id
        """
        all_rows = conn.execute(base_query + order_sql, params).fetchall()

        # Build row dicts and filter hidden (no new detection; only DB fields)
        visible = []
        for row in all_rows:
            row_dict = {
                "id": str(row["id"]) if row["id"] is not None else "",
                "kind": (row["kind"] or "").strip().lower(),
                "status": (row["status"] or "").strip(),
                "error": (row["error"] or "").strip(),
                "final_url": (row["final_url"] or "").strip(),
                "clean_url": (row["clean_url"] or "").strip(),
            }
            if should_hide_resource(row_dict):
                continue
            # Build full resource dict (IDs stay as text)
            rid = str(row["id"]) if row["id"] is not None else ""
            summary = (row["summary"] or "").strip()
            has_summary = bool(summary)  # True iff summary is non-empty
            kind = (row["kind"] or "").strip().lower() or "unknown"
            raw_path = find_raw_path(rid, kind)
            text_path = (row["text_path"] or "").strip()
            final_url = (row["final_url"] or "").strip()
            clean_url = (row["clean_url"] or "").strip()
            display_url = final_url or clean_url
            has_valid_url = bool(display_url and display_url.strip().lower().startswith("http"))
            has_preview = (
                (kind == "pdf" and raw_path) or
                (kind == "html" and (display_url or text_path)) or
                bool(text_path)
            )
            visible.append({
                "id": rid,
                "title": (row["title"] or "").strip(),
                "author": (row["author"] or "").strip() or None,
                "final_url": final_url,
                "clean_url": clean_url,
                "display_url": display_url,
                "kind": kind,
                "raw_path": raw_path,
                "text_path": text_path,
                "summary": summary,
                "has_summary": has_summary,
                "has_preview": has_preview,
                "has_valid_url": has_valid_url,
            })

        total_count = len(visible)
        offset = (page - 1) * page_size
        resources = visible[offset : offset + page_size]
        return resources, total_count


def get_resource_by_id(resource_id: str) -> Optional[dict]:
    """Get a single resource by ID. IDs are compared as text. Returns dict with is_hidden and display_url."""
    with get_db_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT
                id, title, author, final_url, clean_url, kind, status, text_path,
                summary, error
            FROM summaries
            WHERE id = ? AND LOWER(included_excluded) = 'included'
            LIMIT 1
            """,
            (str(resource_id),),
        ).fetchone()

        if not row:
            return None

        rid = str(row["id"]) if row["id"] is not None else ""
        summary = (row["summary"] or "").strip()
        has_summary = bool(summary)  # True iff summary is non-empty
        kind = (row["kind"] or "").strip().lower() or "unknown"
        raw_path = find_raw_path(rid, kind)
        text_path = (row["text_path"] or "").strip()
        final_url = (row["final_url"] or "").strip()
        clean_url = (row["clean_url"] or "").strip()
        display_url = final_url or clean_url
        has_valid_url = bool(display_url and display_url.strip().lower().startswith("http"))
        has_preview = (
            (kind == "pdf" and raw_path) or
            (kind == "html" and (display_url or text_path)) or
            bool(text_path)
        )

        row_for_hide = {
            "kind": kind,
            "status": (row["status"] or "").strip(),
            "error": (row["error"] or "").strip(),
            "final_url": final_url,
            "clean_url": clean_url,
        }
        is_hidden = should_hide_resource(row_for_hide)

        return {
            "id": rid,
            "title": (row["title"] or "").strip(),
            "author": (row["author"] or "").strip() or None,
            "final_url": final_url,
            "clean_url": clean_url,
            "display_url": display_url,
            "kind": kind,
            "raw_path": raw_path,
            "text_path": text_path,
            "summary": summary,
            "has_summary": has_summary,
            "has_preview": has_preview,
            "has_valid_url": has_valid_url,
            "is_hidden": is_hidden,
        }


def get_debug_hidden() -> Tuple[int, int, list[dict]]:
    """Return (total_included, hidden_count, list of hidden row dicts). For /debug/hidden only."""
    with get_db_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, title, author, status, kind, error, final_url, clean_url
            FROM summaries
            WHERE LOWER(included_excluded) = 'included'
            ORDER BY id
            """
        ).fetchall()
    total_included = len(rows)
    hidden_list = []
    for row in rows:
        row_dict = {
            "id": str(row["id"]) if row["id"] is not None else "",
            "title": (row["title"] or "").strip(),
            "author": (row["author"] or "").strip() or None,
            "status": (row["status"] or "").strip(),
            "kind": (row["kind"] or "").strip() or "unknown",
            "error": (row["error"] or "").strip(),
            "final_url": (row["final_url"] or "").strip(),
            "clean_url": (row["clean_url"] or "").strip(),
        }
        d = {
            "kind": row_dict["kind"],
            "status": row_dict["status"],
            "error": row_dict["error"],
            "final_url": row_dict["final_url"],
            "clean_url": row_dict["clean_url"],
        }
        if should_hide_resource(d):
            hidden_list.append(row_dict)
    return total_included, len(hidden_list), hidden_list
