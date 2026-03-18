"""
FastAPI web application for EDI-AI-System.
Provides web UI for browsing resources and Q&A.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "summaries.sqlite"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "downloads"
TEXT_DIR = PROJECT_ROOT / "data" / "processed" / "text"

# FastAPI app
app = FastAPI(title="EDI-AI-System Web UI")

# Templates
templates_dir = PROJECT_ROOT / "web" / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Static files
static_dir = PROJECT_ROOT / "web" / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# =========================
# Database utilities
# =========================
def get_db_connection():
    """Get a database connection."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    return sqlite3.connect(DB_PATH)


def get_all_resources(search_query: Optional[str] = None) -> list[dict]:
    """Get all Included resources, optionally filtered by search query."""
    with get_db_connection() as conn:
        conn.row_factory = sqlite3.Row
        query = """
            SELECT
                id, title, author, final_url, kind, raw_path, text_path,
                summary, error
            FROM summaries
            WHERE included_excluded = 'Included'
            ORDER BY
                CASE
                    WHEN CAST(id AS INTEGER) IS NOT NULL THEN CAST(id AS INTEGER)
                    ELSE 999999
                END,
                id
        """
        params = []
        if search_query:
            query = """
                SELECT
                    id, title, author, final_url, kind, raw_path, text_path,
                    summary, error
                FROM summaries
                WHERE included_excluded = 'Included'
                    AND (
                        title LIKE ? OR author LIKE ?
                    )
                ORDER BY
                    CASE
                        WHEN CAST(id AS INTEGER) IS NOT NULL THEN CAST(id AS INTEGER)
                        ELSE 999999
                    END,
                    id
            """
            search_pattern = f"%{search_query}%"
            params = [search_pattern, search_pattern]

        rows = conn.execute(query, params).fetchall()
        resources = []
        for row in rows:
            summary = (row["summary"] or "").strip()
            error = (row["error"] or "").strip()
            has_summary = bool(summary) and not error

            resources.append({
                "id": row["id"],
                "title": (row["title"] or "").strip(),
                "author": (row["author"] or "").strip() or None,
                "final_url": (row["final_url"] or "").strip(),
                "kind": (row["kind"] or "").strip().lower(),
                "raw_path": (row["raw_path"] or "").strip(),
                "text_path": (row["text_path"] or "").strip(),
                "summary": summary,
                "has_summary": has_summary,
            })
        return resources


def get_resource_by_id(resource_id: str) -> Optional[dict]:
    """Get a single resource by ID."""
    with get_db_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT
                id, title, author, final_url, kind, raw_path, text_path,
                summary, error
            FROM summaries
            WHERE id = ? AND included_excluded = 'Included'
            LIMIT 1
            """,
            (resource_id,),
        ).fetchone()

        if not row:
            return None

        summary = (row["summary"] or "").strip()
        error = (row["error"] or "").strip()
        has_summary = bool(summary) and not error

        return {
            "id": row["id"],
            "title": (row["title"] or "").strip(),
            "author": (row["author"] or "").strip() or None,
            "final_url": (row["final_url"] or "").strip(),
            "kind": (row["kind"] or "").strip().lower(),
            "raw_path": (row["raw_path"] or "").strip(),
            "text_path": (row["text_path"] or "").strip(),
            "summary": summary,
            "has_summary": has_summary,
        }


# =========================
# Routes
# =========================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page."""
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/resources", response_class=HTMLResponse)
async def resources(request: Request, q: Optional[str] = None):
    """Resources listing page with optional search."""
    search_query = (q or "").strip() if q else None
    resources_list = get_all_resources(search_query=search_query)
    return templates.TemplateResponse(
        "resources.html",
        {
            "request": request,
            "resources": resources_list,
            "search_query": search_query or "",
        },
    )


@app.get("/resource/{resource_id}", response_class=HTMLResponse)
async def resource_detail(request: Request, resource_id: str):
    """Resource detail page with preview and summary."""
    resource = get_resource_by_id(resource_id)
    if not resource:
        raise HTTPException(status_code=404, detail="Resource not found")

    # Determine preview type
    preview_type = "none"
    preview_content = None

    kind = resource["kind"]
    raw_path = resource["raw_path"]
    text_path = resource["text_path"]

    # Check for PDF preview
    if kind == "pdf" and raw_path:
        raw_file = PROJECT_ROOT / raw_path
        if raw_file.exists() and raw_file.suffix.lower() == ".pdf":
            preview_type = "pdf"
            preview_content = {"pdf_path": f"/files/pdf/{resource_id}"}

    # Check for HTML/text preview
    elif kind == "html":
        if text_path:
            text_file = PROJECT_ROOT / text_path
            if text_file.exists():
                try:
                    preview_content = {"text": text_file.read_text(encoding="utf-8", errors="ignore")}
                    preview_type = "text"
                except Exception:
                    pass

    # Fallback: try iframe for HTML (may be blocked)
    if preview_type == "none" and kind == "html" and resource["final_url"]:
        preview_type = "iframe"
        preview_content = {"url": resource["final_url"]}

    return templates.TemplateResponse(
        "resource_detail.html",
        {
            "request": request,
            "resource": resource,
            "preview_type": preview_type,
            "preview_content": preview_content,
        },
    )


@app.get("/qa", response_class=HTMLResponse)
async def qa(request: Request):
    """Q&A page (placeholder for now)."""
    return templates.TemplateResponse("qa.html", {"request": request})


# =========================
# File serving (secure)
# =========================
@app.get("/files/pdf/{resource_id}")
async def serve_pdf(resource_id: str):
    """Securely serve PDF files from data/raw/downloads."""
    resource = get_resource_by_id(resource_id)
    if not resource:
        raise HTTPException(status_code=404, detail="Resource not found")

    raw_path = resource["raw_path"]
    if not raw_path:
        raise HTTPException(status_code=404, detail="PDF file not available")

    # Resolve path and ensure it's within RAW_DIR
    pdf_file = PROJECT_ROOT / raw_path
    pdf_file = pdf_file.resolve()

    # Security check: ensure file is within RAW_DIR
    raw_dir_resolved = RAW_DIR.resolve()
    try:
        pdf_file.relative_to(raw_dir_resolved)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    if not pdf_file.exists() or pdf_file.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="PDF file not found")

    return FileResponse(
        pdf_file,
        media_type="application/pdf",
        filename=pdf_file.name,
    )


# =========================
# Health check
# =========================
@app.get("/health")
async def health():
    """Health check endpoint."""
    if not DB_PATH.exists():
        return {"status": "error", "message": "Database not found"}
    return {"status": "ok", "db_path": str(DB_PATH)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
