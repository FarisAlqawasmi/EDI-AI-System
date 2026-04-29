"""
Flask web application for EDI-AI-System.
Provides web UI for browsing resources and Q&A.
Run from project root: python app.py
"""

from __future__ import annotations

import os
import re
import sys
import traceback
from pathlib import Path
from typing import Optional

from flask import Flask, render_template, request, send_file, abort, url_for, jsonify, session
from flask_session import Session

# Add src/ to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import DB helpers
from web.db import get_resources, get_resource_by_id

# Paths (relative to project root)
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "downloads"
TEXT_DIR = PROJECT_ROOT / "data" / "processed" / "text"
SESSION_FILE_DIR = PROJECT_ROOT / "data" / "processed" / "flask_sessions"

# Flask app
app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "web" / "templates"),
    static_folder=str(PROJECT_ROOT / "web" / "static"),
    static_url_path="/static",
)
app.secret_key = "dev-secret-key-change-in-production"  # For session support

# Server-side sessions (avoids cookie size limit; session data stored on disk)
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = str(SESSION_FILE_DIR)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
SESSION_FILE_DIR.mkdir(parents=True, exist_ok=True)
Session(app)

# Add nl2br filter for templates
@app.template_filter('nl2br')
def nl2br_filter(text):
    """Convert newlines to <br> tags."""
    if not text:
        return ""
    return text.replace('\n', '<br>')


def _is_heading_line(line: str) -> bool:
    """Simple heuristic: ALL CAPS, or short Title Case line (likely a heading)."""
    if len(line) > 80:
        return False
    stripped = line.strip()
    if not stripped:
        return False
    # ALL CAPS (at least 2 chars and mostly upper)
    if len(stripped) >= 2 and stripped.isupper() and stripped.isalpha():
        return True
    # Title-like: starts with capital, no more than a few words, mostly letters
    words = stripped.split()
    if 1 <= len(words) <= 8 and all(w[0].isupper() for w in words if w and w[0].isalpha()):
        return True
    return False


def _format_extracted_text(text: str) -> str:
    """Format extracted text into HTML with paragraphs, headings, and list detection."""
    if not text or not text.strip():
        return ""
    import html
    lines = text.strip().split("\n")
    paragraphs = []
    current = []
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            if current:
                paragraphs.append("<p>" + " ".join(html.escape(p) for p in current) + "</p>")
                current = []
        elif _is_heading_line(line_stripped):
            if current:
                paragraphs.append("<p>" + " ".join(html.escape(p) for p in current) + "</p>")
                current = []
            paragraphs.append("<h3 class=\"extract-heading\">" + html.escape(line_stripped) + "</h3>")
        elif line_stripped.startswith(("-", "*", "•", "·")) or (
            len(line_stripped) > 1 and line_stripped[0].isdigit() and line_stripped[1] in ".)"
        ):
            if current:
                paragraphs.append("<p>" + " ".join(html.escape(p) for p in current) + "</p>")
                current = []
            paragraphs.append("<p>" + html.escape(line_stripped) + "</p>")
        else:
            current.append(line_stripped)
    if current:
        paragraphs.append("<p>" + " ".join(html.escape(p) for p in current) + "</p>")
    return "\n".join(paragraphs)


# =========================
# Routes
# =========================
@app.route("/")
def home():
    """Home page."""
    return render_template("home.html")


@app.route("/resources")
def resources():
    """Resources listing page with optional search and pagination."""
    search_query = request.args.get("q", "").strip() or None
    try:
        page = int(request.args.get("page", 1))
    except ValueError:
        page = 1
    # Single scrollable page: fetch all visible resources (no pagination)
    page_size = 5000
    
    resources_list, total_count = get_resources(
        search_query=search_query,
        page=page,
        page_size=page_size,
    )
    
    total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 1
    
    return render_template(
        "resources.html",
        resources=resources_list,
        search_query=search_query or "",
        page=page,
        total_pages=total_pages,
        total_count=total_count,
    )


@app.route("/resource/<resource_id>")
def resource_detail(resource_id: str):
    """Resource detail page with preview and summary."""
    resource = get_resource_by_id(resource_id)
    if not resource:
        abort(404, description="Resource not found")
    if resource.get("is_hidden"):
        return render_template("resource_unavailable.html")

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
            pdf_url = url_for("serve_pdf", resource_id=resource_id)
            extracted_html = None
            if text_path:
                text_file = PROJECT_ROOT / text_path
                if text_file.exists():
                    try:
                        raw_text = text_file.read_text(encoding="utf-8", errors="ignore")
                        extracted_html = _format_extracted_text(raw_text)
                    except Exception:
                        pass
            preview_content = {"pdf_path": pdf_url, "extracted_html": extracted_html}

    # HTML: iframe with live URL; fallback to extracted text when iframe is blocked (X-Frame-Options etc.)
    elif kind == "html":
        extracted_html = None
        if text_path:
            text_file = PROJECT_ROOT / text_path
            if text_file.exists():
                try:
                    raw_text = text_file.read_text(encoding="utf-8", errors="ignore")
                    extracted_html = _format_extracted_text(raw_text)
                except Exception:
                    pass

        # display_url is final_url or clean_url from DB; must be external (never /resource/<id>)
        open_original_url = resource.get("display_url") or resource.get("final_url") or ""
        if open_original_url and not open_original_url.strip().lower().startswith("http"):
            open_original_url = ""  # avoid accidentally using a relative path
        if open_original_url and app.debug:
            print(f"[resource_detail] HTML open_original_url (external): {open_original_url[:80]}...")
        if open_original_url:
            preview_type = "iframe"
            preview_content = {"url": open_original_url, "extracted_html": extracted_html}
        elif extracted_html:
            preview_type = "article"
            preview_content = {"extracted_html": extracted_html}
        else:
            preview_type = "none"
            preview_content = None

    return render_template(
        "resource_detail.html",
        resource=resource,
        preview_type=preview_type,
        preview_content=preview_content,
    )


# Static initial message so /qa always loads without importing RAG/FAISS
QA_INITIAL_MESSAGE = (
    "Welcome to the EDI assistant. I answer questions grounded in the EDI document collection.\n\n"
    "Before we dive in, can I ask you a few quick questions to tailor my answers?"
)


@app.route("/qa")
def qa():
    """Q&A page with chat interface (messages/state loaded via POST /api/qa action=init)."""
    # Start fresh: clear Q&A session so reload always shows welcome message
    for key in list(session.keys()):
        if key in ("qa_messages", "qa_ui_state", "rag_state"):
            session.pop(key, None)
    return render_template("qa.html")


# =========================
# File serving (secure)
# =========================
@app.route("/files/pdf/<resource_id>")
def serve_pdf(resource_id: str):
    """Securely serve PDF files from data/raw/downloads."""
    resource = get_resource_by_id(resource_id)
    if not resource:
        abort(404, description="Resource not found")

    raw_path = resource["raw_path"]
    if not raw_path:
        abort(404, description="PDF file not available")

    # Resolve path and ensure it's within RAW_DIR
    pdf_file = PROJECT_ROOT / raw_path
    pdf_file = pdf_file.resolve()

    # Security check: ensure file is within RAW_DIR
    raw_dir_resolved = RAW_DIR.resolve()
    try:
        pdf_file.relative_to(raw_dir_resolved)
    except ValueError:
        abort(403, description="Access denied")

    if not pdf_file.exists() or pdf_file.suffix.lower() != ".pdf":
        abort(404, description="PDF file not found")

    return send_file(
        pdf_file,
        mimetype="application/pdf",
        as_attachment=False,
    )


# =========================
# Q&A API (button onboarding + timeout to avoid infinite "thinking")
# =========================
QA_API_TIMEOUT_SEC = 300  # 5 minutes

def _default_qa_messages():
    return [{"role": "assistant", "content": QA_INITIAL_MESSAGE}]

def _default_qa_ui_state():
    return {
        "mode": "onboarding_consent",
        "choices": ["Yes", "No"],
        "input_enabled": False,
    }

def _default_rag_state():
    return {
        "awaiting_consent": True,
        "awaiting_role": False,
        "awaiting_familiarity": False,
        "personalisation_enabled": False,
        "user_profile": {"role": None, "edi_familiarity": None},
        "conversation_id": None,
        "previous_response_id": None,
        "last_user_question": None,
        "last_answer": None,
        "last_results": [],
    }


def _slim_sources(sources):
    """Keep only doc_id, title, score, chunk_id for session/modal (no full chunk text)."""
    if not sources:
        return []
    def _safe_score(val):
        # Preserve missing scores so UI can show "N/A".
        if val is None:
            return None
        try:
            return float(val)
        except Exception:
            return None
    return [
        {
            "doc_id": s.get("doc_id") if isinstance(s, dict) else getattr(s, "doc_id", ""),
            "title": s.get("title") if isinstance(s, dict) else getattr(s, "title", "Untitled"),
            "score": _safe_score(s.get("score")) if isinstance(s, dict) else _safe_score(getattr(s, "score", None)),
            "chunk_id": s.get("chunk_id") if isinstance(s, dict) else getattr(s, "chunk_id", ""),
        }
        for s in sources
    ]

@app.route("/api/qa", methods=["POST"])
def api_qa():
    """Single Q&A endpoint: init, choice (onboarding buttons), or message. Never hangs (timeout)."""
    import concurrent.futures

    try:
        data = request.get_json() or {}
        action = (data.get("action") or "init").strip().lower()
        choice = (data.get("choice") or "").strip().lower()
        message = (data.get("message") or "").strip()

        if "qa_messages" not in session:
            session["qa_messages"] = _default_qa_messages()
            session["qa_ui_state"] = _default_qa_ui_state()
            session["rag_state"] = _default_rag_state()

        messages = list(session["qa_messages"])
        ui_state = dict(session["qa_ui_state"])
        rag_state = dict(session["rag_state"])
        sources = []

        if action == "init":
            return jsonify({"messages": messages, "ui_state": ui_state})

        if action == "choice":
            if not choice:
                return jsonify({"error": "choice required"}), 400
            messages.append({"role": "user", "content": choice})
            mode = ui_state.get("mode", "onboarding_consent")

            if mode == "onboarding_consent":
                if choice in ("yes", "y"):
                    rag_state["personalisation_enabled"] = True
                    rag_state["awaiting_consent"] = False
                    rag_state["awaiting_role"] = True
                    messages.append({
                        "role": "assistant",
                        "content": "Thanks — what's your role?",
                    })
                    ui_state["mode"] = "onboarding_role"
                    ui_state["choices"] = ["Student", "Teacher", "Generic user"]
                    ui_state["input_enabled"] = False
                else:
                    rag_state["personalisation_enabled"] = False
                    rag_state["awaiting_consent"] = False
                    rag_state["awaiting_role"] = False
                    rag_state["awaiting_familiarity"] = False
                    messages.append({
                        "role": "assistant",
                        "content": "Thanks — you can now ask your question.",
                    })
                    ui_state["mode"] = "ready"
                    ui_state["choices"] = []
                    ui_state["input_enabled"] = True

            elif mode == "onboarding_role":
                role_map = {"student": "student", "teacher": "teacher", "generic user": "generic user", "generic": "generic user"}
                r = role_map.get(choice) or (choice if choice in role_map.values() else None)
                if r:
                    rag_state["user_profile"]["role"] = r
                    rag_state["awaiting_role"] = False
                    rag_state["awaiting_familiarity"] = True
                    messages.append({
                        "role": "assistant",
                        "content": "And how familiar are you with EDI?",
                    })
                    ui_state["mode"] = "onboarding_familiarity"
                    ui_state["choices"] = ["Low", "Medium", "High"]
                    ui_state["input_enabled"] = False
                else:
                    messages.append({
                        "role": "assistant",
                        "content": "Please choose one of: Student, Teacher, Generic user.",
                    })

            elif mode == "onboarding_familiarity":
                fam_map = {"low": "low", "medium": "medium", "high": "high"}
                f = fam_map.get(choice) or (choice if choice in fam_map.values() else None)
                if f:
                    rag_state["user_profile"]["edi_familiarity"] = f
                    rag_state["awaiting_familiarity"] = False
                    messages.append({
                        "role": "assistant",
                        "content": "Thanks — you can now ask your question.",
                    })
                    ui_state["mode"] = "ready"
                    ui_state["choices"] = []
                    ui_state["input_enabled"] = True
                else:
                    messages.append({
                        "role": "assistant",
                        "content": "Please choose one of: low, medium, high.",
                    })

            session["qa_messages"] = messages
            session["qa_ui_state"] = ui_state
            session["rag_state"] = rag_state
            return jsonify({"messages": messages, "ui_state": ui_state})

        if action == "message":
            if not message:
                return jsonify({"error": "message required"}), 400
            messages.append({"role": "user", "content": message})
            reply = ""
            sources = []

            def _call_rag():
                from rag.rag_service import init_rag, rag_handle_message
                print("[qa] Loading RAG (init_rag)...", flush=True)
                init_rag()
                print("[qa] Running rag_handle_message...", flush=True)
                state_copy = dict(rag_state)
                result = rag_handle_message(message, state_copy)
                return result

            try:
                print("[qa] Calling RAG...", flush=True)
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_call_rag)
                    result = future.result(timeout=QA_API_TIMEOUT_SEC)
                reply = result.get("reply") or ""
                sources = result.get("sources") or []
                rag_state.update(result.get("state") or {})
                print("[qa] RAG completed successfully.", flush=True)
            except concurrent.futures.TimeoutError:
                print(f"[qa] ERROR: Request timed out after {QA_API_TIMEOUT_SEC}s.", flush=True)
                reply = "The request took too long. Please try again or rephrase your question."
                sources = []
            except Exception as e:
                print("[qa] ERROR: Exception during RAG call:", flush=True)
                traceback.print_exc()
                reply = f"Sorry, something went wrong: {str(e)[:200]}. Please try again."
                sources = []

            # Strip [CITE:...] from displayed answer; preserve newlines for lists/bullets
            reply_display = re.sub(r"\[CITE:[^\]]+\]", "", reply).strip()
            # Do not collapse newlines so numbered lists and bullets render on separate lines
            messages.append({"role": "assistant", "content": reply_display, "sources": _slim_sources(sources)})
            session["qa_messages"] = messages
            session["rag_state"] = rag_state
            return jsonify({
                "messages": messages,
                "ui_state": ui_state,
                "sources": sources,
            })

        if action == "reset":
            session["qa_messages"] = _default_qa_messages()
            session["qa_ui_state"] = _default_qa_ui_state()
            session["rag_state"] = _default_rag_state()
            return jsonify({
                "messages": session["qa_messages"],
                "ui_state": session["qa_ui_state"],
            })
    except Exception as e:
        return jsonify({"error": str(e), "messages": session.get("qa_messages", _default_qa_messages()), "ui_state": session.get("qa_ui_state", _default_qa_ui_state())}), 500


@app.route("/health")
def health():
    """Health check endpoint."""
    from web.db import DB_PATH as DB_PATH_CHECK
    if not DB_PATH_CHECK.exists():
        return {"status": "error", "message": "Database not found"}, 500
    return {"status": "ok", "db_path": str(DB_PATH_CHECK)}


@app.route("/debug/hidden")
def debug_hidden():
    """Debug only: show counts and list of hidden resources (when Flask debug=True)."""
    if not app.debug:
        abort(404)
    from web.db import get_debug_hidden
    total_included, hidden_count, hidden_list = get_debug_hidden()
    return jsonify({
        "total_included": total_included,
        "hidden_count": hidden_count,
        "shown_count": total_included - hidden_count,
        "hidden": hidden_list,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="127.0.0.1", port=port, debug=True)
