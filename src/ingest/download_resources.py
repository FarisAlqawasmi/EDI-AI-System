from __future__ import annotations

import re
import csv
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlsplit, urlunsplit, quote, urlparse

import pandas as pd
import requests
from tqdm import tqdm

# HTML extraction
import trafilatura

# PDF extraction
from pypdf import PdfReader


# =========================
# Paths & configuration
# =========================
XLSX_PATH = Path("data/raw/Trimmed Resources.xlsx")
SHEET_NAME = "Resource centre taxonomy and re"

RAW_DIR = Path("data/raw/downloads")          # raw evidence
TEXT_DIR = Path("data/processed/text")        # extracted text
MANIFEST_PATH = Path("data/processed/manifest.csv")

RAW_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

# Column names (must exist in your sheet)
COL_ID = "ID"
COL_TITLE = "Title"
COL_AUTHOR = "Author"
COL_LINK = "Link"

# Optional columns (if present, we log them)
COL_INCLUDED = "Included_Excluded"
COL_TYPE = "Type of Resource (Archived)"
COL_FORMAT = "Resource Format (Modified)"

# Text length thresholds (not “quality truth”, just flags)
MIN_CHARS_FLAG = 200   # below this is very likely junk/blocked/nav-only
MIN_CHARS_OK = 800     # above this is usually “real content”


# =========================
# Media detection (conservative)
# =========================
# We ONLY auto-skip if it's clearly a media-only platform or direct media file.
# We DO NOT skip “bbc.co.uk” etc. because those can be real articles with text.
MEDIA_DOMAINS_STRICT = {
    # YouTube family
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "music.youtube.com", "youtu.be", "www.youtu.be",
    "youtube-nocookie.com", "www.youtube-nocookie.com",

    # Pure video platforms
    "vimeo.com", "www.vimeo.com", "player.vimeo.com",

    # Mostly-audio platforms
    "soundcloud.com", "www.soundcloud.com",
    "open.spotify.com",
    "podcasts.apple.com",

    # Typical short-video/social video (often nav-heavy)
    "tiktok.com", "www.tiktok.com",
    "instagram.com", "www.instagram.com",
}

MEDIA_FILE_EXTENSIONS = {
    ".mp4", ".webm", ".mov", ".m4v", ".avi", ".mkv", ".flv", ".wmv",
    ".mp3", ".m4a", ".wav", ".ogg", ".aac",
}

# URL path patterns that are strong signals *only for strict media domains*
YOUTUBE_PATH_KEYWORDS = ("/watch", "/shorts", "/embed")


def normalize_domain(netloc: str) -> str:
    n = (netloc or "").lower().strip()
    return n.split(":", 1)[0]


def is_strict_media_url(url: str) -> bool:
    """
    Skip only when we're confident it's media-only.
    - strict media domains (YouTube, Vimeo, Spotify, etc.)
    - direct media file extensions
    """
    try:
        p = urlparse(url.strip())
        domain = normalize_domain(p.netloc)
        path = (p.path or "").lower()

        # direct media file links
        if any(path.endswith(ext) for ext in MEDIA_FILE_EXTENSIONS):
            return True

        # strict media platforms
        if domain in MEDIA_DOMAINS_STRICT:
            # extra tightening for YouTube paths (optional, but harmless)
            if "youtube" in domain or "youtu.be" in domain:
                full = (path or "").lower()
                if any(k in full for k in YOUTUBE_PATH_KEYWORDS) or "watch" in (p.query or "").lower():
                    return True
                return True  # still media platform
            return True

        # allow subdomains like foo.youtube.com
        if domain.endswith(".youtube.com") or domain.endswith(".vimeo.com"):
            return True

        return False
    except Exception:
        return False


def save_url_stub(base_name: str, url: str) -> Path:
    """
    Write a small evidence stub instead of downloading full media pages.
    """
    out = RAW_DIR / f"{base_name}.url.txt"
    out.write_text(url.strip() + "\n", encoding="utf-8")
    return out


# =========================
# Included/Excluded filtering
# =========================
def is_included_value(val) -> bool:
    """
    Return True if the Included_Excluded cell means INCLUDED.
    Robust to variants: Included/include/yes/true/1 and extra text.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False

    s = str(val).strip().lower()

    # exact/common
    if s in {"included", "include", "yes", "y", "true", "1"}:
        return True
    if s in {"excluded", "exclude", "no", "n", "false", "0"}:
        return False

    # fuzzy
    if "includ" in s and "exclud" not in s:
        return True
    if "exclud" in s:
        return False

    # default safe: not included
    return False


def filter_included_only(df: pd.DataFrame) -> pd.DataFrame:
    if COL_INCLUDED not in df.columns:
        raise ValueError(
            f"Expected column '{COL_INCLUDED}' but it was not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    # quick breakdown for sanity check
    vc = df[COL_INCLUDED].astype(str).str.strip().value_counts(dropna=False)
    print("[included-filter] Included_Excluded value counts (top 20):")
    print(vc.head(20))

    before = len(df)
    df2 = df[df[COL_INCLUDED].apply(is_included_value)].copy()
    after = len(df2)
    print(f"[included-filter] Kept {after}/{before} rows marked as Included.")
    return df2


# =========================
# Helpers
# =========================
def safe_filename(text: str, max_len: int = 140) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = re.sub(r"[^\w\-(). ]+", "", text)
    text = text.replace(" ", "_")
    return text[:max_len]


def clean_url(url: str) -> str:
    """
    Fix common spreadsheet-copy issues:
    - leading/trailing spaces
    - invisible chars from Excel/Word
    - spaces in URL path/query -> percent encode
    """
    if url is None:
        return ""

    u = str(url).strip()
    u = u.replace("\u00a0", "").replace("\u200b", "").replace("\ufeff", "")

    parts = urlsplit(u)
    path = quote(parts.path, safe="/%")
    query = quote(parts.query, safe="=&%")
    return urlunsplit((parts.scheme, parts.netloc, path, query, parts.fragment))


def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
    })
    return s


def fetch_url(
    session: requests.Session,
    url: str,
    timeout: int = 40,
    retries: int = 2,
) -> Tuple[Optional[requests.Response], Optional[str]]:
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = session.get(url, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            return r, None
        except Exception as e:
            last_err = str(e)
            time.sleep(1.0 * (attempt + 1))
    return None, last_err


def detect_kind(response: requests.Response) -> str:
    """
    Decide content type from headers; fall back to URL.
    """
    ctype = (response.headers.get("Content-Type") or "").lower()

    if "application/pdf" in ctype:
        return "pdf"
    if "text/html" in ctype or "application/xhtml" in ctype:
        return "html"

    # fallback: guess from final URL
    if response.url.lower().endswith(".pdf"):
        return "pdf"
    return "html"


def is_media_response(response: requests.Response) -> bool:
    """
    Detect if content returned is actually media after redirects.
    """
    ctype = (response.headers.get("Content-Type") or "").lower()
    if ctype.startswith("video/") or ctype.startswith("audio/"):
        return True

    if "application/octet-stream" in ctype:
        url_lower = (response.url or "").lower()
        return any(url_lower.endswith(ext) for ext in MEDIA_FILE_EXTENSIONS)

    return False


def save_raw(kind: str, base_name: str, response: requests.Response) -> Path:
    ext = ".pdf" if kind == "pdf" else ".html"
    out = RAW_DIR / f"{base_name}{ext}"
    out.write_bytes(response.content)
    return out


def extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return normalize_text("\n\n".join(pages))


def extract_text_from_html(html: bytes, url: str) -> str:
    downloaded = html.decode("utf-8", errors="ignore")
    extracted = trafilatura.extract(
        downloaded,
        url=url,
        include_comments=False,
        include_tables=True,
    )
    if extracted and extracted.strip():
        return normalize_text(extracted)

    # fallback: very rough
    rough = re.sub(r"<[^>]+>", " ", downloaded)
    return normalize_text(rough)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def write_manifest_row(row: dict, header: list[str]) -> None:
    exists = MANIFEST_PATH.exists()
    with MANIFEST_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def get_optional(row, col_name: str) -> str:
    if col_name in row and pd.notna(row[col_name]):
        return str(row[col_name]).strip()
    return ""


# =========================
# Main pipeline
# =========================
def main(limit: Optional[int] = None):
    df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME)

    # Must have a link
    df = df[df[COL_LINK].notna()].copy()

    # NEW: Only keep Included resources
    df = filter_included_only(df)

    if limit is not None:
        df = df.head(limit)

    # If you re-run, you probably want a fresh manifest
    # Comment this out if you prefer to append.
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()

    session = get_session()

    manifest_header = [
        "id",
        "title",
        "author",
        "included_excluded",
        "resource_type",
        "resource_format",
        "original_url",
        "clean_url",
        "final_url",
        "kind",
        "raw_path",
        "text_path",
        "status",
        "text_len",
        "error",
    ]

    print(f"Processing {len(df)} INCLUDED resources...")

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Fetch+Extract"):
        rid = str(r.get(COL_ID, "")).strip()
        title = str(r.get(COL_TITLE, "untitled")).strip()
        author = str(r.get(COL_AUTHOR, "")).strip()
        original_url = str(r.get(COL_LINK, "")).strip()
        clean = clean_url(original_url)

        included_excluded = get_optional(r, COL_INCLUDED)
        resource_type = get_optional(r, COL_TYPE)
        resource_format = get_optional(r, COL_FORMAT)

        base = safe_filename(f"{rid}_{title}" if rid else title)

        result = {
            "id": rid,
            "title": title,
            "author": author,
            "included_excluded": included_excluded,
            "resource_type": resource_type,
            "resource_format": resource_format,
            "original_url": original_url,
            "clean_url": clean,
            "final_url": "",
            "kind": "",
            "raw_path": "",
            "text_path": "",
            "status": "FAILED",
            "text_len": 0,
            "error": "",
        }

        if not clean.startswith("http"):
            result["error"] = "Invalid URL"
            write_manifest_row(result, manifest_header)
            continue

        # ---- skip only strict media URLs (YouTube/Vimeo/Spotify/etc.) ----
        if is_strict_media_url(clean):
            result["status"] = "SKIPPED_MEDIA"
            result["kind"] = "media"
            result["raw_path"] = str(save_url_stub(base, clean))
            result["error"] = "Strict media-only URL (skipped at ingestion)"
            write_manifest_row(result, manifest_header)
            continue

        resp, err = fetch_url(session, clean)
        if resp is None:
            result["error"] = err or "Fetch failed"
            write_manifest_row(result, manifest_header)
            continue

        result["final_url"] = resp.url

        # If redirect ends up being media, skip but log
        if is_media_response(resp):
            result["status"] = "SKIPPED_MEDIA"
            result["kind"] = "media"
            result["raw_path"] = str(save_url_stub(base, resp.url))
            result["error"] = f"Media content-type: {resp.headers.get('Content-Type')}"
            write_manifest_row(result, manifest_header)
            continue

        kind = detect_kind(resp)
        result["kind"] = kind

        raw_path = save_raw(kind, base, resp)
        result["raw_path"] = str(raw_path)

        text = (
            extract_text_from_pdf(raw_path)
            if kind == "pdf"
            else extract_text_from_html(resp.content, resp.url)
        )

        text_path = TEXT_DIR / f"{base}.txt"
        text_path.write_text(text, encoding="utf-8")
        result["text_path"] = str(text_path)

        tlen = len(text)
        result["text_len"] = tlen

        # Status flags (not hard truth—just helpful labels)
        if tlen < MIN_CHARS_FLAG:
            result["status"] = "TOO_SHORT"
            result["error"] = f"Extracted text length={tlen} (<{MIN_CHARS_FLAG})"
        elif tlen < MIN_CHARS_OK:
            result["status"] = "OK_BUT_SHORT"
            result["error"] = f"Extracted text length={tlen} (<{MIN_CHARS_OK})"
        else:
            result["status"] = "OK"
            result["error"] = ""

        write_manifest_row(result, manifest_header)

    print("Done.")
    print(f"Raw evidence: {RAW_DIR}")
    print(f"Extracted text: {TEXT_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main(limit=None)