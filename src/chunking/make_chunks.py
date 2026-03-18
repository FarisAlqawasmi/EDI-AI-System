from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, List, Tuple


TEXT_DIR = Path("data/processed/text")              # your extracted .txt files
OUT_DIR = Path("data/processed/chunks")             # output chunks
OUT_JSONL = OUT_DIR / "chunks.jsonl"                # one chunk per line


# ---- chunking knobs ----
TARGET_CHARS = 2000
OVERLAP_CHARS = 300

# If a single "block" is bigger than this, we force-split it further
MAX_BLOCK_CHARS = 2500

# Absolute safety: if we still can't split nicely, hard-split at this size
HARD_SPLIT_CHARS = 2200

# Safety: ignore ultra-short docs
MIN_DOC_CHARS = 200


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source_path: str
    chunk_index: int
    char_start: int
    char_end: int
    text: str


# -------------------------
# Normalisation / IDs
# -------------------------
def normalize_text(text: str) -> str:
    # Make whitespace consistent while preserving newlines meaningfully
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    # Keep at most 2 consecutive newlines (paragraph breaks)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def infer_doc_id_and_title(filename_stem: str) -> tuple[str, str]:
    """
    Filenames like: "{ID}_{Title...}"
    """
    if "_" in filename_stem:
        first, rest = filename_stem.split("_", 1)
        if first.strip().isdigit():
            return first.strip(), rest.strip()
    return "unknown", filename_stem


def extract_doc_id_from_path(path: Path) -> str:
    """
    Extract numeric document ID from filename.
    Expected format: {ID}_Something.txt
    """
    stem = path.stem  # filename without .txt
    if "_" in stem:
        first = stem.split("_", 1)[0].strip()
        if first.isdigit():
            return first
    return "unknown"


def sort_key_by_doc_id(path: Path):
    """
    Sort by numeric doc_id if possible, otherwise push to the end.
    """
    doc_id = extract_doc_id_from_path(path)
    if doc_id.isdigit():
        return (int(doc_id), path.name)
    return (10**12, path.name)  # unknown IDs go last


# -------------------------
# Splitting into blocks with indices
# -------------------------
_HEADING_LINE_RE = re.compile(
    r"""^
    (?:[A-Z][A-Za-z0-9 ,\-()]{0,80}|[0-9]+(?:\.[0-9]+){0,3}\s+[A-Za-z].{0,80})
    :?\s*$
    """,
    re.VERBOSE,
)


def _looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s) > 90:
        return False
    if s.endswith(":"):
        return True
    if _HEADING_LINE_RE.match(s):
        return True
    return False


def split_doc_into_blocks(doc_text: str) -> List[Tuple[int, int]]:
    """
    Returns list of (start, end) spans into doc_text.
    Strategy:
      1) Split on blank lines.
      2) Any block still too large: split by heading-like lines.
      3) Still too large: split by sentence boundaries (rough).
      4) Still too large: hard split.
    """
    if not doc_text:
        return []

    # 1) initial split on blank lines, preserving indices
    spans: List[Tuple[int, int]] = []
    start = 0
    for m in re.finditer(r"\n\s*\n+", doc_text):
        end = m.start()
        if end > start:
            spans.append((start, end))
        start = m.end()
    if start < len(doc_text):
        spans.append((start, len(doc_text)))

    def trim_span(a: int, b: int) -> Tuple[int, int]:
        while a < b and doc_text[a].isspace():
            a += 1
        while b > a and doc_text[b - 1].isspace():
            b -= 1
        return a, b

    spans = [trim_span(a, b) for (a, b) in spans]
    spans = [(a, b) for (a, b) in spans if b - a > 0]

    # 2) refine large blocks by heading lines
    refined: List[Tuple[int, int]] = []
    for (a, b) in spans:
        if (b - a) <= MAX_BLOCK_CHARS:
            refined.append((a, b))
            continue

        block = doc_text[a:b]
        lines = block.split("\n")

        # Compute absolute start offset of each line
        line_starts: List[int] = []
        pos = a
        for ln in lines:
            line_starts.append(pos)
            pos += len(ln) + 1  # + "\n"

        cut_points: List[int] = []
        for idx in range(1, len(lines)):
            if _looks_like_heading(lines[idx]):
                cut_points.append(line_starts[idx])

        if not cut_points:
            refined.append((a, b))
            continue

        prev = a
        for cp in cut_points:
            ta, tb = trim_span(prev, cp)
            if tb > ta:
                refined.append((ta, tb))
            prev = cp
        ta, tb = trim_span(prev, b)
        if tb > ta:
            refined.append((ta, tb))

    spans = refined

    # 3) sentence-ish splitting (rough fallback)
    sentence_refined: List[Tuple[int, int]] = []
    sentence_boundary = re.compile(r"(?<=[.!?])\s+")
    for (a, b) in spans:
        if (b - a) <= MAX_BLOCK_CHARS:
            sentence_refined.append((a, b))
            continue

        block = doc_text[a:b]
        pieces = sentence_boundary.split(block)
        if len(pieces) == 1:
            sentence_refined.append((a, b))
            continue

        # Pack pieces by approximate length (still guaranteed by hard split below)
        cur_start = a
        cur_len = 0
        for piece in pieces:
            piece_len = len(piece)
            if cur_len + piece_len > MAX_BLOCK_CHARS and cur_len > 0:
                ta, tb = trim_span(cur_start, cur_start + cur_len)
                if tb > ta:
                    sentence_refined.append((ta, tb))
                cur_start = cur_start + cur_len
                cur_len = 0
            cur_len += piece_len

        ta, tb = trim_span(cur_start, b)
        if tb > ta:
            sentence_refined.append((ta, tb))

    spans = sentence_refined

    # 4) hard-split any span still too large
    final_spans: List[Tuple[int, int]] = []
    for (a, b) in spans:
        if (b - a) <= HARD_SPLIT_CHARS:
            final_spans.append((a, b))
            continue

        cur = a
        while cur < b:
            nxt = min(b, cur + HARD_SPLIT_CHARS)
            ta, tb = trim_span(cur, nxt)
            if tb > ta:
                final_spans.append((ta, tb))
            cur = nxt

    return [(a, b) for (a, b) in final_spans if (b - a) > 0]


# -------------------------
# Packing blocks into chunks (with overlap)
# -------------------------
def pack_blocks_into_chunks(
    doc_text: str,
    block_spans: List[Tuple[int, int]],
    target_chars: int,
    overlap_chars: int,
) -> Iterator[Tuple[int, int, str]]:
    """
    Packs block spans into chunks aiming for target_chars.
    Returns (char_start, char_end, chunk_text) where indices refer to doc_text.
    """
    n = len(block_spans)
    i = 0
    last_emitted_end = -1

    while i < n:
        start = block_spans[i][0]
        end = start
        j = i

        while j < n:
            cand_end = block_spans[j][1]
            cand_len = cand_end - start

            if cand_len > target_chars and j > i:
                break

            end = cand_end
            j += 1

            # Stop once we roughly hit target
            if end - start >= target_chars and j > i:
                break

        # Ensure forward progress
        if end <= last_emitted_end:
            end = block_spans[i][1]
            j = i + 1

        chunk_text = doc_text[start:end].strip()
        yield start, end, chunk_text
        last_emitted_end = end

        if j >= n:
            break

        # Overlap handling: want next chunk to start about overlap_chars before end
        back_target = max(0, end - overlap_chars)

        # Find the block index whose start is <= back_target (closest from the right)
        k = j
        while k > 0 and block_spans[k - 1][0] > back_target:
            k -= 1

        # Make sure we don't get stuck
        if k <= i:
            i = j
        else:
            i = k


# -------------------------
# Main per-file build
# -------------------------
def build_chunks_for_file(path: Path) -> List[Chunk]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    doc_text = normalize_text(raw)

    if len(doc_text) < MIN_DOC_CHARS:
        return []

    block_spans = split_doc_into_blocks(doc_text)
    if not block_spans:
        return []

    doc_id, title = infer_doc_id_and_title(path.stem)

    chunks: List[Chunk] = []
    for idx, (start_pos, end_pos, chunk_text) in enumerate(
        pack_blocks_into_chunks(doc_text, block_spans, TARGET_CHARS, OVERLAP_CHARS)
    ):
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}:{idx}",
                doc_id=doc_id,
                title=title,
                source_path=str(path),
                chunk_index=idx,
                char_start=start_pos,
                char_end=end_pos,
                text=chunk_text,
            )
        )

    return chunks


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sort files by numeric doc_id (1,2,3,...,100,...)
    txt_files = sorted(TEXT_DIR.glob("*.txt"), key=sort_key_by_doc_id)
    if not txt_files:
        raise SystemExit(f"No .txt files found in {TEXT_DIR}. Run your download/extract step first.")

    total_docs = 0
    total_chunks = 0

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for p in txt_files:
            total_docs += 1
            chunks = build_chunks_for_file(p)
            total_chunks += len(chunks)

            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    print("Chunking complete.")
    print(f"Docs scanned: {total_docs}")
    print(f"Chunks written: {total_chunks}")
    print(f"Output: {OUT_JSONL}")
    print(f"TARGET_CHARS={TARGET_CHARS}, OVERLAP_CHARS={OVERLAP_CHARS}, MAX_BLOCK_CHARS={MAX_BLOCK_CHARS}")


if __name__ == "__main__":
    main()