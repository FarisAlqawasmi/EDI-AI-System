"""
Smoke test script to verify database connectivity and data availability.
Run from project root: python -m web.db_check
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# Path relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "summaries.sqlite"


def main() -> None:
    """Print 5 resources with id/title/author/has_summary."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Missing DB file: {DB_PATH.resolve()}")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        # Pull 5 Included resources
        rows = conn.execute(
            """
            SELECT
                id,
                title,
                author,
                summary,
                error,
                final_url,
                kind,
                raw_path,
                text_path
            FROM summaries
            WHERE LOWER(included_excluded) = 'included'
            ORDER BY
                CASE
                    WHEN CAST(id AS INTEGER) IS NOT NULL THEN CAST(id AS INTEGER)
                    ELSE 999999
                END,
                id
            LIMIT 5;
            """
        ).fetchall()

    print(f"✅ DB OK: {DB_PATH}")
    print(f"Showing {len(rows)} Included resources:\n")

    for r in rows:
        rid = (r["id"] or "").strip()
        title = (r["title"] or "").strip()
        author = (r["author"] or "").strip()
        summary = (r["summary"] or "").strip()
        error = (r["error"] or "").strip()
        has_summary = bool(summary) and not error

        print(f"- id: {rid}")
        print(f"  title: {title}")
        print(f"  author: {author if author else '[missing]'}")
        print(f"  kind: {r['kind'] or '[unknown]'}")
        print(f"  has_summary: {has_summary}")
        if error:
            print(f"  error: {error}")
        print()

    # Count total Included resources
    with sqlite3.connect(DB_PATH) as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM summaries WHERE included_excluded = 'Included'"
        ).fetchone()[0]
        with_summary = conn.execute(
            """
            SELECT COUNT(*) FROM summaries
            WHERE LOWER(included_excluded) = 'included'
                AND summary IS NOT NULL
                AND summary != ''
                AND (error IS NULL OR error = '')
            """
        ).fetchone()[0]

    print(f"Total Included resources: {total}")
    print(f"Resources with summaries: {with_summary}")
    print(f"Resources without summaries: {total - with_summary}")
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
