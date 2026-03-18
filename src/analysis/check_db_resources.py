from __future__ import annotations

import sqlite3
from pathlib import Path


DB_PATH = Path("data/processed/summaries.sqlite")


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Missing DB file: {DB_PATH.resolve()}")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        # Pull 5 resources (order by id if possible)
        rows = conn.execute(
            """
            SELECT
                id,
                title,
                author,
                summary,
                error
            FROM summaries
            ORDER BY
                CASE
                    WHEN CAST(id AS INTEGER) IS NOT NULL THEN CAST(id AS INTEGER)
                    ELSE 999999
                END,
                id
            LIMIT 5;
            """
        ).fetchall()

    print(f"DB OK: {DB_PATH}")
    print(f"Showing {len(rows)} rows:\n")

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
        print(f"  has_summary: {has_summary}")
        if error:
            print(f"  error: {error}")
        print()

    print("✅ Done.")


if __name__ == "__main__":
    main()