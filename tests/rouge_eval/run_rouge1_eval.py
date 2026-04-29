#!/usr/bin/env python3
"""
Compare system-generated summaries (generated/) against EDI Hub reference
summaries (edi_generated/) using ROUGE-1 F1 from the rouge-score package.

Install dependency (once), from project root with your venv activated:
    python3 -m pip install rouge-score
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional dependency: fail fast with a clear install hint (no auto-install).
# ---------------------------------------------------------------------------
try:
    from rouge_score import rouge_scorer
except ImportError:
    print(
        "Error: the 'rouge-score' package is not installed.\n"
        "Install it with:\n"
        "  python3 -m pip install rouge-score\n"
        "(Run from your activated virtual environment if you use one.)",
        file=sys.stderr,
    )
    sys.exit(1)

# Resource IDs to evaluate; files are named resource<ID>.txt (no spaces).
RESOURCE_IDS = (9, 40, 52, 54, 62, 76)

# This script lives in tests/rouge_eval/
SCRIPT_DIR = Path(__file__).resolve().parent
GENERATED_DIR = SCRIPT_DIR / "generated"
EDI_DIR = SCRIPT_DIR / "edi_generated"
OUTPUT_CSV = SCRIPT_DIR / "rouge1_results.csv"


def read_text(path: Path) -> str:
    """Load file as UTF-8 text, normalised for scoring."""
    return path.read_text(encoding="utf-8").strip()


def main() -> None:
    # Reference = EDI Hub gold summary; prediction = our system summary.
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

    rows: list[tuple[int, str, str, float]] = []

    for rid in RESOURCE_IDS:
        gen_name = f"resource{rid}.txt"
        gen_path = GENERATED_DIR / gen_name
        edi_path = EDI_DIR / gen_name

        if not gen_path.is_file():
            print(f"Error: missing generated file: {gen_path}", file=sys.stderr)
            sys.exit(1)
        if not edi_path.is_file():
            print(f"Error: missing EDI reference file: {edi_path}", file=sys.stderr)
            sys.exit(1)

        reference = read_text(edi_path)
        prediction = read_text(gen_path)

        scores = scorer.score(reference, prediction)
        rouge_1_f1 = float(scores["rouge1"].fmeasure)

        # Store relative paths from this eval folder for the CSV.
        gen_rel = f"generated/{gen_name}"
        edi_rel = f"edi_generated/{gen_name}"
        rows.append((rid, gen_rel, edi_rel, rouge_1_f1))

    # --- Terminal: per-resource table ---
    print()
    print("ROUGE-1 F1 (reference = EDI Hub, prediction = generated)")
    print("-" * 40)
    print(f"{'Resource ID':<14} | {'ROUGE-1 F1':>12}")
    print("-" * 40)
    for rid, _, _, f1 in rows:
        print(f"{rid:<14} | {f1:>12.6f}")
    print("-" * 40)

    avg_f1 = sum(r[3] for r in rows) / len(rows)
    print(f"{'Average':<14} | {avg_f1:>12.6f}")
    print()

    # --- CSV ---
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Resource ID", "Generated File", "EDI Reference File", "ROUGE-1 F1"]
        )
        for rid, gen_rel, edi_rel, f1 in rows:
            writer.writerow([rid, gen_rel, edi_rel, f"{f1:.6f}"])

    print(f"Wrote: {OUTPUT_CSV}")
    print()


if __name__ == "__main__":
    main()
