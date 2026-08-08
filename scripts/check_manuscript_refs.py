#!/usr/bin/env python3
"""Check the manuscript's internal cross-references.

Elsevier requires every figure and table to be referred to in the text. LaTeX itself only
warns about the reverse failure (a ``\\ref`` with no ``\\label``), so this catches the float
that was written and then never cited.

Run: ``python scripts/check_manuscript_refs.py``
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1] / "paper" / "main.tex"

LABEL = re.compile(r"\\label\{([^}]+)\}")
REF = re.compile(r"\\(?:ref|eqref|autoref)\{([^}]+)\}")
FLOAT_PREFIXES = ("fig:", "tab:", "alg:")


def main() -> int:
    text = MAIN.read_text(encoding="utf-8")
    labels = set(LABEL.findall(text))
    refs = set(REF.findall(text))

    floats = {lab for lab in labels if lab.startswith(FLOAT_PREFIXES)}
    uncited = sorted(floats - refs)
    dangling = sorted(refs - labels)

    print(f"{len(labels)} labels, {len(refs)} distinct references, {len(floats)} floats")
    ok = True
    if uncited:
        print(f"FAIL: floats never referred to in the text: {uncited}")
        ok = False
    if dangling:
        print(f"FAIL: references with no matching label: {dangling}")
        ok = False
    if ok:
        print("OK: every figure, table and algorithm is cited; no dangling references")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
