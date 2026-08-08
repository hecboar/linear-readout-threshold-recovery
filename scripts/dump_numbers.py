#!/usr/bin/env python3
"""Print the generated macros as plain name/value pairs.

Handy when quoting the manuscript's measured quantities outside LaTeX -- in a referee
briefing, a talk, or a response letter -- without re-deriving them by hand.

Run: ``python scripts/dump_numbers.py [substring ...]``
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

NUMBERS = Path(__file__).resolve().parents[1] / "paper" / "generated" / "numbers.tex"
PATTERN = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}")


def main() -> int:
    if not NUMBERS.exists():
        print(f"missing {NUMBERS}; run scripts/make_numbers.py first")
        return 1
    pairs = PATTERN.findall(NUMBERS.read_text(encoding="utf-8"))
    filters = [a.lower() for a in sys.argv[1:]]
    shown = 0
    for name, value in pairs:
        if filters and not any(f in name.lower() for f in filters):
            continue
        print(f"{name:34s} {value}")
        shown += 1
    print(f"\n{shown} of {len(pairs)} macros")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
