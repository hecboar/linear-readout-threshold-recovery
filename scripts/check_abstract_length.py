#!/usr/bin/env python3
"""Count the rendered length of the abstract and the highlights.

The 250-word cap was Elsevier's, from the Guide for Authors of the journal this manuscript
was first written for. The target is now TMLR, which publishes no abstract limit, so 300 is
an editorial choice of our own rather than a rule -- kept as a check because an abstract that
drifts past it stops being an abstract. The highlights file is an Elsevier artefact and TMLR
has no equivalent; the check tolerates its absence.

Historically: Elsevier caps the abstract (commonly 250 words) and each highlight bullet
at 85 characters. Both counts have to be taken *after* the generated numeric macros expand,
which is why this cannot be eyeballed from the source.

The abstract limit is passed in rather than hard-coded, because the authoritative value lives
in the journal's Guide for Authors — see internal/PLAN.md item 0.2.

Run: ``python scripts/check_abstract_length.py [--limit N]``
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "paper"
MAIN = ROOT / "tmlr.tex"
ABSTRACT = ROOT / "sections" / "abstract.tex"
NUMBERS = ROOT / "generated" / "numbers.tex"
HIGHLIGHTS = ROOT / "highlights.txt"

HIGHLIGHT_LIMIT = 85


def expand_macros(text: str) -> str:
    """Substitute the generated \\Foo macros with their values."""
    values = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}",
                             NUMBERS.read_text(encoding="utf-8")))
    for name, value in sorted(values.items(), key=lambda kv: -len(kv[0])):
        text = text.replace("\\" + name + "{}", value).replace("\\" + name, value)
    return text


def word_count(tex: str) -> int:
    tex = expand_macros(tex)
    tex = re.sub(r"\\[a-zA-Z]+\*?", " ", tex)   # drop remaining control sequences
    tex = re.sub(r"[{}$\\~]", " ", tex)
    return len([w for w in tex.split() if any(c.isalnum() for c in w)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300,
                    help="abstract word limit; 300 is our own editorial target, not a venue rule (see below)")
    args = ap.parse_args()

    # tmlr.tex \input{}s the abstract rather than inlining it, so both forms are accepted: the
    # inline one for a single-file source and the file for the split one.
    body = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
                     MAIN.read_text(encoding="utf-8"), re.S)
    if body is not None and "input" in body.group(1) and ABSTRACT.exists():
        body = re.match(r"(.*)", ABSTRACT.read_text(encoding="utf-8"), re.S)
    elif body is None and ABSTRACT.exists():
        body = re.match(r"(.*)", ABSTRACT.read_text(encoding="utf-8"), re.S)
    if body is None:
        print(f"FAIL: no abstract found in {MAIN.name} or sections/abstract.tex")
        return 1

    ok = True
    n = word_count(body.group(1))
    print(f"abstract: {n} words (limit {args.limit})")
    if n > args.limit:
        print(f"FAIL: abstract is {n - args.limit} words over")
        ok = False

    for line in HIGHLIGHTS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"- (.*?)\s*\[(\d+)\]\s*$", line)
        if not m:
            continue
        text, claimed = m.group(1), int(m.group(2))
        length = len(text)
        status = "ok" if length <= HIGHLIGHT_LIMIT else "TOO LONG"
        print(f"highlight: {length:3d} chars [{status}] {text}")
        if length > HIGHLIGHT_LIMIT:
            ok = False
        if length != claimed:
            print(f"FAIL: bracketed count says {claimed}, actual is {length}")
            ok = False

    print("OK: abstract and highlights within limits" if ok else "FAILURES above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
