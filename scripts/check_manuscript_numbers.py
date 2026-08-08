#!/usr/bin/env python3
"""Verify that every measured number in the manuscript comes from the saved results.

Three checks, all of which must pass (exit code 0):

1. ``paper/generated/numbers.tex`` is byte-identical to what
   ``scripts/make_numbers.py`` produces from the current contents of ``results/``.
2. Every macro the manuscript uses from that file is actually defined in it.
3. No macro is defined but unused (a stale macro usually means a claim was deleted from the
   text while its number was left behind).

Run:  ``python scripts/check_manuscript_numbers.py``
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_numbers import OUT, build  # noqa: E402

PAPER = REPO_ROOT / "paper"
TEX_SOURCES = ["main.tex"]
DEFINE_RE = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}")
USE_RE = re.compile(r"\\([A-Za-z]+)(?![A-Za-z])")


def main() -> int:
    failures: list[str] = []

    # ---- 1. the generated file matches the results ----
    if not OUT.exists():
        print(f"FAIL: {OUT.relative_to(REPO_ROOT)} does not exist; run scripts/make_numbers.py")
        return 1
    on_disk = OUT.read_text(encoding="utf-8")
    regenerated = build().render()
    if on_disk != regenerated:
        failures.append(
            "generated/numbers.tex is out of date with respect to results/; "
            "re-run scripts/make_numbers.py")
        d_disk = dict(DEFINE_RE.findall(on_disk) and
                      re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", on_disk))
        d_new = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", regenerated))
        for k in sorted(set(d_disk) | set(d_new)):
            if d_disk.get(k) != d_new.get(k):
                failures.append(f"    {k}: file={d_disk.get(k)!r} results={d_new.get(k)!r}")
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", regenerated))
    print(f"checked {len(defined)} generated macros")

    # ---- 2 and 3. the manuscript uses exactly the macros that are defined ----
    text = ""
    for name in TEX_SOURCES:
        path = PAPER / name
        if not path.exists():
            failures.append(f"manuscript source {path.relative_to(REPO_ROOT)} not found")
            continue
        text += path.read_text(encoding="utf-8")
    for extra in sorted((PAPER / "sections").glob("*.tex")) if (PAPER / "sections").exists() else []:
        text += extra.read_text(encoding="utf-8")

    used = set(USE_RE.findall(text)) & defined
    undefined_used = {m for m in USE_RE.findall(text)
                      if m.startswith(("Eone", "Etwo", "Ethree", "Efour", "Efive", "Esix",
                                       "Dur", "Env"))} - defined
    if undefined_used:
        failures.append(f"manuscript uses undefined generated macros: {sorted(undefined_used)}")
    unused = defined - used
    if unused:
        failures.append(f"generated macros never used in the manuscript: {sorted(unused)}")

    print(f"manuscript uses {len(used)} of them")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("OK: every measured number in the manuscript traces to results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
