"""D8 says every run must record its commit and dirty status, and that a dirty tree must name the
files that make it dirty. A record that misnames those files is worse than one that omits them, so
the parsing gets a test of its own.

The defect this closes: `_git` strips its output, which removes the leading space of a
`" M path"` porcelain line -- but only the *first* line, since later lines keep the newline before
them. The old fixed-offset slice `ln[3:]` therefore ate the first character of exactly one path, and
one wrong entry in a list of correct ones is the kind of thing a reader skims past.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrtr import runlog  # noqa: E402

# Porcelain lines as git emits them, already stripped by `_git`: the first has lost its leading
# space, the rest have not.
STRIPPED_STATUS = "\n".join([
    "M  staged/modified.json",
    " M results/e7/raw/cell.json",
    "?? results/e7_stageB/",
    "A  added.py",
])


def test_dirty_files_survive_the_stripped_first_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runlog, "_git",
                        lambda *a: STRIPPED_STATUS if a[:2] == ("status", "--porcelain") else "x")
    g = runlog.git_provenance()
    assert g["dirty"] is True
    assert g["dirty_file_count"] == 4
    assert g["dirty_files"] == ["staged/modified.json", "results/e7/raw/cell.json",
                                "results/e7_stageB/", "added.py"]


def test_paths_containing_spaces_are_not_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runlog, "_git",
                        lambda *a: " M paper submission/main tex.tex"
                        if a[:2] == ("status", "--porcelain") else "x")
    assert runlog.git_provenance()["dirty_files"] == ["paper submission/main tex.tex"]


def test_clean_tree_reports_not_dirty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runlog, "_git",
                        lambda *a: "" if a[:2] == ("status", "--porcelain") else "x")
    g = runlog.git_provenance()
    assert g["dirty"] is False and g["dirty_files"] == [] and g["dirty_file_count"] == 0
