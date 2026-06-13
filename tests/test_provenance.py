"""Tests for ``frustratometer.utils.provenance.autolog_run``.

No mocks: every test uses ``tmp_path`` and (where relevant) real
``git`` subprocess calls. ``GIT_CEILING_DIRECTORIES`` caps git's upward
search at the tmp directory's parent so tests never accidentally see the
ambient repository they live in.
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from frustratometer.utils import autolog_run


# ─── Scaffolding ────────────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True, capture_output=True)


def _git_init_repo(path: Path) -> None:
    """Init a repo at ``path`` with one committed file (``seed.txt``)."""
    _run(["git", "init", "-q"], path)
    _run(["git", "config", "user.email", "test@test"], path)
    _run(["git", "config", "user.name", "Test"], path)
    (path / "seed.txt").write_text("seed\n")
    _run(["git", "add", "seed.txt"], path)
    _run(["git", "commit", "-q", "-m", "seed"], path)
    # Force-rename to "main" regardless of the host's init.defaultBranch.
    _run(["git", "branch", "-M", "main"], path)


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines()]


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    """chdir into ``tmp_path`` and prevent git from walking above it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    return tmp_path


# ─── Tests ──────────────────────────────────────────────────────────────────

def test_record_shape_and_run_id(in_tmp):
    record = autolog_run()

    [parsed] = _read_jsonl(in_tmp / "run_log.jsonl")
    assert parsed == record

    assert parsed["schema_name"] == "autolog_run"
    assert parsed["schema_version"] == 1
    assert re.fullmatch(r"r\d{6}-[0-9a-f]{4}", parsed["run_id"])

    expected_top = [
        "schema_name", "schema_version", "run_id", "timestamp_utc",
        "cwd", "results_dir", "log_path",
        "hostname", "platform", "pid", "program", "git", "python",
    ]
    assert list(parsed.keys())[: len(expected_top)] == expected_top

    # Timestamp is ISO-8601 and parses
    datetime.fromisoformat(parsed["timestamp_utc"])
    assert parsed["program"]["rerun_command"]


def test_path_resolution_neither(in_tmp):
    autolog_run()
    assert (in_tmp / "run_log.jsonl").exists()
    assert (in_tmp / "commands.log").exists()


def test_path_resolution_results_dir(in_tmp):
    rd = in_tmp / "results" / "r1"
    autolog_run(results_dir=rd)
    assert (rd / "run_log.jsonl").exists()
    assert (in_tmp / "commands.log").exists()
    # commands.log lives in cwd, NOT under results_dir.
    assert not (rd / "commands.log").exists()


def test_path_resolution_log_path(in_tmp):
    target = in_tmp / "deep" / "nested" / "x.jsonl"
    autolog_run(log_path=target)
    assert target.exists()
    assert (in_tmp / "commands.log").exists()


def test_log_path_wins_over_results_dir(in_tmp):
    explicit = in_tmp / "explicit.jsonl"
    rd = in_tmp / "results" / "r1"
    autolog_run(results_dir=rd, log_path=explicit)
    assert explicit.exists()
    assert not (rd / "run_log.jsonl").exists()
    [parsed] = _read_jsonl(explicit)
    # Caller's results_dir is preserved as metadata even though log_path won.
    assert parsed["results_dir"] == str(rd)
    assert parsed["log_path"] == str(explicit)


def test_commands_log_line_format(in_tmp):
    rec1 = autolog_run(results_dir=in_tmp / "r1", extra={"i": 1})
    rec2 = autolog_run(results_dir=in_tmp / "r2", extra={"i": 2})

    lines = (in_tmp / "commands.log").read_text().splitlines()
    assert len(lines) == 2  # back-to-back, no blank separators

    for line, rec in zip(lines, (rec1, rec2)):
        assert line == f"{rec['program']['rerun_command']}  # {rec['run_id']}"


def test_strange_characters(in_tmp, monkeypatch):
    weird = [
        "frustratometer",
        "weird name with spaces",
        '--flag=val"with"quotes',
        "back\\slash",
        "ué",
    ]
    monkeypatch.setattr(sys, "argv", weird)
    autolog_run()

    [parsed] = _read_jsonl(in_tmp / "run_log.jsonl")
    # argv round-trips byte-for-byte through JSON.
    assert parsed["program"]["argv"] == weird

    # commands.log line is single-line, ends with `  # <run_id>`, and
    # the command portion shlex-round-trips.
    [line] = (in_tmp / "commands.log").read_text().splitlines()
    suffix = f"  # {parsed['run_id']}"
    assert line.endswith(suffix)
    assert shlex.split(line.removesuffix(suffix)) == ["frustratometer", *weird[1:]]


def test_git_metadata_clean(in_tmp):
    _git_init_repo(in_tmp)
    record = autolog_run()
    git = record["git"]
    assert git["available"] is True
    assert git["dirty"] is False
    assert re.fullmatch(r"[0-9a-f]{40}", git["commit"])
    assert re.fullmatch(r"[0-9a-f]{7,}", git["short_commit"])
    assert git["branch"] == "main"
    assert git["dirty_patch"] is None
    assert git["dirty_patch_staged"] is None
    assert git["untracked_files"] == []
    assert git["status_porcelain"] == []


def test_git_metadata_dirty_with_inline_patch(in_tmp):
    _git_init_repo(in_tmp)
    # Modify the tracked file with non-ASCII content — exercises the
    # patch-text encoding path through subprocess + JSON.
    (in_tmp / "seed.txt").write_text("seed\nseed modification: ué\n")
    # Add an untracked file.
    (in_tmp / "extra.txt").write_text("untracked\n")

    record = autolog_run()
    git = record["git"]
    assert git["dirty"] is True
    assert git["untracked_files"] == ["extra.txt"]
    assert any("seed.txt" in line and "M" in line for line in git["status_porcelain"])
    assert "?? extra.txt" in git["status_porcelain"]

    assert git["dirty_patch"] is not None
    assert git["dirty_patch"].startswith("diff --git ")
    assert "+seed modification" in git["dirty_patch"]

    # The JSONL line is a single line even though the patch string
    # contains embedded newlines — proves the embedded \n was correctly
    # JSON-escaped to \\n.
    raw = (in_tmp / "run_log.jsonl").read_text()
    assert raw.count("\n") == 1
    json.loads(raw)


def test_git_unavailable(in_tmp):
    record = autolog_run()
    assert record["git"] == {"available": False}
    # Rest of the record is still fully populated.
    assert record["schema_name"] == "autolog_run"
    assert record["schema_version"] == 1
    assert record["program"]["rerun_command"]
