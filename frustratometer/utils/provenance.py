"""Append-only run-manifest provenance logging.

Per call, ``autolog_run`` writes:

- ``commands.log`` — one line per run in the *original CWD*: the
  copy-paste runnable command followed by ``  # <run_id>``. The folder
  context is implicit (the file lives where it was written), and full
  metadata is in ``run_log.jsonl``.
- ``run_log.jsonl`` — one JSON record per run, in the results directory
  if provided, otherwise next to ``commands.log``.

Both files share a short ``run_id`` so a JSONL record and its
``commands.log`` block can be cross-referenced. Patch text from a dirty
git tree is embedded inside the JSONL record (``git.dirty_patch`` /
``git.dirty_patch_staged``).
"""
from __future__ import annotations

import json
import os
import platform
import secrets
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["autolog_run"]

SCHEMA_NAME = "autolog_run"
SCHEMA_VERSION = 1.0


def _run_quiet(cmd: list[str], cwd: Path) -> str | None:
    """Run a command, return stripped stdout, or ``None`` on any failure.

    ``errors="replace"`` keeps binary-tinged output (e.g. patches that touch
    files containing non-UTF-8 bytes) from raising ``UnicodeDecodeError``.
    """
    try:
        return subprocess.check_output(
            cmd,
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
            errors="replace",
        ).strip()
    except Exception:
        return None


def _git_info(cwd: Path) -> dict[str, Any]:
    git_root = _run_quiet(["git", "rev-parse", "--show-toplevel"], cwd)
    if git_root is None:
        return {"available": False}

    root = Path(git_root)
    status = _run_quiet(["git", "status", "--porcelain"], root) or ""
    porcelain = status.splitlines()
    untracked = [line[3:] for line in porcelain if line.startswith("??")]
    dirty = bool(porcelain)

    diff = _run_quiet(["git", "diff"], root) if dirty else None
    diff_staged = _run_quiet(["git", "diff", "--staged"], root) if dirty else None

    return {
        "available": True,
        "describe": _run_quiet(
            ["git", "describe", "--tags", "--always", "--dirty"], root
        ),
        "dirty": dirty,
        "commit": _run_quiet(["git", "rev-parse", "HEAD"], root),
        "short_commit": _run_quiet(["git", "rev-parse", "--short", "HEAD"], root),
        "branch": _run_quiet(["git", "branch", "--show-current"], root),
        "root": str(root),
        "status_porcelain": porcelain,
        "untracked_files": untracked,
        "dirty_patch": diff if diff else None,
        "dirty_patch_staged": diff_staged if diff_staged else None,
    }


def _make_run_id(now: datetime) -> str:
    """Compact, sortable-within-an-hour, collision-resistant id."""
    return f"r{now.strftime('%H%M%S')}-{secrets.token_hex(2)}"


def _resolve_paths(results_dir, log_path, cwd: Path) -> tuple[Path | None, Path]:
    """Resolve ``results_dir`` and ``log_path`` to absolute Paths.

    ``log_path`` wins over ``results_dir``. Returns ``(results_dir_abs,
    log_path_abs)`` where ``results_dir_abs`` is None if the caller did
    not pass one.
    """
    rd_abs: Path | None = None
    if results_dir is not None:
        rd = Path(results_dir)
        rd_abs = rd if rd.is_absolute() else cwd / rd

    if log_path is not None:
        lp = Path(log_path)
        lp_abs = lp if lp.is_absolute() else cwd / lp
    elif rd_abs is not None:
        lp_abs = rd_abs / "run_log.jsonl"
    else:
        lp_abs = cwd / "run_log.jsonl"

    return rd_abs, lp_abs


def autolog_run(
    results_dir=None,
    log_path=None,
    program_version=None,
    extra=None,
):
    """Append a provenance record for the current program run.

    Parameters
    ----------
    results_dir : str | Path | None
        Results directory. When provided (and ``log_path`` is None) the
        JSONL is written at ``results_dir/run_log.jsonl``.
    log_path : str | Path | None
        Explicit JSONL path. Wins over ``results_dir`` if both are given.
    program_version : str | None
        Caller-supplied version string, e.g. ``frustratometer.__version__``.
    extra : dict | None
        Caller-supplied metadata, serialized verbatim. Omitted from the
        record when ``None``.

    Returns
    -------
    dict
        The record that was appended.

    Side effects
    ------------
    - Appends one line to the resolved JSONL path.
    - Appends one line (``<command>  # <run_id>``) to ``<cwd>/commands.log``.
    """
    cwd = Path.cwd()
    rd_abs, lp_abs = _resolve_paths(results_dir, log_path, cwd)
    lp_abs.parent.mkdir(parents=True, exist_ok=True)
    commands_log = cwd / "commands.log"

    now = datetime.now(timezone.utc)
    run_id = _make_run_id(now)
    argv = list(sys.argv)
    rerun_command = shlex.join([Path(argv[0]).name, *argv[1:]])
    orig_argv = getattr(sys, "orig_argv", None)

    record: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp_utc": now.isoformat(),
        "cwd": str(cwd),
        "results_dir": str(rd_abs) if rd_abs is not None else None,
        "log_path": str(lp_abs),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "program": {
            "argv": argv,
            "orig_argv": list(orig_argv) if orig_argv is not None else None,
            "rerun_command": rerun_command,
            "script": str(Path(argv[0]).resolve()),
            "version": program_version,
        },
        "git": _git_info(cwd),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
    }
    if extra is not None:
        record["extra"] = extra

    line = json.dumps(record, sort_keys=False, ensure_ascii=True, default=str)
    with lp_abs.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

    # commands.log: <command>  # <run_id>, one line per run, no separator.
    with commands_log.open("a", encoding="utf-8") as f:
        f.write(f"{rerun_command}  # {run_id}\n")

    return record
