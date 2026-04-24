"""sync.py — Export and import eabrain memory.db across machines.

Two layers:
  - export_db / import_db: content-layer primitives (verbatim copy / content_hash-dedup merge)
  - pull / push: transport layer on top of a git remote, best-effort (never raises)
"""

import os
import shutil
import sqlite3
import subprocess
import time
from datetime import datetime


def export_db(db_path: str, export_path: str) -> None:
    shutil.copy2(db_path, export_path)


def import_db(local_path: str, import_path: str) -> None:
    local = sqlite3.connect(local_path)
    local.row_factory = sqlite3.Row
    remote = sqlite3.connect(import_path)
    remote.row_factory = sqlite3.Row

    local_hashes = {
        row[0] for row in local.execute("SELECT content_hash FROM observations").fetchall()
    }
    local_session_ids = {
        row[0] for row in local.execute("SELECT id FROM sessions").fetchall()
    }

    for row in remote.execute("SELECT * FROM sessions").fetchall():
        row = dict(row)
        if row["id"] not in local_session_ids:
            local.execute(
                "INSERT INTO sessions (id, project, started_at, ended_at, summary) VALUES (?, ?, ?, ?, ?)",
                (row["id"], row["project"], row["started_at"], row["ended_at"], row["summary"]),
            )
        else:
            local_row = local.execute("SELECT summary FROM sessions WHERE id = ?", (row["id"],)).fetchone()
            if row["summary"] and (not local_row["summary"] or len(row["summary"]) > len(local_row["summary"])):
                local.execute("UPDATE sessions SET summary = ? WHERE id = ?", (row["summary"], row["id"]))

    for row in remote.execute("SELECT * FROM observations").fetchall():
        row = dict(row)
        if row["content_hash"] not in local_hashes:
            local.execute(
                "INSERT INTO observations (id, session_id, project, type, content, content_hash, embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (row["id"], row["session_id"], row["project"], row["type"],
                 row["content"], row["content_hash"], row["embedding"], row["created_at"]),
            )
            local_hashes.add(row["content_hash"])

    local.commit()
    local.close()
    remote.close()


# ---------------------------------------------------------------------------
# Transport layer — git-backed sync
# ---------------------------------------------------------------------------

def _git(repo: str, *args: str) -> tuple[int, str]:
    r = subprocess.run(
        ["git", *args], cwd=repo,
        capture_output=True, text=True,
    )
    return r.returncode, (r.stderr or r.stdout).strip()


def _log(eabrain_dir: str, msg: str) -> None:
    os.makedirs(eabrain_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(os.path.join(eabrain_dir, "sync.log"), "a", encoding="utf-8") as f:
        f.write(f"{ts} {msg}\n")


def _is_git_repo(repo: str | None) -> bool:
    return bool(repo) and os.path.isdir(os.path.join(repo, ".git"))


def last_push_timestamp(repo: str | None) -> int | None:
    """Unix timestamp of the most recent commit touching memory.db, or None."""
    if not _is_git_repo(repo):
        return None
    rc, out = _git(repo, "log", "-1", "--format=%ct", "--", "memory.db")
    if rc != 0 or not out:
        return None
    try:
        return int(out)
    except ValueError:
        return None


def pull(db_path: str, repo: str | None, eabrain_dir: str) -> None:
    """Fetch origin/main, reset to it, import merged remote db. Best-effort."""
    if not _is_git_repo(repo):
        return
    rc, err = _git(repo, "fetch", "--quiet", "origin", "main")
    if rc != 0:
        _log(eabrain_dir, f"pull: fetch failed: {err}")
        return
    rc, err = _git(repo, "reset", "--hard", "--quiet", "origin/main")
    if rc != 0:
        _log(eabrain_dir, f"pull: reset failed: {err}")
        return
    repo_db = os.path.join(repo, "memory.db")
    if not os.path.exists(repo_db):
        return
    try:
        import_db(db_path, repo_db)
    except Exception as e:
        _log(eabrain_dir, f"pull: import failed: {e}")


def push(db_path: str, repo: str | None, eabrain_dir: str, attempts: int = 3) -> None:
    """Race-safe push: fetch → reset → import → export → commit → push, with retry.

    Safe under concurrent pushes because import_db is content_hash-idempotent:
    each retry absorbs any newly-pushed remote rows before re-exporting.
    """
    if not _is_git_repo(repo):
        return
    repo_db = os.path.join(repo, "memory.db")

    for attempt in range(attempts):
        rc, err = _git(repo, "fetch", "--quiet", "origin", "main")
        if rc != 0:
            _log(eabrain_dir, f"push: fetch failed: {err}")
            return
        rc, err = _git(repo, "reset", "--hard", "--quiet", "origin/main")
        if rc != 0:
            _log(eabrain_dir, f"push: reset failed: {err}")
            return

        if os.path.exists(repo_db):
            try:
                import_db(db_path, repo_db)
            except Exception as e:
                _log(eabrain_dir, f"push: import failed: {e}")
                return

        export_db(db_path, repo_db)

        rc, _ = _git(repo, "diff", "--quiet", "--", "memory.db")
        if rc == 0:
            return  # nothing to push

        _git(repo, "add", "memory.db")
        msg = f"sync {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}"
        rc, err = _git(repo, "commit", "--quiet", "-m", msg)
        if rc != 0:
            _log(eabrain_dir, f"push: commit failed: {err}")
            return

        rc, err = _git(repo, "push", "--quiet", "origin", "main")
        if rc == 0:
            return  # success

        _log(eabrain_dir, f"push: attempt {attempt + 1}/{attempts} rejected ({err}), retrying")
        time.sleep(1.5 * (attempt + 1))

    _log(eabrain_dir, f"push: all {attempts} attempts failed, giving up")
