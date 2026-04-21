"""sync.py — Export and import eabrain memory.db across machines."""

import os
import shutil
import sqlite3


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
