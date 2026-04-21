"""memory.py — SQLite storage for eabrain observations and sessions."""

import hashlib
import os
import sqlite3
import uuid
from datetime import datetime, timezone


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    project     TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    summary     TEXT
);

CREATE TABLE IF NOT EXISTS observations (
    id           TEXT PRIMARY KEY,
    session_id   TEXT REFERENCES sessions(id),
    project      TEXT NOT NULL,
    type         TEXT NOT NULL,
    content      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding    BLOB,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_obs_project ON observations(project);
CREATE INDEX IF NOT EXISTS idx_obs_type ON observations(type);
CREATE INDEX IF NOT EXISTS idx_obs_created ON observations(created_at);
CREATE INDEX IF NOT EXISTS idx_obs_session ON observations(session_id);
CREATE INDEX IF NOT EXISTS idx_obs_hash ON observations(content_hash);
"""


class MemoryDB:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def close(self):
        self.conn.close()

    def store_observation(
        self,
        project: str,
        obs_type: str,
        content: str,
        session_id: str = None,
        embedding: bytes = None,
    ) -> str:
        obs_id = str(uuid.uuid4())
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO observations (id, session_id, project, type, content, content_hash, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (obs_id, session_id, project, obs_type, content, content_hash, embedding, now),
        )
        self.conn.commit()
        return obs_id

    def query(
        self,
        text: str = "",
        project: str = None,
        obs_type: str = None,
        limit: int = 20,
    ) -> list:
        sql = "SELECT * FROM observations WHERE 1=1"
        params = []
        if text:
            sql += " AND content LIKE ?"
            params.append(f"%{text}%")
        if project:
            sql += " AND project = ?"
            params.append(project)
        if obs_type:
            sql += " AND type = ?"
            params.append(obs_type)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def recent(self, project: str = None, limit: int = 5) -> list:
        if project:
            rows = self.conn.execute(
                "SELECT * FROM observations WHERE project = ? ORDER BY created_at DESC LIMIT ?",
                (project, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM observations ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def simd_search(self, text: str, project: str = None, limit: int = 20) -> list:
        """Search observation content using SIMD substring kernel.

        Falls back to SQL LIKE query if libsubstr.so is not available
        (e.g., kernel not built on this machine).
        """
        try:
            from text_search import find_offsets
        except (ImportError, OSError):
            return self.query(text, project=project, limit=limit)

        needle = text.encode("utf-8")
        sql = "SELECT * FROM observations"
        params = []
        if project:
            sql += " WHERE project = ?"
            params.append(project)
        rows = self.conn.execute(sql, params).fetchall()

        matches = []
        for row in rows:
            content_bytes = row["content"].encode("utf-8")
            if find_offsets(content_bytes, needle):
                matches.append(dict(row))
            if len(matches) >= limit:
                break
        return matches
