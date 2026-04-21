"""memory.py — SQLite storage for eabrain observations and sessions."""

import hashlib
import os
import sqlite3
import uuid
from datetime import datetime, timezone

import numpy as np


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

    def _db_size_bytes(self) -> int:
        total = 0
        for suffix in ("", "-wal", "-shm"):
            p = self.path + suffix
            if os.path.exists(p):
                total += os.path.getsize(p)
        return total

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

    def create_session(self, project: str) -> str:
        sid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO sessions (id, project, started_at) VALUES (?, ?, ?)",
            (sid, project, now),
        )
        self.conn.commit()
        return sid

    def close_session(self, session_id: str, summary: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE sessions SET ended_at = ?, summary = ? WHERE id = ?",
            (now, summary, session_id),
        )
        self.conn.commit()

    def mark_incomplete(self, session_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE sessions SET ended_at = ?, summary = ? WHERE id = ? AND ended_at IS NULL",
            (now, "[incomplete]", session_id),
        )
        self.conn.commit()

    def timeline(self, project: str = None, limit: int = 20, since: str = None) -> list:
        sql = "SELECT * FROM sessions WHERE 1=1"
        params = []
        if project:
            sql += " AND project = ?"
            params.append(project)
        if since:
            sql += " AND started_at >= ?"
            params.append(since)
        sql += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        sessions = self.conn.execute(sql, params).fetchall()
        result = []
        # N+1 is intentional at this scale: sessions descend by started_at,
        # observations ascend by created_at within each session — not expressible
        # in a single JOIN without a subquery.
        for s in sessions:
            obs = self.conn.execute(
                "SELECT * FROM observations WHERE session_id = ? ORDER BY created_at ASC",
                (s["id"],),
            ).fetchall()
            result.append({
                "session": dict(s),
                "observations": [dict(o) for o in obs],
            })
        return result

    def stats(self) -> dict:
        obs_count = self.conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        sess_count = self.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        last = self.conn.execute(
            "SELECT started_at FROM sessions ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        db_size = self._db_size_bytes()
        return {
            "observation_count": obs_count,
            "session_count": sess_count,
            "last_session": last["started_at"] if last else None,
            "db_size_bytes": db_size,
        }

    def load_embeddings(self, project: str = None) -> tuple:
        sql = "SELECT id, embedding FROM observations WHERE embedding IS NOT NULL"
        params = []
        if project:
            sql += " AND project = ?"
            params.append(project)
        rows = self.conn.execute(sql, params).fetchall()
        if not rows:
            return [], np.zeros((0, 256), dtype=np.float32)
        ids = [r["id"] for r in rows]
        vecs = [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
        return ids, np.stack(vecs, axis=0)

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
        sql += " ORDER BY created_at DESC"
        rows = self.conn.execute(sql, params).fetchall()

        matches = []
        for row in rows:
            content_bytes = row["content"].encode("utf-8")
            if find_offsets(content_bytes, needle):
                matches.append(dict(row))
            if len(matches) >= limit:
                break
        return matches
