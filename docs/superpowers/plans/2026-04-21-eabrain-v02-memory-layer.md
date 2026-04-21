# eabrain v0.2 — Persistent Memory Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add persistent cross-session memory to eabrain with SQLite storage, smart injection with configurable preamble, cross-machine sync, and a minimal web viewer — while keeping the existing SIMD kernel index intact.

**Architecture:** SQLite (`memory.db`) stores observations and sessions alongside the existing binary `index.bin` for kernels. The SIMD `fuzzy.ea` kernel powers search across both. A configurable preamble (principles + hard rules) is injected at session start. A single-file web viewer serves via Python's `http.server`.

**Tech Stack:** Python 3 stdlib (sqlite3, http.server, uuid, hashlib, json), numpy (existing), Eä SIMD kernels (existing libfuzzy.so)

**Spec:** `docs/superpowers/specs/2026-04-21-eabrain-v02-design.md`

---

## File Structure

| File | Status | Responsibility |
|------|--------|----------------|
| `memory.py` | NEW | SQLite schema, store/query/timeline operations |
| `inject.py` | NEW | Preamble loading, dynamic context assembly, session lifecycle |
| `sync.py` | NEW | Export/import/merge memory.db |
| `server.py` | NEW | Web viewer: http.server + REST API |
| `web/viewer.html` | NEW | Single-file web UI (HTML + CSS + JS) |
| `eabrain.py` | MODIFY | Add new CLI commands, update search/remember/recall/status |
| `tests/test_memory.py` | NEW | SQLite operations |
| `tests/test_inject.py` | NEW | Injection + budget + session lifecycle |
| `tests/test_sync.py` | NEW | Export/import/merge |
| `tests/test_server.py` | NEW | API endpoints |
| `tests/test_migrate.py` | NEW | v0.1 → v0.2 migration |

---

### Task 1: SQLite Memory Storage (`memory.py`)

**Files:**
- Create: `memory.py`
- Test: `tests/test_memory.py`

- [ ] **Step 1: Write failing tests for schema creation and observation storage**

```python
# tests/test_memory.py
import os
import tempfile
import numpy as np
from memory import MemoryDB

def test_create_schema():
    with tempfile.TemporaryDirectory() as d:
        db = MemoryDB(os.path.join(d, "memory.db"))
        # Tables exist — inserting should not raise
        db.store_observation(
            project="test",
            obs_type="note",
            content="hello",
            session_id=None,
        )
        db.close()

def test_store_and_query_observation():
    with tempfile.TemporaryDirectory() as d:
        db = MemoryDB(os.path.join(d, "memory.db"))
        obs_id = db.store_observation(
            project="eaclaw",
            obs_type="decision",
            content="switched to SQLite for variable-length storage",
            session_id=None,
        )
        results = db.query("SQLite", project="eaclaw")
        assert len(results) == 1
        assert results[0]["id"] == obs_id
        assert results[0]["type"] == "decision"
        assert "SQLite" in results[0]["content"]
        db.close()

def test_query_filters_by_project():
    with tempfile.TemporaryDirectory() as d:
        db = MemoryDB(os.path.join(d, "memory.db"))
        db.store_observation(project="eaclaw", obs_type="note", content="eaclaw note", session_id=None)
        db.store_observation(project="olorin", obs_type="note", content="olorin note", session_id=None)
        results = db.query("note", project="eaclaw")
        assert len(results) == 1
        assert results[0]["project"] == "eaclaw"
        db.close()

def test_query_filters_by_type():
    with tempfile.TemporaryDirectory() as d:
        db = MemoryDB(os.path.join(d, "memory.db"))
        db.store_observation(project="eaclaw", obs_type="bug", content="off-by-one in scan", session_id=None)
        db.store_observation(project="eaclaw", obs_type="decision", content="chose mmap", session_id=None)
        results = db.query("", project="eaclaw", obs_type="bug")
        assert len(results) == 1
        assert results[0]["type"] == "bug"
        db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/dev/eabrain && python3 -m pytest tests/test_memory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'memory'`

- [ ] **Step 3: Implement MemoryDB class**

```python
# memory.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /root/dev/eabrain && python3 -m pytest tests/test_memory.py -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /root/dev/eabrain
git add memory.py tests/test_memory.py
git commit -m "feat: SQLite memory storage with observations and query"
```

---

### Task 2: Session Lifecycle (`memory.py` + `tests/test_memory.py`)

**Files:**
- Modify: `memory.py`
- Modify: `tests/test_memory.py`

- [ ] **Step 1: Write failing tests for session create/close/timeline**

Append to `tests/test_memory.py`:

```python
def test_create_session():
    with tempfile.TemporaryDirectory() as d:
        db = MemoryDB(os.path.join(d, "memory.db"))
        sid = db.create_session(project="eaclaw")
        sessions = db.conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchall()
        assert len(sessions) == 1
        assert sessions[0]["project"] == "eaclaw"
        assert sessions[0]["ended_at"] is None
        db.close()

def test_close_session():
    with tempfile.TemporaryDirectory() as d:
        db = MemoryDB(os.path.join(d, "memory.db"))
        sid = db.create_session(project="eaclaw")
        db.close_session(sid, summary="Fixed the bug")
        row = db.conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
        assert row["ended_at"] is not None
        assert row["summary"] == "Fixed the bug"
        db.close()

def test_timeline():
    with tempfile.TemporaryDirectory() as d:
        db = MemoryDB(os.path.join(d, "memory.db"))
        sid = db.create_session(project="eaclaw")
        db.store_observation(project="eaclaw", obs_type="note", content="first", session_id=sid)
        db.store_observation(project="eaclaw", obs_type="bug", content="second", session_id=sid)
        db.close_session(sid, summary="Done")
        tl = db.timeline(project="eaclaw")
        assert len(tl) == 1
        assert tl[0]["session"]["id"] == sid
        assert tl[0]["session"]["summary"] == "Done"
        assert len(tl[0]["observations"]) == 2
        db.close()

def test_timeline_filters_by_project():
    with tempfile.TemporaryDirectory() as d:
        db = MemoryDB(os.path.join(d, "memory.db"))
        s1 = db.create_session(project="eaclaw")
        db.store_observation(project="eaclaw", obs_type="note", content="a", session_id=s1)
        db.close_session(s1, summary="s1")
        s2 = db.create_session(project="olorin")
        db.store_observation(project="olorin", obs_type="note", content="b", session_id=s2)
        db.close_session(s2, summary="s2")
        tl = db.timeline(project="eaclaw")
        assert len(tl) == 1
        assert tl[0]["session"]["project"] == "eaclaw"
        db.close()

def test_mark_session_incomplete():
    with tempfile.TemporaryDirectory() as d:
        db = MemoryDB(os.path.join(d, "memory.db"))
        sid = db.create_session(project="eaclaw")
        db.mark_incomplete(sid)
        row = db.conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
        assert row["ended_at"] is not None
        assert row["summary"] == "[incomplete]"
        db.close()

def test_stats():
    with tempfile.TemporaryDirectory() as d:
        db = MemoryDB(os.path.join(d, "memory.db"))
        sid = db.create_session(project="eaclaw")
        db.store_observation(project="eaclaw", obs_type="note", content="x", session_id=sid)
        db.store_observation(project="eaclaw", obs_type="bug", content="y", session_id=sid)
        db.close_session(sid, summary="done")
        s = db.stats()
        assert s["observation_count"] == 2
        assert s["session_count"] == 1
        assert s["last_session"] is not None
        db.close()
```

- [ ] **Step 2: Run tests to verify the new tests fail**

Run: `cd /root/dev/eabrain && python3 -m pytest tests/test_memory.py -v`
Expected: new tests FAIL — `AttributeError: 'MemoryDB' object has no attribute 'create_session'`

- [ ] **Step 3: Implement session methods in MemoryDB**

Add to `memory.py` inside the `MemoryDB` class:

```python
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
        db_size = os.path.getsize(self.path) if os.path.exists(self.path) else 0
        return {
            "observation_count": obs_count,
            "session_count": sess_count,
            "last_session": last["started_at"] if last else None,
            "db_size_bytes": db_size,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /root/dev/eabrain && python3 -m pytest tests/test_memory.py -v`
Expected: all 9 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /root/dev/eabrain
git add memory.py tests/test_memory.py
git commit -m "feat: session lifecycle — create, close, timeline, stats"
```

---

### Task 3: Embedding Storage in SQLite (`memory.py`)

**Files:**
- Modify: `memory.py`
- Modify: `tests/test_memory.py`

- [ ] **Step 1: Write failing tests for embedding storage and retrieval**

Append to `tests/test_memory.py`:

```python
def test_store_observation_with_embedding():
    with tempfile.TemporaryDirectory() as d:
        db = MemoryDB(os.path.join(d, "memory.db"))
        emb = np.random.randn(256).astype(np.float32)
        obs_id = db.store_observation(
            project="eaclaw",
            obs_type="note",
            content="test embedding",
            session_id=None,
            embedding=emb.tobytes(),
        )
        results = db.query("embedding")
        assert results[0]["embedding"] is not None
        loaded = np.frombuffer(results[0]["embedding"], dtype=np.float32)
        assert loaded.shape == (256,)
        np.testing.assert_array_almost_equal(loaded, emb)
        db.close()

def test_load_all_embeddings():
    with tempfile.TemporaryDirectory() as d:
        db = MemoryDB(os.path.join(d, "memory.db"))
        emb1 = np.random.randn(256).astype(np.float32)
        emb2 = np.random.randn(256).astype(np.float32)
        id1 = db.store_observation(project="a", obs_type="note", content="one", session_id=None, embedding=emb1.tobytes())
        id2 = db.store_observation(project="a", obs_type="note", content="two", session_id=None, embedding=emb2.tobytes())
        ids, matrix = db.load_embeddings()
        assert len(ids) == 2
        assert matrix.shape == (2, 256)
        assert matrix.dtype == np.float32
        db.close()

def test_load_embeddings_empty():
    with tempfile.TemporaryDirectory() as d:
        db = MemoryDB(os.path.join(d, "memory.db"))
        ids, matrix = db.load_embeddings()
        assert len(ids) == 0
        assert matrix.shape == (0, 256)
        db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/dev/eabrain && python3 -m pytest tests/test_memory.py::test_load_all_embeddings -v`
Expected: FAIL — `AttributeError: 'MemoryDB' object has no attribute 'load_embeddings'`

- [ ] **Step 3: Implement load_embeddings**

Add to `memory.py` at the top: `import numpy as np`

Add to `MemoryDB` class:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /root/dev/eabrain && python3 -m pytest tests/test_memory.py -v`
Expected: all 12 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /root/dev/eabrain
git add memory.py tests/test_memory.py
git commit -m "feat: embedding storage and bulk loading for SIMD search"
```

---

### Task 4: Preamble System + Injection (`inject.py`)

**Files:**
- Create: `inject.py`
- Test: `tests/test_inject.py`

- [ ] **Step 1: Write failing tests for preamble loading and injection**

```python
# tests/test_inject.py
import os
import tempfile
import numpy as np
from inject import load_preamble, build_injection
from memory import MemoryDB

def test_load_preamble_from_dir():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "preamble"))
        with open(os.path.join(d, "preamble", "principles.md"), "w") as f:
            f.write("# Principles\nThink before coding.")
        with open(os.path.join(d, "preamble", "hard_rules.md"), "w") as f:
            f.write("# Hard Rules\nNo file exceeds 500 lines.")
        text = load_preamble(os.path.join(d, "preamble"))
        assert "Think before coding" in text
        assert "No file exceeds 500 lines" in text

def test_load_preamble_missing_dir():
    text = load_preamble("/nonexistent/preamble")
    assert text == ""

def test_build_injection_empty_db():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "preamble"))
        with open(os.path.join(d, "preamble", "principles.md"), "w") as f:
            f.write("# Principles\nBe clear.")
        db = MemoryDB(os.path.join(d, "memory.db"))
        output = build_injection(
            db=db,
            preamble_dir=os.path.join(d, "preamble"),
            project="eaclaw",
        )
        assert "Be clear" in output
        db.close()

def test_build_injection_includes_recent_observations():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "preamble"))
        with open(os.path.join(d, "preamble", "principles.md"), "w") as f:
            f.write("# Principles")
        db = MemoryDB(os.path.join(d, "memory.db"))
        db.store_observation(project="eaclaw", obs_type="decision", content="chose SQLite over binary", session_id=None)
        output = build_injection(
            db=db,
            preamble_dir=os.path.join(d, "preamble"),
            project="eaclaw",
        )
        assert "chose SQLite over binary" in output
        db.close()

def test_build_injection_respects_budget():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "preamble"))
        with open(os.path.join(d, "preamble", "principles.md"), "w") as f:
            f.write("short")
        db = MemoryDB(os.path.join(d, "memory.db"))
        for i in range(50):
            db.store_observation(
                project="eaclaw", obs_type="note",
                content=f"observation number {i} with extra padding text to consume budget",
                session_id=None,
            )
        output = build_injection(db=db, preamble_dir=os.path.join(d, "preamble"), project="eaclaw", budget=500)
        # Budget limits dynamic section — not all 50 observations fit
        assert output.count("observation number") < 50
        db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/dev/eabrain && python3 -m pytest tests/test_inject.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'inject'`

- [ ] **Step 3: Implement inject.py**

```python
# inject.py
"""inject.py — Preamble loading and context injection for eabrain."""

import os

from memory import MemoryDB


def load_preamble(preamble_dir: str) -> str:
    if not os.path.isdir(preamble_dir):
        return ""
    parts = []
    for name in sorted(os.listdir(preamble_dir)):
        if name.endswith(".md"):
            path = os.path.join(preamble_dir, name)
            with open(path, "r", encoding="utf-8") as f:
                parts.append(f.read().strip())
    return "\n\n".join(parts)


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def build_injection(
    db: MemoryDB,
    preamble_dir: str,
    project: str,
    budget: int = 2000,
) -> str:
    sections = []

    # 1. Fixed preamble (always full)
    preamble = load_preamble(preamble_dir)
    if preamble:
        sections.append(preamble)

    # 2. Dynamic context (respects budget)
    dynamic_parts = []
    used = 0

    # Last session summary
    last_sessions = db.timeline(project=project, limit=1)
    if last_sessions and last_sessions[0]["session"].get("summary"):
        summary = last_sessions[0]["session"]["summary"]
        summary_text = f"## Last Session\n{summary}"
        cost = _estimate_tokens(summary_text)
        if used + cost <= budget:
            dynamic_parts.append(summary_text)
            used += cost

    # Recent observations for this project
    recent = db.recent(project=project, limit=10)
    if recent:
        obs_lines = []
        for obs in recent:
            line = f"- [{obs['type']}] {obs['content']}"
            cost = _estimate_tokens(line)
            if used + cost > budget:
                break
            obs_lines.append(line)
            used += cost
        if obs_lines:
            dynamic_parts.append("## Recent Observations\n" + "\n".join(obs_lines))

    if dynamic_parts:
        sections.append("\n\n".join(dynamic_parts))

    return "\n\n".join(sections)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /root/dev/eabrain && python3 -m pytest tests/test_inject.py -v`
Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /root/dev/eabrain
git add inject.py tests/test_inject.py
git commit -m "feat: preamble loading and context injection with token budget"
```

---

### Task 5: Session Lifecycle File (`inject.py` — current_session)

**Files:**
- Modify: `inject.py`
- Modify: `tests/test_inject.py`

- [ ] **Step 1: Write failing tests for session file management**

Append to `tests/test_inject.py`:

```python
from inject import start_session, get_current_session_id, end_session

def test_start_session_creates_file():
    with tempfile.TemporaryDirectory() as d:
        db = MemoryDB(os.path.join(d, "memory.db"))
        session_file = os.path.join(d, "current_session")
        sid = start_session(db, project="eaclaw", session_file=session_file)
        assert os.path.exists(session_file)
        with open(session_file) as f:
            assert f.read().strip() == sid
        db.close()

def test_get_current_session_id():
    with tempfile.TemporaryDirectory() as d:
        db = MemoryDB(os.path.join(d, "memory.db"))
        session_file = os.path.join(d, "current_session")
        sid = start_session(db, project="eaclaw", session_file=session_file)
        assert get_current_session_id(session_file) == sid
        db.close()

def test_get_current_session_id_missing():
    assert get_current_session_id("/nonexistent/file") is None

def test_start_session_marks_orphan_incomplete():
    with tempfile.TemporaryDirectory() as d:
        db = MemoryDB(os.path.join(d, "memory.db"))
        session_file = os.path.join(d, "current_session")
        old_sid = start_session(db, project="eaclaw", session_file=session_file)
        new_sid = start_session(db, project="eaclaw", session_file=session_file)
        assert new_sid != old_sid
        row = db.conn.execute("SELECT * FROM sessions WHERE id = ?", (old_sid,)).fetchone()
        assert row["summary"] == "[incomplete]"
        db.close()

def test_end_session_closes_and_removes_file():
    with tempfile.TemporaryDirectory() as d:
        db = MemoryDB(os.path.join(d, "memory.db"))
        session_file = os.path.join(d, "current_session")
        sid = start_session(db, project="eaclaw", session_file=session_file)
        end_session(db, session_id=sid, summary="all done", session_file=session_file)
        assert not os.path.exists(session_file)
        row = db.conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
        assert row["summary"] == "all done"
        assert row["ended_at"] is not None
        db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/dev/eabrain && python3 -m pytest tests/test_inject.py::test_start_session_creates_file -v`
Expected: FAIL — `ImportError: cannot import name 'start_session'`

- [ ] **Step 3: Implement session lifecycle functions**

Add to `inject.py`:

```python
def start_session(db: MemoryDB, project: str, session_file: str) -> str:
    # Check for orphaned session
    old_sid = get_current_session_id(session_file)
    if old_sid:
        db.mark_incomplete(old_sid)

    sid = db.create_session(project=project)
    os.makedirs(os.path.dirname(os.path.abspath(session_file)), exist_ok=True)
    with open(session_file, "w") as f:
        f.write(sid)
    return sid


def get_current_session_id(session_file: str) -> str:
    if not os.path.exists(session_file):
        return None
    with open(session_file, "r") as f:
        return f.read().strip() or None


def end_session(db: MemoryDB, session_id: str, summary: str, session_file: str) -> None:
    db.close_session(session_id, summary=summary)
    if os.path.exists(session_file):
        os.remove(session_file)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /root/dev/eabrain && python3 -m pytest tests/test_inject.py -v`
Expected: all 10 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /root/dev/eabrain
git add inject.py tests/test_inject.py
git commit -m "feat: session lifecycle with current_session file and orphan detection"
```

---

### Task 6: Sync — Export and Import (`sync.py`)

**Files:**
- Create: `sync.py`
- Test: `tests/test_sync.py`

- [ ] **Step 1: Write failing tests for export and import**

```python
# tests/test_sync.py
import os
import tempfile
import numpy as np
from memory import MemoryDB
from sync import export_db, import_db

def test_export_copies_file():
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, "memory.db")
        db = MemoryDB(db_path)
        db.store_observation(project="a", obs_type="note", content="hello", session_id=None)
        db.close()
        export_path = os.path.join(d, "exported.db")
        export_db(db_path, export_path)
        assert os.path.exists(export_path)
        assert os.path.getsize(export_path) > 0

def test_import_merges_observations():
    with tempfile.TemporaryDirectory() as d:
        # Create "remote" db
        remote_path = os.path.join(d, "remote.db")
        remote = MemoryDB(remote_path)
        remote.store_observation(project="a", obs_type="note", content="from remote", session_id=None)
        remote.close()
        # Create "local" db
        local_path = os.path.join(d, "local.db")
        local = MemoryDB(local_path)
        local.store_observation(project="a", obs_type="note", content="from local", session_id=None)
        local.close()
        # Import remote into local
        import_db(local_path, remote_path)
        merged = MemoryDB(local_path)
        results = merged.query("")
        assert len(results) == 2
        contents = {r["content"] for r in results}
        assert "from remote" in contents
        assert "from local" in contents
        merged.close()

def test_import_deduplicates_by_hash():
    with tempfile.TemporaryDirectory() as d:
        remote_path = os.path.join(d, "remote.db")
        remote = MemoryDB(remote_path)
        remote.store_observation(project="a", obs_type="note", content="same content", session_id=None)
        remote.close()
        local_path = os.path.join(d, "local.db")
        local = MemoryDB(local_path)
        local.store_observation(project="a", obs_type="note", content="same content", session_id=None)
        local.close()
        import_db(local_path, remote_path)
        merged = MemoryDB(local_path)
        results = merged.query("same content")
        assert len(results) == 1
        merged.close()

def test_import_merges_sessions():
    with tempfile.TemporaryDirectory() as d:
        remote_path = os.path.join(d, "remote.db")
        remote = MemoryDB(remote_path)
        sid = remote.create_session(project="a")
        remote.store_observation(project="a", obs_type="note", content="remote obs", session_id=sid)
        remote.close_session(sid, summary="remote session")
        remote.close()
        local_path = os.path.join(d, "local.db")
        local = MemoryDB(local_path)
        local.close()
        import_db(local_path, remote_path)
        merged = MemoryDB(local_path)
        tl = merged.timeline(project="a")
        assert len(tl) == 1
        assert tl[0]["session"]["summary"] == "remote session"
        assert len(tl[0]["observations"]) == 1
        merged.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/dev/eabrain && python3 -m pytest tests/test_sync.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sync'`

- [ ] **Step 3: Implement sync.py**

```python
# sync.py
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

    # Get existing content hashes in local
    local_hashes = {
        row[0] for row in local.execute("SELECT content_hash FROM observations").fetchall()
    }

    # Get existing session IDs in local
    local_session_ids = {
        row[0] for row in local.execute("SELECT id FROM sessions").fetchall()
    }

    # Merge sessions
    for row in remote.execute("SELECT * FROM sessions").fetchall():
        row = dict(row)
        if row["id"] not in local_session_ids:
            local.execute(
                "INSERT INTO sessions (id, project, started_at, ended_at, summary) VALUES (?, ?, ?, ?, ?)",
                (row["id"], row["project"], row["started_at"], row["ended_at"], row["summary"]),
            )
        else:
            # Keep longer summary
            local_row = local.execute("SELECT summary FROM sessions WHERE id = ?", (row["id"],)).fetchone()
            if row["summary"] and (not local_row["summary"] or len(row["summary"]) > len(local_row["summary"])):
                local.execute("UPDATE sessions SET summary = ? WHERE id = ?", (row["summary"], row["id"]))

    # Merge observations (dedup by content_hash)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /root/dev/eabrain && python3 -m pytest tests/test_sync.py -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /root/dev/eabrain
git add sync.py tests/test_sync.py
git commit -m "feat: sync — export and import memory.db with dedup merge"
```

---

### Task 7: Migration from v0.1 (`memory.py`)

**Files:**
- Modify: `memory.py`
- Test: `tests/test_migrate.py`

- [ ] **Step 1: Write failing tests for migration**

```python
# tests/test_migrate.py
import os
import tempfile
import numpy as np
from indexer import write_index
from memory import MemoryDB, migrate_from_index

def test_migrate_creates_observations():
    with tempfile.TemporaryDirectory() as d:
        # Write a v0.1 index.bin with session notes
        idx_path = os.path.join(d, "index.bin")
        sessions = [
            {"text": "Fixed the sign flip bug", "timestamp": 1713700000},
            {"text": "Decided to use SQLite", "timestamp": 1713700100},
        ]
        write_index(idx_path, [], [], sessions, np.zeros((0, 256), dtype=np.float32))
        # Run migration
        db_path = os.path.join(d, "memory.db")
        db = MemoryDB(db_path)
        count = migrate_from_index(db, idx_path, project="legacy")
        assert count == 2
        results = db.query("")
        assert len(results) == 2
        contents = {r["content"] for r in results}
        assert "Fixed the sign flip bug" in contents
        assert "Decided to use SQLite" in contents
        db.close()

def test_migrate_idempotent():
    with tempfile.TemporaryDirectory() as d:
        idx_path = os.path.join(d, "index.bin")
        sessions = [{"text": "hello", "timestamp": 1713700000}]
        write_index(idx_path, [], [], sessions, np.zeros((0, 256), dtype=np.float32))
        db_path = os.path.join(d, "memory.db")
        db = MemoryDB(db_path)
        migrate_from_index(db, idx_path, project="legacy")
        migrate_from_index(db, idx_path, project="legacy")
        results = db.query("")
        assert len(results) == 1
        db.close()

def test_migrate_empty_index():
    with tempfile.TemporaryDirectory() as d:
        idx_path = os.path.join(d, "index.bin")
        write_index(idx_path, [], [], [], np.zeros((0, 256), dtype=np.float32))
        db_path = os.path.join(d, "memory.db")
        db = MemoryDB(db_path)
        count = migrate_from_index(db, idx_path, project="legacy")
        assert count == 0
        db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/dev/eabrain && python3 -m pytest tests/test_migrate.py -v`
Expected: FAIL — `ImportError: cannot import name 'migrate_from_index'`

- [ ] **Step 3: Implement migrate_from_index in memory.py**

Add to `memory.py`:

```python
def migrate_from_index(db: MemoryDB, index_path: str, project: str = "legacy") -> int:
    from indexer import read_index

    idx = read_index(index_path)
    sessions = idx.get("sessions", [])
    if not sessions:
        return 0

    # Check for existing legacy session to make this idempotent
    existing = db.conn.execute(
        "SELECT id FROM sessions WHERE project = ? AND summary = ?",
        (project, "[migrated from index.bin]"),
    ).fetchone()
    if existing:
        return 0

    sid = db.create_session(project=project)
    count = 0
    for s in sessions:
        db.store_observation(
            project=project,
            obs_type="note",
            content=s["text"],
            session_id=sid,
        )
        count += 1
    db.close_session(sid, summary="[migrated from index.bin]")
    return count
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /root/dev/eabrain && python3 -m pytest tests/test_migrate.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /root/dev/eabrain
git add memory.py tests/test_migrate.py
git commit -m "feat: migrate v0.1 session notes from index.bin to SQLite"
```

---

### Task 8: CLI Integration — New Commands (`eabrain.py`)

**Files:**
- Modify: `eabrain.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for new CLI commands**

Append to `tests/test_cli.py`:

```python
def test_inject_prints_preamble(tmp_path, capsys):
    import sys
    preamble_dir = str(tmp_path / "preamble")
    os.makedirs(preamble_dir)
    with open(os.path.join(preamble_dir, "principles.md"), "w") as f:
        f.write("Think before coding.")
    db_path = str(tmp_path / "memory.db")
    session_file = str(tmp_path / "current_session")
    # Simulate inject via the command function
    from eabrain import cmd_inject
    class Args:
        project = str(tmp_path)
        budget = 2000
    cfg = {
        "eabrain_dir": str(tmp_path),
        "index_path": str(tmp_path / "index.bin"),
    }
    cmd_inject(Args(), cfg)
    captured = capsys.readouterr()
    assert "Think before coding" in captured.out

def test_store_creates_observation(tmp_path):
    from memory import MemoryDB
    from eabrain import cmd_store
    db_path = str(tmp_path / "memory.db")
    session_file = str(tmp_path / "current_session")
    db = MemoryDB(db_path)
    sid = db.create_session(project="test")
    with open(session_file, "w") as f:
        f.write(sid)
    db.close()
    class Args:
        content = "chose SQLite"
        type = "decision"
        project = "test"
    cfg = {"eabrain_dir": str(tmp_path)}
    cmd_store(Args(), cfg)
    db = MemoryDB(db_path)
    results = db.query("SQLite")
    assert len(results) == 1
    db.close()

def test_timeline_shows_sessions(tmp_path, capsys):
    from memory import MemoryDB
    from eabrain import cmd_timeline
    db_path = str(tmp_path / "memory.db")
    db = MemoryDB(db_path)
    sid = db.create_session(project="test")
    db.store_observation(project="test", obs_type="note", content="hello", session_id=sid)
    db.close_session(sid, summary="test session")
    db.close()
    class Args:
        project = "test"
        last = 10
        since = None
    cfg = {"eabrain_dir": str(tmp_path)}
    cmd_timeline(Args(), cfg)
    captured = capsys.readouterr()
    assert "test session" in captured.out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/dev/eabrain && python3 -m pytest tests/test_cli.py::test_inject_prints_preamble -v`
Expected: FAIL — `ImportError: cannot import name 'cmd_inject'`

- [ ] **Step 3: Add new commands to eabrain.py**

Add `eabrain_dir` to config defaults in `_load_config`:
```python
cfg.setdefault("eabrain_dir", os.path.expanduser("~/.eabrain"))
```

Add new command functions after the existing ones:

```python
def _get_db(cfg):
    from memory import MemoryDB
    return MemoryDB(os.path.join(cfg["eabrain_dir"], "memory.db"))


def _get_session_file(cfg):
    return os.path.join(cfg["eabrain_dir"], "current_session")


def _get_preamble_dir(cfg):
    return os.path.join(cfg["eabrain_dir"], "preamble")


def cmd_inject(args, cfg):
    from inject import build_injection, start_session
    db = _get_db(cfg)
    project = getattr(args, "project", None) or os.getcwd()
    budget = getattr(args, "budget", 2000) or 2000
    session_file = _get_session_file(cfg)
    sid = start_session(db, project=project, session_file=session_file)
    output = build_injection(
        db=db,
        preamble_dir=_get_preamble_dir(cfg),
        project=project,
        budget=budget,
    )
    print(output)
    db.close()


def cmd_store(args, cfg):
    from inject import get_current_session_id
    from indexer import _byte_histogram
    db = _get_db(cfg)
    session_file = _get_session_file(cfg)
    sid = get_current_session_id(session_file)
    project = getattr(args, "project", None) or os.getcwd()
    content = args.content
    emb = _byte_histogram(content.encode("utf-8"))
    db.store_observation(
        project=project,
        obs_type=args.type,
        content=content,
        session_id=sid,
        embedding=emb.tobytes(),
    )
    print(f"Stored [{args.type}]: {content[:80]}")
    db.close()


def cmd_store_summary(args, cfg):
    from inject import get_current_session_id, end_session
    db = _get_db(cfg)
    session_file = _get_session_file(cfg)
    sid = get_current_session_id(session_file)
    if sid:
        end_session(db, session_id=sid, summary=args.content, session_file=session_file)
        print(f"Session closed: {args.content[:80]}")
    else:
        print("No active session.")
    db.close()


def cmd_timeline(args, cfg):
    db = _get_db(cfg)
    project = getattr(args, "project", None)
    limit = getattr(args, "last", 10) or 10
    since = getattr(args, "since", None)
    tl = db.timeline(project=project, limit=limit, since=since)
    if not tl:
        print("No sessions recorded.")
        db.close()
        return
    for entry in tl:
        s = entry["session"]
        obs = entry["observations"]
        print(f"\n--- Session: {s['started_at'][:16]} [{s['project']}] ---")
        if s.get("summary"):
            print(f"Summary: {s['summary']}")
        for o in obs:
            print(f"  [{o['type']}] {o['content']}")
    db.close()


def cmd_migrate(args, cfg):
    from memory import migrate_from_index
    db = _get_db(cfg)
    idx_path = cfg["index_path"]
    if not os.path.exists(idx_path):
        print("No index.bin found — nothing to migrate.")
        db.close()
        return
    count = migrate_from_index(db, idx_path)
    print(f"Migrated {count} session notes to memory.db")
    db.close()


def cmd_sync(args, cfg):
    from sync import export_db, import_db
    db_path = os.path.join(cfg["eabrain_dir"], "memory.db")
    if args.export_path:
        export_db(db_path, args.export_path)
        print(f"Exported memory.db to {args.export_path}")
    elif args.import_path:
        import_db(db_path, args.import_path)
        print(f"Imported and merged from {args.import_path}")
    else:
        print("Usage: eabrain sync --export <path> or --import <path>")


def cmd_serve(args, cfg):
    from server import serve
    port = getattr(args, "port", 37777) or 37777
    serve(cfg, port=port)
```

- [ ] **Step 4: Register new commands in argparse and dispatch**

In the `main()` function, add new subparsers after the existing ones:

```python
    p_inject = sub.add_parser("inject", help="Inject context for session start")
    p_inject.add_argument("--project", help="Project directory (default: cwd)")
    p_inject.add_argument("--budget", type=int, default=2000, help="Token budget for dynamic section")

    p_store = sub.add_parser("store", help="Store an observation")
    p_store.add_argument("content")
    p_store.add_argument("--type", required=True, choices=["decision", "bug", "architecture", "pattern", "error", "note"])
    p_store.add_argument("--project", help="Project name (default: cwd)")

    p_store_summary = sub.add_parser("store-summary", help="Store session summary and close session")
    p_store_summary.add_argument("content")

    p_timeline = sub.add_parser("timeline", help="Show session timeline")
    p_timeline.add_argument("--project", help="Filter by project")
    p_timeline.add_argument("--last", type=int, default=10, help="Number of sessions")
    p_timeline.add_argument("--since", help="ISO date filter")

    sub.add_parser("migrate", help="Migrate v0.1 notes to memory.db")

    p_sync = sub.add_parser("sync", help="Export or import memory.db")
    p_sync.add_argument("--export", dest="export_path", help="Export to path")
    p_sync.add_argument("--import", dest="import_path", help="Import from path")

    p_serve = sub.add_parser("serve", help="Start web viewer")
    p_serve.add_argument("--port", type=int, default=37777, help="Port number")
```

Update the dispatch dict:

```python
    dispatch = {
        "index": cmd_index,
        "search": cmd_search,
        "ref": cmd_ref,
        "remember": cmd_remember,
        "recall": cmd_recall,
        "status": cmd_status,
        "patterns": cmd_patterns,
        "init": cmd_init,
        "inject": cmd_inject,
        "store": cmd_store,
        "store-summary": cmd_store_summary,
        "timeline": cmd_timeline,
        "migrate": cmd_migrate,
        "sync": cmd_sync,
        "serve": cmd_serve,
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /root/dev/eabrain && python3 -m pytest tests/test_cli.py -v`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
cd /root/dev/eabrain
git add eabrain.py tests/test_cli.py
git commit -m "feat: CLI commands — inject, store, store-summary, timeline, migrate, sync, serve"
```

---

### Task 9: Update Existing Commands (`eabrain.py`)

**Files:**
- Modify: `eabrain.py`

- [ ] **Step 1: Update cmd_remember to use memory.db**

Replace `cmd_remember` in `eabrain.py`:

```python
def cmd_remember(args, cfg):
    from inject import get_current_session_id
    from indexer import _byte_histogram
    db = _get_db(cfg)
    session_file = _get_session_file(cfg)
    sid = get_current_session_id(session_file)
    emb = _byte_histogram(args.note.encode("utf-8"))
    db.store_observation(
        project=os.getcwd(),
        obs_type="note",
        content=args.note,
        session_id=sid,
        embedding=emb.tobytes(),
    )
    print(f"Remembered: {args.note}")
    db.close()
```

- [ ] **Step 2: Update cmd_recall to use memory.db**

Replace `cmd_recall` in `eabrain.py`:

```python
def cmd_recall(args, cfg):
    db = _get_db(cfg)
    n = args.last if args.last else 20
    results = db.recent(limit=n)
    if not results:
        print("No observations.")
        db.close()
        return
    print(f"# Observations ({len(results)} entries)\n")
    for r in results:
        print(f"[{r['created_at'][:16]}] [{r['type']}] {r['content']}")
    db.close()
```

- [ ] **Step 3: Update cmd_search to search both kernels and observations**

In `cmd_search`, after the existing kernel search logic, add observation search. The function should combine results from both sources. Add after the kernel results printing:

```python
    # Search observations
    db = _get_db(cfg)
    if not args.kernels_only:
        if args.fuzzy:
            # SIMD cosine across observation embeddings
            obs_ids, obs_emb = db.load_embeddings()
            if len(obs_ids) > 0:
                obs_scores = np.zeros(len(obs_ids), dtype=np.float32)
                fuzzy_lib = fuzzy_lib if 'fuzzy_lib' in dir() else _load_lib("libfuzzy.so")
                fuzzy_lib.batch_cosine(
                    hist.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_float(norm_val if norm_val > 0 else 1.0),
                    obs_emb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_int32(256),
                    ctypes.c_int32(len(obs_ids)),
                    obs_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                )
                order = np.argsort(obs_scores)[::-1][:5]
                obs_results = []
                for idx_i in order:
                    if obs_scores[idx_i] > 0:
                        obs_row = db.conn.execute(
                            "SELECT * FROM observations WHERE id = ?", (obs_ids[idx_i],)
                        ).fetchone()
                        if obs_row:
                            obs_results.append(dict(obs_row))
                if obs_results:
                    print(f"## Memory ({len(obs_results)} matches)\n")
                    for o in obs_results:
                        print(f"  [{o['type']}] {o['content'][:100]}")
        else:
            obs_results = db.query(query, limit=5)
            if obs_results:
                print(f"## Memory ({len(obs_results)} matches)\n")
                for o in obs_results:
                    print(f"  [{o['type']}] {o['content'][:100]}")
    db.close()
```

Add `--kernels-only` and `--memory-only` flags to the search subparser:

```python
    p_search.add_argument("--kernels-only", action="store_true", help="Search kernels only")
    p_search.add_argument("--memory-only", action="store_true", help="Search observations only")
```

- [ ] **Step 4: Update cmd_status to include memory stats**

Add at the end of `cmd_status`, before the common commands section:

```python
    # Memory stats
    db_path = os.path.join(cfg["eabrain_dir"], "memory.db")
    if os.path.exists(db_path):
        db = _get_db(cfg)
        s = db.stats()
        print(f"Observations: {s['observation_count']}")
        print(f"Sessions: {s['session_count']}")
        if s["last_session"]:
            print(f"Last session: {s['last_session'][:16]}")
        print(f"Memory DB: {s['db_size_bytes'] // 1024}KB")
        db.close()
```

- [ ] **Step 5: Run all existing tests to verify no regressions**

Run: `cd /root/dev/eabrain && python3 -m pytest tests/ -v`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
cd /root/dev/eabrain
git add eabrain.py
git commit -m "feat: update remember/recall/search/status to use memory.db"
```

---

### Task 10: Preamble Content Files

**Files:**
- Create: default preamble files for first-run experience

- [ ] **Step 1: Create the preamble directory and files**

The `eabrain inject` command reads from `~/.eabrain/preamble/`. Create the default files that get installed. Add a `cmd_setup_preamble` helper called from `cmd_inject` if the preamble dir is empty:

Add to `inject.py`:

```python
_DEFAULT_PRINCIPLES = """\
## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them. Don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?"
If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it. Don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" -> "Write tests for invalid inputs, then make them pass"
- "Fix the bug" -> "Write a test that reproduces it, then make it pass"
- "Refactor X" -> "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]

Strong success criteria let you loop independently.
Weak criteria ("make it work") require constant clarification.
"""

_DEFAULT_HARD_RULES = """\
## Hard Rules

Apply these rules to ALL code:

1. **No file exceeds 500 lines.** Split before you hit the limit.
2. **Every feature proven by end-to-end test.** If it's not tested, it doesn't exist.
3. **No fake functions. No stubs.** No `todo!()`, `// TODO`, `// HACK`, `// for now`, `// hardcoded`, `// placeholder`, `// temporary`. If it doesn't compile and pass tests, it's not code.
4. **No premature features.** Don't build what isn't needed yet.
5. **Delete, don't comment.** Dead code gets removed, not commented out.
"""


def ensure_preamble(preamble_dir: str) -> None:
    if os.path.isdir(preamble_dir) and os.listdir(preamble_dir):
        return
    os.makedirs(preamble_dir, exist_ok=True)
    with open(os.path.join(preamble_dir, "01_principles.md"), "w") as f:
        f.write(_DEFAULT_PRINCIPLES)
    with open(os.path.join(preamble_dir, "02_hard_rules.md"), "w") as f:
        f.write(_DEFAULT_HARD_RULES)
```

- [ ] **Step 2: Call ensure_preamble from build_injection**

Add at the start of `build_injection`:

```python
    ensure_preamble(preamble_dir)
```

- [ ] **Step 3: Write a test for ensure_preamble**

Append to `tests/test_inject.py`:

```python
from inject import ensure_preamble

def test_ensure_preamble_creates_defaults():
    with tempfile.TemporaryDirectory() as d:
        preamble_dir = os.path.join(d, "preamble")
        ensure_preamble(preamble_dir)
        assert os.path.exists(os.path.join(preamble_dir, "01_principles.md"))
        assert os.path.exists(os.path.join(preamble_dir, "02_hard_rules.md"))
        with open(os.path.join(preamble_dir, "01_principles.md")) as f:
            content = f.read()
        assert "Think Before Coding" in content

def test_ensure_preamble_does_not_overwrite():
    with tempfile.TemporaryDirectory() as d:
        preamble_dir = os.path.join(d, "preamble")
        os.makedirs(preamble_dir)
        with open(os.path.join(preamble_dir, "custom.md"), "w") as f:
            f.write("My custom rule")
        ensure_preamble(preamble_dir)
        # Should not have created defaults since dir was not empty
        assert not os.path.exists(os.path.join(preamble_dir, "01_principles.md"))
```

- [ ] **Step 4: Run tests**

Run: `cd /root/dev/eabrain && python3 -m pytest tests/test_inject.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
cd /root/dev/eabrain
git add inject.py tests/test_inject.py
git commit -m "feat: default preamble content — principles and hard rules"
```

---

### Task 11: Web Viewer — Server (`server.py`)

**Files:**
- Create: `server.py`
- Test: `tests/test_server.py`

- [ ] **Step 1: Write failing tests for server API**

```python
# tests/test_server.py
import json
import os
import tempfile
import threading
import time
import urllib.request
from memory import MemoryDB
from server import EabrainHandler, make_server

def _setup_db(d):
    db_path = os.path.join(d, "memory.db")
    db = MemoryDB(db_path)
    sid = db.create_session(project="test")
    db.store_observation(project="test", obs_type="decision", content="chose SQLite", session_id=sid)
    db.store_observation(project="test", obs_type="bug", content="off-by-one", session_id=sid)
    db.close_session(sid, summary="test session done")
    db.close()
    return db_path

def test_api_stats():
    with tempfile.TemporaryDirectory() as d:
        db_path = _setup_db(d)
        cfg = {"eabrain_dir": d, "index_path": os.path.join(d, "index.bin")}
        server = make_server(cfg, port=0)  # port=0 picks a free port
        port = server.server_address[1]
        t = threading.Thread(target=server.handle_request)
        t.start()
        resp = urllib.request.urlopen(f"http://localhost:{port}/api/stats")
        data = json.loads(resp.read())
        assert data["observation_count"] == 2
        assert data["session_count"] == 1
        t.join(timeout=2)

def test_api_timeline():
    with tempfile.TemporaryDirectory() as d:
        db_path = _setup_db(d)
        cfg = {"eabrain_dir": d, "index_path": os.path.join(d, "index.bin")}
        server = make_server(cfg, port=0)
        port = server.server_address[1]
        t = threading.Thread(target=server.handle_request)
        t.start()
        resp = urllib.request.urlopen(f"http://localhost:{port}/api/timeline?project=test")
        data = json.loads(resp.read())
        assert len(data) == 1
        assert data[0]["session"]["summary"] == "test session done"
        assert len(data[0]["observations"]) == 2
        t.join(timeout=2)

def test_api_search():
    with tempfile.TemporaryDirectory() as d:
        db_path = _setup_db(d)
        cfg = {"eabrain_dir": d, "index_path": os.path.join(d, "index.bin")}
        server = make_server(cfg, port=0)
        port = server.server_address[1]
        t = threading.Thread(target=server.handle_request)
        t.start()
        resp = urllib.request.urlopen(f"http://localhost:{port}/api/search?q=SQLite")
        data = json.loads(resp.read())
        assert len(data["observations"]) >= 1
        t.join(timeout=2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/dev/eabrain && python3 -m pytest tests/test_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server'`

- [ ] **Step 3: Implement server.py**

```python
# server.py
"""server.py — Minimal web viewer for eabrain memory."""

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from memory import MemoryDB


_WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class EabrainHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Silence request logging

    def _json_response(self, data, status=200):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _get_db(self):
        return MemoryDB(os.path.join(self.server.cfg["eabrain_dir"], "memory.db"))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":
            self._serve_html()
        elif path == "/api/stats":
            self._api_stats()
        elif path == "/api/timeline":
            self._api_timeline(params)
        elif path == "/api/search":
            self._api_search(params)
        elif path.startswith("/api/observations/"):
            obs_id = path.split("/")[-1]
            self._api_observation(obs_id)
        else:
            self.send_error(404)

    def _serve_html(self):
        html_path = os.path.join(_WEB_DIR, "viewer.html")
        if not os.path.exists(html_path):
            self.send_error(404, "viewer.html not found")
            return
        with open(html_path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _api_stats(self):
        db = self._get_db()
        data = db.stats()
        db.close()
        self._json_response(data)

    def _api_timeline(self, params):
        db = self._get_db()
        project = params.get("project", [None])[0]
        limit = int(params.get("last", [20])[0])
        since = params.get("since", [None])[0]
        data = db.timeline(project=project, limit=limit, since=since)
        db.close()
        self._json_response(data)

    def _api_search(self, params):
        db = self._get_db()
        q = params.get("q", [""])[0]
        results = db.query(q, limit=20)
        db.close()
        self._json_response({"observations": results})

    def _api_observation(self, obs_id):
        db = self._get_db()
        row = db.conn.execute("SELECT * FROM observations WHERE id = ?", (obs_id,)).fetchone()
        if not row:
            db.close()
            self._json_response({"error": "not found"}, status=404)
            return
        data = dict(row)
        data.pop("embedding", None)
        db.close()
        self._json_response(data)


def make_server(cfg: dict, port: int = 37777) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", port), EabrainHandler)
    server.cfg = cfg
    return server


def serve(cfg: dict, port: int = 37777):
    server = make_server(cfg, port)
    print(f"eabrain viewer: http://localhost:{port}")
    server.serve_forever()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /root/dev/eabrain && python3 -m pytest tests/test_server.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /root/dev/eabrain
git add server.py tests/test_server.py
git commit -m "feat: web viewer server — REST API for stats, timeline, search"
```

---

### Task 12: Web Viewer — Frontend (`web/viewer.html`)

**Files:**
- Create: `web/viewer.html`

- [ ] **Step 1: Create the web directory**

```bash
mkdir -p /root/dev/eabrain/web
```

- [ ] **Step 2: Create viewer.html**

Create `web/viewer.html` — a single-file web UI with embedded CSS and JS. Catppuccin dark theme. Layout: left panel (timeline), right panel (detail), top bar (search + filters).

The HTML file should contain:

- `<style>` block: CSS grid layout, Catppuccin Mocha colors (`#1e1e2e` base, `#313244` surface, `#cdd6f4` text, `#94e2d5` teal accent, `#b4befe` lavender), monospace font
- Left panel: fetches `/api/timeline`, renders sessions as collapsible groups with observation counts
- Right panel: shows selected observation detail, fetches `/api/observations/<id>` for full content
- Search bar: `<input>` that calls `/api/search?q=...` on Enter, displays results in the right panel
- Filter dropdowns: project (populated from `/api/stats`), type (hardcoded list), date range
- All API calls via `fetch()`, no external dependencies

This is a UI file — the full HTML should be written during implementation based on the API endpoints defined in Task 11. Keep it under 500 lines.

- [ ] **Step 3: Manual test — start the server and verify the viewer**

```bash
cd /root/dev/eabrain
# Store some test data
eabrain remember "test observation for viewer"
# Start server
python3 -c "from server import serve; serve({'eabrain_dir': '$HOME/.eabrain', 'index_path': '$HOME/.eabrain/index.bin'})"
# Open http://localhost:37777 in browser, verify:
# - Timeline shows sessions on the left
# - Clicking an observation shows detail on the right
# - Search returns results
# - Filters work
```

- [ ] **Step 4: Commit**

```bash
cd /root/dev/eabrain
git add web/viewer.html
git commit -m "feat: web viewer frontend — single-file Catppuccin dark UI"
```

---

### Task 13: Update CLAUDE.md and Version

**Files:**
- Modify: `CLAUDE.md`
- Modify: `eabrain.py` (version bump)
- Modify: `setup.py` (version bump)

- [ ] **Step 1: Bump version to 0.2.0**

In `eabrain.py`, change:
```python
_VERSION = "0.2.0"
```

In `setup.py`, change version to `"0.2.0"`.

- [ ] **Step 2: Update CLAUDE.md with new commands**

Update the CLAUDE.md to reflect the new memory commands:

```markdown
# eabrain

Eä-driven context engine + persistent memory for Claude Code.

## Build
1. `./build_kernels.sh` — compile Eä kernels to .so + Python bindings
2. `pip install -e .` — install CLI

## Test
`python3 -m pytest tests/ -v`

## Architecture
- `kernels/*.ea` — pure SIMD kernels (no scalar ops)
- `eabrain.py` — CLI entry point
- `indexer.py` — kernel index builder (index.bin)
- `memory.py` — SQLite observation/session storage (memory.db)
- `inject.py` — preamble loading + context injection
- `server.py` — web viewer (http.server)
- `sync.py` — cross-machine memory sync
- `reference/ea_reference.json` — Eä language reference
- `lib/` — compiled .so files + generated Python bindings (not in git)

## Ea compiler
Located at `/root/dev/eacompute/target/release/ea` (not on PATH).

## Memory
- `eabrain inject` runs at session start (hook)
- `eabrain store` / `eabrain store-summary` run at session end (hook)
- `eabrain remember` for mid-session notes
- `eabrain search` searches kernels AND observations
- `eabrain timeline` for chronological session history
- `eabrain serve` for web viewer
- `eabrain sync --export/--import` for cross-machine transfer
```

- [ ] **Step 3: Run all tests**

Run: `cd /root/dev/eabrain && python3 -m pytest tests/ -v`
Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
cd /root/dev/eabrain
git add eabrain.py setup.py CLAUDE.md
git commit -m "chore: bump version to 0.2.0, update CLAUDE.md with memory commands"
```

---

### Task 14: End-to-End Integration Test

**Files:**
- No new files — manual verification

- [ ] **Step 1: Run the full test suite**

```bash
cd /root/dev/eabrain && python3 -m pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 2: Test the full session lifecycle manually**

```bash
# 1. Inject (creates session)
eabrain inject --project /root/dev/eaclaw

# 2. Mid-session remember
eabrain remember "testing the full v0.2 lifecycle"

# 3. Store an observation
eabrain store "SQLite migration works" --type decision

# 4. Close session
eabrain store-summary "Tested the full eabrain v0.2 lifecycle"

# 5. Verify timeline
eabrain timeline --project /root/dev/eaclaw

# 6. Verify search
eabrain search "lifecycle"
eabrain search --fuzzy "memory storage"

# 7. Verify status
eabrain status

# 8. Test sync
eabrain sync --export /tmp/eabrain-backup.db
eabrain sync --import /tmp/eabrain-backup.db

# 9. Start web viewer
eabrain serve --port 37777
# Open http://localhost:37777, verify UI works
```

- [ ] **Step 3: Test migration from v0.1**

```bash
eabrain migrate
eabrain recall --last 5
```

Expected: old session notes appear as observations

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
cd /root/dev/eabrain
git add -A
git commit -m "fix: integration test fixes for v0.2"
```
