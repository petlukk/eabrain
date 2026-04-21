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
