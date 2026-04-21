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
