import os
import tempfile
import pytest

from memory import MemoryDB

LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "lib", "libsubstr.so")


def test_simd_text_search():
    """Verify memory search uses SIMD when available."""
    if not os.path.exists(LIB_PATH):
        pytest.skip("libsubstr.so not built")

    with tempfile.TemporaryDirectory() as d:
        db = MemoryDB(os.path.join(d, "memory.db"))
        db.store_observation(project="a", obs_type="decision", content="chose SQLite for variable-length storage", session_id=None)
        db.store_observation(project="a", obs_type="bug", content="off-by-one in scan loop", session_id=None)
        db.store_observation(project="a", obs_type="note", content="SIMD search is fast", session_id=None)
        results = db.simd_search("SQLite")
        assert len(results) == 1
        assert "SQLite" in results[0]["content"]
        db.close()


def test_simd_text_search_multiple_hits():
    if not os.path.exists(LIB_PATH):
        pytest.skip("libsubstr.so not built")

    with tempfile.TemporaryDirectory() as d:
        db = MemoryDB(os.path.join(d, "memory.db"))
        db.store_observation(project="a", obs_type="note", content="SIMD kernel compiled", session_id=None)
        db.store_observation(project="a", obs_type="note", content="SIMD search works", session_id=None)
        db.store_observation(project="a", obs_type="note", content="Python is slow", session_id=None)
        results = db.simd_search("SIMD")
        assert len(results) == 2
        db.close()
