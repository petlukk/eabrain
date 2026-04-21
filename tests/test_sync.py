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
        remote_path = os.path.join(d, "remote.db")
        remote = MemoryDB(remote_path)
        remote.store_observation(project="a", obs_type="note", content="from remote", session_id=None)
        remote.close()
        local_path = os.path.join(d, "local.db")
        local = MemoryDB(local_path)
        local.store_observation(project="a", obs_type="note", content="from local", session_id=None)
        local.close()
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
