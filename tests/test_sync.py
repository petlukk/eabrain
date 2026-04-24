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

def _setup_git_user(repo):
    import subprocess
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)

def _run_push_retry_test(conflicts: int):
    import subprocess
    from unittest.mock import patch
    import sync
    import os
    import tempfile
    
    with tempfile.TemporaryDirectory() as d:
        origin_dir = os.path.join(d, "origin")
        os.makedirs(origin_dir)
        subprocess.run(["git", "init", "--bare", "--initial-branch=main"], cwd=origin_dir, check=True)
        
        clone_a = os.path.join(d, "clone_a")
        subprocess.run(["git", "clone", origin_dir, clone_a], check=True)
        _setup_git_user(clone_a)
        
        db_a_path = os.path.join(clone_a, "local_a.db")
        db_a = MemoryDB(db_a_path)
        db_a.store_observation(project="test", obs_type="note", content="init", session_id=None)
        db_a.close()
        
        sync.export_db(db_a_path, os.path.join(clone_a, "memory.db"))
        subprocess.run(["git", "add", "memory.db"], cwd=clone_a, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=clone_a, check=True)
        subprocess.run(["git", "push", "origin", "main"], cwd=clone_a, check=True)
        
        # Clone B will be used to inject conflicting pushes
        clone_b = os.path.join(d, "clone_b")
        subprocess.run(["git", "clone", origin_dir, clone_b], check=True)
        _setup_git_user(clone_b)
        db_b_path = os.path.join(clone_b, "local_b.db")
        
        db_b_temp = MemoryDB(db_b_path)
        db_b_temp.close()
        sync.import_db(db_b_path, os.path.join(clone_b, "memory.db"))
        
        db_a = MemoryDB(db_a_path)
        db_a.store_observation(project="test", obs_type="note", content="A-only", session_id=None)
        db_a.close()
        
        original_git = sync._git
        fetch_count = {"clone_a": 0}
        conflict_count = [0]
        
        def patched_git(repo, *args):
            # Only count fetches done by clone A
            if repo == clone_a and args and args[0] == "fetch":
                fetch_count["clone_a"] += 1
                
            # Inject clone B's push right before clone A pushes
            if repo == clone_a and args and args[0] == "push" and conflict_count[0] < conflicts:
                # Add one new observation for Clone B to cause a conflict
                idx = conflict_count[0] + 1
                db_b = MemoryDB(db_b_path)
                db_b.store_observation(project="test", obs_type="note", content=f"B-only-{idx}", session_id=None)
                db_b.close()
                
                # Push clone B directly to origin
                sync.push(db_b_path, clone_b, eabrain_dir=os.path.join(clone_b, ".eabrain"))
                conflict_count[0] += 1
                
            return original_git(repo, *args)
            
        push_exception = None
        try:
            with patch("sync._git", side_effect=patched_git), patch("time.sleep", return_value=None):
                sync.push(db_a_path, clone_a, eabrain_dir=os.path.join(clone_a, ".eabrain"), attempts=3)
        except Exception as e:
            push_exception = e
            
        # Verify state in clone_c
        clone_c = os.path.join(d, "clone_c")
        subprocess.run(["git", "clone", origin_dir, clone_c], check=True)
        db_c_path = os.path.join(clone_c, "local_c.db")
        db_c_temp = MemoryDB(db_c_path)
        db_c_temp.close()
        sync.import_db(db_c_path, os.path.join(clone_c, "memory.db"))
        
        db_c = MemoryDB(db_c_path)
        results = db_c.query("")
        contents = {r["content"] for r in results}
        db_c.close()
        
        # Read log content if it exists
        log_file = os.path.join(clone_a, ".eabrain", "sync.log")
        log_content = ""
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                log_content = f.read()

        return {
            "fetches": fetch_count["clone_a"],
            "final_contents": contents,
            "log_content": log_content,
            "exception": push_exception,
        }

def test_push_retry_success_on_attempt_2():
    result = _run_push_retry_test(conflicts=1)
    assert result["exception"] is None, "push should not raise an exception on success"
    assert result["fetches"] >= 2, "clone_a should have fetched twice (initial + retry)"
    assert "A-only" in result["final_contents"]
    assert "B-only-1" in result["final_contents"]

def test_push_retry_success_on_attempt_3():
    result = _run_push_retry_test(conflicts=2)
    assert result["exception"] is None, "push should not raise an exception on success"
    assert result["fetches"] >= 3, "clone_a should have fetched three times"
    assert "A-only" in result["final_contents"]
    assert "B-only-1" in result["final_contents"]
    assert "B-only-2" in result["final_contents"]

def test_push_retry_exhausts_attempts():
    result = _run_push_retry_test(conflicts=3)
    assert result["exception"] is None, "push should not raise an exception even on exhaustion"
    # The loop fetches exactly once per attempt. 3 attempts = 3 fetches.
    # We use >= 3 for consistency and flexibility if the underlying loop adds a pre-fetch.
    assert result["fetches"] >= 3, "clone_a should exhaust attempts"
    assert "all 3 attempts failed, giving up" in result["log_content"]
