"""Tests for the web viewer server (server.py)."""

import json
import os
import tempfile
import threading
import urllib.request

from memory import MemoryDB


def _setup_db(d):
    db_path = os.path.join(d, "memory.db")
    db = MemoryDB(db_path)
    sid = db.create_session(project="test")
    db.store_observation(project="test", obs_type="decision", content="chose SQLite", session_id=sid)
    db.store_observation(project="test", obs_type="bug", content="off-by-one", session_id=sid)
    db.close_session(sid, summary="test session done")
    db.close()
    return db_path


def _start(server):
    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()
    return t


def test_api_stats():
    from server import make_server
    with tempfile.TemporaryDirectory() as d:
        _setup_db(d)
        cfg = {"eabrain_dir": d, "index_path": os.path.join(d, "index.bin")}
        server = make_server(cfg, port=0)
        try:
            port = server.server_address[1]
            t = _start(server)
            resp = urllib.request.urlopen(f"http://localhost:{port}/api/stats", timeout=2)
            data = json.loads(resp.read())
            assert data["observation_count"] == 2
            assert data["session_count"] == 1
            t.join(timeout=2)
        finally:
            server.server_close()


def test_api_timeline():
    from server import make_server
    with tempfile.TemporaryDirectory() as d:
        _setup_db(d)
        cfg = {"eabrain_dir": d, "index_path": os.path.join(d, "index.bin")}
        server = make_server(cfg, port=0)
        try:
            port = server.server_address[1]
            t = _start(server)
            resp = urllib.request.urlopen(f"http://localhost:{port}/api/timeline?project=test", timeout=2)
            data = json.loads(resp.read())
            assert len(data) == 1
            assert data[0]["session"]["summary"] == "test session done"
            assert len(data[0]["observations"]) == 2
            t.join(timeout=2)
        finally:
            server.server_close()


def test_api_search():
    from server import make_server
    with tempfile.TemporaryDirectory() as d:
        _setup_db(d)
        cfg = {"eabrain_dir": d, "index_path": os.path.join(d, "index.bin")}
        server = make_server(cfg, port=0)
        try:
            port = server.server_address[1]
            t = _start(server)
            resp = urllib.request.urlopen(f"http://localhost:{port}/api/search?q=SQLite", timeout=2)
            data = json.loads(resp.read())
            assert len(data["observations"]) >= 1
            t.join(timeout=2)
        finally:
            server.server_close()


def test_api_observation_detail():
    from server import make_server
    with tempfile.TemporaryDirectory() as d:
        _setup_db(d)
        cfg = {"eabrain_dir": d, "index_path": os.path.join(d, "index.bin")}
        # Get an obs_id first
        db = MemoryDB(os.path.join(d, "memory.db"))
        obs_id = db.conn.execute("SELECT id FROM observations LIMIT 1").fetchone()[0]
        db.close()

        server = make_server(cfg, port=0)
        try:
            port = server.server_address[1]
            t = _start(server)
            resp = urllib.request.urlopen(f"http://localhost:{port}/api/observations/{obs_id}", timeout=2)
            data = json.loads(resp.read())
            assert data["id"] == obs_id
            assert "embedding" not in data
            t.join(timeout=2)
        finally:
            server.server_close()


def test_api_404():
    from server import make_server
    with tempfile.TemporaryDirectory() as d:
        _setup_db(d)
        cfg = {"eabrain_dir": d, "index_path": os.path.join(d, "index.bin")}
        server = make_server(cfg, port=0)
        try:
            port = server.server_address[1]
            t = _start(server)
            try:
                urllib.request.urlopen(f"http://localhost:{port}/api/nonexistent", timeout=2)
                assert False, "expected 404"
            except urllib.error.HTTPError as e:
                assert e.code == 404
            t.join(timeout=2)
        finally:
            server.server_close()
