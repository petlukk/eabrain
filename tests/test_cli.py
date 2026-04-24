import json
import os
import subprocess
import pytest

EABRAIN = os.path.join(os.path.dirname(__file__), "..", "eabrain.py")

MINIMAL_KERNEL = """// test fixture
export func batch_cosine(query: *f32, vecs: *f32, dim: i32, out: *mut f32) {
    let acc: f32x8 = splat(0.0)
    out[0] = reduce_add(acc)
}
"""


def run(args, config_path=None):
    cmd = ["python3", EABRAIN] + args
    env = os.environ.copy()
    if config_path:
        env["EABRAIN_CONFIG"] = config_path
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


@pytest.fixture
def config_file(tmp_path):
    project = tmp_path / "fake_project"
    (project / "kernels").mkdir(parents=True)
    (project / "kernels" / "sample.ea").write_text(MINIMAL_KERNEL)

    config = {
        "projects": [str(project)],
        "index_path": str(tmp_path / "index.bin"),
        "max_source_lines": 50,
        "max_session_entries": 100,
        "eabrain_dir": str(tmp_path),
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(config))
    return str(p)


def test_index_and_status(config_file):
    r = run(["index"], config_file)
    assert r.returncode == 0
    r = run(["status"], config_file)
    assert r.returncode == 0
    assert "indexed" in r.stdout.lower()


def test_search(config_file):
    run(["index"], config_file)
    r = run(["search", "batch_cosine"], config_file)
    assert r.returncode == 0
    assert "batch_cosine" in r.stdout


def test_ref(config_file):
    r = run(["ref", "reduce_add"], config_file)
    assert r.returncode == 0
    assert "reduce_add" in r.stdout


def test_remember_recall(config_file):
    run(["index"], config_file)
    r = run(["remember", "fixed sign flip"], config_file)
    assert r.returncode == 0
    r = run(["recall"], config_file)
    assert r.returncode == 0
    assert "sign flip" in r.stdout


def test_inject_prints_preamble(tmp_path, capsys):
    preamble_dir = tmp_path / "preamble"
    preamble_dir.mkdir()
    (preamble_dir / "principles.md").write_text("Think before coding.")
    from commands.system import cmd_inject

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
    from commands.memory import cmd_store

    db_path = str(tmp_path / "memory.db")
    session_file = str(tmp_path / "current_session")
    db = MemoryDB(db_path)
    sid = db.create_session(project="test")
    db.close()
    with open(session_file, "w") as f:
        f.write(sid)

    class Args:
        content = "chose SQLite"
        type = "decision"
        project = "test"

    cfg = {"eabrain_dir": str(tmp_path)}
    cmd_store(Args(), cfg)
    db = MemoryDB(db_path)
    results = db.query("SQLite")
    assert len(results) == 1
    assert results[0]["type"] == "decision"
    assert results[0]["session_id"] == sid
    db.close()


def test_timeline_shows_sessions(tmp_path, capsys):
    from memory import MemoryDB
    from commands.memory import cmd_timeline

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
    assert "hello" in captured.out


def test_store_summary_closes_session(tmp_path, capsys):
    from memory import MemoryDB
    from commands.memory import cmd_store_summary

    db_path = str(tmp_path / "memory.db")
    session_file = str(tmp_path / "current_session")
    db = MemoryDB(db_path)
    sid = db.create_session(project="test")
    db.close()
    with open(session_file, "w") as f:
        f.write(sid)

    class Args:
        content = "summary text"

    cfg = {"eabrain_dir": str(tmp_path)}
    cmd_store_summary(Args(), cfg)
    captured = capsys.readouterr()
    assert "Session closed" in captured.out
    assert not os.path.exists(session_file)
    db = MemoryDB(db_path)
    row = db.conn.execute("SELECT summary, ended_at FROM sessions WHERE id = ?", (sid,)).fetchone()
    assert row["summary"] == "summary text"
    assert row["ended_at"] is not None
    db.close()


def test_migrate_command(tmp_path, capsys):
    import numpy as np
    from indexer import write_index
    from commands.memory import cmd_migrate

    idx_path = tmp_path / "index.bin"
    sessions = [{"text": "old note", "timestamp": 1713700000}]
    write_index(str(idx_path), [], [], sessions, np.zeros((0, 256), dtype=np.float32))

    class Args:
        pass

    cfg = {"eabrain_dir": str(tmp_path), "index_path": str(idx_path)}
    cmd_migrate(Args(), cfg)
    captured = capsys.readouterr()
    assert "Migrated 1" in captured.out


def test_sync_export_import(tmp_path, capsys):
    from memory import MemoryDB
    from commands.memory import cmd_sync

    db_path = str(tmp_path / "memory.db")
    db = MemoryDB(db_path)
    sid = db.create_session(project="test")
    db.store_observation(project="test", obs_type="note", content="exported", session_id=sid)
    db.close()

    export_path = str(tmp_path / "out.db")

    class ExportArgs:
        export_path = str(tmp_path / "out.db")
        import_path = None

    class ImportArgs:
        export_path = None
        import_path = str(tmp_path / "out.db")

    cfg = {"eabrain_dir": str(tmp_path)}
    cmd_sync(ExportArgs(), cfg)
    assert os.path.exists(export_path)
    cmd_sync(ImportArgs(), cfg)
    captured = capsys.readouterr()
    assert "Exported" in captured.out
    assert "Imported" in captured.out


def test_default_projects_auto_discovers_parent():
    from eabrain import _default_projects
    import eabrain as eabrain_mod
    expected = os.path.dirname(os.path.dirname(os.path.abspath(eabrain_mod.__file__)))
    assert _default_projects() == [expected]


def test_load_config_missing_projects_triggers_auto_discovery(tmp_path):
    from eabrain import _load_config, _default_projects
    cfg_path = tmp_path / "no_such_config.json"
    cfg = _load_config(str(cfg_path))
    assert cfg["projects"] == _default_projects()


def test_load_config_explicit_empty_projects_stays_empty(tmp_path):
    from eabrain import _load_config
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"projects": []}))
    cfg = _load_config(str(cfg_path))
    assert cfg["projects"] == []


def test_resolve_eacompute_dir_sibling_fallback(tmp_path, monkeypatch):
    import eabrain
    dev = tmp_path / "Dev"
    eabrain_dir = dev / "eabrain"
    eacompute = dev / "eacompute"
    eabrain_dir.mkdir(parents=True)
    (eacompute / "src").mkdir(parents=True)
    monkeypatch.setattr(eabrain, "__file__", str(eabrain_dir / "eabrain.py"))
    monkeypatch.delenv("EACOMPUTE_DIR", raising=False)
    assert eabrain._resolve_eacompute_dir(None) == str(eacompute)


def test_resolve_eacompute_dir_no_sibling_returns_none(tmp_path, monkeypatch):
    import eabrain
    eabrain_dir = tmp_path / "Dev" / "eabrain"
    eabrain_dir.mkdir(parents=True)
    monkeypatch.setattr(eabrain, "__file__", str(eabrain_dir / "eabrain.py"))
    monkeypatch.delenv("EACOMPUTE_DIR", raising=False)
    assert eabrain._resolve_eacompute_dir(None) is None


def test_patterns_list_empty_when_no_history(tmp_path, capsys):
    from commands.search import cmd_patterns
    ar_dir = tmp_path / "autoresearch" / "kernels"
    (ar_dir / "matmul").mkdir(parents=True)
    (ar_dir / "dotprod").mkdir(parents=True)

    class Args:
        query = None
        what_works = False

    cfg = {"autoresearch_dir": str(ar_dir)}
    cmd_patterns(Args(), cfg)
    captured = capsys.readouterr()
    assert "0 of 2 kernels benchmarked" in captured.out
