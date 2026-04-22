import os
import tempfile

import numpy as np

from inject import load_preamble, build_injection, start_session, get_current_session_id, end_session, ensure_preamble
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


def test_start_session_creates_file():
    with tempfile.TemporaryDirectory() as d:
        db = MemoryDB(os.path.join(d, "memory.db"))
        session_file = os.path.join(d, "current_session")
        sid = start_session(db, project="eaclaw", session_file=session_file)
        assert os.path.exists(session_file)
        with open(session_file, encoding="utf-8") as f:
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


def test_ensure_preamble_creates_defaults():
    with tempfile.TemporaryDirectory() as d:
        preamble_dir = os.path.join(d, "preamble")
        ensure_preamble(preamble_dir)
        assert os.path.exists(os.path.join(preamble_dir, "01_principles.md"))
        assert os.path.exists(os.path.join(preamble_dir, "02_hard_rules.md"))
        assert os.path.exists(os.path.join(preamble_dir, "03_commands.md"))
        with open(os.path.join(preamble_dir, "01_principles.md")) as f:
            content = f.read()
        assert "Think Before Coding" in content
        with open(os.path.join(preamble_dir, "03_commands.md")) as f:
            commands = f.read()
        assert "eabrain recall" in commands
        assert "eabrain store-summary" in commands


def test_ensure_preamble_does_not_overwrite():
    with tempfile.TemporaryDirectory() as d:
        preamble_dir = os.path.join(d, "preamble")
        os.makedirs(preamble_dir)
        with open(os.path.join(preamble_dir, "custom.md"), "w") as f:
            f.write("My custom rule")
        ensure_preamble(preamble_dir)
        assert not os.path.exists(os.path.join(preamble_dir, "01_principles.md"))
