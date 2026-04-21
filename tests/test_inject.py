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
