"""Tests for v0.1 → v0.2 migration of session notes from index.bin to memory.db."""

import os
import tempfile

import numpy as np

from indexer import write_index
from memory import MemoryDB, migrate_from_index


def test_migrate_creates_observations():
    with tempfile.TemporaryDirectory() as d:
        idx_path = os.path.join(d, "index.bin")
        sessions = [
            {"text": "Fixed the sign flip bug", "timestamp": 1713700000},
            {"text": "Decided to use SQLite", "timestamp": 1713700100},
        ]
        write_index(idx_path, [], [], sessions, np.zeros((0, 256), dtype=np.float32))
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
