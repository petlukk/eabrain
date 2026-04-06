import os
import pytest
from indexer import build_index, read_index


def test_build_index_finds_kernels(tmp_path):
    index_path = str(tmp_path / "index.bin")
    ref_path = os.path.join(os.path.dirname(__file__), "..", "reference", "ea_reference.json")
    stats = build_index(["/root/dev/eaclaw", "/root/dev/eakv"], ref_path, index_path)
    assert stats["kernel_count"] > 0
    assert stats["ref_count"] > 0


def test_index_contains_known_function(tmp_path):
    index_path = str(tmp_path / "index.bin")
    ref_path = os.path.join(os.path.dirname(__file__), "..", "reference", "ea_reference.json")
    build_index(["/root/dev/eaclaw"], ref_path, index_path)
    idx = read_index(index_path)
    func_names = [k["func_name"] for k in idx["kernels"]]
    assert any("batch" in name for name in func_names), f"No batch* in {func_names[:10]}"


def test_index_has_reference_entries(tmp_path):
    index_path = str(tmp_path / "index.bin")
    ref_path = os.path.join(os.path.dirname(__file__), "..", "reference", "ea_reference.json")
    build_index(["/root/dev/eaclaw"], ref_path, index_path)
    idx = read_index(index_path)
    ref_names = {r["name"] for r in idx["refs"]}
    assert "reduce_add" in ref_names
