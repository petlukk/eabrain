import os
import pytest
from indexer import build_index, read_index

MINIMAL_KERNEL = """// test fixture
export func batch_cosine(query: *f32, vecs: *f32, dim: i32, out: *mut f32) {
    let acc: f32x8 = splat(0.0)
    out[0] = reduce_add(acc)
}

export func batch_dot(query: *f32, vecs: *f32, dim: i32, out: *mut f32) {
    let acc: f32x4 = splat(0.0)
    out[0] = reduce_add(acc)
}
"""

REF_PATH = os.path.join(os.path.dirname(__file__), "..", "reference", "ea_reference.json")


@pytest.fixture
def project_dir(tmp_path):
    root = tmp_path / "fake_project"
    (root / "kernels").mkdir(parents=True)
    (root / "kernels" / "sample.ea").write_text(MINIMAL_KERNEL)
    return str(root)


def test_build_index_finds_kernels(tmp_path, project_dir):
    index_path = str(tmp_path / "index.bin")
    stats = build_index([project_dir], REF_PATH, index_path)
    assert stats["kernel_count"] > 0
    assert stats["ref_count"] > 0


def test_index_contains_known_function(tmp_path, project_dir):
    index_path = str(tmp_path / "index.bin")
    build_index([project_dir], REF_PATH, index_path)
    idx = read_index(index_path)
    func_names = [k["func_name"] for k in idx["kernels"]]
    assert any("batch" in name for name in func_names), f"No batch* in {func_names[:10]}"


def test_index_has_reference_entries(tmp_path, project_dir):
    index_path = str(tmp_path / "index.bin")
    build_index([project_dir], REF_PATH, index_path)
    idx = read_index(index_path)
    ref_names = {r["name"] for r in idx["refs"]}
    assert "reduce_add" in ref_names
