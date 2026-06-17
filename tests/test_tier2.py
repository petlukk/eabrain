import glob
import os

import numpy as np
import pytest

SCAN_LIB = os.path.join(os.path.dirname(__file__), "..", "lib", "libscan.so")
KERNEL_DIR = os.path.join(os.path.dirname(__file__), "..", "kernels")


# ---------- #3 brace-depth kernel (find_brace_offsets) ----------

def _old_line_count(src_bytes, off, n):
    """The byte-by-byte brace walk _scan_ea_file used before find_brace_offsets."""
    body_start = src_bytes.find(b"{", off)
    if body_start < 0:
        return 1
    depth = 0
    pos = body_start
    while pos < n:
        ch = src_bytes[pos:pos + 1]
        if ch == b"{":
            depth += 1
        elif ch == b"}":
            depth -= 1
            if depth == 0:
                break
        pos += 1
    return src_bytes[off:pos + 1].count(b"\n") + 1


def _new_line_counts(src_bytes):
    """Drive the SIMD brace path the way _scan_ea_file does and return the
    list of (off, line_count) for every export."""
    import ctypes

    from indexer import _bind_scan_lib, _load_lib, _match_close_brace

    src = np.frombuffer(src_bytes, dtype=np.uint8)
    n = len(src_bytes)
    lib = _load_lib("libscan.so")
    _bind_scan_lib(lib)

    offsets = np.zeros(256, dtype=np.int32)
    cnt = np.zeros(1, dtype=np.int32)
    lib.find_export_offsets(src, n, offsets, cnt)

    bpos = np.zeros(max(n, 1), dtype=np.int32)
    bkind = np.zeros(max(n, 1), dtype=np.int32)
    bcnt = np.zeros(1, dtype=np.int32)
    lib.find_brace_offsets(src, n, bpos, bkind, n, bcnt)
    bc = int(bcnt[0])
    bpos, bkind = bpos[:bc], bkind[:bc]

    out = []
    for i in range(int(cnt[0])):
        off = int(offsets[i])
        cp = _match_close_brace(bpos, bkind, off, n)
        lc = 1 if cp is None else src_bytes[off:cp + 1].count(b"\n") + 1
        out.append((off, lc))
    return out


def _ref_line_counts(src_bytes):
    """Same export discovery, but old byte-walk for line_count."""
    out = []
    start = 0
    n = len(src_bytes)
    while True:
        off = src_bytes.find(b"export func ", start)
        if off < 0:
            break
        out.append((off, _old_line_count(src_bytes, off, n)))
        start = off + 12
    return out


def test_brace_line_counts_match_on_real_kernels():
    if not os.path.exists(SCAN_LIB):
        pytest.skip("libscan.so not built")
    files = sorted(glob.glob(os.path.join(KERNEL_DIR, "*.ea")))
    assert files, "no kernel sources found"
    for path in files:
        with open(path, "rb") as f:
            src = f.read()
        assert _new_line_counts(src) == _ref_line_counts(src), path


@pytest.mark.parametrize("src", [
    b"export func a() { let x: i32 = 0 }\n",
    b"export func a() {\n  if x {\n    y\n  }\n}\nexport func b() { z }\n",
    b"export func noopen()\n",                 # no '{' -> line_count 1
    b"export func unbalanced() {\n  {\n",      # never rebalances -> runs to EOF
    b"export func a() { } export func b() { }\n",
])
def test_brace_line_counts_match_edge_cases(src):
    if not os.path.exists(SCAN_LIB):
        pytest.skip("libscan.so not built")
    assert _new_line_counts(src) == _ref_line_counts(src)


# ---------- #4 top-k selection ----------

def test_top_k_desc_matches_full_sort():
    from commands.search import _top_k_desc
    rng = np.random.default_rng(0)
    for n in (0, 1, 5, 10, 200):
        for k in (5, 10):
            scores = rng.random(n).astype(np.float32)
            got = _top_k_desc(scores, k)
            ref = np.argsort(scores)[::-1][:k]
            # Same selected scores, in the same descending order.
            np.testing.assert_array_equal(scores[got], scores[ref])


def test_top_k_desc_handles_ties():
    from commands.search import _top_k_desc
    scores = np.array([0.5, 0.5, 0.5, 0.1], dtype=np.float32)
    order = _top_k_desc(scores, 2)
    assert len(order) == 2
    assert np.all(scores[order] == 0.5)


# ---------- index-build directory walk (_find_ea_files) ----------

def test_find_ea_files_prunes_and_recurses(tmp_path):
    from indexer import _find_ea_files

    # Source kernels we expect to find, at varying depths.
    (tmp_path / "a.ea").write_bytes(b"export func a() {}")
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels" / "b.ea").write_bytes(b"export func b() {}")
    (tmp_path / "nested" / "deep").mkdir(parents=True)
    (tmp_path / "nested" / "deep" / "c.ea").write_bytes(b"export func c() {}")

    # .ea files buried in pruned dirs must NOT be found.
    for skip in ("target", ".git", "node_modules", "venv", "__pycache__"):
        (tmp_path / skip).mkdir()
        (tmp_path / skip / "ignore.ea").write_bytes(b"export func z() {}")

    found = {os.path.relpath(p, tmp_path) for p in _find_ea_files(str(tmp_path))}
    assert found == {
        "a.ea",
        os.path.join("kernels", "b.ea"),
        os.path.join("nested", "deep", "c.ea"),
    }


def test_find_ea_files_missing_root():
    from indexer import _find_ea_files
    assert _find_ea_files("/no/such/dir/xyz") == []
