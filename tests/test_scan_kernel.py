import ctypes
import numpy as np
import os
import pytest

LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "lib", "libscan.so")

@pytest.fixture
def lib():
    return ctypes.CDLL(LIB_PATH)

SAMPLE_EA = b"""// test kernel
export func batch_cosine(query: *f32, vecs: *f32, dim: i32, out: *mut f32) {
    let mut acc: f32x8 = splat(0.0)
    for i in 0..dim step 8 {
        let q: f32x8 = load(query, i)
        let d: f32x8 = load(vecs, i)
        acc = fma(q, d, acc)
    }
    let result: f32 = reduce_add(acc)
    out[0] = result
}

export func normalize(vecs: *mut f32, dim: i32, n: i32) {
    let zero: f32x4 = splat(0.0)
    let x: u8x16 = splat(0)
}
"""

def test_count_exports(lib):
    src = np.frombuffer(SAMPLE_EA, dtype=np.uint8)
    count = np.zeros(1, dtype=np.int32)
    lib.count_exports(
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(len(SAMPLE_EA)),
        count.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    assert count[0] == 2

def test_find_export_offsets(lib):
    src = np.frombuffer(SAMPLE_EA, dtype=np.uint8)
    offsets = np.zeros(16, dtype=np.int32)
    count = np.zeros(1, dtype=np.int32)
    lib.find_export_offsets(
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(len(SAMPLE_EA)),
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        count.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    assert count[0] == 2
    pos0 = SAMPLE_EA.index(b"export func batch_cosine")
    pos1 = SAMPLE_EA.index(b"export func normalize")
    assert offsets[0] == pos0
    assert offsets[1] == pos1

def test_detect_simd_types(lib):
    src = np.frombuffer(SAMPLE_EA, dtype=np.uint8)
    mask = np.zeros(1, dtype=np.int32)
    lib.detect_simd_types(
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(len(SAMPLE_EA)),
        mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    m = mask[0]
    assert m & 0b01 != 0, "f32x4 should be detected"
    assert m & 0b10 != 0, "f32x8 should be detected"
    assert m & 0b100 == 0, "f32x16 should not be detected"
    assert m & 0b1000000 != 0, "u8x16 should be detected"

def test_detect_intrinsics(lib):
    src = np.frombuffer(SAMPLE_EA, dtype=np.uint8)
    mask = np.zeros(1, dtype=np.int32)
    lib.detect_intrinsics(
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(len(SAMPLE_EA)),
        mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    m = mask[0]
    assert m & 1 != 0, "reduce_add should be detected"
    assert m & 2 != 0, "fma should be detected"
    assert m & 4 != 0, "splat should be detected"
    assert m & 8 != 0, "load should be detected"

def test_count_exports_real_file(lib):
    """Test against a real .ea kernel file."""
    real_path = "/root/dev/eaclaw/kernels/search.ea"
    if not os.path.exists(real_path):
        pytest.skip("eaclaw not available")
    with open(real_path, "rb") as f:
        data = f.read()
    src = np.frombuffer(data, dtype=np.uint8)
    count = np.zeros(1, dtype=np.int32)
    lib.count_exports(
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(len(data)),
        count.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    # eaclaw/search.ea has multiple exports (batch_dot, batch_cosine, etc.)
    assert count[0] >= 4, f"Expected >=4 exports, got {count[0]}"
