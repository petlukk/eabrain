import ctypes
import numpy as np
import os
import pytest

LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "lib", "libfuzzy.so")

@pytest.fixture
def lib():
    return ctypes.CDLL(LIB_PATH)

def test_byte_histogram_embed(lib):
    """Byte histogram should count byte frequencies."""
    dim = 256
    text = b"aab"  # a=97 appears 2x, b=98 appears 1x
    out = np.zeros(dim, dtype=np.float32)
    lib.byte_histogram_embed(
        text,
        ctypes.c_int32(len(text)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    assert out[97] == 2.0  # 'a'
    assert out[98] == 1.0  # 'b'
    assert out[0] == 0.0

def test_batch_cosine_identical(lib):
    """Identical vectors should have cosine similarity ~1.0."""
    dim = 256
    n_vecs = 4
    query = np.random.randn(dim).astype(np.float32)
    query /= np.linalg.norm(query)
    vecs = np.tile(query, n_vecs).astype(np.float32)
    out = np.zeros(n_vecs, dtype=np.float32)
    q_norm = float(np.linalg.norm(query))
    lib.batch_cosine(
        query.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(q_norm),
        vecs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int32(dim),
        ctypes.c_int32(n_vecs),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    for i in range(n_vecs):
        assert abs(out[i] - 1.0) < 0.01, f"vec {i}: {out[i]}"

def test_batch_cosine_orthogonal(lib):
    """Orthogonal vectors should have cosine similarity ~0.0."""
    dim = 256
    query = np.zeros(dim, dtype=np.float32)
    query[0] = 1.0
    vec = np.zeros(dim, dtype=np.float32)
    vec[1] = 1.0
    out = np.zeros(1, dtype=np.float32)
    lib.batch_cosine(
        query.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(1.0),
        vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int32(dim),
        ctypes.c_int32(1),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    assert abs(out[0]) < 0.01

def test_normalize_vectors(lib):
    """Normalized vectors should have unit length."""
    dim = 256
    n_vecs = 3
    vecs = np.random.randn(n_vecs * dim).astype(np.float32)
    lib.normalize_vectors(
        vecs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int32(dim),
        ctypes.c_int32(n_vecs),
    )
    for i in range(n_vecs):
        norm = np.linalg.norm(vecs[i * dim:(i + 1) * dim])
        assert abs(norm - 1.0) < 0.01, f"vec {i} norm: {norm}"


# ---------------------------------------------------------------------------
# Regression: scalar-tail loops must read floats directly, not via to_i32.
#
# `to_f32(to_i32(ptr[i]))` truncates the fractional part — silently corrupting
# any element handled by the scalar tail (dim not a multiple of the SIMD
# width: 8 on x86, 4 on aarch64). The SIMD body is unaffected; only
# misaligned dims trigger the bug, so the existing dim=256 tests didn't
# catch it.
# ---------------------------------------------------------------------------

def test_normalize_vectors_misaligned_dim(lib):
    """All-tail path: dim=7 forces every element through the scalar loop.

    Before the fix, `to_f32(to_i32(vecs[i]))` truncated 0.5 → 0, so the
    norm came out 0 and the kernel left the vector untouched.
    """
    dim = 7
    vecs = np.full(dim, 0.5, dtype=np.float32)
    lib.normalize_vectors(
        vecs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int32(dim),
        ctypes.c_int32(1),
    )
    norm = float(np.linalg.norm(vecs))
    assert abs(norm - 1.0) < 1e-5, f"norm={norm}, vecs={vecs}"


def test_batch_cosine_misaligned_dim(lib):
    """Identical fractional vectors with dim=10 must score ~1.0.

    dim=10 puts elements 0..7 in the SIMD body and 8..9 in the scalar tail.
    Before the fix, the tail elements were truncated to 0, so the dot
    product missed two contributions and cosine dropped to ~0.894.
    """
    dim = 10
    query = np.full(dim, 0.5, dtype=np.float32)
    vec = query.copy()
    q_norm = float(np.linalg.norm(query))
    out = np.zeros(1, dtype=np.float32)
    lib.batch_cosine(
        query.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(q_norm),
        vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int32(dim),
        ctypes.c_int32(1),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    assert abs(out[0] - 1.0) < 1e-5, f"cosine={out[0]}"
