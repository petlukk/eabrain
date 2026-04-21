import ctypes
import os
import numpy as np
import pytest

LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "lib", "libsubstr.so")

@pytest.fixture
def lib():
    if not os.path.exists(LIB_PATH):
        pytest.skip("libsubstr.so not built")
    lib = ctypes.CDLL(LIB_PATH)
    lib.substr_search.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),  # haystack
        ctypes.c_int32,                   # haystack_len
        ctypes.POINTER(ctypes.c_uint8),  # needle
        ctypes.c_int32,                   # needle_len
        ctypes.POINTER(ctypes.c_int32),  # out_offsets
        ctypes.c_int32,                   # out_max
        ctypes.POINTER(ctypes.c_int32),  # out_count
    ]
    lib.substr_search.restype = None
    return lib

def test_find_single_match(lib):
    haystack = b"hello world SQLite is great"
    needle = b"SQLite"
    hay = np.frombuffer(haystack, dtype=np.uint8)
    ndl = np.frombuffer(needle, dtype=np.uint8)
    offsets = np.zeros(64, dtype=np.int32)
    count = np.zeros(1, dtype=np.int32)
    lib.substr_search(
        hay.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(len(haystack)),
        ndl.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(len(needle)),
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(64),
        count.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    assert int(count[0]) == 1
    assert int(offsets[0]) == 12  # "SQLite" starts at byte 12

def test_find_multiple_matches(lib):
    haystack = b"abc abc abc"
    needle = b"abc"
    hay = np.frombuffer(haystack, dtype=np.uint8)
    ndl = np.frombuffer(needle, dtype=np.uint8)
    offsets = np.zeros(64, dtype=np.int32)
    count = np.zeros(1, dtype=np.int32)
    lib.substr_search(
        hay.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(len(haystack)),
        ndl.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(len(needle)),
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(64),
        count.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    assert int(count[0]) == 3
    assert int(offsets[0]) == 0
    assert int(offsets[1]) == 4
    assert int(offsets[2]) == 8

def test_no_match(lib):
    haystack = b"hello world"
    needle = b"xyz"
    hay = np.frombuffer(haystack, dtype=np.uint8)
    ndl = np.frombuffer(needle, dtype=np.uint8)
    offsets = np.zeros(64, dtype=np.int32)
    count = np.zeros(1, dtype=np.int32)
    lib.substr_search(
        hay.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(len(haystack)),
        ndl.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(len(needle)),
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(64),
        count.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    assert int(count[0]) == 0

def test_caps_at_out_max(lib):
    haystack = b"aaa"
    needle = b"a"
    hay = np.frombuffer(haystack, dtype=np.uint8)
    ndl = np.frombuffer(needle, dtype=np.uint8)
    offsets = np.zeros(2, dtype=np.int32)
    count = np.zeros(1, dtype=np.int32)
    lib.substr_search(
        hay.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(len(haystack)),
        ndl.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(len(needle)),
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(2),
        count.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    assert int(count[0]) == 2

def test_long_haystack(lib):
    """Test with haystack > 16 bytes to exercise SIMD path."""
    haystack = b"x" * 100 + b"NEEDLE" + b"x" * 100 + b"NEEDLE" + b"x" * 50
    needle = b"NEEDLE"
    hay = np.frombuffer(haystack, dtype=np.uint8)
    ndl = np.frombuffer(needle, dtype=np.uint8)
    offsets = np.zeros(64, dtype=np.int32)
    count = np.zeros(1, dtype=np.int32)
    lib.substr_search(
        hay.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(len(haystack)),
        ndl.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(len(needle)),
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(64),
        count.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    assert int(count[0]) == 2
    assert int(offsets[0]) == 100
    assert int(offsets[1]) == 206
