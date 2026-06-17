"""text_search.py — SIMD substring search helper around libsubstr.so.

Loads the shared library at import time (OSError propagates if it's missing,
so callers can wrap the import in a try/except to fall back to SQL LIKE).
`argtypes`/`restype` are set once on the cached handle.
"""

import ctypes
import os

import numpy as np

_LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")

_lib = ctypes.CDLL(os.path.join(_LIB_DIR, "libsubstr.so"))
_lib.substr_search.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
_lib.substr_search.restype = None

_lib.batch_contains.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
_lib.batch_contains.restype = None


def find_offsets(haystack: bytes, needle: bytes, max_results: int = 256) -> list[int]:
    """Return byte offsets of every occurrence of `needle` in `haystack`.

    Empty inputs return an empty list. Matches are capped at `max_results`.
    """
    if not needle or not haystack:
        return []
    hay = np.frombuffer(haystack, dtype=np.uint8)
    ndl = np.frombuffer(needle, dtype=np.uint8)
    offsets = np.zeros(max_results, dtype=np.int32)
    count = np.zeros(1, dtype=np.int32)
    _lib.substr_search(
        hay.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(len(haystack)),
        ndl.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(len(needle)),
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(max_results),
        count.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    return offsets[:int(count[0])].tolist()


def batch_contains(blobs, needle: bytes) -> list:
    """Return a list of bools: blobs[r] contains `needle`.

    Packs every blob into one buffer and runs the SIMD membership scan in a
    single call, replacing a per-blob Python loop. Empty needle or empty
    `blobs` yields all-False."""
    n = len(blobs)
    if not needle or n == 0:
        return [False] * n
    offsets = np.zeros(n + 1, dtype=np.int32)
    acc = 0
    for i, b in enumerate(blobs):
        acc += len(b)
        offsets[i + 1] = acc
    data = np.frombuffer(b"".join(blobs), dtype=np.uint8)
    ndl = np.frombuffer(needle, dtype=np.uint8)
    flags = np.zeros(n, dtype=np.int32)
    if data.size == 0:
        return [False] * n
    _lib.batch_contains(
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(n),
        ndl.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(len(needle)),
        flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    return [bool(x) for x in flags]
