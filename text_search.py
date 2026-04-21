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
