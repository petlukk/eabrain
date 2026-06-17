import os

import numpy as np
import pytest

FUZZY_LIB = os.path.join(os.path.dirname(__file__), "..", "lib", "libfuzzy.so")
SUBSTR_LIB = os.path.join(os.path.dirname(__file__), "..", "lib", "libsubstr.so")


# ---------- batch_byte_histogram (#1) ----------

def test_batch_histogram_matches_per_blob():
    """Each row of the batched histogram must match the single-blob Python
    reference _byte_histogram, which is what the index loop used before."""
    if not os.path.exists(FUZZY_LIB):
        pytest.skip("libfuzzy.so not built")

    from indexer import _byte_histogram, _simd_batch_byte_histogram

    blobs = [
        b"export func batch_cosine(query: *f32) { let acc: f32x8 = splat(0.0) }",
        b"aab",
        b"",  # empty record in the middle must stay a zero row
        b"\x00\x01\x02\xff\xff" * 40,  # spans multiple 16-byte windows, high bytes
    ]
    batched = _simd_batch_byte_histogram(blobs)
    assert batched.shape == (len(blobs), 256)
    for i, b in enumerate(blobs):
        np.testing.assert_array_almost_equal(batched[i], _byte_histogram(b), decimal=5)


def test_batch_histogram_empty_list():
    if not os.path.exists(FUZZY_LIB):
        pytest.skip("libfuzzy.so not built")
    from indexer import _simd_batch_byte_histogram
    out = _simd_batch_byte_histogram([])
    assert out.shape == (0, 256)


def test_batch_histogram_all_empty_blobs():
    if not os.path.exists(FUZZY_LIB):
        pytest.skip("libfuzzy.so not built")
    from indexer import _simd_batch_byte_histogram
    out = _simd_batch_byte_histogram([b"", b"", b""])
    assert out.shape == (3, 256)
    assert np.sum(out) == 0.0


# ---------- batch_contains (#2) ----------

def _ref_contains(blobs, needle):
    return [needle in b for b in blobs]


def test_batch_contains_matches_reference():
    if not os.path.exists(SUBSTR_LIB):
        pytest.skip("libsubstr.so not built")
    from text_search import batch_contains

    blobs = [
        b"the quick brown fox",
        b"no match here",
        b"needle in a haystack",
        b"",
        b"short",
        b"x" * 50 + b"needle" + b"y" * 50,  # match past the first SIMD windows
    ]
    needle = b"needle"
    assert batch_contains(blobs, needle) == _ref_contains(blobs, needle)


def test_batch_contains_no_cross_record_match():
    """A needle split across two adjacent records must NOT match — records are
    packed contiguously, so the per-record bound is what prevents a false hit."""
    if not os.path.exists(SUBSTR_LIB):
        pytest.skip("libsubstr.so not built")
    from text_search import batch_contains
    # "abcdef" only exists if you read across the boundary of these two records.
    blobs = [b"abc", b"def"]
    assert batch_contains(blobs, b"abcdef") == [False, False]


def test_batch_contains_empty_needle_and_list():
    if not os.path.exists(SUBSTR_LIB):
        pytest.skip("libsubstr.so not built")
    from text_search import batch_contains
    assert batch_contains([b"abc"], b"") == [False]
    assert batch_contains([], b"x") == []
