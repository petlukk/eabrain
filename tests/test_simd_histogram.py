import os
import numpy as np

LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "lib", "libfuzzy.so")

def test_simd_histogram_matches_python():
    """SIMD byte_histogram_embed must produce same result as the Python _byte_histogram."""
    if not os.path.exists(LIB_PATH):
        import pytest
        pytest.skip("libfuzzy.so not built")

    from indexer import _byte_histogram as python_hist
    from indexer import _simd_byte_histogram as simd_hist

    text = b"export func batch_cosine(query: *f32, vecs: *f32) { let acc: f32x8 = splat(0.0) }"
    py_result = python_hist(text)
    simd_result = simd_hist(text)
    np.testing.assert_array_almost_equal(py_result, simd_result, decimal=5)

def test_simd_histogram_empty():
    if not os.path.exists(LIB_PATH):
        import pytest
        pytest.skip("libfuzzy.so not built")

    from indexer import _simd_byte_histogram as simd_hist
    result = simd_hist(b"")
    assert result.shape == (256,)
    assert np.sum(result) == 0.0
