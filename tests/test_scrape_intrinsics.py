"""Tests for the eacompute intrinsic scraper.

Verifies that `scrape_eacompute_intrinsics` finds the doc-commented
signatures in the Eä compiler's typeck sources, and that `build_index`
merges them into the ref list without clobbering the curated JSON.
"""

import os
import tempfile

import pytest

from indexer import scrape_eacompute_intrinsics

EACOMPUTE_DIR = "/root/dev/eacompute"
HAS_EACOMPUTE = os.path.isdir(os.path.join(EACOMPUTE_DIR, "src", "typeck"))


@pytest.mark.skipif(not HAS_EACOMPUTE, reason="eacompute source not available")
def test_scrape_finds_known_intrinsics():
    """Scraping eacompute src/typeck must surface recently-added intrinsics
    that do NOT appear in the static reference JSON."""
    results = scrape_eacompute_intrinsics([EACOMPUTE_DIR])
    names = {r["name"] for r in results}

    # These were added over the past weeks and must be picked up automatically.
    expected = {
        "maddubs_i16",
        "madd_i16",
        "cvt_f16_f32",
        "cvt_f32_f16",
        "smmla_i32",
        "ummla_i32",
    }
    missing = expected - names
    assert not missing, f"Scraper missed known intrinsics: {missing}"


@pytest.mark.skipif(not HAS_EACOMPUTE, reason="eacompute source not available")
def test_scraped_entry_shape():
    results = scrape_eacompute_intrinsics([EACOMPUTE_DIR])
    assert results, "expected at least one scraped intrinsic"
    for entry in results:
        assert set(entry.keys()) == {"name", "category", "signature", "description"}
        assert entry["category"] == "intrinsic"
        assert entry["name"]
        assert "->" in entry["signature"]
        # Must fit in the binary ref record slots.
        assert len(entry["signature"].encode("utf-8")) <= 127
        assert len(entry["description"].encode("utf-8")) <= 255


@pytest.mark.skipif(not HAS_EACOMPUTE, reason="eacompute source not available")
def test_scraper_collapses_width_variants():
    """maddubs_i16 has three width variants in the source; they must be
    collapsed into a single entry whose signature mentions all of them."""
    results = scrape_eacompute_intrinsics([EACOMPUTE_DIR])
    by_name = {r["name"]: r for r in results}
    assert "maddubs_i16" in by_name
    sig = by_name["maddubs_i16"]["signature"]
    # All three width variants must be present (u8x16, u8x32, u8x64).
    assert "u8x16" in sig
    assert "u8x32" in sig
    assert "u8x64" in sig


def test_scraper_handles_missing_project_dir():
    """Scraper must not crash when given a nonexistent directory."""
    with tempfile.TemporaryDirectory() as empty:
        results = scrape_eacompute_intrinsics([empty, "/nonexistent/path"])
        assert results == []


def test_find_arrow_pairs_kernel():
    """Direct test of the scan_rust.ea SIMD kernel: every "->" in the input
    must appear in the returned offsets, and no false positives."""
    from indexer import _find_arrow_offsets, _load_scan_rust_lib

    lib = _load_scan_rust_lib()
    # Kernel may be absent in pre-build state; test both paths below.

    # Crafted input with known arrow positions.
    src = b"fn a() -> i32 { 1 }\n/// foo(a) -> b\n// minus- sign\n"
    expected = [
        src.index(b"->"),
        src.index(b"->", src.index(b"->") + 1),
    ]

    # SIMD path (if kernel is built).
    if lib is not None:
        simd_offsets = _find_arrow_offsets(src, lib)
        assert simd_offsets == expected, (
            f"SIMD kernel returned {simd_offsets}, expected {expected}"
        )

    # Pure-Python fallback path (force by passing None).
    py_offsets = _find_arrow_offsets(src, None)
    assert py_offsets == expected, (
        f"Python fallback returned {py_offsets}, expected {expected}"
    )


@pytest.mark.skipif(not HAS_EACOMPUTE, reason="eacompute source not available")
def test_simd_and_python_scrapers_agree():
    """The SIMD-backed scraper and the pure-Python fallback must return
    byte-identical results for the same input (modulo the lib being None)."""
    from indexer import _find_arrow_offsets, _load_scan_rust_lib

    lib = _load_scan_rust_lib()
    if lib is None:
        pytest.skip("scan_rust.ea kernel not built")

    # Test on a real eacompute intrinsics source file.
    rs_path = os.path.join(
        EACOMPUTE_DIR, "src", "typeck", "intrinsics_dotprod.rs"
    )
    if not os.path.isfile(rs_path):
        pytest.skip("intrinsics_dotprod.rs not present")
    with open(rs_path, "rb") as f:
        src_bytes = f.read()

    simd_offsets = _find_arrow_offsets(src_bytes, lib)
    py_offsets = _find_arrow_offsets(src_bytes, None)
    assert simd_offsets == py_offsets, (
        f"SIMD/Python disagreement:\n"
        f"  SIMD ({len(simd_offsets)}): {simd_offsets[:10]}...\n"
        f"  Py   ({len(py_offsets)}):   {py_offsets[:10]}..."
    )
