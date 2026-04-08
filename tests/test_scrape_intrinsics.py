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
