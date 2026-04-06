import os
import struct
import tempfile
import numpy as np
from indexer import (
    MAGIC, VERSION, HEADER_SIZE, KERNEL_RECORD_SIZE, REF_RECORD_SIZE,
    write_header, read_header,
    pack_kernel_record, unpack_kernel_record,
    pack_ref_record, unpack_ref_record,
    write_index, read_index,
)

def test_header_roundtrip():
    buf = write_header(kernel_count=10, ref_count=5, session_count=3)
    assert len(buf) == HEADER_SIZE
    assert buf[:8] == MAGIC
    hdr = read_header(buf)
    assert hdr["kernel_count"] == 10
    assert hdr["ref_count"] == 5
    assert hdr["session_count"] == 3
    assert hdr["version"] == VERSION

def test_kernel_record_roundtrip():
    rec = {
        "path": "eaclaw/kernels/search.ea",
        "func_name": "batch_cosine",
        "arch": "x86_64",
        "simd_width": 8,
        "line_start": 87,
        "line_count": 34,
        "intrinsics_mask": 0b11,
        "flags": 1,
    }
    packed = pack_kernel_record(rec)
    assert len(packed) == KERNEL_RECORD_SIZE
    unpacked = unpack_kernel_record(packed)
    assert unpacked["path"] == rec["path"]
    assert unpacked["func_name"] == rec["func_name"]
    assert unpacked["arch"] == rec["arch"]
    assert unpacked["simd_width"] == rec["simd_width"]
    assert unpacked["line_start"] == rec["line_start"]
    assert unpacked["line_count"] == rec["line_count"]
    assert unpacked["intrinsics_mask"] == rec["intrinsics_mask"]

def test_ref_record_roundtrip():
    rec = {
        "name": "reduce_add",
        "category": "intrinsic",
        "signature": "reduce_add(v: f32xN) -> f32",
        "description": "Horizontal sum of all lanes.",
    }
    packed = pack_ref_record(rec)
    assert len(packed) == REF_RECORD_SIZE
    unpacked = unpack_ref_record(packed)
    assert unpacked["name"] == rec["name"]
    assert unpacked["category"] == rec["category"]

def test_full_index_roundtrip():
    kernels = [
        {
            "path": "eaclaw/kernels/search.ea",
            "func_name": "batch_cosine",
            "arch": "x86_64",
            "simd_width": 8,
            "line_start": 87,
            "line_count": 34,
            "intrinsics_mask": 0b11,
            "flags": 1,
        },
        {
            "path": "olorin/kernels/search.ea",
            "func_name": "batch_dot",
            "arch": "aarch64",
            "simd_width": 4,
            "line_start": 7,
            "line_count": 60,
            "intrinsics_mask": 0b11,
            "flags": 1,
        },
    ]
    refs = [
        {
            "name": "reduce_add",
            "category": "intrinsic",
            "signature": "reduce_add(v: f32xN) -> f32",
            "description": "Horizontal sum of all lanes.",
        },
    ]
    sessions = [
        {"text": "Fixed sign flip in layer 0"},
    ]
    embeddings = np.random.randn(2, 256).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = f.name
    try:
        write_index(path, kernels, refs, sessions, embeddings)
        loaded = read_index(path)
        assert len(loaded["kernels"]) == 2
        assert len(loaded["refs"]) == 1
        assert len(loaded["sessions"]) == 1
        assert loaded["kernels"][0]["func_name"] == "batch_cosine"
        assert loaded["kernels"][1]["func_name"] == "batch_dot"
        assert loaded["refs"][0]["name"] == "reduce_add"
        assert loaded["sessions"][0]["text"] == "Fixed sign flip in layer 0"
        assert loaded["embeddings"].shape == (2, 256)
    finally:
        os.unlink(path)
