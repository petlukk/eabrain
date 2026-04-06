"""
indexer.py — Binary index format for eabrain.

Index layout:
  1. Header            (64 bytes)
  2. Kernel records    (kernel_count * 256 bytes)
  3. Ref records       (ref_count * 512 bytes)
  4. Session entries   (variable length)
  5. Embeddings        (kernel_count * 256 * 4 bytes, 64-byte aligned)
"""

import struct
import time
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAGIC = b"EABRAIN\0"
VERSION = 1
HEADER_SIZE = 64
KERNEL_RECORD_SIZE = 256
REF_RECORD_SIZE = 512

# Struct formats (little-endian)
# Header: magic(8s) + version(I) + kernel_count(I) + ref_count(I) +
#         session_count(I) + offsets(4Q = 4*8 = 32 bytes) + timestamp(Q)
# Total: 8+4+4+4+4+32+8 = 64 bytes
_HEADER_FMT = "<8sIIII4QQ"
assert struct.calcsize(_HEADER_FMT) == HEADER_SIZE

# Kernel record: path(128s) + func_name(64s) + arch(8s) +
#   simd_width(I) + line_start(I) + line_count(I) +
#   intrinsics_mask(Q) + flags(I) + padding(32x)
# Total: 128+64+8+4+4+4+8+4+32 = 256 bytes
_KERNEL_FMT = "<128s64s8sIIIQI32x"
assert struct.calcsize(_KERNEL_FMT) == KERNEL_RECORD_SIZE

# Ref record: name(64s) + category(32s) + signature(128s) +
#   description(256s) + flags(I) + padding(28x)
# Total: 64+32+128+256+4+28 = 512 bytes
_REF_FMT = "<64s32s128s256sI28x"
assert struct.calcsize(_REF_FMT) == REF_RECORD_SIZE

# Session entry header: timestamp(Q) + length(I)
_SESSION_HDR_FMT = "<QI"
_SESSION_HDR_SIZE = struct.calcsize(_SESSION_HDR_FMT)  # 12 bytes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad(s: str, n: int) -> bytes:
    """Encode string to UTF-8 and null-pad (or truncate) to exactly n bytes."""
    b = s.encode("utf-8")
    if len(b) >= n:
        b = b[:n - 1] + b"\x00"
    return b.ljust(n, b"\x00")


def _unpad(b: bytes) -> str:
    """Strip null bytes and decode from UTF-8."""
    return b.split(b"\x00", 1)[0].decode("utf-8")


def _align64(offset: int) -> int:
    """Round up offset to the next 64-byte boundary."""
    return (offset + 63) & ~63


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def write_header(
    kernel_count: int,
    ref_count: int,
    session_count: int,
    offsets: tuple = (0, 0, 0, 0),
    timestamp: int = None,
) -> bytes:
    """Pack and return a 64-byte header."""
    if timestamp is None:
        timestamp = int(time.time())
    return struct.pack(
        _HEADER_FMT,
        MAGIC,
        VERSION,
        kernel_count,
        ref_count,
        session_count,
        *offsets,
        timestamp,
    )


def read_header(buf: bytes) -> dict:
    """Unpack a 64-byte header buffer into a dict."""
    magic, version, kernel_count, ref_count, session_count, o0, o1, o2, o3, ts = \
        struct.unpack(_HEADER_FMT, buf[:HEADER_SIZE])
    if magic != MAGIC:
        raise ValueError(f"Bad magic: {magic!r}")
    return {
        "magic": magic,
        "version": version,
        "kernel_count": kernel_count,
        "ref_count": ref_count,
        "session_count": session_count,
        "offsets": (o0, o1, o2, o3),
        "timestamp": ts,
    }


# ---------------------------------------------------------------------------
# Kernel records
# ---------------------------------------------------------------------------

def pack_kernel_record(rec: dict) -> bytes:
    """Pack a kernel record dict into KERNEL_RECORD_SIZE bytes."""
    return struct.pack(
        _KERNEL_FMT,
        _pad(rec.get("path", ""), 128),
        _pad(rec.get("func_name", ""), 64),
        _pad(rec.get("arch", ""), 8),
        rec.get("simd_width", 0),
        rec.get("line_start", 0),
        rec.get("line_count", 0),
        rec.get("intrinsics_mask", 0),
        rec.get("flags", 0),
    )


def unpack_kernel_record(buf: bytes) -> dict:
    """Unpack KERNEL_RECORD_SIZE bytes into a kernel record dict."""
    path_b, func_b, arch_b, simd_width, line_start, line_count, intrinsics_mask, flags = \
        struct.unpack(_KERNEL_FMT, buf[:KERNEL_RECORD_SIZE])
    return {
        "path": _unpad(path_b),
        "func_name": _unpad(func_b),
        "arch": _unpad(arch_b),
        "simd_width": simd_width,
        "line_start": line_start,
        "line_count": line_count,
        "intrinsics_mask": intrinsics_mask,
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# Ref records
# ---------------------------------------------------------------------------

def pack_ref_record(rec: dict) -> bytes:
    """Pack a ref record dict into REF_RECORD_SIZE bytes."""
    return struct.pack(
        _REF_FMT,
        _pad(rec.get("name", ""), 64),
        _pad(rec.get("category", ""), 32),
        _pad(rec.get("signature", ""), 128),
        _pad(rec.get("description", ""), 256),
        rec.get("flags", 0),
    )


def unpack_ref_record(buf: bytes) -> dict:
    """Unpack REF_RECORD_SIZE bytes into a ref record dict."""
    name_b, cat_b, sig_b, desc_b, flags = \
        struct.unpack(_REF_FMT, buf[:REF_RECORD_SIZE])
    return {
        "name": _unpad(name_b),
        "category": _unpad(cat_b),
        "signature": _unpad(sig_b),
        "description": _unpad(desc_b),
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# Session entries (variable length)
# ---------------------------------------------------------------------------

def _pack_session(session: dict, ts: int = None) -> bytes:
    """Pack a session entry: timestamp(u64) + length(u32) + text(bytes)."""
    if ts is None:
        ts = int(time.time())
    text_bytes = session.get("text", "").encode("utf-8")[:512]
    hdr = struct.pack(_SESSION_HDR_FMT, ts, len(text_bytes))
    return hdr + text_bytes


def _unpack_session(buf: bytes, offset: int) -> tuple:
    """Unpack one session entry from buf at offset. Returns (dict, new_offset)."""
    ts, length = struct.unpack_from(_SESSION_HDR_FMT, buf, offset)
    offset += _SESSION_HDR_SIZE
    text = buf[offset:offset + length].decode("utf-8")
    offset += length
    return {"timestamp": ts, "text": text}, offset


# ---------------------------------------------------------------------------
# Full index read/write
# ---------------------------------------------------------------------------

def write_index(
    path: str,
    kernels: list,
    refs: list,
    sessions: list,
    embeddings: np.ndarray,
) -> None:
    """Write a complete index.bin file.

    Layout:
      [0]    Header (64 bytes)
      [off0] Kernel records
      [off1] Ref records
      [off2] Session entries
      [off3] Embeddings (64-byte aligned)
    """
    kernel_count = len(kernels)
    ref_count = len(refs)
    session_count = len(sessions)

    off0 = HEADER_SIZE  # kernel records start
    off1 = off0 + kernel_count * KERNEL_RECORD_SIZE  # ref records start
    off2 = off1 + ref_count * REF_RECORD_SIZE  # sessions start

    # Pre-pack sessions to know their total size
    packed_sessions = [_pack_session(s) for s in sessions]
    sessions_size = sum(len(p) for p in packed_sessions)

    off3 = _align64(off2 + sessions_size)  # embeddings start (64-byte aligned)

    offsets = (off0, off1, off2, off3)

    with open(path, "wb") as f:
        # 1. Header
        f.write(write_header(kernel_count, ref_count, session_count, offsets))

        # 2. Kernel records
        for k in kernels:
            f.write(pack_kernel_record(k))

        # 3. Ref records
        for r in refs:
            f.write(pack_ref_record(r))

        # 4. Session entries
        for ps in packed_sessions:
            f.write(ps)

        # Pad to 64-byte alignment before embeddings
        current = off2 + sessions_size
        pad_bytes = off3 - current
        if pad_bytes > 0:
            f.write(b"\x00" * pad_bytes)

        # 5. Embeddings
        emb = np.asarray(embeddings, dtype=np.float32)
        f.write(emb.tobytes())


def read_index(path: str) -> dict:
    """Read a complete index.bin file. Returns dict with kernels, refs, sessions, embeddings."""
    with open(path, "rb") as f:
        data = f.read()

    hdr = read_header(data)
    kernel_count = hdr["kernel_count"]
    ref_count = hdr["ref_count"]
    session_count = hdr["session_count"]
    off0, off1, off2, off3 = hdr["offsets"]

    # Kernel records
    kernels = []
    for i in range(kernel_count):
        start = off0 + i * KERNEL_RECORD_SIZE
        kernels.append(unpack_kernel_record(data[start:start + KERNEL_RECORD_SIZE]))

    # Ref records
    refs = []
    for i in range(ref_count):
        start = off1 + i * REF_RECORD_SIZE
        refs.append(unpack_ref_record(data[start:start + REF_RECORD_SIZE]))

    # Session entries
    sessions = []
    offset = off2
    for _ in range(session_count):
        session, offset = _unpack_session(data, offset)
        sessions.append(session)

    # Embeddings
    emb_bytes = len(data) - off3
    emb_count = emb_bytes // (256 * 4)
    embeddings = np.frombuffer(data[off3:off3 + emb_count * 256 * 4], dtype=np.float32)
    if emb_count > 0:
        embeddings = embeddings.reshape(emb_count, 256)

    return {
        "header": hdr,
        "kernels": kernels,
        "refs": refs,
        "sessions": sessions,
        "embeddings": embeddings,
    }
