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


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

import ctypes
import glob
import json
import os
import re


def _load_lib(name: str) -> ctypes.CDLL:
    """Load a .so from the lib/ directory relative to this file."""
    lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
    return ctypes.CDLL(os.path.join(lib_dir, name))


def _scan_ea_file(src_bytes: bytes, scan_lib) -> list:
    """Scan a .ea file and return list of kernel dicts."""
    src = np.frombuffer(src_bytes, dtype=np.uint8)
    n = len(src_bytes)

    # Find export offsets
    offsets = np.zeros(256, dtype=np.int32)
    count = np.zeros(1, dtype=np.int32)
    scan_lib.find_export_offsets(
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(n),
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        count.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )

    # Detect SIMD types
    type_mask = np.zeros(1, dtype=np.int32)
    scan_lib.detect_simd_types(
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(n),
        type_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )

    # Detect intrinsics
    intr_mask = np.zeros(1, dtype=np.int32)
    scan_lib.detect_intrinsics(
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(n),
        intr_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )

    tm = int(type_mask[0])
    if tm & 0b100:
        simd_width = 16
    elif tm & 0b010:
        simd_width = 8
    elif tm & 0b001:
        simd_width = 4
    else:
        simd_width = 0

    # Detect arch
    text_lower = src_bytes.lower()
    if b"aarch64" in text_lower or b"cfg(aarch64)" in text_lower:
        arch = "aarch64"
    else:
        arch = "x86_64"

    lines = src_bytes.split(b"\n")

    kernels = []
    num_exports = int(count[0])
    for i in range(num_exports):
        off = int(offsets[i])
        # Skip "export func " (12 bytes)
        name_start = off + 12
        name_end = src_bytes.find(b"(", name_start)
        if name_end < 0:
            continue
        func_name = src_bytes[name_start:name_end].decode("utf-8", errors="replace").strip()

        # Compute line number (1-based)
        line_start = src_bytes[:off].count(b"\n") + 1

        # Count lines until closing brace at column 0
        body_start = src_bytes.find(b"{", off)
        if body_start < 0:
            line_count = 1
        else:
            depth = 0
            pos = body_start
            while pos < n:
                ch = src_bytes[pos:pos+1]
                if ch == b"{":
                    depth += 1
                elif ch == b"}":
                    depth -= 1
                    if depth == 0:
                        break
                pos += 1
            line_count = src_bytes[off:pos+1].count(b"\n") + 1

        kernels.append({
            "func_name": func_name,
            "line_start": line_start,
            "line_count": line_count,
            "arch": arch,
            "simd_width": simd_width,
            "intrinsics_mask": int(intr_mask[0]),
            "flags": 0,
        })

    return kernels


def _byte_histogram(text_bytes: bytes) -> np.ndarray:
    """Compute 256-dim byte histogram and L2-normalize."""
    hist = np.zeros(256, dtype=np.float32)
    for b in text_bytes:
        hist[b] += 1.0
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm
    return hist


# Matches a single signature within a doc comment. Doc-comment lines may
# contain either one signature:
#   /// maddubs_i16(u8x64, i8x64) -> i16x32  (AVX-512BW vpmaddubsw)
# or several comma-separated variants on one line:
#   /// cvt_f16_f32(i16x4) -> f32x4, cvt_f16_f32(i16x8) -> f32x8
#
# The SIMD kernel in scan_rust.ea locates every "->" byte pair in the
# source; this regex then pulls name/args/ret from the context around
# each hit. The regex runs on a tiny line-sized slice, not the whole file.
_SIG_RE = re.compile(r"(\w+)\(([^)]*)\)\s*->\s*([A-Za-z_][\w]*)")
_TAG_RE = re.compile(r"\(([^()]*)\)")

# Lane-pattern (e.g. "8x16", "32x4") marks a paren as an argument list,
# not a descriptive tag. Same for ": Type" forms (`acc: i32x4`).
_ARGLIST_RE = re.compile(r"\d+x\d+|:\s*\w")

# Matches string literals used as match arms in the intrinsics dispatch:
#   "vdot_i32" => Some(...)
#   "reduce_add" | "reduce_max" | "reduce_min" => Some(...)
_DISPATCH_ARM_RE = re.compile(r'"([a-z][\w]*)"\s*(?:\||=>|if)')


def _load_scan_rust_lib():
    """Try to load libscan_rust.so. Returns None if unavailable (pre-first
    kernel-build state), in which case the scraper falls back to a pure-Python
    byte scan."""
    try:
        lib = _load_lib("libscan_rust.so")
        lib.find_arrow_pairs.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        lib.find_arrow_pairs.restype = None
        return lib
    except OSError:
        return None


def _find_arrow_offsets(src_bytes: bytes, lib) -> list:
    """Return byte offsets of every "->" occurrence in src_bytes.

    Uses the SIMD Eä kernel when available, falls back to str.find() when
    the shared library hasn't been built yet.
    """
    if lib is None:
        # Pure-Python fallback — straightforward byte search.
        offsets = []
        start = 0
        while True:
            idx = src_bytes.find(b"->", start)
            if idx < 0:
                break
            offsets.append(idx)
            start = idx + 1
        return offsets

    src = np.frombuffer(src_bytes, dtype=np.uint8)
    n = len(src_bytes)
    # Generous upper bound: a typical intrinsics_*.rs has tens of arrows,
    # but the kernel caps at out_max regardless.
    out_max = max(1024, n // 32)
    offsets = np.zeros(out_max, dtype=np.int32)
    count = np.zeros(1, dtype=np.int32)
    lib.find_arrow_pairs(
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(n),
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(out_max),
        count.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    return offsets[: int(count[0])].tolist()


def _line_bounds(src_bytes: bytes, pos: int) -> tuple:
    """Return (start, end) byte offsets of the line containing `pos`."""
    start = src_bytes.rfind(b"\n", 0, pos) + 1  # -1 + 1 = 0 if not found
    end = src_bytes.find(b"\n", pos)
    if end < 0:
        end = len(src_bytes)
    return start, end


def _is_doc_comment_line(src_bytes: bytes, line_start: int, arrow_pos: int) -> bool:
    """True if the byte range [line_start, arrow_pos) begins (after leading
    whitespace) with `///`."""
    i = line_start
    while i < arrow_pos and src_bytes[i : i + 1] in (b" ", b"\t"):
        i += 1
    return src_bytes[i : i + 3] == b"///"


def scrape_eacompute_intrinsics(project_dirs: list) -> list:
    """Scrape intrinsic signatures from eacompute typeck source.

    Walks each project_dir looking for `src/typeck/intrinsics_*.rs` files
    (the Eä compiler's per-family intrinsic checkers). Doc comments on the
    `check_*` functions follow a consistent pattern:

        /// name(args) -> return_type  (optional descriptive tag)

    Returns a list of ref-shaped dicts, one per unique intrinsic name, with
    all width variants collapsed into a single semicolon-joined signature
    (truncated to fit the 128-byte binary record).
    """
    found = {}  # name -> {"signatures": [...], "tags": [...]}
    dispatch_names = set()  # names from the intrinsics dispatch match
    scan_lib = _load_scan_rust_lib()

    for project_dir in project_dirs:
        project_dir = os.path.expanduser(project_dir)
        typeck_dir = os.path.join(project_dir, "src", "typeck")
        if not os.path.isdir(typeck_dir):
            continue

        # Pass 1: scan intrinsics_*.rs for doc-comment signatures.
        #
        # The SIMD Eä kernel `find_arrow_pairs` locates every "->" byte pair
        # in the source. Only doc-comment lines (prefix "///") produce
        # ref entries — we check each hit's surrounding line.
        for rs_path in sorted(glob.glob(os.path.join(typeck_dir, "intrinsics_*.rs"))):
            try:
                with open(rs_path, "rb") as f:
                    src_bytes = f.read()
            except OSError:
                continue

            arrow_offsets = _find_arrow_offsets(src_bytes, scan_lib)
            seen_lines = set()  # avoid double-processing lines with multiple arrows
            for pos in arrow_offsets:
                line_start, line_end = _line_bounds(src_bytes, pos)
                if line_start in seen_lines:
                    continue
                seen_lines.add(line_start)
                if not _is_doc_comment_line(src_bytes, line_start, pos):
                    continue

                # Decode just this line. Strip the "///" prefix for regex use.
                line = src_bytes[line_start:line_end].decode("utf-8", "ignore")
                body = line.lstrip().lstrip("/").lstrip()
                matches = _SIG_RE.findall(body)
                if not matches:
                    continue
                tags = [
                    t.strip() for t in _TAG_RE.findall(body)
                    if "->" not in t and not _ARGLIST_RE.search(t)
                ]
                for name, args, ret in matches:
                    sig = f"{name}({args.strip()}) -> {ret}"
                    entry = found.setdefault(name, {"signatures": [], "tags": []})
                    if sig not in entry["signatures"]:
                        entry["signatures"].append(sig)
                    for tag in tags:
                        if tag and tag not in entry["tags"]:
                            entry["tags"].append(tag)

        # Pass 2: scan intrinsics.rs dispatch match for authoritative names.
        dispatch_path = os.path.join(typeck_dir, "intrinsics.rs")
        if os.path.isfile(dispatch_path):
            try:
                with open(dispatch_path, "r", encoding="utf-8") as f:
                    dispatch_src = f.read()
            except OSError:
                dispatch_src = ""
            for m in _DISPATCH_ARM_RE.finditer(dispatch_src):
                dispatch_names.add(m.group(1))

    # Anything in the dispatch match but not yet in `found` is a known
    # intrinsic with no doc-comment signature — still surface it so
    # `eabrain ref` doesn't say "No results found".
    for name in dispatch_names:
        if name not in found:
            found[name] = {
                "signatures": [f"{name}(...) -> ?"],
                "tags": ["no doc comment; see src/typeck/intrinsics_*.rs"],
            }

    results = []
    for name, data in sorted(found.items()):
        # Join width variants with "; " and truncate to fit the 127-byte limit
        # (leaving room for a terminator).
        sig_joined = "; ".join(data["signatures"])
        if len(sig_joined.encode("utf-8")) > 127:
            sig_joined = sig_joined.encode("utf-8")[:124].decode("utf-8", "ignore") + "..."
        # Description: the first few tags, or a fallback marker
        if data["tags"]:
            desc = "; ".join(data["tags"])
            if len(desc.encode("utf-8")) > 255:
                desc = desc.encode("utf-8")[:252].decode("utf-8", "ignore") + "..."
        else:
            desc = "auto-extracted from eacompute src/typeck (not yet curated)"
        results.append({
            "name": name,
            "category": "intrinsic",
            "signature": sig_joined,
            "description": desc,
        })
    return results


def build_index(project_dirs: list, ref_path: str, index_path: str) -> dict:
    """Build and write the binary index.

    Returns stats dict: kernel_count, ref_count, file_count.
    """
    scan_lib = _load_lib("libscan.so")

    # Load reference entries (static curated JSON)
    with open(ref_path, "r", encoding="utf-8") as f:
        ref_data = json.load(f)
    refs = ref_data.get("entries", [])

    # Merge in auto-scraped intrinsics from eacompute source. Static JSON wins
    # on name collisions — curated descriptions are authoritative. Scraped
    # entries only fill gaps for intrinsics not yet documented in the JSON.
    known_names = {r["name"] for r in refs}
    for scraped in scrape_eacompute_intrinsics(project_dirs):
        if scraped["name"] not in known_names:
            refs.append(scraped)
            known_names.add(scraped["name"])

    kernels = []
    embeddings = []
    file_count = 0

    for project_dir in project_dirs:
        project_dir = os.path.expanduser(project_dir)
        ea_files = glob.glob(os.path.join(project_dir, "**", "*.ea"), recursive=True)
        for ea_path in sorted(ea_files):
            try:
                with open(ea_path, "rb") as f:
                    src_bytes = f.read()
            except OSError:
                continue

            file_count += 1
            file_kernels = _scan_ea_file(src_bytes, scan_lib)
            emb = _byte_histogram(src_bytes)

            for k in file_kernels:
                k["path"] = ea_path
                kernels.append(k)
                embeddings.append(emb)

    if embeddings:
        emb_array = np.stack(embeddings, axis=0)
    else:
        emb_array = np.zeros((0, 256), dtype=np.float32)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)

    write_index(index_path, kernels, refs, [], emb_array)

    return {
        "kernel_count": len(kernels),
        "ref_count": len(refs),
        "file_count": file_count,
    }
