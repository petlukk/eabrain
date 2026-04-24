"""safety.py — Injection + secret-leak scanning via fused Eä SIMD kernel.

Ported from Olorin's src/core/safety.rs. Shared kernel source
(kernels/fused_safety.ea) produces `libfused_safety.so`; the pattern tables
and verification logic are translated to Python here.

Scan runs automatically on every write to the memory DB (see memory.py), so
API keys, bearer tokens, PEM blocks, and prompt-injection attempts never
land in observations that later get replayed into Claude's context via
`eabrain inject`.
"""

import ctypes
import os
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List

import numpy as np

_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "lib", "libfused_safety.so")
_lib = ctypes.CDLL(_LIB_PATH)
_lib.scan_safety_fused.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
]
_lib.scan_safety_fused.restype = None


def _call_kernel(text: np.ndarray, inject_masks: np.ndarray,
                 leak_masks: np.ndarray, n_out: np.ndarray) -> None:
    _lib.scan_safety_fused(
        text.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(text.size),
        inject_masks.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        leak_masks.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        n_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )


class WarningKind(Enum):
    INJECTION = "injection"
    SECRET_LEAK = "secret_leak"


@dataclass
class SafetyWarning:
    kind: WarningKind
    pattern: str
    position: int

    def __str__(self) -> str:
        return f"{self.kind.value}: {self.pattern} @ offset {self.position}"


class SafetyScanError(Exception):
    """Raised when content fails the injection/leak scan on write."""

    def __init__(self, warnings: List[SafetyWarning]):
        self.warnings = warnings
        body = "\n".join(f"  - {w}" for w in warnings)
        super().__init__(
            f"Content rejected by safety scan:\n{body}\n"
            f"Redact the offending token and retry. The memory DB syncs to a "
            f"git remote, so secrets persisted here are effectively public."
        )


# ── Injection patterns (case-insensitive) ─────────────────────────────────

_INJECTION_PATTERNS: List[tuple] = [
    (b"ignore previous",      "override previous instructions"),
    (b"ignore all previous",  "override ALL previous instructions"),
    (b"disregard",            "potential instruction override"),
    (b"forget everything",    "attempt to reset context"),
    (b"you are now",          "role change attempt"),
    (b"act as",               "role manipulation"),
    (b"pretend to be",        "role manipulation"),
    (b"system:",              "system message injection"),
    (b"assistant:",           "assistant response injection"),
    (b"user:",                "user message injection"),
    (b"<|",                   "special token injection"),
    (b"|>",                   "special token injection"),
    (b"[INST]",               "instruction token injection"),
    (b"[/INST]",              "instruction token injection"),
    (b"new instructions",     "new instruction attempt"),
    (b"updated instructions", "instruction update attempt"),
]


def _match_case_insensitive(text: bytes, pos: int, pattern: bytes) -> bool:
    if pos + len(pattern) > len(text):
        return False
    return text[pos:pos + len(pattern)].lower() == pattern.lower()


def _verify_injection_at(text: bytes, pos: int, out: List[SafetyWarning]) -> None:
    for pattern, desc in _INJECTION_PATTERNS:
        if _match_case_insensitive(text, pos, pattern):
            out.append(SafetyWarning(WarningKind.INJECTION, desc, pos))
            return


# ── Leak patterns (case-sensitive) ────────────────────────────────────────

def _is_alnum(b: int) -> bool:
    return (48 <= b <= 57) or (65 <= b <= 90) or (97 <= b <= 122)


def _valid_alnum_dash(tail: bytes) -> bool:
    return all(_is_alnum(b) or b in (0x2D, 0x5F) for b in tail)  # - _


def _valid_upper_alnum(tail: bytes) -> bool:
    return all((65 <= b <= 90) or (48 <= b <= 57) for b in tail)


def _valid_alnum_underscore(tail: bytes) -> bool:
    return all(_is_alnum(b) or b == 0x5F for b in tail)  # _


def _valid_alnum_dash_dot(tail: bytes) -> bool:
    return all(_is_alnum(b) or b in (0x2D, 0x5F, 0x2E) for b in tail)  # - _ .


def _always_true(_: bytes) -> bool:
    return True


_LEAK_PATTERNS = [
    (b"sk-ant-api",   20, "Anthropic API key",          _valid_alnum_dash),
    (b"sk-proj-",     20, "OpenAI API key (project)",   _valid_alnum_dash),
    (b"sk-",          20, "OpenAI API key",             _valid_alnum_dash),
    (b"AKIA",         20, "AWS access key",             _valid_upper_alnum),
    (b"ghp_",         40, "GitHub personal token",      _valid_alnum_underscore),
    (b"gho_",         40, "GitHub OAuth token",         _valid_alnum_underscore),
    (b"ghu_",         40, "GitHub user token",          _valid_alnum_underscore),
    (b"ghs_",         40, "GitHub server token",        _valid_alnum_underscore),
    (b"ghr_",         40, "GitHub refresh token",       _valid_alnum_underscore),
    (b"github_pat_",  40, "GitHub fine-grained PAT",    _valid_alnum_underscore),
    (b"xoxb-",        15, "Slack bot token",            _valid_alnum_dash),
    (b"xoxp-",        15, "Slack user token",           _valid_alnum_dash),
    (b"xoxa-",        15, "Slack app token",            _valid_alnum_dash),
    (b"SG.",          40, "SendGrid API key",           _valid_alnum_dash_dot),
    (b"sk_live_",     24, "Stripe live API key",        _valid_alnum_underscore),
    (b"sk_test_",     24, "Stripe test API key",        _valid_alnum_underscore),
    (b"sess_",        32, "Session token",              _valid_alnum_underscore),
    (b"-----BEGIN",   10, "PEM private key",            _always_true),
    (b"AIza",         39, "Google API key",             _valid_alnum_dash),
    (b"Bearer ",      27, "Bearer token",               _always_true),
]


def _verify_leak_at(text: bytes, pos: int, out: List[SafetyWarning]) -> None:
    for prefix, min_total_len, desc, validate in _LEAK_PATTERNS:
        if pos + len(prefix) > len(text):
            continue
        if text[pos:pos + len(prefix)] != prefix:
            continue
        remaining = text[pos + len(prefix):]
        tail_needed = max(0, min_total_len - len(prefix))
        if len(remaining) < tail_needed:
            continue
        tail_end = len(remaining)
        for i, b in enumerate(remaining):
            if chr(b).isspace():
                tail_end = i
                break
        tail = remaining[:tail_end]
        if len(tail) < tail_needed:
            continue
        if validate(tail):
            out.append(SafetyWarning(WarningKind.SECRET_LEAK, desc, pos))
            return


# ── SIMD bitmap iteration ─────────────────────────────────────────────────

def _for_each_candidate(masks: np.ndarray, n_blocks: int,
                        fn: Callable[[int], None]) -> None:
    count = min(n_blocks, len(masks))
    for block in range(count):
        m = int(masks[block]) & 0xFFFFFFFF
        while m:
            bit = (m & -m).bit_length() - 1
            fn(block * 16 + bit)
            m &= m - 1


# ── Public entry point ────────────────────────────────────────────────────

def scan(text: bytes) -> List[SafetyWarning]:
    """Scan bytes for prompt-injection and secret-leak patterns.

    Returns the list of warnings (empty → clean). The SIMD kernel emits
    16-byte-block bitmaps of positions where the first byte of some pattern
    prefix may appear; Python verifies each candidate against the full
    pattern table.
    """
    if not text:
        return []

    n_blocks = (len(text) + 15) // 16

    text_arr = np.frombuffer(text, dtype=np.uint8)
    inject_masks = np.zeros(n_blocks, dtype=np.int32)
    leak_masks = np.zeros(n_blocks, dtype=np.int32)
    n_out = np.zeros(1, dtype=np.int32)

    _call_kernel(text_arr, inject_masks, leak_masks, n_out)

    warnings: List[SafetyWarning] = []

    # The fused SIMD kernel loads 16-byte pairs (curr + next), so it can only
    # cover positions up to len-17. Anything past that is checked scalar.
    simd_covered = ((len(text) - 17) // 16 + 1) * 16 if len(text) >= 17 else 0

    _for_each_candidate(inject_masks, int(n_out[0]),
                        lambda pos: _verify_injection_at(text, pos, warnings))
    for pos in range(simd_covered, len(text)):
        _verify_injection_at(text, pos, warnings)

    seen: set = set()
    def _verify_leak_dedup(pos: int) -> None:
        if pos in seen:
            return
        seen.add(pos)
        _verify_leak_at(text, pos, warnings)

    _for_each_candidate(leak_masks, int(n_out[0]), _verify_leak_dedup)
    for pos in range(simd_covered, len(text)):
        _verify_leak_dedup(pos)

    return warnings


def check_or_raise(text: str) -> None:
    """Scan `text` and raise SafetyScanError if it contains a secret leak.

    Injection warnings are NOT blocked — memory notes can legitimately contain
    phrases like "pretend to be a senior reviewer" or discussion of prompt
    injection mitigations. Callers who need injection blocking should consume
    `scan()` directly and filter on `WarningKind.INJECTION`.
    """
    leaks = [w for w in scan(text.encode("utf-8"))
             if w.kind == WarningKind.SECRET_LEAK]
    if leaks:
        raise SafetyScanError(leaks)
