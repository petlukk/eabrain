import os
import shutil
import subprocess

import pytest

EABRAIN = os.path.join(os.path.dirname(__file__), "..", "eabrain.py")


def _resolve_autoresearch_dir() -> str | None:
    env = os.environ.get("AUTORESEARCH_DIR")
    if env and os.path.isdir(env):
        return env
    ec_env = os.environ.get("EACOMPUTE_DIR")
    if ec_env:
        candidate = os.path.join(ec_env, "autoresearch", "kernels")
        if os.path.isdir(candidate):
            return candidate
    ea = shutil.which("ea") or os.environ.get("EA")
    if ea and os.path.isfile(ea):
        parent = os.path.dirname(os.path.dirname(os.path.dirname(ea)))
        candidate = os.path.join(parent, "autoresearch", "kernels")
        if os.path.isdir(candidate):
            return candidate
    return None


def _has_benchmark_data(ar_dir: str | None) -> bool:
    if not ar_dir:
        return False
    for name in os.listdir(ar_dir):
        if os.path.isfile(os.path.join(ar_dir, name, "history.json")):
            return True
    return False


pytestmark = pytest.mark.skipif(
    not _has_benchmark_data(_resolve_autoresearch_dir()),
    reason="autoresearch benchmark data (history.json) not present on this machine",
)


def run(args):
    return subprocess.run(
        ["python3", EABRAIN] + args,
        capture_output=True, text=True,
    )

def test_patterns_list():
    r = run(["patterns"])
    assert r.returncode == 0
    assert "matmul" in r.stdout
    assert "batch_cosine" in r.stdout

def test_patterns_what_works():
    r = run(["patterns", "--what-works"])
    assert r.returncode == 0
    assert "Proven optimization patterns" in r.stdout
    assert "Common winning patterns" in r.stdout

def test_patterns_specific_kernel():
    r = run(["patterns", "matmul"])
    assert r.returncode == 0
    assert "Strategy Space" in r.stdout
    assert "History" in r.stdout
    assert "Best Kernel" in r.stdout
    assert "matmul_f32" in r.stdout

def test_patterns_substring_match():
    r = run(["patterns", "dot"])
    assert r.returncode == 0
    # Should match dot_product, dot_u8i8, batch_dot
    assert "dot" in r.stdout.lower()

def test_patterns_no_match():
    r = run(["patterns", "nonexistent_kernel_xyz"])
    assert r.returncode == 0
    assert "No autoresearch benchmark" in r.stdout
