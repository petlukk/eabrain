import os
import subprocess

EABRAIN = os.path.join(os.path.dirname(__file__), "..", "eabrain.py")

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
