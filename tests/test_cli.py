import json
import os
import subprocess
import pytest

EABRAIN = os.path.join(os.path.dirname(__file__), "..", "eabrain.py")

MINIMAL_KERNEL = """// test fixture
export func batch_cosine(query: *f32, vecs: *f32, dim: i32, out: *mut f32) {
    let acc: f32x8 = splat(0.0)
    out[0] = reduce_add(acc)
}
"""


def run(args, config_path=None):
    cmd = ["python3", EABRAIN] + args
    env = os.environ.copy()
    if config_path:
        env["EABRAIN_CONFIG"] = config_path
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


@pytest.fixture
def config_file(tmp_path):
    project = tmp_path / "fake_project"
    (project / "kernels").mkdir(parents=True)
    (project / "kernels" / "sample.ea").write_text(MINIMAL_KERNEL)

    config = {
        "projects": [str(project)],
        "index_path": str(tmp_path / "index.bin"),
        "max_source_lines": 50,
        "max_session_entries": 100,
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(config))
    return str(p)


def test_index_and_status(config_file):
    r = run(["index"], config_file)
    assert r.returncode == 0
    r = run(["status"], config_file)
    assert r.returncode == 0
    assert "indexed" in r.stdout.lower()


def test_search(config_file):
    run(["index"], config_file)
    r = run(["search", "batch_cosine"], config_file)
    assert r.returncode == 0
    assert "batch_cosine" in r.stdout


def test_ref(config_file):
    r = run(["ref", "reduce_add"], config_file)
    assert r.returncode == 0
    assert "reduce_add" in r.stdout


def test_remember_recall(config_file):
    run(["index"], config_file)
    r = run(["remember", "fixed sign flip"], config_file)
    assert r.returncode == 0
    r = run(["recall"], config_file)
    assert r.returncode == 0
    assert "sign flip" in r.stdout
