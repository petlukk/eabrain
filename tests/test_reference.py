import json
import os

REF_PATH = os.path.join(os.path.dirname(__file__), "..", "reference", "ea_reference.json")

def test_reference_loads():
    with open(REF_PATH) as f:
        ref = json.load(f)
    assert "entries" in ref
    assert len(ref["entries"]) > 0

def test_reference_entry_fields():
    with open(REF_PATH) as f:
        ref = json.load(f)
    required = {"name", "category", "signature", "description"}
    for entry in ref["entries"]:
        assert required.issubset(entry.keys()), f"Missing fields in {entry.get('name', '?')}"

def test_reference_categories():
    with open(REF_PATH) as f:
        ref = json.load(f)
    valid = {"intrinsic", "type", "operator", "compiler_flag", "gotcha"}
    for entry in ref["entries"]:
        assert entry["category"] in valid, f"Bad category: {entry['category']}"

def test_reference_no_duplicates():
    with open(REF_PATH) as f:
        ref = json.load(f)
    names = [e["name"] for e in ref["entries"]]
    assert len(names) == len(set(names)), f"Duplicates: {[n for n in names if names.count(n) > 1]}"

def test_reference_known_entries():
    with open(REF_PATH) as f:
        ref = json.load(f)
    names = {e["name"] for e in ref["entries"]}
    assert "reduce_add" in names
    assert "fma" in names
    assert "f32x8" in names
    assert "--lib" in names
    assert "to_f32(ptr[i]) not to_f32(to_i32(ptr[i]))" in names
