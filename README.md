# eabrain

A context engine for AI coding assistants. It indexes all your [Ea](https://github.com/petlukk/eacompute) kernel files and gives your LLM instant access to function lookups, language reference, and session memory.

**The problem:** Every time you start a new Claude Code (or Cursor/Copilot) session, it forgets everything about your codebase. It re-scans files, re-discovers how the Ea compiler works, and repeats the same mistakes.

**The fix:** `eabrain` builds a searchable index of your code. Your AI runs `eabrain search` instead of grepping, `eabrain ref` instead of guessing, and `eabrain recall` to remember what you worked on last time.

Built with Ea SIMD kernels for fast search. Python for everything else.

---

## Quick Start

### 1. Clone it

```bash
git clone git@github.com:petlukk/eabrain.git
cd eabrain
```

### 2. Build the SIMD kernels

You need the [Ea compiler](https://github.com/petlukk/eacompute) built first. Then:

```bash
# Point to your ea binary (default: /root/dev/eacompute/target/release/ea)
export EA=/path/to/ea

./build_kernels.sh
```

This compiles `fuzzy.ea` and `scan.ea` into shared libraries in `lib/`.

### 3. Install

```bash
pip install -e .
```

Now `eabrain` is available as a CLI command.

### 4. Tell it where your projects are

Create `~/.eabrain/config.json`:

```json
{
    "projects": [
        "/home/you/dev/my-ea-project",
        "/home/you/dev/another-project"
    ],
    "index_path": "~/.eabrain/index.bin",
    "max_source_lines": 50,
    "max_session_entries": 100
}
```

Each path should be the root of a project that contains `.ea` files somewhere inside it.

### 5. Build the index

```bash
eabrain index
```

This scans all `.ea` files in your projects, extracts every exported function, detects SIMD types and intrinsics used, and writes a binary index to `~/.eabrain/index.bin`.

### 6. Done

```bash
eabrain status
```

---

## Commands

### `eabrain index`

Scans all `.ea` files across your configured projects and builds the binary index.

```bash
eabrain index

# Override project list for a one-off scan:
eabrain index --projects /path/to/project1,/path/to/project2
```

Run this again after you've added or changed `.ea` files.

### `eabrain search <query>`

Find kernels by function name or file path.

```bash
# Exact/substring match on function names and paths
eabrain search "batch_cosine"

# Filter by architecture
eabrain search "batch" --arch arm
eabrain search "batch" --arch x86

# Fuzzy search (byte-histogram cosine similarity via SIMD)
eabrain search --fuzzy "cosine similarity"
```

Output includes the full source code for small kernels (under 50 lines), so your AI doesn't need a second round-trip to read the file.

Example output:

```
# eabrain search: batch_cosine

11 results (indexed 2m ago, 503 kernels)

1. eaclaw/kernels/search.ea:73
   export func batch_cosine(...)
   arch: x86_64  simd: f32x8  lines: 36
   [source: 36 lines]
   export func batch_cosine(
       query: *f32,
       query_norm: f32,
       ...
```

### `eabrain ref <name>`

Look up Ea language reference: intrinsics, SIMD types, operators, compiler flags, and known gotchas.

```bash
eabrain ref "reduce_add"
eabrain ref "f32x8"
eabrain ref "target-triple"
eabrain ref "to_f32"
```

Example output:

```
# eabrain ref: reduce_add

Name: reduce_add
Category: intrinsic
Signature: reduce_add(v: f32xN) -> f32
Description: Horizontal sum of all lanes in a SIMD vector
```

The reference includes ~50 entries covering:

| Category | Examples |
|---|---|
| Intrinsics | `reduce_add`, `fma`, `splat`, `load`, `store`, `select`, `sqrt` |
| Types | `f32x4`, `f32x8`, `f32x16`, `i32x4`, `u8x16` |
| Operators | `.&`, `.\|`, `.^`, `.<<`, `.>>`, `.==`, `.>` |
| Compiler flags | `--lib`, `--target-triple`, `--avx512`, `ea bind --python` |
| Gotchas | float read pattern, no scalar bitwise, compiler path |

### `eabrain remember <note>`

Save a note to the session journal. These persist across sessions in the binary index.

```bash
eabrain remember "fixed the sign flip in layer 0 dequant kernel"
eabrain remember "ported batch_cosine to AVX-512, lane ordering was wrong"
```

### `eabrain recall`

Show saved session notes.

```bash
# Show all notes
eabrain recall

# Show last 5
eabrain recall --last 5
```

### `eabrain status`

Quick orientation. Run this at the start of every session.

```bash
eabrain status
```

```
eabrain v0.1.0
Index: ~/.eabrain/index.bin
Last indexed: 2h ago
Kernels: 503
Refs: 51
Projects: 6
Session notes: 3
```

### `eabrain init`

Generate a CLAUDE.md snippet (or append to an existing one) that tells your AI assistant how to use eabrain.

```bash
# Add instructions to a project's CLAUDE.md
eabrain init --project-dir /path/to/my-project

# Current directory
eabrain init
```

This appends something like:

```markdown
## eabrain
Use `eabrain search <query>` to find Ea kernels across all projects.
Use `eabrain ref <name>` to look up Ea language reference.
Use `eabrain remember <note>` to save context between sessions.
```

---

## How It Works

### Architecture

```
eabrain.py          CLI (argparse, ~300 lines)
indexer.py          Binary index format + builder (~480 lines)
kernels/fuzzy.ea    SIMD cosine search (batch_cosine, normalize, byte_histogram)
kernels/scan.ea     SIMD source parser (find exports, detect types/intrinsics)
reference/          Pre-baked Ea language reference (JSON)
lib/                Compiled .so files (not in git, built by build_kernels.sh)
~/.eabrain/         Config + binary index (shared across all projects)
```

### The Binary Index (`index.bin`)

A flat binary file with four sections:

1. **Header** (64 bytes) -- magic, version, counts, section offsets
2. **Kernel records** (256 bytes each, fixed-width) -- path, function name, arch, SIMD width, line numbers, intrinsics bitmask
3. **Reference entries** (512 bytes each, fixed-width) -- name, category, signature, description
4. **Session journal** (variable-length) -- timestamped text notes
5. **Embeddings** (256 x f32 per kernel, 64-byte aligned) -- byte-histogram vectors for fuzzy search

Fixed-width records let the SIMD kernels stride through at known offsets without parsing.

### The Ea Kernels

**`fuzzy.ea`** -- Adapted from [eaclaw](https://github.com/petlukk/eaclaw)'s battle-tested vector search:
- `byte_histogram_embed` -- count byte frequencies into a 256-dim vector
- `normalize_vectors` -- in-place L2 normalization (f32x8 on x86, f32x4 on ARM)
- `batch_cosine` -- cosine similarity of query against all stored embeddings

**`scan.ea`** -- SIMD source code parser:
- `count_exports` / `find_export_offsets` -- find `export func` declarations using u8x16 byte scanning
- `detect_simd_types` -- scan for `f32x4`, `f32x8`, `f32x16`, `i32x4`, `u8x16` usage
- `detect_intrinsics` -- scan for `reduce_add`, `fma`, `splat`, `load`, `store`, `select`, `sqrt` usage

All kernels are pure SIMD with scalar tails for alignment. No scalar-only code paths.

---

## For AI Assistants

If you're Claude Code (or another AI) reading this, here's how to use eabrain:

1. **Start of session:** Run `eabrain status` to orient yourself
2. **Looking for a kernel:** Run `eabrain search "name"` before grepping manually
3. **Ea syntax question:** Run `eabrain ref "intrinsic_name"` instead of guessing
4. **Cross-compile question:** Run `eabrain ref "target-triple"`
5. **End of session:** Run `eabrain remember "what we worked on"` to save context
6. **After editing .ea files:** Run `eabrain index` to refresh the index

---

## Configuration

### Config file: `~/.eabrain/config.json`

```json
{
    "projects": [
        "/root/dev/eacompute",
        "/root/dev/eakv",
        "/root/dev/eaclaw",
        "/root/dev/olorin",
        "/root/dev/Cougar",
        "/root/dev/eachacha"
    ],
    "index_path": "~/.eabrain/index.bin",
    "max_source_lines": 50,
    "max_session_entries": 100
}
```

| Field | Default | Description |
|---|---|---|
| `projects` | (built-in list) | Directories to scan for `.ea` files |
| `index_path` | `~/.eabrain/index.bin` | Where to write the binary index |
| `max_source_lines` | 50 | Include full source in search results for kernels shorter than this |
| `max_session_entries` | 100 | Max session notes before oldest are dropped |

### Environment variable

Set `EABRAIN_CONFIG` to point to an alternative config file:

```bash
EABRAIN_CONFIG=/tmp/test-config.json eabrain index
```

---

## Development

### Prerequisites

- Python 3.10+
- numpy
- [Ea compiler](https://github.com/petlukk/eacompute) (built from source)

### Build

```bash
./build_kernels.sh      # compile Ea kernels to .so
pip install -e ".[dev]" # install with dev deps (pytest)
```

### Test

```bash
python3 -m pytest tests/ -v
```

25 tests covering: kernel compilation and correctness, binary format roundtrips, index building against real project data, CLI integration.

### Project structure

```
eabrain/
  kernels/
    fuzzy.ea            SIMD cosine search kernel
    scan.ea             SIMD source parser kernel
  reference/
    ea_reference.json   Ea language reference (51 entries)
  tests/
    test_fuzzy_kernel.py
    test_scan_kernel.py
    test_index_format.py
    test_indexer.py
    test_reference.py
    test_cli.py
  lib/                  (built, not in git)
    libfuzzy.so
    libscan.so
  eabrain.py            CLI entry point
  indexer.py            Binary format + index builder
  build_kernels.sh      Kernel build script
  setup.py              Package config
  CLAUDE.md             Instructions for AI assistants
```

---

## License

MIT
