# eabrain

Eä-driven context engine + persistent memory for Claude Code. SIMD-accelerated
search over Eä kernels, language reference, and session memory.

## Build
1. `./build_kernels.sh` — compile Eä kernels to .so + Python bindings
2. `pip install -e .` — install CLI

## Test
`python3 -m pytest tests/ -v`

## Architecture
- `kernels/*.ea` — pure SIMD kernels (no scalar ops)
- `eabrain.py` — CLI entry point (argparse + ctypes)
- `indexer.py` — kernel index builder (writes `~/.eabrain/index.bin`)
- `memory.py` — SQLite observation/session storage (`~/.eabrain/memory.db`)
- `inject.py` — preamble loading + context injection
- `server.py` — web viewer (stdlib `http.server`)
- `sync.py` — cross-machine memory.db export/import with dedup merge
- `reference/ea_reference.json` — pre-baked Eä language reference
- `web/viewer.html` — single-file Catppuccin Mocha UI
- `lib/` — compiled .so files + generated Python bindings (not in git)

## Ea compiler
Resolved at runtime in this order:
1. `$EA` env var (must point at the `ea` binary)
2. `ea` on `PATH`

`$EACOMPUTE_DIR` overrides the eacompute source location used by the
intrinsic scraper; otherwise it's derived from the compiler's install path
(`.../eacompute/target/release/ea` → `.../eacompute`).

## Memory commands
- `eabrain inject` — emit preamble + recent context (call at session start)
- `eabrain remember <note>` — store a quick observation
- `eabrain store <text> --type {decision|bug|architecture|pattern|error|note}`
- `eabrain store-summary <text>` — close the current session with a summary
- `eabrain recall [--last N]` — show recent observations
- `eabrain timeline [--project P] [--last N] [--since DATE]`
- `eabrain search <q> [--fuzzy] [--kernels-only|--memory-only]`
- `eabrain serve [--port 37777]` — web viewer
- `eabrain sync --export PATH | --import PATH`
- `eabrain migrate` — port v0.1 session notes from index.bin → memory.db
