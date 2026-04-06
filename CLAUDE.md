# eabrain

Eä-driven context engine for Claude Code. SIMD-accelerated search over Eä kernels, language reference, and session memory.

## Build
1. `./build_kernels.sh` — compile Eä kernels to .so + Python bindings
2. `pip install -e .` — install CLI

## Test
`python3 -m pytest tests/ -v`

## Architecture
- `kernels/*.ea` — pure SIMD kernels (no scalar ops)
- `eabrain.py` — CLI entry point (argparse + ctypes calls)
- `indexer.py` — index builder (walks projects, calls scan.ea, writes index.bin)
- `reference/ea_reference.json` — pre-baked Eä language reference
- `lib/` — compiled .so files + generated Python bindings (not in git)

## Ea compiler
Located at `/root/dev/eacompute/target/release/ea` (not on PATH).
All kernels are pure SIMD — no scalar code paths allowed.
