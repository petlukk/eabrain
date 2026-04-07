#!/usr/bin/env python3
"""eabrain — Eä-driven context engine for Claude Code."""

import argparse
import ctypes
import json
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_PROJECTS = [
    "/root/dev/eacompute",
    "/root/dev/eakv",
    "/root/dev/eaclaw",
    "/root/dev/Olorin",
    "/root/dev/Cougar",
    "/root/dev/eachacha",
]
_DEFAULT_INDEX = os.path.expanduser("~/.eabrain/index.bin")
_EA_COMPILER = "/root/dev/eacompute/target/release/ea"
_VERSION = "0.1.0"
_LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")


def _load_config(env_path: str = None) -> dict:
    path = env_path or os.environ.get("EABRAIN_CONFIG") or os.path.expanduser("~/.eabrain/config.json")
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
    else:
        cfg = {}
    cfg.setdefault("projects", _DEFAULT_PROJECTS)
    cfg.setdefault("index_path", _DEFAULT_INDEX)
    cfg.setdefault("max_source_lines", 50)
    cfg.setdefault("max_session_entries", 100)
    cfg.setdefault("ea_compiler", _EA_COMPILER)
    return cfg


def _ref_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference", "ea_reference.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_lib(name: str) -> ctypes.CDLL:
    return ctypes.CDLL(os.path.join(_LIB_DIR, name))


def _read_source(path: str, line_start: int, line_count: int, max_lines: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        start = max(0, line_start - 1)
        end = min(len(all_lines), start + line_count)
        if (end - start) > max_lines:
            return None
        return "".join(all_lines[start:end]).rstrip()
    except OSError:
        return None


def _time_ago(ts: int) -> str:
    delta = int(time.time()) - ts
    if delta < 60:
        return f"{delta}s ago"
    if delta < 3600:
        return f"{delta // 60}m ago"
    if delta < 86400:
        return f"{delta // 3600}h ago"
    return f"{delta // 86400}d ago"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_index(args, cfg):
    from indexer import build_index
    projects = cfg["projects"]
    if args.projects:
        projects = [p.strip() for p in args.projects.split(",")]
    stats = build_index(projects, _ref_path(), cfg["index_path"])
    print(f"Indexed {stats['kernel_count']} kernels from {stats['file_count']} files, "
          f"{stats['ref_count']} ref entries -> {cfg['index_path']}")


def cmd_search(args, cfg):
    from indexer import read_index
    idx = read_index(cfg["index_path"])
    kernels = idx["kernels"]
    query = args.query
    max_lines = cfg["max_source_lines"]

    if args.fuzzy:
        fuzzy_lib = _load_lib("libfuzzy.so")
        # Build byte histogram for query
        query_bytes = query.encode("utf-8")
        hist = np.zeros(256, dtype=np.float32)
        for b in query_bytes:
            hist[b] += 1.0
        norm_val = float(np.linalg.norm(hist))
        if norm_val > 0:
            hist /= norm_val

        emb = idx["embeddings"]
        n = len(kernels)
        if n == 0 or emb.shape[0] == 0:
            results = []
        else:
            scores = np.zeros(n, dtype=np.float32)
            fuzzy_lib.batch_cosine(
                hist.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_float(norm_val if norm_val > 0 else 1.0),
                emb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int32(256),
                ctypes.c_int32(n),
                scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            order = np.argsort(scores)[::-1][:10]
            results = [kernels[i] for i in order if scores[i] > 0]
    else:
        ql = query.lower()
        results = [k for k in kernels
                   if ql in k["func_name"].lower() or ql in k["path"].lower()]

    if args.arch:
        results = [k for k in results if k["arch"] == args.arch]

    ts = idx["header"]["timestamp"]
    print(f"# eabrain search: {query}\n")
    print(f"{len(results)} results (indexed {_time_ago(ts)}, {len(kernels)} kernels)\n")
    for i, k in enumerate(results[:10], 1):
        rel = k["path"]
        print(f"{i}. {rel}:{k['line_start']}")
        print(f"   export func {k['func_name']}(...)")
        print(f"   arch: {k['arch']}  simd: f32x{k['simd_width']}  lines: {k['line_count']}")
        src = _read_source(k["path"], k["line_start"], k["line_count"], max_lines)
        if src is not None:
            print(f"   [source: {k['line_count']} lines]")
            for line in src.split("\n"):
                print(f"   {line}")
        print()


def cmd_ref(args, cfg):
    with open(_ref_path()) as f:
        data = json.load(f)
    entries = data.get("entries", [])
    query = args.query.lower()

    # Exact match first, then substring
    exact = [e for e in entries if e["name"].lower() == query]
    substr = [e for e in entries if query in e["name"].lower() or query in e.get("description", "").lower()]
    results = exact if exact else substr

    print(f"# eabrain ref: {args.query}\n")
    if not results:
        print("No results found.")
        return
    for e in results[:5]:
        print(f"Name: {e['name']}")
        print(f"Category: {e.get('category', '')}")
        print(f"Signature: {e.get('signature', '')}")
        print(f"Description: {e.get('description', '')}")
        print()


def cmd_remember(args, cfg):
    from indexer import read_index, write_index
    idx = read_index(cfg["index_path"])
    sessions = idx["sessions"]
    sessions.append({"text": args.note, "timestamp": int(time.time())})
    max_s = cfg["max_session_entries"]
    if len(sessions) > max_s:
        sessions = sessions[-max_s:]
    write_index(cfg["index_path"], idx["kernels"], idx["refs"], sessions, idx["embeddings"])
    print(f"Remembered: {args.note}")


def cmd_recall(args, cfg):
    from indexer import read_index
    idx = read_index(cfg["index_path"])
    sessions = idx["sessions"]
    n = args.last if args.last else len(sessions)
    shown = sessions[-n:] if n < len(sessions) else sessions
    if not shown:
        print("No session notes.")
        return
    print(f"# Session notes ({len(shown)} entries)\n")
    for s in shown:
        print(f"[{_time_ago(s['timestamp'])}] {s['text']}")


def cmd_status(args, cfg):
    index_path = cfg["index_path"]
    if os.path.exists(index_path):
        from indexer import read_index
        idx = read_index(index_path)
        hdr = idx["header"]
        print(f"eabrain v{_VERSION}")
        print(f"Index: {index_path}")
        print(f"Last indexed: {_time_ago(hdr['timestamp'])}")
        print(f"Kernels: {hdr['kernel_count']}")
        print(f"Refs: {hdr['ref_count']}")
        print(f"Projects: {len(cfg['projects'])}")
        print(f"Session notes: {hdr['session_count']}")
    else:
        print(f"eabrain v{_VERSION}  (no index — run: eabrain index)")
        print(f"Projects: {len(cfg['projects'])}")
    print(f"Ea compiler: {cfg['ea_compiler']}")
    print()
    print("Common commands:")
    print("  eabrain index              # build index")
    print("  eabrain search <query>     # search kernels")
    print("  eabrain ref <name>         # look up Ea reference")
    print("  eabrain remember <note>    # save session note")
    print("  eabrain recall             # show session notes")


def cmd_patterns(args, cfg):
    ar_dir = cfg.get("autoresearch_dir", "/root/dev/eacompute/autoresearch/kernels")
    if not os.path.isdir(ar_dir):
        print(f"Autoresearch not found at {ar_dir}")
        print("Set autoresearch_dir in ~/.eabrain/config.json")
        sys.exit(1)

    kernel_dirs = sorted(os.listdir(ar_dir))
    query = args.query.lower() if args.query else None

    if args.what_works:
        # Summary of all proven optimization patterns
        print("# eabrain patterns --what-works\n")
        print("Proven optimization patterns from autoresearch (28 benchmarks):\n")
        wins = []
        for name in kernel_dirs:
            hist_path = os.path.join(ar_dir, name, "history.json")
            if not os.path.exists(hist_path):
                continue
            with open(hist_path) as f:
                history = json.load(f)
            accepted = [x for x in history if x.get("accepted")]
            if not accepted:
                continue
            best = accepted[-1]
            wins.append((name, best, len(history)))

        if not wins:
            print("No accepted optimizations found.")
            return

        # Group by pattern type
        print(f"{'Kernel':25s} {'Gain':>8s}  Strategy")
        print("-" * 80)
        for name, best, n_iters in wins:
            t = f"{best['time_us']:.0f}us" if best.get("time_us") else "n/a"
            hyp = best["hypothesis"][:70] if best.get("hypothesis") else ""
            print(f"{name:25s} {t:>8s}  {hyp}")

        print(f"\n{len(wins)} kernels improved out of {len(kernel_dirs)} benchmarked.")
        print("\nCommon winning patterns:")
        all_hyps = " ".join(w[1].get("hypothesis", "") for w in wins).lower()
        patterns = [
            ("unroll", "Loop unrolling (2x-8x)"),
            ("prefetch", "Prefetch tuning (add or remove)"),
            ("f32x8", "SIMD width: f32x4 -> f32x8"),
            ("f32x16", "SIMD width: f32x8 -> f32x16 (AVX-512)"),
            ("stream_store", "Non-temporal stores (bypass cache)"),
            ("accumulator", "Multiple independent accumulators"),
            ("restrict", "Pointer restrict for alias analysis"),
            ("fuse", "Operation fusion"),
        ]
        for keyword, desc in patterns:
            if keyword in all_hyps:
                print(f"  - {desc}")
        return

    if query is None:
        # List all benchmarks
        print("# eabrain patterns\n")
        print("Available autoresearch benchmarks:\n")
        print(f"{'Kernel':25s} {'Iters':>6s} {'Accepted':>9s} {'Best':>10s}")
        print("-" * 55)
        for name in kernel_dirs:
            hist_path = os.path.join(ar_dir, name, "history.json")
            if not os.path.exists(hist_path):
                continue
            with open(hist_path) as f:
                history = json.load(f)
            accepted = [x for x in history if x.get("accepted")]
            best = accepted[-1] if accepted else None
            t = f"{best['time_us']:.0f}us" if best and best.get("time_us") else "-"
            print(f"{name:25s} {len(history):>6d} {len(accepted):>9d} {t:>10s}")
        print(f"\nUse: eabrain patterns <name> for details")
        print("Use: eabrain patterns --what-works for proven optimization patterns")
        return

    # Find matching kernel
    matches = [n for n in kernel_dirs if query in n.lower()]
    if not matches:
        print(f"No autoresearch benchmark matching '{args.query}'.")
        print(f"Available: {', '.join(kernel_dirs)}")
        return

    for name in matches:
        kdir = os.path.join(ar_dir, name)
        print(f"# eabrain patterns: {name}\n")

        # Strategy space from program.md
        prog_path = os.path.join(kdir, "program.md")
        if os.path.exists(prog_path):
            with open(prog_path) as f:
                prog = f.read()
            # Extract strategy section
            lines = prog.split("\n")
            in_strategy = False
            strategy_lines = []
            for line in lines:
                if "strategy" in line.lower() and line.startswith("#"):
                    in_strategy = True
                    continue
                elif line.startswith("#") and in_strategy:
                    break
                elif in_strategy:
                    strategy_lines.append(line)
            if strategy_lines:
                print("## Strategy Space\n")
                print("\n".join(strategy_lines).strip())
                print()

        # Optimization history
        hist_path = os.path.join(kdir, "history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = json.load(f)
            accepted = [x for x in history if x.get("accepted")]
            rejected = [x for x in history if not x.get("accepted")]

            print(f"## History ({len(history)} iterations, {len(accepted)} accepted)\n")
            for h in history:
                status = "ACCEPTED" if h["accepted"] else "rejected"
                t = f"{h['time_us']:.0f}us" if h.get("time_us") else "n/a"
                correct = "correct" if h.get("correct") else "WRONG"
                hyp = h.get("hypothesis", "")[:90]
                print(f"  {h['iteration']:2d}. [{status:8s}] {t:>8s} {correct:7s}  {hyp}")
            print()

            # What worked / what failed summary
            if accepted:
                print("## What Worked\n")
                for h in accepted:
                    print(f"  - {h.get('hypothesis', 'n/a')}")
                print()
            if rejected:
                print("## What Failed\n")
                for h in rejected[-5:]:
                    hyp = h.get("hypothesis", "n/a")[:100]
                    reason = "wrong output" if not h.get("correct") else "no improvement"
                    if not h.get("time_us"):
                        reason = "compile error or timeout"
                    print(f"  - [{reason}] {hyp}")
                print()

        # Best kernel source
        best_path = os.path.join(kdir, "best_kernel.ea")
        if os.path.exists(best_path):
            with open(best_path) as f:
                src = f.read()
            lines = src.strip().split("\n")
            print(f"## Best Kernel ({len(lines)} lines)\n")
            print(src.strip())
            print()


def cmd_init(args, cfg):
    target_dir = args.project_dir or os.getcwd()
    claude_md = os.path.join(target_dir, "CLAUDE.md")
    snippet = (
        "\n## eabrain\n"
        "Use `eabrain search <query>` to find Ea kernels across all projects.\n"
        "Use `eabrain ref <name>` to look up Ea language reference.\n"
        "Use `eabrain patterns <kernel>` to see autoresearch-proven optimization patterns.\n"
        "Use `eabrain patterns --what-works` for a summary of all winning optimizations.\n"
        "Use `eabrain remember <note>` to save context between sessions.\n"
    )
    with open(claude_md, "a", encoding="utf-8") as f:
        f.write(snippet)
    print(f"Appended eabrain snippet to {claude_md}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(prog="eabrain", description="Eä-driven context engine")
    parser.add_argument("--config", help="Path to config.json")
    sub = parser.add_subparsers(dest="command")

    p_index = sub.add_parser("index", help="Build index")
    p_index.add_argument("--projects", help="Comma-separated project dirs")

    p_search = sub.add_parser("search", help="Search kernels")
    p_search.add_argument("query")
    p_search.add_argument("--fuzzy", action="store_true", help="Use cosine similarity")
    p_search.add_argument("--arch", choices=["arm", "x86", "aarch64", "x86_64"],
                          help="Filter by arch")

    p_ref = sub.add_parser("ref", help="Look up Ea reference")
    p_ref.add_argument("query")

    p_remember = sub.add_parser("remember", help="Save session note")
    p_remember.add_argument("note")

    p_recall = sub.add_parser("recall", help="Show session notes")
    p_recall.add_argument("--last", type=int, help="Show last N entries")

    sub.add_parser("status", help="Show index status")

    p_patterns = sub.add_parser("patterns", help="Autoresearch optimization patterns")
    p_patterns.add_argument("query", nargs="?", help="Kernel name (e.g. matmul, dot_product)")
    p_patterns.add_argument("--what-works", action="store_true",
                            help="Summary of all proven optimization patterns")

    p_init = sub.add_parser("init", help="Add eabrain snippet to CLAUDE.md")
    p_init.add_argument("--project-dir", help="Target project directory")

    args = parser.parse_args()
    cfg = _load_config(args.config if hasattr(args, "config") else None)

    dispatch = {
        "index": cmd_index,
        "search": cmd_search,
        "ref": cmd_ref,
        "remember": cmd_remember,
        "recall": cmd_recall,
        "status": cmd_status,
        "patterns": cmd_patterns,
        "init": cmd_init,
    }

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    fn = dispatch.get(args.command)
    if fn is None:
        parser.print_help()
        sys.exit(1)

    fn(args, cfg)


if __name__ == "__main__":
    main()
