#!/usr/bin/env python3
"""eabrain — Eä-driven context engine for Claude Code."""

import argparse
import ctypes
import json
import os
import shutil
import sys
import time

import numpy as np

_DEFAULT_INDEX = os.path.expanduser("~/.eabrain/index.bin")
_VERSION = "0.2.0"
_LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")


def _resolve_ea_compiler() -> str | None:
    """Locate the ea compiler via $EA or PATH. Returns None if not found."""
    env = os.environ.get("EA")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    return shutil.which("ea")


def _resolve_eacompute_dir(ea_compiler: str | None) -> str | None:
    """Locate eacompute source tree via $EACOMPUTE_DIR, derive from the ea
    compiler's install path, or fall back to a sibling of the eabrain
    install (.../Dev/eabrain → .../Dev/eacompute)."""
    env = os.environ.get("EACOMPUTE_DIR")
    if env and os.path.isdir(env):
        return env
    if ea_compiler:
        # .../eacompute/target/release/ea → .../eacompute
        parent = os.path.dirname(os.path.dirname(os.path.dirname(ea_compiler)))
        if os.path.isdir(os.path.join(parent, "src")):
            return parent
    install_dir = os.path.dirname(os.path.abspath(__file__))
    if "site-packages" not in install_dir and "dist-packages" not in install_dir:
        sibling = os.path.join(os.path.dirname(install_dir), "eacompute")
        if os.path.isdir(os.path.join(sibling, "src")):
            return sibling
    return None


def _default_projects() -> list:
    """Default projects list: the parent directory of the eabrain install.
    When eabrain is cloned into ~/Dev/eabrain, this returns ["~/Dev"], so
    sibling projects are indexed automatically. Returns [] for site-packages
    installs — those users must set `projects` in config.json explicitly."""
    install_dir = os.path.dirname(os.path.abspath(__file__))
    if "site-packages" in install_dir or "dist-packages" in install_dir:
        return []
    parent = os.path.dirname(install_dir)
    return [parent] if parent and os.path.isdir(parent) else []


def _load_config(env_path: str = None) -> dict:
    path = env_path or os.environ.get("EABRAIN_CONFIG") or os.path.expanduser("~/.eabrain/config.json")
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
    else:
        cfg = {}
    if "projects" not in cfg:
        cfg["projects"] = _default_projects()
    cfg.setdefault("index_path", _DEFAULT_INDEX)
    cfg.setdefault("max_source_lines", 50)
    cfg.setdefault("max_session_entries", 100)
    cfg.setdefault("eabrain_dir", os.path.expanduser("~/.eabrain"))

    if "ea_compiler" not in cfg:
        cfg["ea_compiler"] = _resolve_ea_compiler()
    if "eacompute_dir" not in cfg:
        cfg["eacompute_dir"] = _resolve_eacompute_dir(cfg.get("ea_compiler"))
    if "autoresearch_dir" not in cfg:
        ec = cfg.get("eacompute_dir")
        cfg["autoresearch_dir"] = os.path.join(ec, "autoresearch", "kernels") if ec else None
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
    if not projects:
        print("No projects configured. Set 'projects' in ~/.eabrain/config.json "
              "or pass --projects /path1,/path2", file=sys.stderr)
        sys.exit(1)
    stats = build_index(projects, _ref_path(), cfg["index_path"])
    print(f"Indexed {stats['kernel_count']} kernels from {stats['file_count']} files, "
          f"{stats['ref_count']} ref entries -> {cfg['index_path']}")


def cmd_search(args, cfg):
    from indexer import read_index, _simd_byte_histogram
    query = args.query
    max_lines = cfg["max_source_lines"]
    memory_only = getattr(args, "memory_only", False)
    kernels_only = getattr(args, "kernels_only", False)

    fuzzy_lib = None
    hist = None

    # Kernel search
    if not memory_only:
        idx = read_index(cfg["index_path"])
        kernels = idx["kernels"]
        if args.fuzzy:
            fuzzy_lib = _load_lib("libfuzzy.so")
            hist = _simd_byte_histogram(query.encode("utf-8"))
            emb = idx["embeddings"]
            n = len(kernels)
            if n == 0 or emb.shape[0] == 0:
                results = []
            else:
                scores = np.zeros(n, dtype=np.float32)
                fuzzy_lib.batch_cosine(
                    hist.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_float(1.0),
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
        print(f"{len(results)} kernel results (indexed {_time_ago(ts)}, {len(kernels)} kernels)\n")
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
    else:
        print(f"# eabrain search: {query}\n")

    # Observation search
    if not kernels_only:
        db = _get_db(cfg)
        obs_results = []
        if args.fuzzy:
            obs_ids, obs_emb = db.load_embeddings()
            if len(obs_ids) > 0:
                if fuzzy_lib is None:
                    fuzzy_lib = _load_lib("libfuzzy.so")
                if hist is None:
                    hist = _simd_byte_histogram(query.encode("utf-8"))
                obs_scores = np.zeros(len(obs_ids), dtype=np.float32)
                fuzzy_lib.batch_cosine(
                    hist.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_float(1.0),
                    obs_emb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_int32(256),
                    ctypes.c_int32(len(obs_ids)),
                    obs_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                )
                order = np.argsort(obs_scores)[::-1][:5]
                for idx_i in order:
                    if obs_scores[idx_i] > 0:
                        obs_row = db.conn.execute(
                            "SELECT * FROM observations WHERE id = ?", (obs_ids[idx_i],)
                        ).fetchone()
                        if obs_row:
                            obs_results.append(dict(obs_row))
        else:
            obs_results = db.simd_search(query, limit=5)

        if obs_results:
            print(f"## Memory ({len(obs_results)} matches)\n")
            for o in obs_results:
                print(f"  [{o['type']}] {o['content'][:100]}")
        db.close()


def cmd_ref(args, cfg):
    # Read from the binary index, which contains the curated JSON entries
    # merged with auto-scraped intrinsics from eacompute source. Fall back
    # to the static JSON if the index is missing (pre-first-index state).
    from indexer import read_index
    if os.path.exists(cfg["index_path"]):
        idx = read_index(cfg["index_path"])
        entries = idx["refs"]
    else:
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
    from inject import get_current_session_id
    from indexer import _simd_byte_histogram
    db = _get_db(cfg)
    sid = get_current_session_id(_get_session_file(cfg))
    emb = _simd_byte_histogram(args.note.encode("utf-8"))
    db.store_observation(
        project=os.getcwd(),
        obs_type="note",
        content=args.note,
        session_id=sid,
        embedding=emb.tobytes(),
    )
    print(f"Remembered: {args.note}")
    db.close()


def cmd_recall(args, cfg):
    db = _get_db(cfg)
    n = args.last if args.last else 20
    results = db.recent(limit=n)
    if not results:
        print("No observations.")
        db.close()
        return
    print(f"# Observations ({len(results)} entries)\n")
    for r in results:
        print(f"[{r['created_at'][:16]}] [{r['type']}] {r['content']}")
    db.close()


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
    print(f"Ea compiler: {cfg['ea_compiler'] or '(not found — set $EA or put ea on PATH)'}")

    db_path = os.path.join(cfg["eabrain_dir"], "memory.db")
    if os.path.exists(db_path):
        db = _get_db(cfg)
        s = db.stats()
        print(f"Observations: {s['observation_count']}")
        print(f"Sessions: {s['session_count']}")
        if s["last_session"]:
            print(f"Last session: {s['last_session'][:16]}")
        print(f"Memory DB: {s['db_size_bytes'] // 1024}KB")
        db.close()

    print()
    print("Common commands:")
    print("  eabrain index              # build index")
    print("  eabrain search <query>     # search kernels and observations")
    print("  eabrain ref <name>         # look up Ea reference")
    print("  eabrain remember <note>    # save observation")
    print("  eabrain recall             # show recent observations")
    print("  eabrain timeline           # show session timeline")
    print("  eabrain serve              # start web viewer")


def cmd_patterns(args, cfg):
    ar_dir = cfg.get("autoresearch_dir")
    if not ar_dir or not os.path.isdir(ar_dir):
        shown = ar_dir or "(not resolved)"
        print(f"Autoresearch not found at {shown}")
        print("Set autoresearch_dir in ~/.eabrain/config.json, or EACOMPUTE_DIR env var, "
              "or put ea on PATH so the location can be derived.")
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


def _get_db(cfg):
    from memory import MemoryDB
    return MemoryDB(os.path.join(cfg["eabrain_dir"], "memory.db"))


def _get_session_file(cfg):
    return os.path.join(cfg["eabrain_dir"], "current_session")


def _get_preamble_dir(cfg):
    return os.path.join(cfg["eabrain_dir"], "preamble")


def cmd_inject(args, cfg):
    from inject import build_injection, start_session
    db = _get_db(cfg)
    project = getattr(args, "project", None) or os.getcwd()
    budget = getattr(args, "budget", 2000) or 2000
    start_session(db, project=project, session_file=_get_session_file(cfg))
    output = build_injection(
        db=db,
        preamble_dir=_get_preamble_dir(cfg),
        project=project,
        budget=budget,
    )
    print(output)
    db.close()


def cmd_store(args, cfg):
    from inject import get_current_session_id
    from indexer import _simd_byte_histogram
    db = _get_db(cfg)
    sid = get_current_session_id(_get_session_file(cfg))
    project = getattr(args, "project", None) or os.getcwd()
    content = args.content
    emb = _simd_byte_histogram(content.encode("utf-8"))
    db.store_observation(
        project=project,
        obs_type=args.type,
        content=content,
        session_id=sid,
        embedding=emb.tobytes(),
    )
    print(f"Stored [{args.type}]: {content[:80]}")
    db.close()


def cmd_store_summary(args, cfg):
    from inject import get_current_session_id, end_session
    db = _get_db(cfg)
    session_file = _get_session_file(cfg)
    sid = get_current_session_id(session_file)
    if sid:
        end_session(db, session_id=sid, summary=args.content, session_file=session_file)
        print(f"Session closed: {args.content[:80]}")
    else:
        print("No active session.")
    db.close()


def cmd_timeline(args, cfg):
    db = _get_db(cfg)
    project = getattr(args, "project", None)
    limit = getattr(args, "last", 10) or 10
    since = getattr(args, "since", None)
    tl = db.timeline(project=project, limit=limit, since=since)
    if not tl:
        print("No sessions recorded.")
        db.close()
        return
    for entry in tl:
        s = entry["session"]
        obs = entry["observations"]
        print(f"\n--- Session: {s['started_at'][:16]} [{s['project']}] ---")
        if s.get("summary"):
            print(f"Summary: {s['summary']}")
        for o in obs:
            print(f"  [{o['type']}] {o['content']}")
    db.close()


def cmd_migrate(args, cfg):
    from memory import migrate_from_index
    db = _get_db(cfg)
    idx_path = cfg["index_path"]
    if not os.path.exists(idx_path):
        print("No index.bin found — nothing to migrate.")
        db.close()
        return
    count = migrate_from_index(db, idx_path)
    print(f"Migrated {count} session notes to memory.db")
    db.close()


def cmd_sync(args, cfg):
    from sync import export_db, import_db
    db_path = os.path.join(cfg["eabrain_dir"], "memory.db")
    if args.export_path:
        export_db(db_path, args.export_path)
        print(f"Exported memory.db to {args.export_path}")
    elif args.import_path:
        import_db(db_path, args.import_path)
        print(f"Imported and merged from {args.import_path}")
    else:
        print("Usage: eabrain sync --export <path> or --import <path>")


def cmd_serve(args, cfg):
    from server import serve
    port = getattr(args, "port", 37777) or 37777
    serve(cfg, port=port)


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

    p_search = sub.add_parser("search", help="Search kernels and observations")
    p_search.add_argument("query")
    p_search.add_argument("--fuzzy", action="store_true", help="Use cosine similarity")
    p_search.add_argument("--arch", choices=["arm", "x86", "aarch64", "x86_64"],
                          help="Filter by arch")
    scope = p_search.add_mutually_exclusive_group()
    scope.add_argument("--kernels-only", action="store_true", help="Search kernels only")
    scope.add_argument("--memory-only", action="store_true", help="Search observations only")

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

    p_inject = sub.add_parser("inject", help="Inject context for session start")
    p_inject.add_argument("--project", help="Project directory (default: cwd)")
    p_inject.add_argument("--budget", type=int, default=2000, help="Token budget for dynamic section")

    p_store = sub.add_parser("store", help="Store an observation")
    p_store.add_argument("content")
    p_store.add_argument("--type", required=True,
                         choices=["decision", "bug", "architecture", "pattern", "error", "note"])
    p_store.add_argument("--project", help="Project name (default: cwd)")

    p_store_summary = sub.add_parser("store-summary", help="Store session summary and close session")
    p_store_summary.add_argument("content")

    p_timeline = sub.add_parser("timeline", help="Show session timeline")
    p_timeline.add_argument("--project", help="Filter by project")
    p_timeline.add_argument("--last", type=int, default=10, help="Number of sessions")
    p_timeline.add_argument("--since", help="ISO date filter")

    sub.add_parser("migrate", help="Migrate v0.1 notes to memory.db")

    p_sync = sub.add_parser("sync", help="Export or import memory.db")
    p_sync.add_argument("--export", dest="export_path", help="Export to path")
    p_sync.add_argument("--import", dest="import_path", help="Import from path")

    p_serve = sub.add_parser("serve", help="Start web viewer")
    p_serve.add_argument("--port", type=int, default=37777, help="Port number")

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
        "inject": cmd_inject,
        "store": cmd_store,
        "store-summary": cmd_store_summary,
        "timeline": cmd_timeline,
        "migrate": cmd_migrate,
        "sync": cmd_sync,
        "serve": cmd_serve,
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
