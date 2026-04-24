"""Search commands: index, search, ref, patterns."""

import ctypes
import json
import os
import sys

import numpy as np


def cmd_index(args, cfg):
    from eabrain import _ref_path
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
    from eabrain import _get_db, _load_lib, _read_source, _time_ago
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
    from eabrain import _ref_path
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
        benchmarked = 0
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
            benchmarked += 1
        if benchmarked == 0:
            print(f"(0 of {len(kernel_dirs)} kernels benchmarked — no history.json files "
                  f"under {ar_dir}. Run autoresearch to populate.)")
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
