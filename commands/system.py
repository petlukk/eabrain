"""System commands: status, inject, init, serve."""

import os


def cmd_status(args, cfg):
    from eabrain import _VERSION, _get_db, _time_ago
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


def cmd_inject(args, cfg):
    from eabrain import _get_db, _get_preamble_dir, _get_session_file, _time_ago
    from inject import build_injection, start_session
    from sync import last_push_timestamp
    db = _get_db(cfg)
    project = getattr(args, "project", None) or os.getcwd()
    budget = getattr(args, "budget", 2000) or 2000
    start_session(db, project=project, session_file=_get_session_file(cfg))

    sync_status = None
    ts = last_push_timestamp(cfg.get("sync_repo"))
    if ts is not None:
        sync_status = f"Sync: auto · last push {_time_ago(ts)}"

    output = build_injection(
        db=db,
        preamble_dir=_get_preamble_dir(cfg),
        project=project,
        budget=budget,
        sync_status=sync_status,
    )
    print(output)
    db.close()


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


def cmd_serve(args, cfg):
    from server import serve
    port = getattr(args, "port", 37777) or 37777
    serve(cfg, port=port)
