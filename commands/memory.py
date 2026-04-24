"""Memory-DB commands: remember, recall, store, store-summary, timeline, migrate, sync."""

import os
import sys


def cmd_remember(args, cfg):
    from eabrain import _get_db, _get_session_file
    from inject import get_current_session_id
    from indexer import _simd_byte_histogram
    from safety import SafetyScanError
    db = _get_db(cfg)
    try:
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
    except SafetyScanError as e:
        print(f"eabrain remember: {e}", file=sys.stderr)
        sys.exit(2)
    finally:
        db.close()


def cmd_recall(args, cfg):
    from eabrain import _get_db
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


def cmd_store(args, cfg):
    from eabrain import _get_db, _get_session_file
    from inject import get_current_session_id
    from indexer import _simd_byte_histogram
    from safety import SafetyScanError
    db = _get_db(cfg)
    try:
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
    except SafetyScanError as e:
        print(f"eabrain store: {e}", file=sys.stderr)
        sys.exit(2)
    finally:
        db.close()


def cmd_store_summary(args, cfg):
    from eabrain import _get_db, _get_session_file
    from inject import get_current_session_id, end_session
    from safety import SafetyScanError
    db = _get_db(cfg)
    try:
        session_file = _get_session_file(cfg)
        sid = get_current_session_id(session_file)
        if sid:
            end_session(db, session_id=sid, summary=args.content, session_file=session_file)
            print(f"Session closed: {args.content[:80]}")
        else:
            print("No active session.")
    except SafetyScanError as e:
        print(f"eabrain store-summary: {e}", file=sys.stderr)
        sys.exit(2)
    finally:
        db.close()


def cmd_timeline(args, cfg):
    from eabrain import _get_db
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
    from eabrain import _get_db
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
    import sync
    db_path = os.path.join(cfg["eabrain_dir"], "memory.db")
    eabrain_dir = cfg["eabrain_dir"]
    action = getattr(args, "action", None)

    if action == "pull":
        sync.pull(db_path, cfg.get("sync_repo"), eabrain_dir)
        return
    if action == "push":
        sync.push(db_path, cfg.get("sync_repo"), eabrain_dir)
        return

    if args.export_path:
        sync.export_db(db_path, args.export_path)
        print(f"Exported memory.db to {args.export_path}")
    elif args.import_path:
        sync.import_db(db_path, args.import_path)
        print(f"Imported and merged from {args.import_path}")
    else:
        print("Usage: eabrain sync {pull|push} | --export <path> | --import <path>")
