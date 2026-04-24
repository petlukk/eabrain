#!/usr/bin/env python3
"""eabrain — Eä-driven context engine for Claude Code."""

import argparse
import ctypes
import json
import os
import shutil
import sys
import time

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


def _resolve_sync_repo() -> str | None:
    """Auto-detect an eabrain-memory sibling checkout of the install.
    Users with a different layout can override by setting `sync_repo` in
    config.json. Returns None if no clone is found — sync becomes a no-op."""
    install_dir = os.path.dirname(os.path.abspath(__file__))
    if "site-packages" in install_dir or "dist-packages" in install_dir:
        return None
    sibling = os.path.join(os.path.dirname(install_dir), "eabrain-memory")
    return sibling if os.path.isdir(os.path.join(sibling, ".git")) else None


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
    if "sync_repo" not in cfg:
        cfg["sync_repo"] = _resolve_sync_repo()
    return cfg


def _ref_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference", "ea_reference.json")


# ---------------------------------------------------------------------------
# Helpers (shared across commands/* via function-local imports)
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


def _get_db(cfg):
    from memory import MemoryDB
    return MemoryDB(os.path.join(cfg["eabrain_dir"], "memory.db"))


def _get_session_file(cfg):
    return os.path.join(cfg["eabrain_dir"], "current_session")


def _get_preamble_dir(cfg):
    return os.path.join(cfg["eabrain_dir"], "preamble")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    from commands.memory import (
        cmd_remember, cmd_recall, cmd_store, cmd_store_summary,
        cmd_timeline, cmd_migrate, cmd_sync,
    )
    from commands.search import cmd_index, cmd_search, cmd_ref, cmd_patterns
    from commands.system import cmd_status, cmd_inject, cmd_init, cmd_serve

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

    p_sync = sub.add_parser("sync", help="Sync memory.db with git remote")
    p_sync.add_argument("action", nargs="?", choices=["pull", "push"],
                        help="Git-backed sync action (requires sync_repo configured)")
    p_sync.add_argument("--export", dest="export_path", help="Export local db to path")
    p_sync.add_argument("--import", dest="import_path", help="Merge db at path into local")

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
