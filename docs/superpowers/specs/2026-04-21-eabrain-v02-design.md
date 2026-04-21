# eabrain v0.2 — Persistent Memory Layer

## Overview

eabrain evolves from an Ea kernel indexer into a general-purpose persistent memory system for Claude Code. It auto-captures session context via hooks, stores structured observations in SQLite, and injects relevant memory into new sessions — so Claude picks up where it left off.

The Ea kernel index stays. SIMD-accelerated search stays. A new memory layer, smart injection, sync, and a web viewer are added alongside.

## Goals

1. **Cross-session memory** — decisions, bugs, architecture context survive between Claude sessions
2. **General-purpose** — works across all projects, not just Ea
3. **Portable** — memory.db transfers between machines, Claude resumes anywhere
4. **SIMD-fast** — fuzzy.ea cosine similarity powers search across kernels and observations
5. **Zero framework dependencies** — stdlib Python, vanilla JS, Ea SIMD kernels

## Non-Goals

- MCP server (CLI + hooks is sufficient)
- Polyglot code indexing (Claude can read files directly)
- Async runtime or external HTTP frameworks
- AI-powered compression via API calls (Claude compresses its own sessions at zero cost)

---

## Architecture

```
eabrain v0.2

  Storage:
    index.bin   — kernel records + kernel embeddings (unchanged from v0.1)
    memory.db   — SQLite: observations, sessions, observation embeddings
    config.json — projects list, preamble paths, budget, port

  SIMD Kernels (unchanged):
    fuzzy.ea    — byte-histogram cosine similarity (x86_64 + aarch64)
    scan.ea     — source code parser (find exports, detect types/intrinsics)
    scan_rust.ea — Rust doc comment scanner

  Hooks (Claude Code settings.json):
    SessionStart  → eabrain inject
    SessionEnd    → Claude summarizes → eabrain store + store-summary

  Web Viewer:
    eabrain serve → stdlib http.server, single HTML file, REST API
```

### Component Boundaries

| Component | Responsibility | Interface |
|-----------|---------------|-----------|
| `eabrain.py` | CLI entry point, argument parsing, dispatches to modules | CLI commands |
| `indexer.py` | Kernel index builder (unchanged) | `build_index()`, `load_index()` |
| `memory.py` (NEW) | SQLite operations: store, query, timeline, migrate | `store_observation()`, `query()`, `timeline()`, `store_summary()` |
| `inject.py` (NEW) | Assembles preamble + dynamic context, respects token budget | `inject(project, budget)` |
| `server.py` (NEW) | Web viewer: http.server subclass, REST API | `serve(port)` |
| `sync.py` (NEW) | Export/import/merge memory.db across machines | `export_db()`, `import_db()` |
| `fuzzy.py` | SIMD cosine search bindings (unchanged) | `batch_cosine()`, `byte_histogram_embed()` |
| `scan.py` | SIMD source parser bindings (unchanged) | `count_exports()`, `find_export_offsets()` |

---

## Storage

### SQLite Schema (memory.db)

```sql
CREATE TABLE sessions (
    id          TEXT PRIMARY KEY,  -- UUID
    project     TEXT NOT NULL,
    started_at  TEXT NOT NULL,     -- ISO 8601
    ended_at    TEXT,              -- NULL if incomplete
    summary     TEXT               -- Claude's end-of-session compression
);

CREATE TABLE observations (
    id          TEXT PRIMARY KEY,  -- UUID
    session_id  TEXT REFERENCES sessions(id),
    project     TEXT NOT NULL,
    type        TEXT NOT NULL,     -- decision | bug | architecture | pattern | error | note
    content     TEXT NOT NULL,
    content_hash TEXT NOT NULL,    -- SHA-256 of content, for dedup on merge
    embedding   BLOB,             -- 256 x f32, computed by fuzzy.ea
    created_at  TEXT NOT NULL      -- ISO 8601
);

CREATE INDEX idx_obs_project ON observations(project);
CREATE INDEX idx_obs_type ON observations(type);
CREATE INDEX idx_obs_created ON observations(created_at);
CREATE INDEX idx_obs_session ON observations(session_id);
CREATE INDEX idx_obs_hash ON observations(content_hash);
```

### Kernel Index (index.bin — unchanged)

Binary format from v0.1 stays for kernel records and kernel embeddings. No changes.

### Embedding Storage

Kernel embeddings remain in index.bin (fixed-count, SIMD-aligned).

Observation embeddings are stored as BLOBs in SQLite. For SIMD search, `memory.py` loads all embeddings into a contiguous numpy array at query time, passes to `fuzzy.ea batch_cosine()`. With thousands of observations this is sub-millisecond.

---

## Injection System

### Preamble

Two fixed sections stored as text files in `~/.eabrain/preamble/`:

1. `principles.md` — working principles (think before coding, simplicity first, surgical changes, goal-driven execution)
2. `hard_rules.md` — hard rules (500 line limit, tested features, no stubs, no premature features, delete don't comment)

Stored as config files, not hardcoded. Editable by the user.

### inject command

`eabrain inject [--project DIR] [--budget N]`

Output order (always):
1. Fixed preamble: principles.md (full text)
2. Fixed preamble: hard_rules.md (full text)
3. Dynamic: last session summary for this project (if exists and recent)
4. Dynamic: top 5 observations by recency for this project
5. Dynamic: top 5 observations by SIMD cosine similarity to project name + recent context

Budget (default ~2000 tokens) applies to the dynamic section only. Preamble is always included in full.

### Session Lifecycle

`eabrain inject` creates a new session row in memory.db and writes the session ID to `~/.eabrain/current_session`. All subsequent commands (`remember`, `store`) read the session ID from this file automatically — no need to pass `--session-id` manually. `store-summary` closes the session (sets ended_at) and removes the file.

If `current_session` already exists when `inject` runs (orphaned session), the old session is marked incomplete before a new one starts.

### Hook Configuration

Added to Claude Code `settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      { "command": "eabrain inject --project $PWD" }
    ],
    "Stop": [
      { "command": "echo 'Summarize this session for eabrain. For each notable decision, bug, architecture choice, or pattern, run: eabrain store \"<observation>\" --type <type>. Then run: eabrain store-summary \"<summary>\"'" }
    ]
  }
}
```

The Stop hook prompts Claude to self-summarize. Claude generates the observations and pipes them to `eabrain store`. No external API calls.

---

## CLI Commands

### Existing (unchanged)

| Command | Description |
|---------|-------------|
| `eabrain index` | Build kernel index from .ea files |
| `eabrain ref <name>` | Ea language reference lookup |
| `eabrain patterns [kernel]` | Autoresearch optimization patterns |
| `eabrain init --project-dir DIR` | Append CLAUDE.md snippet |

### Updated

| Command | Change |
|---------|--------|
| `eabrain status` | Now also shows: observation count, session count, last session date, memory.db size |
| `eabrain search <query>` | Searches kernels AND observations. `--fuzzy` runs SIMD cosine across both. `--kernels-only` / `--memory-only` to narrow scope |
| `eabrain remember <note>` | Now stores as observation (type=note) in memory.db with embedding. Was: text append to index.bin |
| `eabrain recall [--last N]` | Now queries memory.db. Was: reads from index.bin |

### New

| Command | Description |
|---------|-------------|
| `eabrain inject [--project DIR] [--budget N]` | Print preamble + relevant context to stdout. Called by SessionStart hook. |
| `eabrain store <content> --type <type> [--project DIR]` | Store an observation with auto-computed embedding. Session ID read from `~/.eabrain/current_session`. |
| `eabrain store-summary <content>` | Store session summary, close session, remove current_session file. |
| `eabrain timeline [--project DIR] [--last N] [--since DATE]` | Chronological view of sessions + observations, grouped by session. |
| `eabrain serve [--port 37777]` | Start web viewer. |
| `eabrain sync --export <path>` | Copy memory.db to path. |
| `eabrain sync --import <path>` | Merge imported memory.db into local. Dedup by content_hash + timestamp. |
| `eabrain migrate` | One-time: move session notes from index.bin to memory.db. |

---

## Web Viewer

### Server

Python `http.server` subclass in `server.py`. Single-threaded, localhost only.

### API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve HTML file |
| `/api/timeline` | GET | Sessions + observations, params: project, since, last |
| `/api/search` | GET | Search results, params: q, fuzzy (0/1), scope (all/kernels/memory) |
| `/api/observations/<id>` | GET | Full observation + similar observations (SIMD cosine) |
| `/api/stats` | GET | Counts, projects, recent activity |

### Frontend

Single HTML file with embedded CSS and JS. No frameworks.

Layout:
- **Left panel:** timeline grouped by session, collapsible
- **Right panel:** observation detail + "similar observations" (SIMD-powered)
- **Top bar:** search input, triggers fuzzy search
- **Filters:** project dropdown, type dropdown, date range
- **Theme:** dark, Catppuccin palette (matches Olorin)

---

## Sync

### Export

`eabrain sync --export <path>` copies `~/.eabrain/memory.db` to the target path.

### Import

`eabrain sync --import <path>` merges the imported database:

1. Open both databases
2. For each observation in imported DB:
   - Check if content_hash exists in local DB
   - If not, insert (preserving original UUID and timestamps)
3. For each session in imported DB:
   - Check if session ID exists in local DB
   - If not, insert
   - If yes, merge: keep longer summary, union observations

Merge is additive. Never deletes local data.

### Portability

- UUIDs for all IDs (no auto-increment collisions)
- content_hash (SHA-256) for dedup
- ISO 8601 timestamps (timezone-aware)
- Embeddings are deterministic (same content = same embedding)

---

## Migration (v0.1 to v0.2)

`eabrain migrate`:

1. Read session entries from index.bin
2. Create a single "legacy" session in memory.db
3. Convert each note to an observation (type=note)
4. Compute embeddings for each migrated observation
5. Write to memory.db
6. Remove session section from index.bin (kernel records and embeddings stay)

Run once. Idempotent (checks if migration already happened).

---

## Edge Cases

**Empty state (first run):**
- `inject` prints preamble only, no dynamic context
- `search` falls back to kernel-only results
- memory.db created on first `store` or `remember`

**Orphaned sessions (Claude killed mid-session):**
- Session has no ended_at or summary
- Next SessionStart detects orphan, marks incomplete
- Mid-session `remember` observations are preserved

**Budget overflow:**
- Preamble always included in full
- Dynamic section respects token budget
- Recency + SIMD relevance scoring decides what fits
- Overflow items available via `eabrain search` / `eabrain timeline`

**Sync conflicts:**
- UUID + content_hash prevent collisions
- Additive merge, no deletions
- Duplicate content silently skipped

---

## File Structure (new/changed files)

```
eabrain/
  eabrain.py          CHANGED  — new commands: inject, store, store-summary, timeline, serve, sync, migrate
  indexer.py           UNCHANGED
  memory.py            NEW      — SQLite operations
  inject.py            NEW      — preamble loading + dynamic context assembly
  server.py            NEW      — web viewer (http.server + REST API)
  sync.py              NEW      — export/import/merge
  fuzzy.py             UNCHANGED
  scan.py              UNCHANGED
  scan_rust.py         UNCHANGED
  kernels/             UNCHANGED
  reference/           UNCHANGED
  web/
    viewer.html        NEW      — single-file web UI (HTML + CSS + JS)
  tests/
    test_memory.py     NEW      — SQLite operations
    test_inject.py     NEW      — injection assembly + budget
    test_server.py     NEW      — API endpoints
    test_sync.py       NEW      — export/import/merge
    test_timeline.py   NEW      — timeline queries
    test_migrate.py    NEW      — v0.1 to v0.2 migration
```

---

## Dependencies

No new external dependencies. Everything uses Python stdlib + numpy (already required):

- `sqlite3` — stdlib
- `http.server` — stdlib
- `uuid` — stdlib
- `hashlib` — stdlib
- `json` — stdlib
- `numpy` — already in setup.py
- `fuzzy.ea` / `scan.ea` — already compiled

---

## Testing Strategy

Every new module gets a test file. Tests use a temporary memory.db (`:memory:` or tmpdir).

| Test File | Covers |
|-----------|--------|
| test_memory.py | Store, query, timeline, schema creation, embedding storage |
| test_inject.py | Preamble loading, budget enforcement, relevance ranking |
| test_server.py | API endpoints return correct JSON, search integration |
| test_sync.py | Export, import, merge, dedup, UUID collision handling |
| test_timeline.py | Chronological ordering, project filtering, session grouping |
| test_migrate.py | index.bin notes converted to observations, idempotency |
