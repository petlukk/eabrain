# eabrain

A context engine + persistent memory layer for AI coding assistants. It indexes
all your [Ea](https://github.com/petlukk/eacompute) kernel files for instant
function lookups and language reference, and it gives Claude Code (or any AI)
a SQLite-backed memory that survives across sessions.

**The problem:** Every new AI session forgets your codebase. It re-scans files,
re-discovers how the Ea compiler works, and forgets what you decided last week.

**The fix:** `eabrain` keeps two stores:

- **`index.bin`** — searchable index of every `.ea` kernel + Ea language reference
- **`memory.db`** — SQLite of session journals, observations (decisions, bugs,
  architecture notes) tagged by project and timestamp

All searches go through SIMD kernels written in Ea (byte histograms, cosine
similarity, substring matching). Python for everything else.

---

## Quick Start

```bash
git clone git@github.com:petlukk/eabrain.git
cd eabrain

# 1. Build SIMD kernels (needs the Ea compiler — set $EA or put `ea` on PATH)
export EA=/path/to/eacompute/target/release/ea
./build_kernels.sh

# 2. Install
#    On Debian/Ubuntu 23.04+, plain `pip install` is blocked by PEP 668.
#    Either: `pip install -e . --user --break-system-packages`
#        or: `pipx install -e .`
#        or: create a venv first (note: `eabrain` CLI lives inside it, so
#            hooks that call `eabrain` only work when the venv is active).
pip install -e .

# 3. (Optional) Configure
#    By default, eabrain scans the parent directory of this clone — so if
#    you cloned to ~/Dev/eabrain, sibling projects under ~/Dev/ are indexed
#    automatically. Only create a config.json if you want to override:
mkdir -p ~/.eabrain
cat > ~/.eabrain/config.json <<'EOF'
{
  "projects": ["/path/to/your/ea-project", "/path/to/another-project"]
}
EOF

# 4. Build the kernel index
eabrain index

# 5. Check status
eabrain status
```

Configuration defaults: `index_path=~/.eabrain/index.bin`,
`eabrain_dir=~/.eabrain`. The Ea compiler is resolved from `$EA`, then PATH.
`$EACOMPUTE_DIR` overrides the source-tree location used by the intrinsic scraper.

---

## Commands

### Kernel + reference search

```bash
eabrain index                       # rebuild the kernel index
eabrain search <query>              # substring search across kernels and observations
eabrain search <query> --fuzzy      # SIMD cosine similarity (byte histograms)
eabrain search <query> --arch arm   # filter by architecture
eabrain search <query> --kernels-only | --memory-only
eabrain ref <name>                  # Ea language reference (intrinsics, types, flags)
eabrain status                      # one-shot orientation
```

### Persistent memory (v0.2+)

```bash
eabrain inject                      # session start: print preamble + recent context
eabrain remember <note>             # quick observation (type=note)
eabrain store <text> --type {decision|bug|architecture|pattern|error|note}
eabrain store-summary <text>        # session end: close current session with summary
eabrain recall [--last N]           # show recent observations
eabrain timeline [--project P] [--last N] [--since DATE]
eabrain migrate                     # port v0.1 session notes from index.bin → memory.db
```

The session lifecycle is:

```
inject (start)  →  remember/store (during)  →  store-summary (end)
```

`inject` writes a session id to `~/.eabrain/current_session`; `store` and
`remember` link observations to that session; `store-summary` closes it and
removes the marker file. An orphaned session is auto-marked `[incomplete]`
the next time `inject` runs.

### Web viewer

```bash
eabrain serve [--port 37777]
```

Single-file Catppuccin Mocha UI at `web/viewer.html`. Left panel: timeline of
sessions; right panel: observation detail; top bar: search + project/type filters.

### Cross-machine sync

The raw primitives:

```bash
eabrain sync --export /tmp/backup.db    # dump memory.db
eabrain sync --import /tmp/backup.db    # merge by content hash, dedup safely
```

#### Pattern: private memory repo + Claude Code hooks

For continuous sync across machines (instead of manual export/import), create
a **private** git repo for your memory database and wire two scripts into
Claude Code's `SessionStart` / `SessionEnd` hooks. The `eabrain` tool itself
stays public (it's a shared CLI); your cross-project notes stay private.

```bash
# 1. Create an empty PRIVATE GitHub repo for your memory
gh repo create <you>/eabrain-memory --private --clone=false

# 2. Clone it somewhere stable on this machine
git clone https://github.com/<you>/eabrain-memory.git ~/eabrain-memory

# 3. Seed it with an initial export
cd ~/eabrain-memory
eabrain sync --export memory.db
git add memory.db
git commit -m "initial sync"
git push -u origin main
```

Drop these two scripts into the cloned repo:

`~/eabrain-memory/sync-pull.sh`:

```bash
#!/usr/bin/env bash
# SessionStart: pull latest memory.db and import into local eabrain.
# Best-effort — never fails the caller.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO" 2>/dev/null || exit 0
command -v eabrain >/dev/null 2>&1 || exit 0
[ -d "$REPO/.git" ] || exit 0

# --ff-only refuses to overwrite unpushed local commits.
git pull -q --ff-only origin main >/dev/null 2>&1 || exit 0
[ -f "$REPO/memory.db" ] || exit 0
eabrain sync --import "$REPO/memory.db" >/dev/null 2>&1 || exit 0
exit 0
```

`~/eabrain-memory/sync-push.sh`:

```bash
#!/usr/bin/env bash
# SessionEnd: export memory.db and push if it changed.
# Best-effort — never fails the caller.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO" 2>/dev/null || exit 0
command -v eabrain >/dev/null 2>&1 || exit 0
[ -d "$REPO/.git" ] || exit 0

eabrain sync --export "$REPO/memory.db" >/dev/null 2>&1 || exit 0
git add memory.db >/dev/null 2>&1
if ! git diff --cached --quiet 2>/dev/null; then
    git commit -q -m "sync $(date -u +%Y-%m-%dT%H:%M:%SZ)" >/dev/null 2>&1
    git push -q origin main >/dev/null 2>&1
fi
exit 0
```

```bash
chmod +x ~/eabrain-memory/sync-pull.sh ~/eabrain-memory/sync-push.sh
git add sync-pull.sh sync-push.sh
git commit -m "add sync scripts"
git push
```

Wire them into `~/.claude/settings.json` alongside the existing `eabrain
inject` / `store-summary` hooks (see *Wire it into Claude Code* section
below for the base form):

```json
"SessionStart": [{ "hooks": [{ "type": "command",
  "command": "~/eabrain-memory/sync-pull.sh; out=$(eabrain inject --project \"${CLAUDE_PROJECT_DIR:-$PWD}\" 2>/dev/null) && python3 -c '…' \"$out\" || true"
}]}],
"SessionEnd": [{ "hooks": [{ "type": "command",
  "command": "eabrain store-summary \"session ended\" >/dev/null 2>&1; ~/eabrain-memory/sync-push.sh; true"
}]}]
```

On every session:
- **Start**: pull + import (`--ff-only` keeps unpushed local commits safe)
  runs before `eabrain inject`, so the preamble gets the freshest context.
- **End**: `store-summary` archives, then `sync-push.sh` exports + commits +
  pushes — only if the DB changed, so no noise commits.

#### Setting up a second machine

```bash
git clone https://github.com/<you>/eabrain-memory.git <local-path>
chmod +x <local-path>/sync-{pull,push}.sh
eabrain sync --import <local-path>/memory.db
# Edit ~/.claude/settings.json with the hook snippets above,
# substituting <local-path> for ~/eabrain-memory.
```

Both scripts resolve their repo via `BASH_SOURCE`, so the clone location
doesn't have to match across machines — only the hook's absolute path does.

#### Multi-machine caveat

`memory.db` is one SQLite file and `eabrain` has no merge mode. If two
machines modify the DB in parallel between syncs, the second-to-push's
session data is only preserved if the first push was pulled by the
second machine before it exported.

**Use sequentially** (finish on one machine before starting on another)
and the auto-sync keeps them consistent. For true parallel use you'd
need manual conflict resolution — or contributions welcome for a
timestamp-driven merge mode in `sync.py`.

### Autoresearch patterns

If you have the [eacompute](https://github.com/petlukk/eacompute) autoresearch
suite locally, `eabrain patterns` browses proven optimization patterns:

```bash
eabrain patterns                  # list all benchmarks
eabrain patterns --what-works     # summary of winning strategies
eabrain patterns matmul           # deep-dive: strategy space, history, best kernel
```

Override the location with `autoresearch_dir` in config or `$EACOMPUTE_DIR`.

### `eabrain init`

Appends a one-paragraph eabrain snippet to a project's `CLAUDE.md` so the AI
knows the commands are available.

---

## Architecture

```
eabrain.py          CLI (argparse + ctypes dispatch)
indexer.py          Kernel index format + builder (writes index.bin)
memory.py           SQLite storage (sessions, observations, embeddings)
inject.py           Preamble loader + session lifecycle
sync.py             Export/import memory.db with hash dedup
server.py           stdlib http.server, REST API for the viewer
web/viewer.html     Single-file frontend (no deps)
kernels/*.ea        Pure SIMD kernels — fuzzy, scan, scan_rust, substr
reference/          Pre-baked Ea language reference (JSON)
lib/                Compiled .so files + Python bindings (built, not in git)
~/.eabrain/         Config, index.bin, memory.db, current_session, preamble/
```

### The two stores

**`index.bin`** is a flat binary file with a fixed-width header and four sections:
kernel records, ref entries, session journal (legacy v0.1, kept for migration),
and 256-dim byte-histogram embeddings (64-byte aligned for SIMD).

**`memory.db`** is SQLite with two tables:

- `sessions(id, project, started_at, ended_at, summary)`
- `observations(id, session_id, project, type, content, content_hash, embedding, created_at)`

Content hashing makes `eabrain sync --import` idempotent across machines.
Embeddings are stored alongside observations so SIMD cosine search runs against
your memory the same way it runs against kernels.

### Ea kernels

- **`fuzzy.ea`** — `byte_histogram_embed`, `normalize_vectors`, `batch_cosine`
- **`scan.ea`** — `count_exports`, `find_export_offsets`, `detect_simd_types`, `detect_intrinsics`
- **`scan_rust.ea`** — SIMD byte-pair search for the Ea intrinsic scraper
- **`substr.ea`** — generalized SIMD substring search powering `text_search` over observations

All kernels are pure SIMD with scalar tails for alignment. No scalar-only paths.

---

## For AI Assistants

If you're Claude Code (or another AI) reading this:

1. **Session start:** `eabrain inject --project $(pwd)` — emits the preamble
   (principles + hard rules from `~/.eabrain/preamble/`) plus recent context
   for this project, and creates a session
2. **During work:**
   - `eabrain search "name"` before grepping
   - `eabrain ref "intrinsic"` before guessing Ea syntax
   - `eabrain patterns "kernel_type"` before designing a new kernel
   - `eabrain store "decided X because Y" --type decision` to record load-bearing context
3. **Session end:** `eabrain store-summary "what we shipped today"` — closes the
   session and removes the `current_session` marker
4. **After editing .ea files:** `eabrain index`

Use `eabrain timeline --last 10` to recall what happened across recent sessions.

### Wire it into Claude Code (auto-inject + auto-close)

Add this to `~/.claude/settings.json` so Claude Code calls `inject` at session
start and `store-summary` at session end automatically. Both hooks are
best-effort: if `eabrain` isn't on PATH the session continues normally.

The `SessionStart` envelope uses `python3` (always present since eabrain is
a Python tool) rather than `jq`, which is not installed on many systems —
using `jq` here means the hook silently no-ops whenever `jq` is missing.

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "out=$(eabrain inject --project \"${CLAUDE_PROJECT_DIR:-$PWD}\" 2>/dev/null) && python3 -c 'import json,sys; print(json.dumps({\"hookSpecificOutput\":{\"hookEventName\":\"SessionStart\",\"additionalContext\":sys.argv[1]}}))' \"$out\" || true"
          }
        ]
      }
    ],
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "eabrain store-summary \"session ended\" >/dev/null 2>&1 || true"
          }
        ]
      }
    ]
  }
}
```

The `SessionStart` hook prints `inject`'s output as `additionalContext`, so
the preamble + recent observations land in Claude's context window before
your first message. The `SessionEnd` hook closes the open session so the
next `inject` has a real "Last Session" summary to surface.

After editing `~/.claude/settings.json`, open `/hooks` in Claude Code once
to reload the config (the file watcher only watches dirs that already had
a settings file at launch).

---

## Configuration

`~/.eabrain/config.json`:

```json
{
  "projects": ["/path/to/project1", "/path/to/project2"],
  "index_path": "~/.eabrain/index.bin",
  "eabrain_dir": "~/.eabrain",
  "max_source_lines": 50,
  "max_session_entries": 100
}
```

| Field | Default | Description |
|---|---|---|
| `projects` | auto (parent dir of install) | Directories to scan for `.ea` files. If unset, defaults to the parent directory of wherever eabrain is cloned — so sibling repos are indexed automatically. Set to `[]` to disable auto-discovery. |
| `index_path` | `~/.eabrain/index.bin` | Kernel index location |
| `eabrain_dir` | `~/.eabrain` | Where memory.db, current_session, preamble/ live |
| `max_source_lines` | 50 | Inline source for kernels shorter than this |
| `max_session_entries` | 100 | (legacy) max v0.1 session notes |

`EABRAIN_CONFIG=/tmp/test-config.json eabrain ...` overrides the config path.

---

## Development

```bash
./build_kernels.sh              # compile Ea kernels to .so
pip install -e ".[dev]"         # install with pytest
python3 -m pytest tests/ -v     # 83 tests
```

Test coverage spans kernel correctness, binary format roundtrips, indexer
behavior on real project trees, the SQLite memory layer, the inject/session
lifecycle, sync export/import dedup, the REST API, and end-to-end CLI flows.

## License

MIT
