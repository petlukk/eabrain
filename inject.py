"""inject.py — Preamble loading and context injection for eabrain."""

import os
import uuid

from memory import MemoryDB


_DEFAULT_PRINCIPLES = """\
## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them. Don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?"
If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it. Don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" -> "Write tests for invalid inputs, then make them pass"
- "Fix the bug" -> "Write a test that reproduces it, then make it pass"
- "Refactor X" -> "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]

Strong success criteria let you loop independently.
Weak criteria ("make it work") require constant clarification.
"""

_DEFAULT_HARD_RULES = """\
## Hard Rules

Apply these rules to ALL code:

1. **No file exceeds 500 lines.** Split before you hit the limit.
2. **Every feature proven by end-to-end test.** If it's not tested, it doesn't exist.
3. **No fake functions. No stubs.** No `todo!()`, `// TODO`, `// HACK`, `// for now`, `// hardcoded`, `// placeholder`, `// temporary`. If it doesn't compile and pass tests, it's not code.
4. **No premature features.** Don't build what isn't needed yet.
5. **Delete, don't comment.** Dead code gets removed, not commented out.
"""


def ensure_preamble(preamble_dir: str) -> None:
    if os.path.isdir(preamble_dir) and os.listdir(preamble_dir):
        return
    os.makedirs(preamble_dir, exist_ok=True)
    with open(os.path.join(preamble_dir, "01_principles.md"), "w", encoding="utf-8") as f:
        f.write(_DEFAULT_PRINCIPLES)
    with open(os.path.join(preamble_dir, "02_hard_rules.md"), "w", encoding="utf-8") as f:
        f.write(_DEFAULT_HARD_RULES)


def load_preamble(preamble_dir: str) -> str:
    if not os.path.isdir(preamble_dir):
        return ""
    parts = []
    for name in sorted(os.listdir(preamble_dir)):
        if name.endswith(".md"):
            path = os.path.join(preamble_dir, name)
            with open(path, "r", encoding="utf-8") as f:
                parts.append(f.read().strip())
    return "\n\n".join(parts)


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def build_injection(
    db: MemoryDB,
    preamble_dir: str,
    project: str,
    budget: int = 2000,
) -> str:
    sections = []

    # 1. Fixed preamble (always full)
    ensure_preamble(preamble_dir)
    preamble = load_preamble(preamble_dir)
    if preamble:
        sections.append(preamble)

    # 2. Dynamic context (respects budget)
    dynamic_parts = []
    used = 0

    # Last session summary
    last_sessions = db.timeline(project=project, limit=1)
    if last_sessions and last_sessions[0]["session"].get("summary"):
        summary = last_sessions[0]["session"]["summary"]
        summary_text = f"## Last Session\n{summary}"
        cost = _estimate_tokens(summary_text)
        if used + cost <= budget:
            dynamic_parts.append(summary_text)
            used += cost

    # Recent observations for this project
    recent = db.recent(project=project, limit=10)
    if recent:
        obs_lines = []
        for obs in recent:
            line = f"- [{obs['type']}] {obs['content']}"
            cost = _estimate_tokens(line)
            if used + cost > budget:
                break
            obs_lines.append(line)
            used += cost
        if obs_lines:
            dynamic_parts.append("## Recent Observations\n" + "\n".join(obs_lines))

    if dynamic_parts:
        sections.append("\n\n".join(dynamic_parts))

    return "\n\n".join(sections)


def start_session(db: MemoryDB, project: str, session_file: str) -> str:
    # Check for orphaned session
    old_sid = get_current_session_id(session_file)
    if old_sid:
        db.mark_incomplete(old_sid)

    sid = str(uuid.uuid4())
    os.makedirs(os.path.dirname(os.path.abspath(session_file)), exist_ok=True)
    with open(session_file, "w", encoding="utf-8") as f:
        f.write(sid)
    db.create_session(project=project, session_id=sid)
    return sid


def get_current_session_id(session_file: str) -> str:
    if not os.path.exists(session_file):
        return None
    with open(session_file, "r", encoding="utf-8") as f:
        return f.read().strip() or None


def end_session(db: MemoryDB, session_id: str, summary: str, session_file: str) -> None:
    db.close_session(session_id, summary=summary)
    if os.path.exists(session_file):
        os.remove(session_file)
