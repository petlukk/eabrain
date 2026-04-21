"""inject.py — Preamble loading and context injection for eabrain."""

import os

from memory import MemoryDB


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
