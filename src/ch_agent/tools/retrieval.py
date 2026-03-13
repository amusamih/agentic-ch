from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ch_agent.core.schemas import Evidence, ToolResult
from ch_agent.core.tracing import utc_now_iso


def _simple_relevance_sections(text: str, query: str) -> List[str]:
    """
    Very simple, deterministic retrieval:
    - Split by headings '##'
    - Score by keyword overlap
    - Return top sections
    """
    q = query.lower()
    sections = text.split("## ")
    scored = []

    keywords = [w for w in q.replace("?", " ").replace(".", " ").split() if len(w) >= 4]
    keywords = list(dict.fromkeys(keywords))[:12]  # unique, capped

    for sec in sections:
        if not sec.strip():
            continue
        sec_lower = sec.lower()
        score = 0
        for kw in keywords:
            if kw in sec_lower:
                score += 1
        # small boost if "sleep" appears
        if "sleep" in sec_lower:
            score += 1
        scored.append((score, sec.strip()))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [s for score, s in scored if score > 0][:3]
    if not top:
        # fallback: return first two sections after title
        top = [s for _, s in scored[:2]]

    # Keep sections short-ish
    out = []
    for s in top:
        # cap lines
        lines = s.splitlines()
        out.append("\n".join(lines[:18]).strip())
    return out


def retrieve_sleep_guidance_tool(kb_path: Path, user_query: str) -> ToolResult:
    if not kb_path.exists():
        return ToolResult(tool_name="retrieve_sleep_guidance", outputs={}, ok=False, error=f"KB not found: {kb_path}")

    text = kb_path.read_text(encoding="utf-8-sig")
    snippets = _simple_relevance_sections(text, user_query)

    ev = Evidence(
        source_type="local_kb",
        source_id=str(kb_path),
        locator="markdown sections",
        snippet=f"snippets_returned={len(snippets)}",
        retrieved_at_utc=utc_now_iso(),
    )

    outputs = {"query": user_query, "snippets": snippets}
    return ToolResult(tool_name="retrieve_sleep_guidance", outputs=outputs, evidence=[ev], ok=True)
