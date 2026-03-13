from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ch_agent.core.llm import make_openai_client, openai_chat


@dataclass
class AgentRole:
    name: str
    description: str
    allowed_tools: List[str]


def select_specialist_agent(
    user_query: str,
    context: Dict[str, Any],
    roles: List[AgentRole],
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    LLM selects a specialist agent given USER_QUERY + CONTEXT + available roles.
    Returns:
      {
        "selected_agent": "SleepAgent",
        "allowed_tools": [...],
        "selector_raw": {...},
        "selector_text": "..."
      }
    """
    roles_text = "\n".join(
        [f"- {r.name}: {r.description} | tools: {r.allowed_tools}" for r in roles]
    )

    system_prompt = (
        "You are an agent-router (specialist selector).\n"
        "Choose exactly ONE specialist agent role that best fits the USER_QUERY.\n"
        "Rules:\n"
        "1) Output MUST be valid JSON ONLY (no markdown).\n"
        "2) JSON MUST have keys: selected_agent.\n"
        "3) selected_agent must be one of the provided roles.\n"
        "4) Be conservative: if unclear, choose the most general/appropriate role.\n"
    )

    user_prompt = (
        f"USER_QUERY:\n{user_query}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"AVAILABLE_ROLES:\n{roles_text}\n\n"
        "Return JSON now."
    )

    client = make_openai_client()
    reply = openai_chat(
        client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=0.0,
        max_tokens=120,
    )

    text = (reply.text or "").strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # fallback: extract {...}
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(text[start : end + 1])
        else:
            return {
                "selected_agent": roles[0].name if roles else "Unknown",
                "allowed_tools": roles[0].allowed_tools if roles else [],
                "selector_error": "Could not parse JSON from selector output.",
                "selector_text": text,
            }

    selected = obj.get("selected_agent")
    allowed_map = {r.name: r.allowed_tools for r in roles}

    if selected not in allowed_map:
        # fallback to first role
        selected = roles[0].name if roles else "Unknown"

    return {
        "selected_agent": selected,
        "allowed_tools": allowed_map.get(selected, []),
        "selector_raw": obj,
        "selector_text": text,
    }
