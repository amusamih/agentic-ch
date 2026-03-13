from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from ch_agent.core.llm import make_openai_client, openai_chat


@dataclass
class ToolInfo:
    name: str
    description: str
    input_schema_hint: str  # short hint (JSON-ish)


def plan_tool_calls(
    user_query: str,
    context: Dict[str, Any],
    tools: List[ToolInfo],
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Returns:
      {"tool_calls": [{"tool_name": "...", "inputs": {...}}, ...], ...}

    Used only in AGENTIC_MULTI.
    """
    tool_list_text = "\n".join(
        [f"- {t.name}: {t.description} | inputs: {t.input_schema_hint}" for t in tools]
    )

    system_prompt = (
        "You are a tool-planning assistant.\n"
        "Your job is to choose which tools to call to answer USER_QUERY.\n"
        "Rules:\n- If the user requests a visit brief / visit prep brief and generate_visit_brief_from_parsed is available, plan to call it (typically after parse_phr_bundle).\n"
        "1) Output MUST be valid JSON ONLY (no markdown), with key: tool_calls.\n"
        "2) tool_calls is a list of objects with keys: tool_name, inputs.\n"
        "3) Only choose from AVAILABLE_TOOLS.\n"
        "4) Use values from CONTEXT to fill inputs when needed.\n"
        "5) If no tool is needed, return {\"tool_calls\": []}.\n"
        "6) Be conservative: do NOT call tools unless they add clear value.\n7) Prefer specialized artifact tools when available (e.g., visit brief tool) rather than generating free-form.\n8) If you call check_interactions and the user asks what to do / recommendations / next steps, also call retrieve_meds_guidance.\n"
    )

    user_prompt = (
        f"USER_QUERY:\n{user_query}\n\n"
        f"CONTEXT (values you may use in tool inputs):\n{context}\n\n"
        f"AVAILABLE_TOOLS:\n{tool_list_text}\n\n"
        "Return JSON now."
    )

    client = make_openai_client()
    reply = openai_chat(
        client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=0.0,
        max_tokens=350,
    )

    text = (reply.text or "").strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try extracting first {...}
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(text[start : end + 1])
        else:
            return {"tool_calls": [], "planner_error": "Could not parse JSON from planner output.", "planner_text": text}

    if not isinstance(obj, dict) or "tool_calls" not in obj or not isinstance(obj["tool_calls"], list):
        return {"tool_calls": [], "planner_error": "Planner returned invalid schema.", "planner_raw": obj, "planner_text": text}

    allowed = {t.name for t in tools}
    cleaned = []
    for call in obj["tool_calls"]:
        if not isinstance(call, dict):
            continue
        tn = call.get("tool_name")
        inputs = call.get("inputs", {})
        if tn in allowed and isinstance(inputs, dict):
            cleaned.append({"tool_name": tn, "inputs": inputs})

    return {"tool_calls": cleaned, "planner_raw": obj, "planner_text": text}
