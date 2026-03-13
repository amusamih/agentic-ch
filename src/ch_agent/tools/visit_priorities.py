from __future__ import annotations

from typing import Any, Dict, List, Optional

from ch_agent.core.schemas import Evidence, ToolResult
from ch_agent.core.tracing import utc_now_iso


def _lab_priority(lab: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    status = str(lab.get("status") or "").lower()
    if status not in {"high", "low", "abnormal"}:
        return None
    return {
        "category": "lab",
        "title": f"Review abnormal lab: {lab.get('name')}",
        "reason": f"{lab.get('name')} was {lab.get('status')} ({lab.get('value')} {lab.get('unit', '')}) on {lab.get('date')}.",
    }


def _symptom_priority(sym: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sev = sym.get("severity_1to5")
    try:
        sev = int(sev)
    except Exception:
        return None
    if sev < 4:
        return None
    return {
        "category": "symptom",
        "title": f"Discuss symptom burden: {sym.get('symptom')}",
        "reason": f"Reported severity {sev}/5 on {sym.get('date')}.",
    }


def extract_visit_priorities_tool(parsed_phr: Optional[Dict[str, Any]] = None) -> ToolResult:
    parsed_phr = parsed_phr or {}
    if not parsed_phr:
        return ToolResult(tool_name="extract_visit_priorities", outputs={}, ok=False, error="No parsed PHR provided.")

    priorities: List[Dict[str, Any]] = []

    # Active problems
    for p in parsed_phr.get("problems", [])[:3]:
        if str(p.get("status", "active")).lower() == "active":
            priorities.append({
                "category": "problem",
                "title": f"Review active condition: {p.get('name')}",
                "reason": f"Active problem with onset {p.get('onset_date')}.",
            })

    # Abnormal labs
    for lab in parsed_phr.get("recent_labs", []):
        pr = _lab_priority(lab)
        if pr:
            priorities.append(pr)

    # Recent symptoms
    for sym in parsed_phr.get("recent_symptoms", []):
        pr = _symptom_priority(sym)
        if pr:
            priorities.append(pr)

    # Med reconciliation
    meds = parsed_phr.get("medications", [])
    if meds:
        priorities.append({
            "category": "medication",
            "title": "Confirm medication regimen and adherence",
            "reason": f"{len(meds)} listed medications should be reviewed during the visit.",
        })

    # Deduplicate by title
    seen = set()
    uniq = []
    for p in priorities:
        k = p["title"]
        if k not in seen:
            seen.add(k)
            uniq.append(p)

    suggested_questions = []
    for p in uniq[:5]:
        if p["category"] == "lab":
            suggested_questions.append(f"What is driving the abnormal {p['title'].split(': ', 1)[-1].lower()} result, and what follow-up is needed?")
        elif p["category"] == "problem":
            suggested_questions.append(f"What is the current plan for managing {p['title'].split(': ', 1)[-1].lower()}?")
        elif p["category"] == "symptom":
            suggested_questions.append(f"What could be contributing to {p['title'].split(': ', 1)[-1].lower()} and how should it be monitored?")
        elif p["category"] == "medication":
            suggested_questions.append("Are my current medications still appropriate, and do any need adjustment or closer monitoring?")

    outputs = {
        "priorities": uniq[:6],
        "suggested_questions": suggested_questions[:6],
    }

    ev = Evidence(
        source_type="derived_analysis",
        source_id="parse_phr_bundle.outputs",
        locator="problems/recent_labs/recent_symptoms/medications",
        snippet=f"priorities={len(outputs['priorities'])}",
        retrieved_at_utc=utc_now_iso(),
    )

    return ToolResult(tool_name="extract_visit_priorities", outputs=outputs, evidence=[ev], ok=True)
