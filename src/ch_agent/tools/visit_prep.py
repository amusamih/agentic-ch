from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ch_agent.core.llm import make_openai_client, openai_chat
from ch_agent.core.schemas import Evidence, ToolResult
from ch_agent.core.tracing import utc_now_iso
from ch_agent.tools.phr import parse_phr_bundle_tool


def generate_visit_brief_tool(
    bundle_path: Path,
    visit_reason: str,
    patient_goals: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> ToolResult:
    """
    Produces a visit-prep brief (Markdown) grounded in the parsed PHR bundle.
    This is NOT diagnosis; it is organization + question generation + trend summarization.
    """

    parsed = parse_phr_bundle_tool(bundle_path=bundle_path)
    if not parsed.ok:
        return ToolResult(tool_name="generate_visit_brief", outputs={}, evidence=parsed.evidence, ok=False, error=parsed.error)

    counts = parsed.outputs.get("counts", {})
    patient = parsed.outputs.get("patient", {})
    problems = parsed.outputs.get("problems", [])
    meds = parsed.outputs.get("medications", [])
    allergies = parsed.outputs.get("allergies", [])
    recent_labs = parsed.outputs.get("recent_labs", [])
    recent_vitals = parsed.outputs.get("recent_vitals", [])
    recent_encounters = parsed.outputs.get("recent_encounters", [])
    recent_symptoms = parsed.outputs.get("recent_symptoms", [])
    timeline = parsed.outputs.get("timeline", [])

    system_prompt = (
        "You are a consumer-health assistant helping a user prepare for a clinician visit.\n"
        "You MUST NOT diagnose or recommend medication changes.\n"
        "You MUST use only the provided PHR_DATA and avoid inventing numbers.\n"
        "Output MUST be in Markdown.\n"
        "Provide:\n"
        "  (1) A concise one-page 'Visit Brief' with clear headings.\n"
        "  (2) A prioritized question list (5-10 questions).\n"
        "  (3) A short 'Safety & Scope' note (what you can/can't do; encourage clinician confirmation).\n"
    )

    user_prompt = (
        f"VISIT_REASON:\n{visit_reason}\n\n"
        f"PATIENT_GOALS (optional):\n{patient_goals or ''}\n\n"
        f"PHR_DATA (structured):\n"
        f"- patient: {patient}\n"
        f"- counts: {counts}\n"
        f"- problems: {problems}\n"
        f"- medications: {meds}\n"
        f"- allergies: {allergies}\n"
        f"- recent_labs: {recent_labs}\n"
        f"- recent_vitals: {recent_vitals}\n"
        f"- recent_encounters: {recent_encounters}\n"
        f"- recent_symptoms: {recent_symptoms}\n"
        f"- timeline (last): {timeline}\n"
    )

    client = make_openai_client()
    reply = openai_chat(
        client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=0.2,
        max_tokens=700,
    )

    ev = Evidence(
        source_type="phr_json",
        source_id=str(bundle_path),
        locator="Used parsed sections: problems/medications/allergies/labs/vitals/encounters/symptom_diary",
        snippet=f"visit_reason={visit_reason} timeline_events_used={len(timeline)}",
        retrieved_at_utc=utc_now_iso(),
    )

    outputs = {
        "visit_reason": visit_reason,
        "patient_goals": patient_goals,
        "visit_brief_markdown": reply.text,
        "llm_model": reply.model,
        "llm_usage": reply.usage,
    }

    return ToolResult(tool_name="generate_visit_brief", outputs=outputs, evidence=[ev], ok=True)
