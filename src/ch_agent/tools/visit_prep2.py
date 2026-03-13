from __future__ import annotations

from typing import Any, Dict, Optional

from ch_agent.core.llm import make_openai_client, openai_chat
from ch_agent.core.schemas import Evidence, ToolResult
from ch_agent.core.tracing import utc_now_iso


def generate_visit_brief_from_parsed_tool(
    parsed_phr: Dict[str, Any],
    visit_reason: str,
    patient_goals: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> ToolResult:
    """
    Generate visit brief from already-parsed PHR output.
    This makes multi-step traces explicit: parse -> brief.
    """

    counts = parsed_phr.get("counts", {})
    patient = parsed_phr.get("patient", {})
    problems = parsed_phr.get("problems", [])
    meds = parsed_phr.get("medications", [])
    allergies = parsed_phr.get("allergies", [])
    recent_labs = parsed_phr.get("recent_labs", [])
    recent_vitals = parsed_phr.get("recent_vitals", [])
    recent_encounters = parsed_phr.get("recent_encounters", [])
    recent_symptoms = parsed_phr.get("recent_symptoms", [])
    timeline = parsed_phr.get("timeline", [])

    system_prompt = (
        "You are a consumer-health assistant helping a user prepare for a clinician visit.\n"
        "You MUST NOT diagnose or recommend medication changes.\n"
        "You MUST use only the provided PHR_DATA and avoid inventing numbers.\n"
        "Output MUST be in Markdown.\n"
        "Provide:\n"
        "  (1) A concise one-page 'Visit Brief' with clear headings.\n"
        "  (2) A prioritized question list (5-10 questions).\n"
        "  (3) A short 'Safety & Scope' note.\n"
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

    # Evidence here points to "parsed PHR" rather than original file.
    ev = Evidence(
        source_type="derived",
        source_id="parse_phr_bundle.outputs",
        locator="counts/patient/problems/medications/allergies/labs/vitals/encounters/symptom_diary/timeline",
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

    return ToolResult(tool_name="generate_visit_brief_from_parsed", outputs=outputs, evidence=[ev], ok=True)
