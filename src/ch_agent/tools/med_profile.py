from __future__ import annotations

from pathlib import Path
from typing import List

from ch_agent.core.schemas import Evidence, ToolResult
from ch_agent.core.tracing import utc_now_iso


def load_med_profile_tool(profile_path: Path) -> ToolResult:
    if not profile_path.exists():
        return ToolResult(tool_name="load_med_profile", outputs={}, ok=False, error=f"Profile not found: {profile_path}")

    import json
    obj = json.loads(profile_path.read_text(encoding="utf-8-sig"))

    meds = [m.get("name", "").strip().lower() for m in obj.get("active_medications", [])]
    sups = [m.get("name", "").strip().lower() for m in obj.get("supplements", [])]
    otc = [m.get("name", "").strip().lower() for m in obj.get("otc_or_as_needed", [])]

    med_list: List[str] = [x for x in meds + sups + otc if x]

    ev = Evidence(
        source_type="synthetic_profile",
        source_id=str(profile_path),
        locator="active_medications/supplements/otc_or_as_needed",
        snippet=f"med_list={med_list}",
        retrieved_at_utc=utc_now_iso(),
    )

    outputs = {
        "profile_id": obj.get("profile_id"),
        "patient_label": obj.get("patient_label"),
        "med_list": med_list,
        "active_medications": obj.get("active_medications", []),
        "supplements": obj.get("supplements", []),
        "otc_or_as_needed": obj.get("otc_or_as_needed", []),
        "allergies": obj.get("allergies", []),
    }

    return ToolResult(tool_name="load_med_profile", outputs=outputs, evidence=[ev], ok=True)
