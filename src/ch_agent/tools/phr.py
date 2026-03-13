from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ch_agent.core.schemas import Evidence, ToolResult
from ch_agent.core.tracing import utc_now_iso


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"PHR bundle not found: {path}")
    # Use utf-8-sig just in case BOM appears in other files later
    txt = path.read_text(encoding="utf-8-sig")
    return json.loads(txt)


def _add_event(events: List[Dict[str, Any]], date_str: str, kind: str, title: str, details: Dict[str, Any]) -> None:
    events.append(
        {
            "date": date_str,
            "kind": kind,
            "title": title,
            "details": details,
        }
    )


def build_timeline(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    for p in bundle.get("problems", []):
        _add_event(events, p.get("onset_date", ""), "problem", f"Problem: {p.get('name','')}", p)

    for m in bundle.get("medications", []):
        _add_event(events, m.get("start_date", ""), "medication", f"Medication started: {m.get('name','')}", m)

    for e in bundle.get("encounters", []):
        _add_event(events, e.get("date", ""), "encounter", f"Encounter: {e.get('type','')}", e)

    for l in bundle.get("labs", []):
        _add_event(events, l.get("date", ""), "lab", f"Lab: {l.get('test','')}", l)

    for v in bundle.get("vitals", []):
        _add_event(events, v.get("date", ""), "vital", f"Vital: {v.get('type','')}", v)

    for s in bundle.get("symptom_diary", []):
        _add_event(events, s.get("date", ""), "symptom", f"Symptom: {s.get('symptom','')}", s)

    # Sort by date string (ISO dates sort lexicographically)
    events = [ev for ev in events if ev.get("date")]
    events.sort(key=lambda x: x["date"])

    return events


def parse_phr_bundle_tool(bundle_path: Path) -> ToolResult:
    bundle = _read_json(bundle_path)

    patient = bundle.get("patient", {})
    problems = bundle.get("problems", [])
    medications = bundle.get("medications", [])
    allergies = bundle.get("allergies", [])
    labs = bundle.get("labs", [])
    vitals = bundle.get("vitals", [])
    encounters = bundle.get("encounters", [])
    symptom_diary = bundle.get("symptom_diary", [])

    timeline = build_timeline(bundle)

    outputs = {
        "patient": patient,
        "counts": {
            "problems": len(problems),
            "medications": len(medications),
            "allergies": len(allergies),
            "labs": len(labs),
            "vitals": len(vitals),
            "encounters": len(encounters),
            "symptom_diary": len(symptom_diary),
            "timeline_events": len(timeline),
        },
        "problems": problems,
        "medications": medications,
        "allergies": allergies,
        "recent_labs": sorted(labs, key=lambda x: x.get("date",""))[-4:],
        "recent_vitals": sorted(vitals, key=lambda x: x.get("date",""))[-6:],
        "recent_encounters": sorted(encounters, key=lambda x: x.get("date",""))[-3:],
        "recent_symptoms": sorted(symptom_diary, key=lambda x: x.get("date",""))[-5:],
        "timeline": timeline[-40:],  # cap for UI/trace readability
    }

    ev = Evidence(
        source_type="phr_json",
        source_id=str(bundle_path),
        locator="top-level keys: patient/problems/medications/labs/vitals/encounters/symptom_diary",
        snippet=f"timeline_events={len(timeline)} problems={len(problems)} meds={len(medications)}",
        retrieved_at_utc=utc_now_iso(),
    )

    return ToolResult(tool_name="parse_phr_bundle", outputs=outputs, evidence=[ev], ok=True)
