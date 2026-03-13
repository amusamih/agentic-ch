from __future__ import annotations

from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

from ch_agent.core.schemas import Evidence, ToolResult
from ch_agent.core.tracing import utc_now_iso


def analyze_sleep_patterns_tool(nights: Optional[List[Dict[str, Any]]] = None, user_query: str = "") -> ToolResult:
    nights = nights or []
    if not nights:
        return ToolResult(tool_name="analyze_sleep_patterns", outputs={}, ok=False, error="No sleep nights provided.")

    hours = []
    short_nights = []
    best_night = None
    worst_night = None
    flagged_notes = []

    for n in nights:
        h = n.get("sleep_hours")
        if isinstance(h, (int, float)):
            hours.append(float(h))
            if best_night is None or h > best_night.get("sleep_hours", -1):
                best_night = n
            if worst_night is None or h < worst_night.get("sleep_hours", 999):
                worst_night = n
            if h < 6.0:
                short_nights.append({"date": n.get("date"), "sleep_hours": h})

        notes = str(n.get("notes") or "").strip()
        if notes:
            nl = notes.lower()
            if any(k in nl for k in ["stress", "travel", "late", "caffeine", "screen"]):
                flagged_notes.append({"date": n.get("date"), "notes": notes})

    avg_sleep = round(mean(hours), 2) if hours else None
    variability = round(pstdev(hours), 2) if len(hours) > 1 else 0.0

    patterns = []
    if avg_sleep is not None:
        patterns.append(f"Average sleep duration was {avg_sleep} hours.")
    patterns.append(f"Night-to-night variability was {variability} hours.")
    if short_nights:
        patterns.append(f"{len(short_nights)} nights were shorter than 6 hours.")
    if best_night:
        patterns.append(f"Best night: {best_night.get('date')} with {best_night.get('sleep_hours')} hours.")
    if worst_night:
        patterns.append(f"Worst night: {worst_night.get('date')} with {worst_night.get('sleep_hours')} hours.")
    if flagged_notes:
        patterns.append(f"{len(flagged_notes)} nights had contextual notes that may explain disrupted sleep.")

    outputs = {
        "summary": {
            "avg_sleep_hours": avg_sleep,
            "variability_hours": variability,
            "short_night_count": len(short_nights),
        },
        "patterns": patterns,
        "short_nights": short_nights,
        "flagged_notes": flagged_notes,
        "best_night": best_night,
        "worst_night": worst_night,
        "user_query": user_query,
    }

    ev = Evidence(
        source_type="derived_analysis",
        source_id="get_sleep_series.outputs",
        locator="nights",
        snippet=f"nights={len(nights)} avg={avg_sleep} variability={variability}",
        retrieved_at_utc=utc_now_iso(),
    )

    return ToolResult(tool_name="analyze_sleep_patterns", outputs=outputs, evidence=[ev], ok=True)
