from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ch_agent.core.schemas import Evidence, ToolResult
from ch_agent.core.tracing import utc_now_iso


@dataclass
class SleepNight:
    date: str
    sleep_hours: float
    bedtime_local: str
    waketime_local: str
    hrv_rmssd_ms: int
    resting_hr_bpm: int
    notes: str = ""


class CsvSleepDataSource:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Sleep CSV not found: {self.csv_path}")

    def get_last_n_days(self, user_id: str, days: int) -> List[SleepNight]:
        df = pd.read_csv(self.csv_path)
        df = df[df["user_id"] == user_id].copy()
        if df.empty:
            return []

        # Ensure date sorting
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        df = df.tail(days)

        nights: List[SleepNight] = []
        for _, r in df.iterrows():
            nights.append(
                SleepNight(
                    date=r["date"].date().isoformat(),
                    sleep_hours=float(r["sleep_hours"]),
                    bedtime_local=str(r["bedtime_local"]),
                    waketime_local=str(r["waketime_local"]),
                    hrv_rmssd_ms=int(r["hrv_rmssd_ms"]),
                    resting_hr_bpm=int(r["resting_hr_bpm"]),
                    notes=str(r.get("notes", "")) if not pd.isna(r.get("notes", "")) else "",
                )
            )
        return nights


def summarize_sleep(nights: List[SleepNight]) -> Dict[str, float]:
    if not nights:
        return {"avg_sleep_hours": 0.0, "min_sleep_hours": 0.0, "max_sleep_hours": 0.0}

    vals = [n.sleep_hours for n in nights]
    return {
        "avg_sleep_hours": round(sum(vals) / len(vals), 2),
        "min_sleep_hours": round(min(vals), 2),
        "max_sleep_hours": round(max(vals), 2),
    }


def get_sleep_series_tool(csv_path: Path, user_id: str, days: int) -> ToolResult:
    ds = CsvSleepDataSource(csv_path)
    nights = ds.get_last_n_days(user_id=user_id, days=days)
    summary = summarize_sleep(nights)

    evidence = Evidence(
        source_type="wearable_csv",
        source_id=str(csv_path),
        locator=f"user_id={user_id}, last_days={days}",
        snippet=f"rows={len(nights)} avg_sleep_hours={summary.get('avg_sleep_hours')}",
        retrieved_at_utc=utc_now_iso(),
    )

    outputs = {
        "user_id": user_id,
        "days": days,
        "summary": summary,
        "nights": [n.__dict__ for n in nights],
    }

    ok = len(nights) > 0
    err = None if ok else f"No data found for user_id={user_id}"

    return ToolResult(tool_name="get_sleep_series", outputs=outputs, evidence=[evidence], ok=ok, error=err)
