import json
import os
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunMetadata:
    run_id: str
    app_name: str
    scenario_id: Optional[str]
    created_at_utc: str
    notes: Optional[str] = None


@dataclass
class TraceEvent:
    run_id: str
    ts_utc: str
    step_type: str  # PLAN | TOOL_CALL | OBSERVATION | CHECK | SAFETY | OUTPUT
    payload: Dict[str, Any]


class JsonlTracer:
    """
    Writes structured trace events into:
      runs/demos/<run_id>/trace.jsonl
    plus metadata.json
    """

    def __init__(self, project_root: Path, app_name: str, scenario_id: Optional[str] = None, notes: Optional[str] = None):
        self.project_root = project_root
        self.run_id = uuid.uuid4().hex
        self.run_dir = self.project_root / "runs" / "demos" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.trace_path = self.run_dir / "trace.jsonl"
        self.meta_path = self.run_dir / "metadata.json"

        self.meta = RunMetadata(
            run_id=self.run_id,
            app_name=app_name,
            scenario_id=scenario_id,
            created_at_utc=utc_now_iso(),
            notes=notes,
        )
        self._write_json(self.meta_path, asdict(self.meta))

    def log(self, step_type: str, payload: Dict[str, Any]) -> None:
        evt = TraceEvent(
            run_id=self.run_id,
            ts_utc=utc_now_iso(),
            step_type=step_type,
            payload=payload,
        )
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(evt), ensure_ascii=False) + "\n")

    def _write_json(self, path: Path, obj: Dict[str, Any]) -> None:
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")