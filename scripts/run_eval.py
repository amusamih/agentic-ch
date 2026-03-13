from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
import yaml
from pydantic import BaseModel, Field

from ch_agent.core.agent import AgentRunner, AgentConfig
from ch_agent.core.llm import load_env
from ch_agent.core.safety import SafetyPolicy
from ch_agent.core.tools import ToolRegistry, ToolSpec

from ch_agent.tools.wearable import get_sleep_series_tool
from ch_agent.tools.retrieval import retrieve_sleep_guidance_tool
from ch_agent.tools.meds import check_interactions_tool
from ch_agent.tools.retrieval_meds import retrieve_meds_guidance_tool
from ch_agent.tools.phr import parse_phr_bundle_tool
from ch_agent.tools.visit_prep2 import generate_visit_brief_from_parsed_tool


app = typer.Typer(add_completion=False)


# -------------------------
# Tool input schemas
# -------------------------
class SleepInputs(BaseModel):
    user_id: str
    days: int = Field(..., ge=1, le=90)


class SleepGuidanceInputs(BaseModel):
    user_query: str = Field(..., min_length=3)


class InteractionInputs(BaseModel):
    med_list: List[str]


class MedGuidanceInputs(BaseModel):
    user_query: str = Field(..., min_length=3)


class ParseInputs(BaseModel):
    dummy: str = "ok"


class BriefFromParsedInputs(BaseModel):
    visit_reason: Optional[str] = None  # auto-filled by AgentRunner if missing
    patient_goals: str = ""
    parsed_phr: Optional[Dict[str, Any]] = None  # auto-wired by AgentRunner


def build_registry(project_root: Path, app_name: str) -> ToolRegistry:
    reg = ToolRegistry()

    if app_name == "sleep":
        csv_path = project_root / "data" / "synthetic" / "wearable_sleep.csv"
        kb_path = project_root / "data" / "knowledge" / "sleep_guidance.md"

        reg.register(
            ToolSpec(
                name="get_sleep_series",
                description="Get sleep series from wearable CSV data.",
                input_model=SleepInputs,
                handler=lambda inp: get_sleep_series_tool(
                    csv_path=csv_path, user_id=inp.user_id, days=inp.days
                ),
            )
        )
        reg.register(
            ToolSpec(
                name="retrieve_sleep_guidance",
                description="Retrieve sleep guidance snippets from local KB.",
                input_model=SleepGuidanceInputs,
                handler=lambda inp: retrieve_sleep_guidance_tool(
                    kb_path=kb_path, user_query=inp.user_query
                ),
            )
        )

    elif app_name == "meds":
        kb_path = project_root / "data" / "knowledge" / "med_interactions_minikb.json"
        reg.register(
            ToolSpec(
                name="check_interactions",
                description="Check medication/supplement interactions using a local KB.",
                input_model=InteractionInputs,
                handler=lambda inp: check_interactions_tool(
                    kb_path=kb_path, med_list=inp.med_list
                ),
            )
        )


        reg.register(
            ToolSpec(
                name="retrieve_meds_guidance",
                description="Retrieve medication safety guidance snippets from a local KB (actions, red flags, clinician questions).",
                input_model=MedGuidanceInputs,
                handler=lambda inp: retrieve_meds_guidance_tool(
                    kb_path=(project_root / "data" / "knowledge" / "meds_guidance.md"),
                    user_query=inp.user_query,
                ),
            )
        )

    elif app_name == "visit":
        bundle_path = project_root / "data" / "synthetic" / "phr_bundle.json"

        reg.register(
            ToolSpec(
                name="parse_phr_bundle",
                description="Parse the PHR bundle and return structured summary + timeline.",
                input_model=ParseInputs,
                handler=lambda _inp: parse_phr_bundle_tool(bundle_path=bundle_path),
            )
        )
        reg.register(
            ToolSpec(
                name="generate_visit_brief_from_parsed",
                description=(
                    "Generate a clinician visit-prep brief using parsed PHR output. "
                    "If parsed_phr or visit_reason are omitted, the agent will auto-fill them from parse output / context."
                ),
                input_model=BriefFromParsedInputs,
                handler=lambda inp: generate_visit_brief_from_parsed_tool(
                    parsed_phr=inp.parsed_phr or {},
                    visit_reason=inp.visit_reason,
                    patient_goals=inp.patient_goals,
                    model="gpt-4o-mini",
                ),
            )
        )

    else:
        raise typer.BadParameter("app must be: sleep | meds | visit")

    return reg


def safe_decision_for_query(query: str) -> str:
    return SafetyPolicy().evaluate_user_query(query).decision


def ensure_eval_dir(project_root: Path, eval_id: str) -> Path:
    p = project_root / "runs" / "evals" / eval_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in summary.items():
            w.writerow([k, v])


def tools_match(actual: List[str], scenario: Dict[str, Any], mode_key: str) -> bool:
    any_of = (scenario.get("expected_tools_any_of", {}) or {}).get(mode_key)
    if any_of is not None:
        return any(actual == opt for opt in any_of)

    expected = (scenario.get("expected_tools", {}) or {}).get(mode_key, [])
    return actual == expected


@app.command()
def run(
    app_name: str = typer.Option(..., "--app", help="sleep | meds | visit"),
    mode: str = typer.Option(..., "--mode", help="llm_only | tool_oneshot | agentic_multi"),
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_env(str(project_root))

    app_key = app_name.strip().lower()
    mode_key = mode.strip().lower()

    if app_key not in ("sleep", "meds", "visit"):
        raise typer.BadParameter("app must be: sleep | meds | visit")
    if mode_key not in ("llm_only", "tool_oneshot", "agentic_multi"):
        raise typer.BadParameter("mode must be: llm_only | tool_oneshot | agentic_multi")

    scen_path = project_root / "data" / "scenarios" / f"{app_key}.yaml"
    spec = yaml.safe_load(scen_path.read_text(encoding="utf-8-sig"))
    scenarios = spec.get("scenarios", [])

    reg = build_registry(project_root, app_key)

    if mode_key == "llm_only":
        cfg = AgentConfig(mode="LLM_ONLY", model="gpt-4o-mini", max_tokens=500)  # type: ignore
    elif mode_key == "tool_oneshot":
        cfg = AgentConfig(mode="TOOL_ONESHOT", model="gpt-4o-mini", max_tokens=500)  # type: ignore
    else:
        cfg = AgentConfig(mode="AGENTIC_MULTI", model="gpt-4o-mini", max_tokens=650)  # type: ignore

    runner = AgentRunner(registry=reg, config=cfg)

    eval_id = uuid.uuid4().hex
    out_dir = ensure_eval_dir(project_root, eval_id)

    results: List[Dict[str, Any]] = []
    pass_count = 0

    for s in scenarios:
        sid = s["id"]
        query = s["query"]
        context = s.get("context", {}) or {}

        if mode_key == "tool_oneshot":
            oneshot = s.get("oneshot", {})
            if oneshot:
                context = dict(context)
                context["oneshot_tool"] = oneshot.get("tool")
                context["oneshot_inputs"] = oneshot.get("inputs", {})

        expected_safety = s.get("expected_safety", "ALLOW")
        actual_safety = safe_decision_for_query(query)

        run_res = runner.run(
            project_root=project_root,
            app_name=f"eval_{app_key}_{mode_key}",
            user_query=query,
            scenario_id=sid,
            context=context,
        )

        actual_tools = run_res.get("used_tools", [])
        tools_ok = tools_match(actual_tools, s, mode_key)
        safety_ok = (actual_safety == expected_safety)

        ok = tools_ok and safety_ok
        if ok:
            pass_count += 1

        expected_tools_field = s.get("expected_tools_any_of") or s.get("expected_tools") or {}

        results.append(
            {
                "scenario_id": sid,
                "app": app_key,
                "mode": mode_key,
                "run_id": run_res.get("run_id"),
                "expected_tools_spec": expected_tools_field,
                "actual_tools": actual_tools,
                "tools_ok": tools_ok,
                "expected_safety": expected_safety,
                "actual_safety": actual_safety,
                "safety_ok": safety_ok,
                "pass": ok,
            }
        )

        print(f"{sid} | pass={ok} | tools_ok={tools_ok} | safety_ok={safety_ok} | run_id={run_res.get('run_id')}")

    summary = {
        "eval_id": eval_id,
        "app": app_key,
        "mode": mode_key,
        "total_scenarios": len(results),
        "passed": pass_count,
        "pass_rate": round(pass_count / max(1, len(results)), 3),
        "results_jsonl": str(out_dir / "results.jsonl"),
        "summary_csv": str(out_dir / "summary.csv"),
    }

    write_jsonl(out_dir / "results.jsonl", results)
    write_summary_csv(out_dir / "summary.csv", summary)

    print("\nSaved:")
    print(summary["results_jsonl"])
    print(summary["summary_csv"])
    print("Pass rate:", summary["pass_rate"])


if __name__ == "__main__":
    app()
