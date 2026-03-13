from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from pydantic import BaseModel, Field

from ch_agent.core.agent import AgentRunner, AgentConfig
from ch_agent.core.llm import load_env
from ch_agent.core.tools import ToolRegistry, ToolSpec

from ch_agent.tools.wearable import get_sleep_series_tool
from ch_agent.tools.sleep_analysis import analyze_sleep_patterns_tool
from ch_agent.tools.retrieval import retrieve_sleep_guidance_tool
from ch_agent.tools.meds import check_interactions_tool
from ch_agent.tools.retrieval_meds import retrieve_meds_guidance_tool
from ch_agent.tools.med_profile import load_med_profile_tool
from ch_agent.tools.phr import parse_phr_bundle_tool
from ch_agent.tools.visit_prep2 import generate_visit_brief_from_parsed_tool
from ch_agent.tools.visit_priorities import extract_visit_priorities_tool


app = typer.Typer(add_completion=False)


class SleepInputs(BaseModel):
    user_id: str
    days: int = Field(..., ge=1, le=90)


class SleepGuidanceInputs(BaseModel):
    user_query: str = Field(..., min_length=3)


class SleepAnalysisInputs(BaseModel):
    nights: Optional[List[Dict[str, Any]]] = None
    user_query: str = ""


class InteractionInputs(BaseModel):
    med_list: List[str]


class LoadMedProfileInputs(BaseModel):
    dummy: str = "ok"


class MedGuidanceInputs(BaseModel):
    user_query: str = Field(..., min_length=3)


class ParseInputs(BaseModel):
    dummy: str = "ok"


class VisitPrioritiesInputs(BaseModel):
    parsed_phr: Optional[Dict[str, Any]] = None


class BriefFromParsedInputs(BaseModel):
    visit_reason: Optional[str] = None  # auto-filled by AgentRunner if missing
    patient_goals: str = ""
    parsed_phr: Optional[Dict[str, Any]] = None
    priorities: Optional[List[Dict[str, Any]]] = None


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
                handler=lambda inp: get_sleep_series_tool(csv_path=csv_path, user_id=inp.user_id, days=inp.days),
            )
        )

        reg.register(
            ToolSpec(
                name="analyze_sleep_patterns",
                description="Analyze retrieved sleep nights to detect variability, short nights, and notable patterns.",
                input_model=SleepAnalysisInputs,
                handler=lambda inp: analyze_sleep_patterns_tool(
                    nights=inp.nights,
                    user_query=inp.user_query,
                ),
            )
        )

        reg.register(
            ToolSpec(
                name="retrieve_sleep_guidance",
                description="Retrieve sleep guidance snippets from local KB.",
                input_model=SleepGuidanceInputs,
                handler=lambda inp: retrieve_sleep_guidance_tool(kb_path=kb_path, user_query=inp.user_query),
            )
        )

    elif app_name == "meds":
        kb_path = project_root / "data" / "knowledge" / "med_interactions_minikb.json"
        reg.register(
            ToolSpec(
                name="check_interactions",
                description="Check medication/supplement interactions using a local KB.",
                input_model=InteractionInputs,
                handler=lambda inp: check_interactions_tool(kb_path=kb_path, med_list=inp.med_list),
            )
        )



        reg.register(
            ToolSpec(
                name="load_med_profile",
                description="Load the user's saved medication/supplement profile from a local synthetic dataset.",
                input_model=LoadMedProfileInputs,
                handler=lambda _inp: load_med_profile_tool(
                    profile_path=(project_root / "data" / "synthetic" / "med_profile.json")
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
                name="extract_visit_priorities",
                description="Extract the most important visit agenda items from parsed PHR data.",
                input_model=VisitPrioritiesInputs,
                handler=lambda inp: extract_visit_priorities_tool(
                    parsed_phr=inp.parsed_phr or {},
                ),
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
                    priorities=inp.priorities or [],
                    model="gpt-4o-mini",
                ),
            )
        )
    else:
        raise ValueError(f"Unknown app: {app_name}")

    return reg


def preset_query_and_context(app_name: str, paper_demo: bool) -> tuple[str, Dict[str, Any]]:
    if app_name == "sleep":
        if paper_demo:
            q = "Analyze my sleep over the last two weeks and give evidence-based suggestions. Use the sleep guidance knowledge base."
        else:
            q = "Analyze my sleep over the last two weeks and give suggestions. Use the sleep guidance knowledge base if helpful."
        ctx = {"user_id": "U1"}
        return q, ctx

    if app_name == "meds":
        q = "Check interactions between my medications and supplements and tell me what to do."
        ctx = {}
        return q, ctx

    if app_name == "visit":
        if paper_demo:
            q = (
                "I have a doctor appointment soon. Use my PHR records to prepare a visit brief and a prioritized question list. "
                "Parse the PHR first, then use the dedicated visit-brief tool."
            )
        else:
            q = "I have a doctor appointment soon. Use my PHR records to prepare a visit brief and a prioritized question list."
        ctx = {
            "visit_reason": "Follow-up on blood pressure, cholesterol, and recent fatigue/stress.",
            "patient_goals": "Improve sleep and reduce work stress; understand lab trends; ensure medications are appropriate.",
        }
        return q, ctx

    raise ValueError(f"Unknown app: {app_name}")


@app.command()
def run(
    app_name: str = typer.Option(..., "--app", help="sleep | meds | visit"),
    mode: str = typer.Option("agentic_multi", "--mode", help="llm_only | tool_oneshot | agentic_multi"),
    paper_demo: bool = typer.Option(False, "--paper-demo", help="Use curated prompts that reliably trigger intended tool chains"),
    oneshot_tool: Optional[str] = typer.Option(None, "--oneshot-tool", help="Tool name to invoke in tool_oneshot mode"),
    oneshot_inputs_json: Optional[str] = typer.Option(None, "--oneshot-inputs-json", help="JSON dict for tool inputs in tool_oneshot mode"),
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_env(str(project_root))

    app_key = app_name.strip().lower()
    mode_key = mode.strip().lower()

    reg = build_registry(project_root, app_key)
    query, ctx = preset_query_and_context(app_key, paper_demo=paper_demo)

    if mode_key == "llm_only":
        cfg = AgentConfig(mode="LLM_ONLY", model="gpt-4o-mini", max_tokens=450)  # type: ignore

    elif mode_key == "tool_oneshot":
        if not oneshot_tool:
            raise typer.BadParameter("In tool_oneshot mode, you must provide --oneshot-tool.")

        oneshot_inputs: Dict[str, Any] = {}
        if oneshot_inputs_json:
            oneshot_inputs = json.loads(oneshot_inputs_json)

        if not oneshot_inputs:
            if oneshot_tool == "get_sleep_series":
                oneshot_inputs = {"user_id": ctx.get("user_id", "U1"), "days": 14}
            elif oneshot_tool == "check_interactions":
                default_med_list = ctx.get("med_list", [])
                if not default_med_list:
                    prof = project_root / "data" / "synthetic" / "med_profile.json"
                    if prof.exists():
                        profile = json.loads(prof.read_text(encoding="utf-8-sig"))
                        meds = [m.get("name", "") for m in profile.get("active_medications", [])]
                        sups = [s.get("name", "") for s in profile.get("supplements", [])]
                        otc = [o.get("name", "") for o in profile.get("otc_or_as_needed", [])]
                        default_med_list = [x for x in meds + sups + otc if x]
                oneshot_inputs = {"med_list": default_med_list}
            elif oneshot_tool == "parse_phr_bundle":
                oneshot_inputs = {"dummy": "ok"}
            elif oneshot_tool == "generate_visit_brief_from_parsed":
                oneshot_inputs = {
                    "visit_reason": ctx.get("visit_reason", query),
                    "patient_goals": ctx.get("patient_goals", ""),
                }

        ctx = dict(ctx)
        ctx["oneshot_tool"] = oneshot_tool
        ctx["oneshot_inputs"] = oneshot_inputs

        cfg = AgentConfig(mode="TOOL_ONESHOT", model="gpt-4o-mini", max_tokens=450)  # type: ignore

    elif mode_key == "agentic_multi":
        cfg = AgentConfig(mode="AGENTIC_MULTI", model="gpt-4o-mini", max_tokens=600)  # type: ignore

    else:
        raise typer.BadParameter("mode must be: llm_only | tool_oneshot | agentic_multi")

    runner = AgentRunner(registry=reg, config=cfg)

    result = runner.run(
        project_root=project_root,
        app_name=f"demo_{app_key}_{mode_key}",
        user_query=query,
        scenario_id=f"{app_key.upper()}_{mode_key.upper()}",
        context=ctx,
    )

    print("run_id:", result["run_id"])
    print("mode:", mode_key)
    print("app:", app_key)
    print("selected_agent:", result.get("selected_agent"))
    print("used_tools:", result["used_tools"])
    print("\n--- output (preview) ---\n")
    print((result["final_answer"] or "")[:900])


if __name__ == "__main__":
    app()
