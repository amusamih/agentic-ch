from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Literal, List

from ch_agent.core.llm import make_openai_client, openai_chat
from ch_agent.core.planner import ToolInfo, plan_tool_calls
from ch_agent.core.multiagent import AgentRole, select_specialist_agent
from ch_agent.core.safety import SafetyPolicy
from ch_agent.core.tools import ToolRegistry
from ch_agent.core.tracing import JsonlTracer


AgentMode = Literal["LLM_ONLY", "TOOL_ONESHOT", "AGENTIC_MULTI"]


@dataclass
class AgentConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 500
    mode: AgentMode = "AGENTIC_MULTI"


class AgentRunner:
    """
    Modes:
      - LLM_ONLY: never uses tools
      - TOOL_ONESHOT: user explicitly specifies a tool (context['oneshot_tool'] + context['oneshot_inputs'])
      - AGENTIC_MULTI: multi-agent routing + planning
          (1) LLM selects specialist agent role
          (2) planner sees only that agent's allowed tools
          (3) planner chooses tool calls
    """

    def __init__(self, registry: ToolRegistry, safety: Optional[SafetyPolicy] = None, config: Optional[AgentConfig] = None):
        self.registry = registry
        self.safety = safety or SafetyPolicy()
        self.config = config or AgentConfig()

    def run(
        self,
        project_root: Path,
        app_name: str,
        user_query: str,
        scenario_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        tracer = JsonlTracer(project_root=project_root, app_name=app_name, scenario_id=scenario_id)
        context = context or {}

        tracer.log("PLAN", {"user_query": user_query, "context_keys": list(context.keys()), "mode": self.config.mode})

        # 1) Safety
        decision = self.safety.evaluate_user_query(user_query)
        tracer.log(
            "SAFETY",
            {
                "decision": decision.decision,
                "reason": decision.reason,
                "user_message": decision.user_message,
                "recommended_next_step": decision.recommended_next_step,
            },
        )
        if decision.decision in ("REFUSE", "ESCALATE"):
            final = decision.user_message
            if decision.recommended_next_step:
                final += " " + decision.recommended_next_step
            tracer.log("OUTPUT", {"final_answer": final})
            return {"run_id": tracer.run_id, "final_answer": final, "used_tools": [], "tool_results": []}

        # 2) Decide tool calls
        tool_calls: List[Dict[str, Any]] = []
        selected_agent: Optional[str] = None
        allowed_tools: List[str] = []

        if self.config.mode == "LLM_ONLY":
            tool_calls = []

        elif self.config.mode == "TOOL_ONESHOT":
            tname = context.get("oneshot_tool")
            tinputs = context.get("oneshot_inputs", {})
            if tname and isinstance(tinputs, dict):
                tool_calls = [{"tool_name": tname, "inputs": tinputs}]
            tracer.log("PLAN", {"oneshot_tool_calls": tool_calls})

        elif self.config.mode == "AGENTIC_MULTI":
            roles = self._agent_roles()
            sel = select_specialist_agent(
                user_query=user_query,
                context=context,
                roles=roles,
                model=self.config.model,
            )
            selected_agent = sel.get("selected_agent")
            allowed_tools = sel.get("allowed_tools", [])
            tracer.log("PLAN", {"selected_agent": selected_agent, "allowed_tools": allowed_tools, "selector": sel})

            tools_info = self._tools_for_planner(allowed_tools=allowed_tools)
            planner_plan = plan_tool_calls(user_query=user_query, context=context, tools=tools_info, model=self.config.model)
            tracer.log("PLAN", {"planner_plan": planner_plan})
            tool_calls = planner_plan.get("tool_calls", [])

        
        # --- Plan repair (policy): if user asks for a visit brief, ensure we call the dedicated brief tool ---
        if self.config.mode == "AGENTIC_MULTI" and selected_agent == "VisitPrepAgent":
            ql = (user_query or "").lower()
            wants_brief = any(k in ql for k in ["visit brief", "doctor appointment", "appointment", "visit prep", "visit preparation"])
            if wants_brief:
                has_parse = any(c.get("tool_name") == "parse_phr_bundle" for c in tool_calls)
                has_brief = any(c.get("tool_name") == "generate_visit_brief_from_parsed" for c in tool_calls)
                if (not has_brief) and ("generate_visit_brief_from_parsed" in self.registry.get_tool_specs()):
                    # Ensure parsing happens first
                    if not has_parse and ("parse_phr_bundle" in self.registry.get_tool_specs()):
                        tool_calls.insert(0, {"tool_name": "parse_phr_bundle", "inputs": {"dummy": "ok"}})
                    # Add the brief tool call with empty inputs; AgentRunner will auto-fill visit_reason/parsed_phr
                    tool_calls.append({"tool_name": "generate_visit_brief_from_parsed", "inputs": {}})
                    tracer.log("CHECK", {"plan_repair": "added generate_visit_brief_from_parsed for visit brief request"})

        
        # --- Plan repair (policy): for meds requests, load saved profile when needed ---
        if self.config.mode == "AGENTIC_MULTI" and selected_agent == "MedSafetyAgent":
            ql = (user_query or "").lower()
            wants_meds = any(k in ql for k in ["medication", "medications", "meds", "supplement", "supplements", "interaction"])
            wants_guidance = any(k in ql for k in ["what to do", "recommend", "next step", "next steps", "should i do", "what should i do"])
            has_context_meds = bool(context.get("med_list"))
            has_profile = any(c.get("tool_name") == "load_med_profile" for c in tool_calls)
            has_check = any(c.get("tool_name") == "check_interactions" for c in tool_calls)
            has_guidance = any(c.get("tool_name") == "retrieve_meds_guidance" for c in tool_calls)

            if wants_meds and (not has_context_meds) and ("load_med_profile" in self.registry.get_tool_specs()):
                if not has_profile:
                    tool_calls.insert(0, {"tool_name": "load_med_profile", "inputs": {"dummy": "ok"}})
                    tracer.log("CHECK", {"plan_repair": "added load_med_profile for meds request"})
                if not has_check and ("check_interactions" in self.registry.get_tool_specs()):
                    tool_calls.append({"tool_name": "check_interactions", "inputs": {}})

            if wants_guidance and ("retrieve_meds_guidance" in self.registry.get_tool_specs()) and not has_guidance:
                tool_calls.append({"tool_name": "retrieve_meds_guidance", "inputs": {"user_query": user_query}})
                tracer.log("CHECK", {"plan_repair": "added retrieve_meds_guidance for meds next-step request"})

        
        # --- Plan repair (policy): for sleep requests, ensure structured analysis before guidance ---
        if self.config.mode == "AGENTIC_MULTI" and selected_agent == "SleepAgent":
            ql = (user_query or "").lower()
            wants_analysis = any(k in ql for k in ["my sleep", "last two weeks", "my data", "analyze", "pattern", "insight", "disrupted"])
            wants_guidance = any(k in ql for k in ["suggest", "guidance", "improve", "recommend"])
            has_series = any(c.get("tool_name") == "get_sleep_series" for c in tool_calls)
            has_analysis = any(c.get("tool_name") == "analyze_sleep_patterns" for c in tool_calls)
            has_guidance = any(c.get("tool_name") == "retrieve_sleep_guidance" for c in tool_calls)

            if wants_analysis:
                if not has_series and ("get_sleep_series" in self.registry.get_tool_specs()):
                    tool_calls.insert(0, {"tool_name": "get_sleep_series", "inputs": {"user_id": context.get("user_id", "U1"), "days": 14}})
                if not has_analysis and ("analyze_sleep_patterns" in self.registry.get_tool_specs()):
                    # place after get_sleep_series if present
                    idx = 1 if tool_calls and tool_calls[0].get("tool_name") == "get_sleep_series" else len(tool_calls)
                    tool_calls.insert(idx, {"tool_name": "analyze_sleep_patterns", "inputs": {"user_query": user_query}})
                    tracer.log("CHECK", {"plan_repair": "added analyze_sleep_patterns for sleep analysis request"})
            if wants_guidance and ("retrieve_sleep_guidance" in self.registry.get_tool_specs()) and not has_guidance:
                tool_calls.append({"tool_name": "retrieve_sleep_guidance", "inputs": {"user_query": user_query}})
                tracer.log("CHECK", {"plan_repair": "added retrieve_sleep_guidance for sleep guidance request"})

        # --- Plan repair (policy): for visit requests, add priorities before brief generation ---
        if self.config.mode == "AGENTIC_MULTI" and selected_agent == "VisitPrepAgent":
            ql = (user_query or "").lower()
            wants_brief = any(k in ql for k in ["visit brief", "doctor appointment", "appointment", "visit prep", "visit preparation"])
            if wants_brief:
                has_parse = any(c.get("tool_name") == "parse_phr_bundle" for c in tool_calls)
                has_priorities = any(c.get("tool_name") == "extract_visit_priorities" for c in tool_calls)
                has_brief = any(c.get("tool_name") == "generate_visit_brief_from_parsed" for c in tool_calls)
                if not has_parse and ("parse_phr_bundle" in self.registry.get_tool_specs()):
                    tool_calls.insert(0, {"tool_name": "parse_phr_bundle", "inputs": {"dummy": "ok"}})
                if not has_priorities and ("extract_visit_priorities" in self.registry.get_tool_specs()):
                    idx = 1 if tool_calls and tool_calls[0].get("tool_name") == "parse_phr_bundle" else len(tool_calls)
                    tool_calls.insert(idx, {"tool_name": "extract_visit_priorities", "inputs": {}})
                    tracer.log("CHECK", {"plan_repair": "added extract_visit_priorities for visit brief request"})
                if not has_brief and ("generate_visit_brief_from_parsed" in self.registry.get_tool_specs()):
                    tool_calls.append({"tool_name": "generate_visit_brief_from_parsed", "inputs": {}})
                    tracer.log("CHECK", {"plan_repair": "added generate_visit_brief_from_parsed for visit brief request"})

        # 3) Execute tool calls with explicit auto-wiring
        tool_results = []
        last_outputs: Dict[str, Dict[str, Any]] = {}

        for call in tool_calls:
            tname = call["tool_name"]
            inputs = dict(call.get("inputs", {}) or {})

            # ---- Visit-prep auto-wiring ----
            if tname == "generate_visit_brief_from_parsed":
                # Ensure required fields exist BEFORE validation
                if not inputs.get("visit_reason"):
                    inputs["visit_reason"] = context.get("visit_reason") or user_query
                if inputs.get("patient_goals") is None:
                    inputs["patient_goals"] = context.get("patient_goals") or ""
                if not inputs.get("parsed_phr") and "parse_phr_bundle" in last_outputs:
                    inputs["parsed_phr"] = last_outputs["parse_phr_bundle"]
                if not inputs.get("priorities") and "extract_visit_priorities" in last_outputs:
                    inputs["priorities"] = last_outputs["extract_visit_priorities"].get("priorities", [])

                tracer.log(
                    "CHECK",
                    {
                        "auto_wiring": "visit brief inputs ensured",
                        "visit_reason_present": bool(inputs.get("visit_reason")),
                        "has_parsed_phr": bool(inputs.get("parsed_phr")),
                        "has_priorities": bool(inputs.get("priorities")),
                    },
                )

            # ---- Sleep auto-wiring ----
            if tname == "analyze_sleep_patterns":
                if not inputs.get("nights") and "get_sleep_series" in last_outputs:
                    inputs["nights"] = last_outputs["get_sleep_series"].get("nights", [])
                if not inputs.get("user_query"):
                    inputs["user_query"] = user_query
                tracer.log("CHECK", {"sleep_analysis_autowiring": bool(inputs.get("nights"))})

            # ---- Visit priorities auto-wiring ----
            if tname == "extract_visit_priorities":
                if not inputs.get("parsed_phr") and "parse_phr_bundle" in last_outputs:
                    inputs["parsed_phr"] = last_outputs["parse_phr_bundle"]
                tracer.log("CHECK", {"visit_priorities_autowiring": bool(inputs.get("parsed_phr"))})

            # ---- Med-profile auto-wiring ----
            if tname == "check_interactions":
                if not inputs.get("med_list") and "load_med_profile" in last_outputs:
                    inputs["med_list"] = last_outputs["load_med_profile"].get("med_list", [])
                tracer.log("CHECK", {"med_profile_autowiring": bool(inputs.get("med_list"))})

            res = self.registry.run(tname, inputs, tracer=tracer)
            tool_results.append(res.model_dump())

            if res.ok and isinstance(res.outputs, dict):
                last_outputs[tname] = res.outputs

        # 4) Final response grounded in tool outputs
        system_prompt = (
            "You are a consumer-health assistant. You must be helpful, cautious, and avoid diagnosis.\n"
            "Use the provided TOOL_OUTPUTS as the basis for your response.\n"
            "If TOOL_OUTPUTS are empty, answer generally.\n"
            "Never invent measurements or cite evidence that is not in TOOL_OUTPUTS.\n"
        )
        user_prompt = self._build_user_prompt(user_query, tool_results)

        tracer.log("CHECK", {"system_prompt": system_prompt, "user_prompt_preview": user_prompt[:320], "selected_agent": selected_agent})

        client = make_openai_client()
        reply = openai_chat(
            client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        tracer.log("OBSERVATION", {"model": reply.model, "usage": reply.usage, "text": reply.text})
        tracer.log("OUTPUT", {"final_answer": reply.text})

        return {
            "run_id": tracer.run_id,
            "final_answer": reply.text,
            "selected_agent": selected_agent,
            "allowed_tools": allowed_tools,
            "used_tools": [c["tool_name"] for c in tool_calls],
            "tool_results": tool_results,
        }

    def _agent_roles(self) -> List[AgentRole]:
        return [
            AgentRole(
                name="SleepAgent",
                description="Handles sleep/recovery coaching using wearable sleep data and sleep guidance knowledge.",
                allowed_tools=["get_sleep_series", "analyze_sleep_patterns", "retrieve_sleep_guidance"],
            ),
            AgentRole(
                name="MedSafetyAgent",
                description="Handles medication/supplement interaction checks and safety-focused guidance.",
                allowed_tools=["load_med_profile", "check_interactions", "retrieve_meds_guidance"],
            ),
            AgentRole(
                name="VisitPrepAgent",
                description="Handles visit preparation using PHR parsing and visit-brief generation.",
                allowed_tools=["parse_phr_bundle", "extract_visit_priorities", "generate_visit_brief_from_parsed"],
            ),
        ]

    def _tools_for_planner(self, allowed_tools: List[str]) -> list[ToolInfo]:
        infos: list[ToolInfo] = []
        allowed_set = set(allowed_tools)

        for name, spec in self.registry.get_tool_specs().items():
            if allowed_set and name not in allowed_set:
                continue

            schema = spec.input_model.model_json_schema()
            props = schema.get("properties", {}) or {}
            required = schema.get("required", []) or []
            hint_obj = {k: props.get(k, {}).get("type", "any") for k in props.keys()}
            hint = {"required": required, "fields": hint_obj}

            infos.append(ToolInfo(name=name, description=spec.description, input_schema_hint=str(hint)))

        return infos

    def _build_user_prompt(self, user_query: str, tool_results: list[dict]) -> str:
        return f"USER_QUERY:\n{user_query}\n\nTOOL_OUTPUTS:\n{tool_results}\n"
