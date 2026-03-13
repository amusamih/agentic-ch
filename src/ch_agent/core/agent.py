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

                tracer.log(
                    "CHECK",
                    {
                        "auto_wiring": "visit brief inputs ensured",
                        "visit_reason_present": bool(inputs.get("visit_reason")),
                        "has_parsed_phr": bool(inputs.get("parsed_phr")),
                    },
                )

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
                allowed_tools=["get_sleep_series", "retrieve_sleep_guidance"],
            ),
            AgentRole(
                name="MedSafetyAgent",
                description="Handles medication/supplement interaction checks and safety-focused guidance.",
                allowed_tools=["check_interactions", "retrieve_meds_guidance"],
            ),
            AgentRole(
                name="VisitPrepAgent",
                description="Handles visit preparation using PHR parsing and visit-brief generation.",
                allowed_tools=["parse_phr_bundle", "generate_visit_brief_from_parsed"],
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
