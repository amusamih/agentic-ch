from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from ch_agent.core.schemas import ToolResult
from ch_agent.core.tracing import JsonlTracer


InputT = TypeVar("InputT", bound=BaseModel)


@dataclass
class ToolSpec(Generic[InputT]):
    name: str
    description: str
    input_model: Type[InputT]
    handler: Callable[[InputT], ToolResult]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec[Any]] = {}

    def register(self, spec: ToolSpec[Any]) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> Dict[str, str]:
        return {name: spec.description for name, spec in self._tools.items()}

    def get_tool_specs(self) -> Dict[str, ToolSpec[Any]]:
        # Public accessor for planner / UI layers
        return dict(self._tools)

    def run(self, tool_name: str, raw_inputs: Dict[str, Any], tracer: Optional[JsonlTracer] = None) -> ToolResult:
        if tool_name not in self._tools:
            return ToolResult(tool_name=tool_name, outputs={}, ok=False, error=f"Unknown tool: {tool_name}")

        spec = self._tools[tool_name]

        if tracer:
            tracer.log("TOOL_CALL", {"tool_name": tool_name, "inputs": raw_inputs})

        try:
            parsed = spec.input_model(**raw_inputs)
        except ValidationError as e:
            res = ToolResult(tool_name=tool_name, outputs={}, ok=False, error=str(e))
            if tracer:
                tracer.log("OBSERVATION", res.model_dump())
            return res

        try:
            res = spec.handler(parsed)
        except Exception as e:
            res = ToolResult(tool_name=tool_name, outputs={}, ok=False, error=f"{type(e).__name__}: {e}")

        if tracer:
            tracer.log("OBSERVATION", res.model_dump())

        return res
