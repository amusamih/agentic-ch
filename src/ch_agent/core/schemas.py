from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


StepType = Literal["PLAN", "TOOL_CALL", "OBSERVATION", "CHECK", "SAFETY", "OUTPUT"]


class Evidence(BaseModel):
    """
    Minimal grounding metadata. Every tool should return evidence even if synthetic/local.
    """
    source_type: str = Field(..., examples=["wearable_csv", "local_kb", "interaction_kb", "phr_json"])
    source_id: str = Field(..., examples=["data/synthetic/wearable_sleep.csv", "knowledge/sleep_guidance.md"])
    locator: Optional[str] = Field(None, description="Row range, JSON pointer, section id, etc.")
    snippet: Optional[str] = Field(None, description="Small excerpt for display/debug (keep short).")
    retrieved_at_utc: Optional[str] = Field(None, description="UTC timestamp when evidence was retrieved.")


class ToolCall(BaseModel):
    tool_name: str
    inputs: Dict[str, Any]


class ToolResult(BaseModel):
    tool_name: str
    outputs: Dict[str, Any]
    evidence: List[Evidence] = Field(default_factory=list)
    ok: bool = True
    error: Optional[str] = None


class AgentOutput(BaseModel):
    text: str
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[Evidence] = Field(default_factory=list)
