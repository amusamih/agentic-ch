from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ch_agent.core.schemas import Evidence, ToolResult
from ch_agent.core.tracing import utc_now_iso


def _norm(s: str) -> str:
    return " ".join(s.lower().strip().split())


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    a_n = _norm(a)
    b_n = _norm(b)
    return tuple(sorted([a_n, b_n]))  # type: ignore[return-value]


@dataclass
class InteractionHit:
    a: str
    b: str
    severity: str
    summary: str
    recommendation: str


class MedInteractionKB:
    def __init__(self, kb_path: Path):
        self.kb_path = kb_path
        if not self.kb_path.exists():
            raise FileNotFoundError(f"Interaction KB not found: {self.kb_path}")
        self._index: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        obj = json.loads(self.kb_path.read_text(encoding="utf-8-sig"))
        items = obj.get("items", [])
        for it in items:
            pair = it.get("pair", [])
            if len(pair) != 2:
                continue
            key = _pair_key(pair[0], pair[1])
            self._index[key] = it

    def lookup_pairs(self, terms: List[str]) -> List[InteractionHit]:
        terms_n = [_norm(t) for t in terms if t and t.strip()]
        hits: List[InteractionHit] = []
        seen = set()

        for i in range(len(terms_n)):
            for j in range(i + 1, len(terms_n)):
                key = _pair_key(terms_n[i], terms_n[j])
                if key in seen:
                    continue
                seen.add(key)
                it = self._index.get(key)
                if not it:
                    continue
                # Return original normalized term strings for readability
                a, b = key[0], key[1]
                hits.append(
                    InteractionHit(
                        a=a,
                        b=b,
                        severity=it.get("severity", "unknown"),
                        summary=it.get("summary", ""),
                        recommendation=it.get("recommendation", ""),
                    )
                )

        return hits


def check_interactions_tool(kb_path: Path, med_list: List[str]) -> ToolResult:
    kb = MedInteractionKB(kb_path)
    hits = kb.lookup_pairs(med_list)

    outputs = {
        "med_list": [_norm(m) for m in med_list],
        "interactions_found": [
            {
                "a": h.a,
                "b": h.b,
                "severity": h.severity,
                "summary": h.summary,
                "recommendation": h.recommendation,
            }
            for h in hits
        ],
        "count": len(hits),
    }

    # Evidence includes KB file and which pairs matched
    matched_pairs = [f"{h.a} + {h.b}" for h in hits] if hits else []
    ev = Evidence(
        source_type="interaction_kb",
        source_id=str(kb_path),
        locator="items[] (pair matching)",
        snippet=f"matched_pairs={matched_pairs}" if matched_pairs else "matched_pairs=[]",
        retrieved_at_utc=utc_now_iso(),
    )

    return ToolResult(tool_name="check_interactions", outputs=outputs, evidence=[ev], ok=True)
