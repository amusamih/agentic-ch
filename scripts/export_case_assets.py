#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

MODE_ORDER = ["LLM_ONLY", "TOOL_ONESHOT", "AGENTIC_MULTI"]
MODE_LABELS = {
    "LLM_ONLY": "LLM only",
    "TOOL_ONESHOT": "Tool augmented",
    "AGENTIC_MULTI": "Agentic AI",
}

PERSONAL_DATA_TOOLS = {
    "parse_phr_bundle",
    "load_med_profile",
    "get_sleep_series",
}

AGENT_INFERENCE = {
    frozenset({"parse_phr_bundle", "extract_visit_priorities", "generate_visit_brief_from_parsed"}): "VisitPrepAgent",
    frozenset({"load_med_profile", "check_interactions", "retrieve_meds_guidance"}): "MedSafetyAgent",
    frozenset({"get_sleep_series", "analyze_sleep_patterns", "retrieve_sleep_guidance"}): "SleepAgent",
}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def first_nonempty(*values: Any) -> Optional[Any]:
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None


def clean_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def excerpt(text: str, limit: int = 320) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def find_run_dir(project_root: Path, run_id: str) -> Optional[Path]:
    direct = project_root / "runs" / "demos" / run_id
    if direct.exists():
        return direct
    for p in (project_root / "runs").rglob(run_id):
        if p.is_dir():
            return p
    return None


def infer_agent_from_tools(tools: List[str]) -> Optional[str]:
    toolset = frozenset(tools)
    for known_tools, agent in AGENT_INFERENCE.items():
        if known_tools.issubset(toolset):
            return agent
    return None


def classify_output_style(app: str, mode: str, tools: List[str], output_text: str) -> str:
    if app == "visit":
        if "generate_visit_brief_from_parsed" in tools:
            return "Structured visit brief"
        if "parse_phr_bundle" in tools or "extract_visit_priorities" in tools:
            return "Partially structured support"
        return "Generic guidance"
    if app == "meds":
        if "retrieve_meds_guidance" in tools and "check_interactions" in tools:
            return "Grounded safety guidance"
        if "check_interactions" in tools:
            return "Interaction check response"
        return "Generic guidance"
    if app == "sleep":
        if "analyze_sleep_patterns" in tools and "retrieve_sleep_guidance" in tools:
            return "Grounded sleep support"
        if "get_sleep_series" in tools:
            return "Data-based summary"
        return "Generic guidance"
    return "User-facing response"


def extract_from_trace(app: str, mode: str, trace_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    safety = None
    selected_agent = None
    tools: List[str] = []
    final_output = None
    step_counts = {"SAFETY": 0, "TOOL_CALL": 0, "OBSERVATION": 0, "OUTPUT": 0}

    for rec in trace_records:
        step_type = str(rec.get("step_type") or rec.get("type") or "").strip()
        payload = rec.get("payload") or {}
        item = rec.get("item")
        details = rec.get("details")

        if step_type in step_counts:
            step_counts[step_type] += 1

        if not selected_agent:
            selected_agent = first_nonempty(
                payload.get("selected_agent"),
                payload.get("agent_name"),
                payload.get("specialist"),
                payload.get("specialist_agent"),
                payload.get("agent"),
            )

        if step_type == "SAFETY" and not safety:
            safety = first_nonempty(
                payload.get("decision"),
                payload.get("result"),
                payload.get("item"),
                item,
            )

        if step_type == "TOOL_CALL":
            tool_name = first_nonempty(
                payload.get("tool_name"),
                payload.get("name"),
                item,
            )
            if tool_name and tool_name not in tools:
                tools.append(str(tool_name))

        if step_type == "OUTPUT":
            final_output = first_nonempty(
                payload.get("final_answer"),
                payload.get("answer"),
                payload.get("text"),
                payload.get("content"),
                details,
            )
            outputs = payload.get("outputs")
            if not final_output and isinstance(outputs, dict):
                for key in [
                    "final_answer",
                    "visit_brief_markdown",
                    "response",
                    "answer",
                    "output",
                ]:
                    if outputs.get(key):
                        final_output = outputs[key]
                        break

    if not selected_agent:
        selected_agent = infer_agent_from_tools(tools)

    final_output = str(final_output or "").strip()
    output_style = classify_output_style(app, mode, tools, final_output)
    uses_personal_data = any(t in PERSONAL_DATA_TOOLS for t in tools)

    return {
        "safety": safety or "",
        "selected_agent": selected_agent or "",
        "tools_used": tools,
        "tool_count": len(tools),
        "uses_personal_data": uses_personal_data,
        "output_style": output_style,
        "output_excerpt": excerpt(final_output, 320),
        "step_counts": step_counts,
    }


def build_case_summary(project_root: Path, bundle_path: Path) -> Dict[str, Any]:
    bundle = load_json(bundle_path)
    app = bundle.get("app", "")
    prompt = bundle.get("prompt", "")
    runs = bundle.get("runs", {})

    summary: Dict[str, Any] = {
        "bundle": str(bundle_path),
        "app": app,
        "prompt": prompt,
        "modes": {},
    }

    for mode in MODE_ORDER:
        run_id = runs.get(mode)
        if not run_id:
            continue
        run_dir = find_run_dir(project_root, run_id)
        trace_path = run_dir / "trace.jsonl" if run_dir else None
        trace_records = read_jsonl(trace_path) if trace_path else []
        extracted = extract_from_trace(app, mode, trace_records)

        summary["modes"][mode] = {
            "run_id": run_id,
            "run_dir": str(run_dir) if run_dir else "",
            "trace_path": str(trace_path) if trace_path else "",
            **extracted,
        }

    return summary


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def make_chatgpt_text(summary: Dict[str, Any]) -> str:
    lines = []
    lines.append("CASE SUMMARY FOR PAPER WRITE-UP")
    lines.append("")
    lines.append(f"App: {summary.get('app', '')}")
    lines.append(f"Prompt: {summary.get('prompt', '')}")
    lines.append("")

    for mode in MODE_ORDER:
        mode_info = summary["modes"].get(mode)
        if not mode_info:
            continue
        lines.append(f"[{MODE_LABELS[mode]}]")
        lines.append(f"Run ID: {mode_info.get('run_id', '')}")
        lines.append(f"Safety: {mode_info.get('safety', '')}")
        lines.append(f"Selected agent: {mode_info.get('selected_agent', '') or 'None'}")
        tools = mode_info.get("tools_used", [])
        lines.append(f"Tools used: {', '.join(tools) if tools else 'None'}")
        lines.append(f"Output style: {mode_info.get('output_style', '')}")
        lines.append(f"Output excerpt: {mode_info.get('output_excerpt', '')}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def make_case_excerpt_tex(summary: Dict[str, Any]) -> str:
    prompt = latex_escape(summary.get("prompt", ""))
    agentic = summary["modes"].get("AGENTIC_MULTI", {})
    selected_agent = latex_escape(agentic.get("selected_agent", "") or "None")
    tools = agentic.get("tools_used", [])
    tools_text = latex_escape(", ".join(tools) if tools else "None")
    output_excerpt = latex_escape(agentic.get("output_excerpt", ""))

    text = rf"""\noindent\textbf{{Illustrative request.}} {prompt}

\noindent\textbf{{Selected path.}} {selected_agent}

\noindent\textbf{{Tools used.}} {tools_text}

\noindent\textbf{{Output excerpt.}} {output_excerpt}
"""
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper-ready assets from a saved comparison bundle.")
    parser.add_argument("--bundle", required=True, help="Path to comparison JSON file.")
    parser.add_argument("--out", required=True, help="Output directory for generated files.")
    parser.add_argument("--project-root", default=".", help="Project root directory. Default is current directory.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    bundle_path = Path(args.bundle).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_case_summary(project_root, bundle_path)

    json_path = out_dir / "case_summary.json"
    chatgpt_path = out_dir / "case_for_chatgpt.txt"
    excerpt_path = out_dir / "case_excerpt.tex"

    write_json(json_path, summary)
    chatgpt_text = make_chatgpt_text(summary)
    chatgpt_path.write_text(chatgpt_text, encoding="utf-8")
    excerpt_path.write_text(make_case_excerpt_tex(summary), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {chatgpt_path}")
    print(f"Wrote {excerpt_path}")
    print("")
    print("=== COPY FOR CHATGPT ===")
    print(chatgpt_text)


if __name__ == "__main__":
    main()