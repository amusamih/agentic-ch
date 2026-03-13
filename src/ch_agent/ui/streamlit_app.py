from __future__ import annotations

import json
import re
import subprocess
import sys
import uuid
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def runs_dir(root: Path) -> Path:
    p = root / "runs" / "demos"
    p.mkdir(parents=True, exist_ok=True)
    return p


def comparisons_dir(root: Path) -> Path:
    p = root / "runs" / "comparisons"
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_run_ids(p: Path) -> List[str]:
    return sorted([d.name for d in p.iterdir() if d.is_dir()], reverse=True) if p.exists() else []


def list_comparisons(root: Path) -> List[str]:
    p = comparisons_dir(root)
    return sorted([f.name for f in p.glob("compare_*.json")], reverse=True)


def save_comparison(root: Path, payload: Dict[str, Any]) -> str:
    cid = "compare_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8] + ".json"
    path = comparisons_dir(root) / cid
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return cid


def read_trace(run_dir: Path) -> List[Dict[str, Any]]:
    trace_path = run_dir / "trace.jsonl"
    if not trace_path.exists():
        return []
    events = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def read_metadata(run_dir: Path) -> Dict[str, Any]:
    return read_json(run_dir / "metadata.json")


def extract_user_query(events: List[Dict[str, Any]]) -> str:
    for e in events:
        if e.get("step_type") == "PLAN":
            pl = e.get("payload", {}) or {}
            if "user_query" in pl:
                return str(pl.get("user_query") or "")
    return ""


def summarize_from_trace(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    selected_agent = None
    used_tools: List[str] = []
    final_answer = ""
    safety_payload = None

    for e in events:
        stype = e.get("step_type")
        payload = e.get("payload", {}) or {}

        if stype == "PLAN" and "selected_agent" in payload:
            selected_agent = payload.get("selected_agent")

        if stype == "TOOL_CALL":
            tn = payload.get("tool_name")
            if tn:
                used_tools.append(tn)

        if stype == "SAFETY":
            safety_payload = payload

        if stype == "OUTPUT":
            final_answer = payload.get("final_answer", "") or ""

    return {
        "selected_agent": selected_agent,
        "used_tools": used_tools,
        "final_answer": final_answer,
        "safety": safety_payload,
    }


def run_demo_subprocess(app_name: str, mode: str, paper_demo: bool, oneshot_tool: Optional[str]) -> Dict[str, Any]:
    root = project_root()
    script = root / "scripts" / "run_demo.py"

    cmd = [sys.executable, str(script), "--app", app_name, "--mode", mode]
    if paper_demo:
        cmd.append("--paper-demo")

    if mode == "tool_oneshot":
        if not oneshot_tool:
            return {"ok": False, "error": "tool_oneshot requires a oneshot tool."}
        cmd += ["--oneshot-tool", oneshot_tool]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return {"ok": False, "error": p.stderr or p.stdout or "Unknown error"}

    m = re.search(r"run_id:\s*([0-9a-f]{32})", p.stdout)
    if not m:
        return {"ok": False, "error": "Could not parse run_id from output."}
    return {"ok": True, "run_id": m.group(1)}


def build_mini_trace(events: List[Dict[str, Any]], max_rows: int = 6) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for e in events:
        stype = e.get("step_type")
        pl = e.get("payload", {}) or {}

        if stype == "SAFETY":
            rows.append({"step": "SAFETY", "item": pl.get("decision"), "details": (pl.get("reason", "") or "")[:140]})

        elif stype == "TOOL_CALL":
            rows.append({
                "step": "TOOL_CALL",
                "item": pl.get("tool_name"),
                "details": json.dumps(pl.get("inputs", {}), ensure_ascii=False)[:160],
            })

        elif stype == "OBSERVATION":
            tn = pl.get("tool_name")
            if not tn:
                continue
            ok = pl.get("ok")
            err = pl.get("error")
            rows.append({
                "step": "OBSERVATION",
                "item": f"{tn} (ok={ok})",
                "details": (err or "")[:160],
            })

        elif stype == "OUTPUT":
            fa = (pl.get("final_answer") or "").replace("\n", " ").strip()
            rows.append({"step": "OUTPUT", "item": "final_answer", "details": fa[:160]})

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.head(max_rows)
    return df


def fixed_preview(text: str, lines: int = 10, width: int = 92) -> str:
    raw = " ".join((text or "").split())
    if not raw:
        return "\n" * (lines - 1)
    wrapped = textwrap.wrap(raw, width=width)
    out = wrapped[:lines]
    if len(wrapped) > lines and out:
        out[-1] = (out[-1][: max(0, width - 1)] + "…").rstrip()
    while len(out) < lines:
        out.append("")
    return "\n".join(out)


def default_oneshot_tool_for_app(app_name: str) -> str:
    if app_name == "sleep":
        return "get_sleep_series"
    if app_name == "meds":
        return "check_interactions"
    return "generate_visit_brief_from_parsed"


def pretty_title(mode_label: str) -> str:
    return {
        "LLM_ONLY": "LLM-only baseline (no tools)",
        "TOOL_ONESHOT": "Tool-augmented baseline (user-directed one-shot)",
        "AGENTIC_MULTI": "Agentic system (specialist routing + multi-step tools)",
    }.get(mode_label, mode_label)


def subtitle(mode_label: str) -> str:
    return {
        "LLM_ONLY": "No tools / no personal data access",
        "TOOL_ONESHOT": "User explicitly invokes one tool (consent/intent-driven)",
        "AGENTIC_MULTI": "LLM selects specialist agent + plans tool chain",
    }.get(mode_label, "")


def render_panel(mode_label: str, run_id: str, run_dir: Path) -> None:
    events = read_trace(run_dir)
    meta = read_metadata(run_dir)
    summary = summarize_from_trace(events)

    st.markdown(f"### {pretty_title(mode_label)}")
    st.caption(subtitle(mode_label))

    st.markdown(
        f"**Safety:** `{(summary['safety'] or {}).get('decision')}`  \n"
        f"**Selected specialist agent:** `{summary['selected_agent']}`  \n"
        f"**Tools used:** `{summary['used_tools']}`"
    )

    st.markdown("**Final output (preview)**")
    st.code(fixed_preview(summary["final_answer"], lines=10, width=92), language=None)

    st.markdown("**Instrumented trace (compact)**")
    st.dataframe(build_mini_trace(events, max_rows=6), use_container_width=True, height=220)

    with st.expander("Run details (optional)"):
        st.write({"run_id": run_id, "app_label": meta.get("app_name", ""), "scenario_id": meta.get("scenario_id", "")})


# ---------------- UI ----------------
st.set_page_config(page_title="CEMAG Agentic Consumer Health Demo", layout="wide")
st.title("CEMAG Agentic Consumer Health Demo")

root = project_root()
rdir = runs_dir(root)

with st.sidebar:
    view = st.radio("View", ["Compare modes (paper figure)", "Single run"], index=0)
    st.divider()

    app_name = st.selectbox("App", ["sleep", "meds", "visit"])
    paper_demo = st.checkbox("Curated prompt variant", value=False)

    if view.startswith("Compare"):
        st.info(f"TOOL_ONESHOT baseline: `{default_oneshot_tool_for_app(app_name)}`")
        run_compare = st.button("Run comparison (3 modes)", type="primary")
        st.divider()
        comp_files = list_comparisons(root)
        comp_selected = st.selectbox("Open previous comparison", [""] + comp_files)
    else:
        run_ids = list_run_ids(rdir)
        selected_run = st.selectbox("Open run_id", [""] + run_ids)


# ---- Compare view ----
if view.startswith("Compare"):
    # load old comparison
    if 'comp_selected' in locals() and comp_selected:
        comp_payload = read_json(comparisons_dir(root) / comp_selected)
        st.session_state["compare_results"] = {
            "LLM_ONLY": {"ok": True, "run_id": comp_payload["runs"]["LLM_ONLY"]},
            "TOOL_ONESHOT": {"ok": True, "run_id": comp_payload["runs"]["TOOL_ONESHOT"]},
            "AGENTIC_MULTI": {"ok": True, "run_id": comp_payload["runs"]["AGENTIC_MULTI"]},
        }
        st.session_state["compare_prompt"] = comp_payload.get("prompt", "")

    if 'run_compare' in locals() and run_compare:
        oneshot = default_oneshot_tool_for_app(app_name)
        results = {
            "LLM_ONLY": run_demo_subprocess(app_name, "llm_only", paper_demo, None),
            "TOOL_ONESHOT": run_demo_subprocess(app_name, "tool_oneshot", paper_demo, oneshot),
            "AGENTIC_MULTI": run_demo_subprocess(app_name, "agentic_multi", paper_demo, None),
        }
        st.session_state["compare_results"] = results

        prompt = ""
        if results["LLM_ONLY"].get("ok"):
            prompt = extract_user_query(read_trace(rdir / results["LLM_ONLY"]["run_id"]))
        if not prompt and results["AGENTIC_MULTI"].get("ok"):
            prompt = extract_user_query(read_trace(rdir / results["AGENTIC_MULTI"]["run_id"]))
        st.session_state["compare_prompt"] = prompt

        if all(results[k].get("ok") for k in ["LLM_ONLY", "TOOL_ONESHOT", "AGENTIC_MULTI"]):
            bundle = {
                "app": app_name,
                "paper_demo": paper_demo,
                "prompt": prompt,
                "runs": {k: results[k]["run_id"] for k in ["LLM_ONLY", "TOOL_ONESHOT", "AGENTIC_MULTI"]},
            }
            fname = save_comparison(root, bundle)
            st.success(f"Saved comparison bundle: {fname}")

    results = st.session_state.get("compare_results")
    if not results:
        st.info("Run a comparison or open a previous comparison.")
        st.stop()

    st.subheader("Prompt (same across all modes)")
    st.code(st.session_state.get("compare_prompt") or "(prompt not found)")

    cols = st.columns(3)
    order = ["LLM_ONLY", "TOOL_ONESHOT", "AGENTIC_MULTI"]
    for i, label in enumerate(order):
        with cols[i]:
            rr = results.get(label, {})
            if not rr.get("ok"):
                st.error(rr.get("error", "Run failed"))
                continue
            rid = rr["run_id"]
            render_panel(label, rid, rdir / rid)

# ---- Single run view ----
else:
    if not selected_run:
        st.info("Select a run_id from the sidebar.")
        st.stop()

    rid = selected_run
    run_dir = rdir / rid
    events = read_trace(run_dir)
    meta = read_metadata(run_dir)
    summary = summarize_from_trace(events)

    st.subheader("Run summary")
    st.write({"run_id": rid, "app_label": meta.get("app_name"), "scenario_id": meta.get("scenario_id")})
    st.subheader("Prompt")
    st.code(extract_user_query(events) or "(prompt not found)")
    st.subheader("Final output")
    st.markdown(summary["final_answer"] or "")
    st.subheader("Instrumented trace (compact)")
    st.dataframe(build_mini_trace(events, max_rows=20), use_container_width=True)
