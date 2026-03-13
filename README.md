# Agentic AI for Consumer Health

Prototype implementation for a paper on **LLM-only, tool-augmented, and agentic AI systems for consumer health**.

## Overview
This repository contains a synthetic-data prototype for comparing three system modes across three consumer-health applications:

- **LLM-only baseline**
- **Tool-augmented baseline (user-directed one-shot)**
- **Agentic system (specialist routing + multi-step tools)**

The prototype emphasizes:
- **consumer-facing, non-diagnostic support**
- **tool grounding and traceability**
- **evaluation through scenario-based benchmarks**
- **safe handling of red-flag and out-of-scope requests**

## Applications
### 1) Sleep coaching
Agentic 3-tool chain:
1. `get_sleep_series`
2. `analyze_sleep_patterns`
3. `retrieve_sleep_guidance`

### 2) Medication safety
Agentic 3-tool chain:
1. `load_med_profile`
2. `check_interactions`
3. `retrieve_meds_guidance`

### 3) Visit preparation
Agentic 3-tool chain:
1. `parse_phr_bundle`
2. `extract_visit_priorities`
3. `generate_visit_brief_from_parsed`

## Project structure
```text
CEMAG/
├── data/
│   ├── knowledge/            # local guidance / safety knowledge bases
│   ├── scenarios/            # evaluation scenario YAML files
│   └── synthetic/            # synthetic personal-data assets
├── runs/                     # local traces, eval outputs, and demo artifacts (gitignored)
├── scripts/
│   ├── run_demo.py           # run demo scenarios
│   └── run_eval.py           # run evaluation scenarios
├── src/
│   └── ch_agent/
│       ├── core/             # agent logic, planning, safety, tracing
│       ├── tools/            # tool implementations
│       └── ui/               # Streamlit UI
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

## Data assets
The prototype currently uses **three synthetic personal-data assets**:
- `wearable_sleep.csv`
- `phr_bundle.json`
- `med_profile.json`

It also uses local synthetic knowledge sources for:
- sleep guidance
- medication interactions
- medication safety guidance

## Setup
Create a virtual environment, install dependencies, and configure API keys in a local `.env` file.

`.env` is **not committed**. Use `.env.example` as a template.

## Run the Streamlit app
```bash
streamlit run src/ch_agent/ui/streamlit_app.py
```

## Run demos
```bash
python scripts/run_demo.py --app sleep --mode agentic_multi
python scripts/run_demo.py --app meds --mode agentic_multi
python scripts/run_demo.py --app visit --mode agentic_multi
```

## Run evaluations
```bash
python scripts/run_eval.py --app sleep --mode agentic_multi
python scripts/run_eval.py --app meds --mode agentic_multi
python scripts/run_eval.py --app visit --mode agentic_multi
```

## Notes
- The current prototype uses **synthetic data only**
- `runs/` contains local artifacts and is excluded from git
- The project is designed so that real datasets or connectors could be added later if desired

## Intended research contribution
This prototype supports a paper studying:
- when consumer-health tasks benefit from tool grounding
- when multi-step agentic orchestration adds value over one-shot tool use
- how safety policies and escalation can be integrated
- how such systems can be evaluated through scenario-based testing
