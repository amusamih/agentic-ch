# CEMAG — Agentic AI for Consumer Health

Prototype implementation for a paper on **Agentic AI and LLMs for Consumer Health**.

## Overview
This repository contains a prototype framework for studying how **LLM-only systems**, **tool-augmented baselines**, and **agentic multi-step systems** can support consumer-health use cases in a safe, traceable, and evaluation-driven way.

## Applications
The current prototype demonstrates three consumer-health applications:

- **Sleep coaching from wearable-style data**
- **Medication interaction safety guidance**
- **Visit preparation from personal health records (PHR)**

## Compared modes
The system supports three comparison modes:

- **LLM-only baseline**  
  No tools are used. The model responds from its general knowledge only.

- **Tool-augmented baseline (user-directed one-shot)**  
  The user explicitly invokes one tool.

- **Agentic system (specialist routing + multi-step tools)**  
  The system selects a specialist role, plans tool use, and can execute multi-step workflows.

## Repository structure
- `src/` — core implementation
- `data/` — synthetic datasets and local knowledge bases
- `scripts/run_demo.py` — run demo scenarios
- `scripts/run_eval.py` — run evaluation scenarios

## Current applications and intended agentic behavior

### 1. Sleep coaching
- Tool baseline: retrieve wearable-style sleep statistics
- Agentic mode: combine personal sleep metrics with retrieved sleep guidance

### 2. Medication safety
- Tool baseline: check medication interactions
- Agentic mode: combine interaction lookup with retrieved medication-safety guidance

### 3. Visit preparation
- Tool baseline: generate a visit brief directly
- Agentic mode: parse PHR data first, then generate a structured visit brief

## Setup
Create a virtual environment, install dependencies, and configure API keys in a local `.env` file.

The `.env` file is **not committed**. Use `.env.example` as a template.

## Example demo commands
```bash
python scripts/run_demo.py --app sleep --mode agentic_multi
python scripts/run_demo.py --app meds --mode agentic_multi
python scripts/run_demo.py --app visit --mode agentic_multi
```

## Example evaluation commands
```bash
python scripts/run_eval.py --app sleep --mode agentic_multi
python scripts/run_eval.py --app meds --mode agentic_multi
python scripts/run_eval.py --app visit --mode agentic_multi
```

## Notes
- Data is currently **synthetic**
- `runs/` contains local execution artifacts and is excluded from git
- `.env` is excluded from git
- The project is designed to support later extension to real datasets if needed

## Intended research use
This prototype is designed to support a paper studying:

- when consumer-health tasks benefit from tool use,
- when multi-step agentic orchestration adds value,
- how safety and traceability should be integrated,
- and how such systems can be evaluated across realistic usage modes.
