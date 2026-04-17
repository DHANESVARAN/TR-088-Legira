# LEGIRA Setup Guide

This document explains how to set up the required environment, install dependencies, and run the LEGIRA project from the repository root.

## Recommended File Location

This file is stored at the root of the project:

- [HOW_TO_SETUP.md](/mnt/d/LEGIRA/HOW_TO_SETUP.md)

This is the correct place for it because setup instructions should be easy to find immediately after opening the repository.

The recommended startup script is also stored at the root:

- [run_legira.sh](/mnt/d/LEGIRA/run_legira.sh)

That placement makes it easy to launch the project without navigating into subfolders first.

---

## What This Project Contains

This repository has three relevant runnable areas:

1. `Phases/Phase - 1`
   Baseline legal case prioritization engine and Streamlit dashboard.

2. `Phases/Phase - 2 & 3/hackathon2`
   Enhanced legal prioritization pipeline, improved dashboard, and model-comparison workflow.

3. `Phases/Phase - 2 & 3/js-crypto-model1`
   Separate Node.js blockchain prototype.

If your goal is to run the **main LEGIRA application**, you should start with:

- `Phases/Phase - 2 & 3/hackathon2`

That is the most complete version of the legal prioritization system in this repository.

---

## System Requirements

## 1. Operating System

Recommended:

- Windows 10 or Windows 11
- WSL Ubuntu, Git Bash, or any POSIX-style shell if using the `.sh` script

Also supported in practice:

- Linux

## 2. Python

Install:

- Python `3.11+` recommended

The repository appears to have been used with Python `3.13` in some local artifacts, but you do not need that exact version. Python `3.11` or `3.12` is a safer general choice.

## 3. Node.js

Needed only for the blockchain prototype:

- Node.js `18+`
- npm `9+`

## 4. Optional Local LLM Support

The enhanced dashboard contains hooks for local summarization through an Ollama-compatible endpoint:

- default URL: `http://localhost:11434`
- default model name in code: `qwen2.5:latest`

This is optional. The dashboard can still run without it, but any local LLM summarization feature will require that service.

---

## Project Paths

Main root:

```text
D:\LEGIRA
```

Phase 1 application:

```text
D:\LEGIRA\Phases\Phase - 1
```

Phase 2 application:

```text
D:\LEGIRA\Phases\Phase - 2 & 3\hackathon2
```

Blockchain prototype:

```text
D:\LEGIRA\Phases\Phase - 2 & 3\js-crypto-model1
```

---

## Python Dependencies

The checked-in `requirements.txt` files currently list:

```text
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.5.0
streamlit>=1.36.0
```

However, the Phase 2 dashboard also imports `altair`, and the model comparison flow references optional packages such as:

- `xgboost`
- `lightgbm`
- `imbalanced-learn`
- `joblib`

For a safer full setup, install:

- `pandas`
- `numpy`
- `scikit-learn`
- `streamlit`
- `altair`
- `joblib`

Optional but recommended for the improved model workflow:

- `xgboost`
- `lightgbm`
- `imbalanced-learn`

---

## Setup Strategy

Use separate virtual environments for each Python application area:

1. one environment for `Phase - 1`
2. one environment for `Phase - 2 & 3/hackathon2`

This keeps dependencies clean and avoids path confusion.

---

## Step-By-Step Setup For Phase 2

This is the recommended setup because Phase 2 is the most complete LEGIRA workflow.

## 1. Open a terminal in the project root

```bash
cd /mnt/d/LEGIRA
```

On Windows Command Prompt, this is conceptually:

```bat
cd /d D:\LEGIRA
```

## 2. Move into the Phase 2 application folder

```bash
cd "Phases/Phase - 2 & 3/hackathon2"
```

## 3. Create a virtual environment

```bash
python3 -m venv .venv
```

If your system uses `python` instead of `python3`:

```bash
python -m venv .venv
```

## 4. Activate the virtual environment

### Linux / WSL / Git Bash

```bash
source .venv/bin/activate
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```bat
.venv\Scripts\activate.bat
```

## 5. Upgrade pip

```bash
python -m pip install --upgrade pip
```

## 6. Install the base dependencies

```bash
pip install -r requirements.txt
pip install altair joblib
```

## 7. Install optional improved-model dependencies

If you want the strongest Phase 2 setup, also install:

```bash
pip install xgboost lightgbm imbalanced-learn
```

If one of those packages fails due to platform/compiler issues, the core dashboard and baseline workflow can still run. The improved model path may then fall back or lose some functionality.

## 8. Verify that the data files exist

Check these paths:

```text
Phases/Phase - 2 & 3/hackathon2/data/dataset1_core_cases.csv
Phases/Phase - 2 & 3/hackathon2/data/dataset2_detention.csv
Phases/Phase - 2 & 3/hackathon2/data/dataset3_temporal.csv
Phases/Phase - 2 & 3/hackathon2/data/dataset4_demographics.csv
Phases/Phase - 2 & 3/hackathon2/data/dataset5_nlp.csv
Phases/Phase - 2 & 3/hackathon2/data/ipc_bns_max_sentence_lookup.csv
```

These are required for the main legal prioritization pipeline.

## 9. Generate outputs with the prioritization engine

Run:

```bash
python prioritization_engine.py --data-dir data --output-dir outputs --augment-demo
```

This should generate:

- `prioritized_cases.csv`
- `flagged_undertrials.csv`
- `cluster_summary.csv`
- `synthetic_demo_cases.csv`
- `evaluation_metrics.json`

inside:

```text
Phases/Phase - 2 & 3/hackathon2/outputs
```

## 10. Launch the Streamlit dashboard

```bash
streamlit run dashboard.py
```

Once running, Streamlit usually prints a local URL such as:

```text
http://localhost:8501
```

Open that URL in your browser.

---

## Step-By-Step Setup For Phase 1

Use this if you specifically want to run the baseline version.

## 1. Change folder

```bash
cd /mnt/d/LEGIRA/"Phases/Phase - 1"
```

## 2. Create virtual environment

```bash
python3 -m venv .venv
```

## 3. Activate it

```bash
source .venv/bin/activate
```

Or on Windows:

```bat
.venv\Scripts\activate.bat
```

## 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install altair joblib
```

## 5. Generate outputs

```bash
python prioritization_engine.py --data-dir data --output-dir outputs
```

## 6. Start dashboard

```bash
streamlit run dashboard.py
```

---

## Setup For Blockchain Prototype

This part is independent from the main legal application.

## 1. Change folder

```bash
cd "/mnt/d/LEGIRA/Phases/Phase - 2 & 3/js-crypto-model1"
```

## 2. Install dependencies

```bash
npm install
```

## 3. Run tests

```bash
npm test
```

Note:

- the configured test command uses `jest --watchAll`
- that is interactive and may not be ideal for automated runs

## 4. Start development server

```bash
npm run dev
```

The Express app in `index.js` listens on:

```text
http://localhost:3001
```

And the P2P server defaults to:

```text
port 5001
```

---

## Optional Local LLM Setup

The enhanced dashboard contains a helper for local case summarization.

If you want that feature:

1. install Ollama or an equivalent compatible local service
2. start the service on:

```text
http://localhost:11434
```

3. make sure the configured model exists, for example:

```text
qwen2.5:latest
```

Without this, the main dashboard can still run, but local summarization requests may fail or be skipped.

---

## Recommended Storage Layout

Store the files like this:

```text
D:\LEGIRA
├── README.md
├── HOW_TO_SETUP.md
├── run_legira.sh
├── docs
├── Reference
└── Phases
```

Why this is recommended:

- `README.md` explains the project
- `HOW_TO_SETUP.md` explains installation and runtime
- `run_legira.sh` provides a single start command from the root

This is the cleanest onboarding layout for a judge, teammate, or evaluator.

---

## How To Start Quickly

If you only want the main application:

1. install Python
2. create and activate the virtual environment in `Phases/Phase - 2 & 3/hackathon2`
3. install dependencies
4. return to the root
5. run:

```bash
./run_legira.sh phase2
```

---

## Troubleshooting

## 1. `python` or `python3` not found

Install Python and make sure it is available on your PATH.

## 2. `streamlit` command not found

Your virtual environment is probably not activated, or Streamlit is not installed in it.

Fix:

```bash
source .venv/bin/activate
pip install streamlit
```

## 3. `ModuleNotFoundError: No module named altair`

Install Altair:

```bash
pip install altair
```

## 4. `xgboost` or `lightgbm` installation fails

These are optional for the improved model workflow. You can still run the baseline application and much of the Phase 2 dashboard without them.

## 5. Streamlit opens but data is missing

Confirm that the `data/` folder is present and that output CSV files were generated in `outputs/`.

## 6. Local LLM summarization fails

Check:

- Ollama service is running
- endpoint is `http://localhost:11434`
- configured model exists

## 7. Permission denied on `run_legira.sh`

Run:

```bash
chmod +x run_legira.sh
```

---

## Commands Summary

### Phase 2 install

```bash
cd "/mnt/d/LEGIRA/Phases/Phase - 2 & 3/hackathon2"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install altair joblib xgboost lightgbm imbalanced-learn
```

### Phase 2 start

```bash
python prioritization_engine.py --data-dir data --output-dir outputs --augment-demo
streamlit run dashboard.py
```

### Root-level quick start

```bash
cd /mnt/d/LEGIRA
./run_legira.sh phase2
```

---

## Final Recommendation

For submission, demo, and judge evaluation, use:

- `Phase - 2 & 3/hackathon2` as the main application
- `HOW_TO_SETUP.md` as the installation guide
- `run_legira.sh` as the startup shortcut

That is the cleanest and most professional setup path for this repository.
