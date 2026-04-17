# <div align="center">LEGIRA</div>

<div align="center">
  <img width="677" height="369" alt="Gemini_Generated_Image_ic00psic00psic00__1_-removebg-preview" src="https://github.com/user-attachments/assets/0a204ead-4610-43fa-af14-cf9ee0042cf1" />

</div>

<div align="center">
  <img src="https://img.shields.io/badge/TENSOR'26-Hackathon-ff4d6d?style=for-the-badge&logo=rocket&logoColor=white" alt="Hackathon" />
  <img src="https://img.shields.io/badge/Domain-Legal%20AI-4f8df7?style=for-the-badge&logo=scale-balanced&logoColor=white" alt="Legal AI" />
  <img src="https://img.shields.io/badge/Focus-Undertrial%20Justice-00b894?style=for-the-badge&logo=shield&logoColor=white" alt="Undertrial Justice" />
  <img src="https://img.shields.io/badge/Interface-Streamlit-f97316?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Models-scikit--learn%20%7C%20XGBoost-f4c430?style=for-the-badge&logo=scikitlearn&logoColor=black" alt="Models" />
 

</div>

<div align="center">
  <h3>
    <span style="color:#2f5597;">AI-assisted judicial triage for</span>
    <span style="color:#c00000;">over-detained</span>,
    <span style="color:#ed7d31;">delayed</span>, and
    <span style="color:#70ad47;">vulnerable</span> cases.
  </h3>
</div>

---

## 🌈 Overview

LEGIRA is a fairness-aware legal prioritization system built to help courts move beyond simple filing-order queues. It identifies cases that deserve earlier review by combining detention history, statutory punishment limits, stale-hearing patterns, vulnerability indicators, and ML-assisted classification.

This repository contains two major implementation stages:

- `Phase - 1`: baseline legal case prioritization and dashboard system
- `Phase - 2 & 3`: enhanced model-comparison system, improved artifacts, and expanded technical workflow

It also contains a separate blockchain prototype that can support future auditability.

---

## 👥 Team

<div align="center">
  <img src="https://img.shields.io/badge/DHANESVARAN%20M-Team%20Member-1f77b4?style=for-the-badge" alt="DHANESVARAN M" />
  <img src="https://img.shields.io/badge/ASWIN%20M-Team%20Member-ff7f0e?style=for-the-badge" alt="ASWIN M" />
  <img src="https://img.shields.io/badge/DHARSHAN%20R-Team%20Member-2ca02c?style=for-the-badge" alt="DHARSHAN R" />
  <img src="https://img.shields.io/badge/GANESH%20M-Team%20Member-d62728?style=for-the-badge" alt="GANESH M" />
</div>

---

## ⚖️ Problem Statement

Judicial backlog is not only an operational delay problem. It is also a fairness problem.

Cases that should be reviewed urgently often get buried in ordinary queues, especially when they involve:

- undertrial prisoners detained beyond likely or statutory sentence exposure
- repeated adjournments and stale hearing histories
- vulnerable persons with age, disability, illness, or special care concerns
- bailable or low-severity matters that remain pending too long

LEGIRA is designed to surface these cases early and explain why they were prioritized.

---

## ✨ Core Capabilities

<table>
  <tr>
    <th align="left">Capability</th>
    <th align="left">Description</th>
  </tr>
  <tr>
    <td>📥 Multi-source ingestion</td>
    <td>Merges structured case, detention, temporal, demographic, and NLP-derived datasets.</td>
  </tr>
  <tr>
    <td>📚 Legal enrichment</td>
    <td>Uses IPC/BNS statute lookup tables to derive sentence ceilings and bailability context.</td>
  </tr>
  <tr>
    <td>🧮 Explainable scoring</td>
    <td>Builds a bounded <code>priority_score</code> using overstay, delay, severity, and vulnerability signals.</td>
  </tr>
  <tr>
    <td>🚨 Undertrial overstay detection</td>
    <td>Applies a critical override for fairness-sensitive detention cases.</td>
  </tr>
  <tr>
    <td>🤖 ML triage</td>
    <td>Predicts urgency and case-routing labels using trained classifiers.</td>
  </tr>
  <tr>
    <td>📈 Backlog segmentation</td>
    <td>Clusters cases into operational groups such as critical backlog and routine queue.</td>
  </tr>
  <tr>
    <td>🖥️ Dashboard review</td>
    <td>Supports filtering, ranking, flagged-case inspection, and new-case scoring in Streamlit.</td>
  </tr>
</table>

---

## 🏗️ Phase-Wise System Architecture

### 🟦 Phase 1 Architecture

Phase 1 is the **baseline prioritization system**. It focuses on direct case triage from structured data and presents the outputs in an initial Streamlit dashboard.

```mermaid
flowchart LR
    A1[dataset1_core_cases.csv] --> B1[load_and_merge]
    A2[dataset2_detention.csv] --> B1
    A3[dataset3_temporal.csv] --> B1
    A4[dataset4_demographics.csv] --> B1
    A5[dataset5_nlp.csv] --> B1
    A6[ipc_bns_max_sentence_lookup.csv] --> C1[engineer_features]
    B1 --> C1
    C1 --> D1[priority_score and recommended_track]
    C1 --> E1[RandomForest urgency and track models]
    D1 --> F1[KMeans clustering]
    D1 --> G1[prioritized_cases.csv]
    D1 --> H1[flagged_undertrials.csv]
    F1 --> I1[cluster_summary.csv]
    G1 --> J1[dashboard.py]
    H1 --> J1
    I1 --> J1
```

### 🟩 Phase 2 Architecture

Phase 2 extends the system into an **evaluation and comparison architecture**. It preserves the core prioritization engine while adding explicit model benchmarking and artifact export.

```mermaid
flowchart LR
    P1[Raw legal datasets] --> P2[hackathon2/prioritization_engine.py]
    P2 --> P3[feature engineering and baseline scoring]
    P3 --> P4[SimilarModel: RandomForest plus KMeans]
    P3 --> P5[ImprovedModel: XGBoost or boosting plus hierarchical clustering]
    P4 --> P6[metrics_similar.json]
    P5 --> P7[metrics_improved.json]
    P4 --> P8[saved model artifacts]
    P5 --> P8
    P4 --> P9[sample predictions]
    P5 --> P9
    P6 --> P10[compare_models.py]
    P7 --> P10
    P10 --> P11[model_comparison.csv]
    P3 --> P12[enhanced outputs and dashboard]
```

---

## 📂 File Path Architecture From `D:\LEGIRA\Phases`

The following architecture is derived from the main project folder `D:\LEGIRA\Phases`, with hidden `.git` content excluded. For readability, generated/vendor-heavy folders such as `node_modules`, virtual environments, and `__pycache__` are also omitted.

### 📁 `D:\LEGIRA\Phases\Phase - 1`

```text
D:\LEGIRA\Phases\Phase - 1
├── README.md
├── dashboard.py
├── prioritization_engine.py
├── generate_statute_lookup.py
├── requirements.txt
├── data
│   ├── dataset1_core_cases.csv
│   ├── dataset2_detention.csv
│   ├── dataset3_temporal.csv
│   ├── dataset4_demographics.csv
│   ├── dataset5_nlp.csv
│   ├── ipc_bns_max_sentence_lookup.csv
│   └── ipc_bns_statutes_master.csv
└── outputs
    ├── cluster_summary.csv
    ├── evaluation_metrics.json
    ├── flagged_undertrials.csv
    ├── prioritized_cases.csv
    └── synthetic_demo_cases.csv
```

### 📁 `D:\LEGIRA\Phases\Phase - 2 & 3`

```text
D:\LEGIRA\Phases\Phase - 2 & 3
├── logo.jpeg
├── Gemini_Generated_Image_ic00psic00psic00.png
├── package.json
├── package-lock.json
├── hackathon2
│   ├── MODELS_README.md
│   ├── MODEL_CREATION_SUMMARY.md
│   ├── compare_models.py
│   ├── dashboard.py
│   ├── generate_statute_lookup.py
│   ├── model_improved.py
│   ├── model_similar.py
│   ├── prioritization_engine.py
│   ├── requirements.txt
│   ├── data
│   │   ├── dataset1_core_cases.csv
│   │   ├── dataset2_detention.csv
│   │   ├── dataset3_temporal.csv
│   │   ├── dataset4_demographics.csv
│   │   ├── dataset5_nlp.csv
│   │   ├── ipc_bns_max_sentence_lookup.csv
│   │   └── ipc_bns_statutes_master.csv
│   ├── outputs
│   │   ├── cluster_summary.csv
│   │   ├── evaluation_metrics.json
│   │   ├── flagged_undertrials.csv
│   │   ├── prioritized_cases.csv
│   │   └── synthetic_demo_cases.csv
│   └── comparison_outputs
│       ├── detailed_metrics.json
│       ├── model_comparison.csv
│       ├── sample_predictions.csv
│       ├── improved_models
│       └── similar_models
└── js-crypto-model1
    ├── .gitignore
    ├── Block
    │   ├── block.js
    │   └── block.test.js
    ├── blockchain.js
    │   ├── blockchain.js
    │   └── blockchain.test.js
    ├── chain-util.js
    ├── config.js
    ├── dev-test.js
    ├── index.js
    ├── miner.js
    ├── p2p-server.js
    ├── package.json
    ├── package-lock.json
    ├── tester.js
    └── wallet
        ├── index.js
        ├── index.test.js
        ├── transaction-pool.js
        ├── transaction-poot.test.js
        ├── transaction.js
        └── transaction.test.js
```

---

## 📊 Checked-In Results

<table>
  <tr>
    <th align="left">Metric</th>
    <th align="left">Result</th>
  </tr>
  <tr>
    <td>Urgency classification accuracy</td>
    <td><b>88.43%</b></td>
  </tr>
  <tr>
    <td>Case-type classification accuracy</td>
    <td><b>73.97%</b></td>
  </tr>
  <tr>
    <td>Undertrial detection recall</td>
    <td><b>100%</b></td>
  </tr>
  <tr>
    <td>Dashboard filter response time</td>
    <td><b>3.26 ms</b></td>
  </tr>
  <tr>
    <td>Ranked-list agreement (Spearman)</td>
    <td><b>0.8942</b></td>
  </tr>
</table>

### 📌 Output Inventory

- `prioritized_cases.csv`: `1,206` ranked cases
- `flagged_undertrials.csv`: `321` flagged undertrial cases
- `cluster_summary.csv`: `4` clusters
- `synthetic_demo_cases.csv`: `6` demo cases

---

## 🤖 Model Comparison

<table>
  <tr>
    <th>Metric</th>
    <th>Similar Model</th>
    <th>Improved Model</th>
    <th>Winner</th>
  </tr>
  <tr>
    <td>Urgency Accuracy</td>
    <td>88.43%</td>
    <td><b>88.84%</b></td>
    <td>🟢 Improved</td>
  </tr>
  <tr>
    <td>Urgency Weighted F1</td>
    <td>88.46%</td>
    <td><b>88.98%</b></td>
    <td>🟢 Improved</td>
  </tr>
  <tr>
    <td>Track Accuracy</td>
    <td><b>73.14%</b></td>
    <td>72.31%</td>
    <td>🔵 Similar</td>
  </tr>
  <tr>
    <td>Track Weighted F1</td>
    <td>68.78%</td>
    <td><b>70.39%</b></td>
    <td>🟢 Improved</td>
  </tr>
  <tr>
    <td>Undertrial Recall</td>
    <td><b>100%</b></td>
    <td><b>100%</b></td>
    <td>🟡 Tie</td>
  </tr>
</table>

---

## 🚀 Run Instructions

### Phase 1 baseline

```bash
cd "Phases/Phase - 1"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python prioritization_engine.py --data-dir data --output-dir outputs
streamlit run dashboard.py
```

### Phase 2 enhanced pipeline

```bash
cd "Phases/Phase - 2 & 3/hackathon2"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python prioritization_engine.py --data-dir data --output-dir outputs --augment-demo
python compare_models.py
streamlit run dashboard.py
```

### Blockchain prototype

```bash
cd "Phases/Phase - 2 & 3/js-crypto-model1"
npm install
npm test
npm run dev
```

---

## 📸 Screenshot Guide

<img width="1600" height="815" alt="WhatsApp Image 2026-04-17 at 9 25 24 AM" src="https://github.com/user-attachments/assets/ffcd1a34-2626-4184-8485-d6fe9e7a3959" />

<img width="1600" height="814" alt="WhatsApp Image 2026-04-17 at 9 25 39 AM" src="https://github.com/user-attachments/assets/608d0f0c-4ba6-4771-8e4e-b826302669f1" />

<img width="1600" height="775" alt="WhatsApp Image 2026-04-17 at 9 29 27 AM" src="https://github.com/user-attachments/assets/7b6ebc46-769e-4b47-899c-e93ea3ac279d" />

<img width="1600" height="776" alt="WhatsApp Image 2026-04-17 at 9 42 27 AM" src="https://github.com/user-attachments/assets/ed1dcb0a-2ed7-4076-bc5f-72568ae77157" />

<img width="1600" height="771" alt="WhatsApp Image 2026-04-17 at 9 42 44 AM" src="https://github.com/user-attachments/assets/5567bdd2-fc86-4c51-aa8d-84c1592d500d" />

<img width="1600" height="797" alt="WhatsApp Image 2026-04-17 at 9 43 38 AM" src="https://github.com/user-attachments/assets/b412637f-8ad1-4b51-94a0-a1f0e68c208c" />

<img width="1600" height="803" alt="WhatsApp Image 2026-04-17 at 9 44 16 AM" src="https://github.com/user-attachments/assets/72632742-a67b-4afd-ba53-91f7912a1adf" />

<img width="1600" height="754" alt="WhatsApp Image 2026-04-17 at 9 46 29 AM" src="https://github.com/user-attachments/assets/5471d9ef-5853-46fe-970b-b666042740b0" />



---

## 📘 Documentation

- IEEE paper: [docs/LEGIRA_IEEE_Paper.md](docs/LEGIRA_IEEE_Paper.md)
- Word-friendly paper: [docs/LEGIRA_IEEE_Paper_Word_Friendly.rtf](docs/LEGIRA_IEEE_Paper_Word_Friendly.rtf)

---

## 🏁 Final Note

LEGIRA is designed as a legal intelligence layer for smarter hearing-order decisions. Its core contribution is not only prediction, but explainable prioritization for fairness-sensitive judicial review.
