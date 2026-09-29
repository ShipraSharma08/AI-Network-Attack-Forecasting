# AI Network Attack Forecasting from Network Traffic Data

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-World%20Model-EE4C2C.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-SOC%20Console-FF4B4B.svg)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Tests-34%20Passed%20(100%25)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep temporal learning and explainable forecasting framework for proactive network intrusion detection and advance threat warning, developed for **NTRO (National Technical Research Organisation)** and evaluated on the **CSE-CIC-IDS2018** infiltration dataset.

> **System Scope & Capability Statement:**  
> This system is an **offline PCAP-based and CSV-based attack forecasting prototype** engineered for SOC analysts and incident response teams. It models discrete 60-second temporal network state transitions to forecast multi-step risk and physical network state evolution ahead of attack onset.

---

## 🚀 Key Highlights & Empirical Performance

- **Dual-Head Temporal World Model:** Jointly predicts next-state network flow dynamics $\hat{S}(t+1) \in \mathbb{R}^{29}$ (regression, **MSE = 0.0886**) and future attack risk $P(\text{attack}_{t+1})$ (**75.76% F1-score**, **78.12% Precision**, **73.53% Recall**), outperforming the standard Logistic Regression baseline (**52.83% F1**, **+43.4% relative gain**).
- **True Recursive K-Step Autoregressive Rollout:** At each horizon step $k$, the predicted state $\hat{S}(t+k)$ is appended to the input buffer, achieving **0.8718 AUC-ROC at $K=1$ min** and **0.8176 AUC-ROC at $K=5$ min**.
- **Behavior-Driven MITRE ATT&CK Mapping:** Rule-based classifier grounded in real traffic behaviors (SYN scanning, ACK ratios, RST failed connection rates, PSH/URG payload spikes) covering *Reconnaissance* (`TA0043`), *Initial Access* (`TA0001`), *Lateral Movement* (`TA0008`), and *Command & Control* (`TA0011`).
- **Forensic Flagged Flow Evidence:** Extracts real 5-tuple forensic flows (`data/processed/flagged_flows_lookup.csv`, 3,150 flows) with timestamp, source/dest IPs, ports, protocols, and anomaly drivers with one-click CSV download.
- **Explainability (SHAP):** KernelExplainer on the LSTM temporal representation provides attribution scores across the 10-minute history window with percentage deviations from baseline.
- **Interactive SOC Console:** Signal Intelligence-themed Streamlit dashboard (`src/dashboard/app.py`) for live PCAP/CSV ingestion, threat gauge monitoring, forecast trajectory curves, forensic flow inspection, and structured JSON incident log export.

---

## 📂 Project Structure

```text
.
├── configs/                  # Pipeline configuration files
├── data/
│   ├── processed/            # 60s cleaned temporal transition states & flagged flows lookup
│   └── raw/                  # Raw PCAP & CSV records
├── docs/                     # Visualizations & architecture figures
│   └── kstep_forecasting_examples.png
├── models/                   # Saved trained weights (.h5 / .pt)
│   ├── lstm_worldmodel_model.h5  # Dual-Head Temporal LSTM World Model
│   ├── lstm_worldmodel_model.pt  # PyTorch checkpoint
│   ├── lstm_model.h5             # Flow-only classifier baseline
│   └── lstm_fused_model.h5       # Multi-modal Fused LSTM (43 features)
├── reports/                  # Detailed metrics & technical documentation
│   ├── kstep_worldmodel_rollout_metrics.json
│   ├── kstep_rollout_metrics.json
│   └── technical_report.md
├── src/
│   ├── dashboard/            # Streamlit SOC incident console (app.py)
│   ├── explainability/       # MITRE mapping & SHAP attribution
│   ├── features/             # Canonical 29-flow schema & feature extraction
│   ├── forecasting/          # Autoregressive K-step World Model rollout engine
│   └── models/               # Temporal LSTM World Model architecture
└── tests/                    # 34 comprehensive unit & integration tests
```

---

## ⚡ Quickstart

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/vaibhav-09/AI-Network-Attack-Forecasting.git
cd AI-Network-Attack-Forecasting

# Activate virtual environment
source myvenv/bin/activate
pip install -r requirements.txt
```

### 2. Run Comprehensive Test Suite
```bash
PYTHONPATH=. python -m unittest discover tests
# Ran 34 tests in 0.427s — OK (100% pass)
```

### 3. Launch Interactive SOC Dashboard
```bash
streamlit run src/dashboard/app.py
```
*(Access console at `http://localhost:8501`)*

### 4. Evaluate K-Step World Model Rollout
```bash
PYTHONPATH=. python src/forecasting/kstep_worldmodel_rollout.py
```

### 5. Run Behavioral MITRE Stage Prediction
```bash
PYTHONPATH=. python src/explainability/mitre_stage_mapping.py --prob 0.85 --elapsed 20.0
```

---

## 📊 Empirical Benchmarks Summary

Evaluated on strict chronological test split ($w \ge 630$, $N=89$ windows):

| Model Architecture | Feature Space | Precision | Recall | F1-Score | FPR | Next-State Std MSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 29 (Flow) | 0.7368 | 0.4118 | 0.5283 | 0.0909 | N/A |
| **Classifier-Only Temporal LSTM** | 29 (Flow) | 0.7812 | 0.7353 | 0.7576 | 0.1273 | N/A |
| **Temporal LSTM World Model (Dual-Head)** | 29 (Flow) | **0.7812** | **0.7353** | **0.7576** | 0.1273 | **0.0886** |
| **Fused Multi-Modal LSTM** | 43 (Flow+Pkt) | 0.6923 | 0.2647 | 0.3830 | **0.0727** | N/A |

### Out-of-Distribution / Unseen Event Generalization:
- **Unseen Attack Holdout Evaluation:** F1 = **91.18%**, Precision = **96.10%**, Recall = **86.76%**, OOD FPR = **1.14%**.

---

## 📑 Full Technical Documentation
For complete architectural details, mathematical formulation, and ablation analyses, see:  
👉 [reports/technical_report.md](reports/technical_report.md)
