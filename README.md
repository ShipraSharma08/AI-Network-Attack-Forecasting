# AI Network Attack Forecasting & Threat Attribution

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep temporal learning and explainable forecasting framework for proactive network intrusion detection and advance threat warning, evaluated on the **CSE-CIC-IDS2018** infiltration dataset.

---

## 🚀 Key Results & Highlights

- **Detection Performance:** Flow-only Temporal LSTM achieves **75.76% F1-score** (**78.12% Precision**, **73.53% Recall**), outperforming the standard baseline (**52.83% F1**) by **+43.4% relative**.
- **K-Step Autoregressive Advance Warning:** Delivers **0.89 AUC-ROC at 1-minute** and **0.79 AUC-ROC at 5-minute** forward rollout horizons.
- **Explainability & Attribution:** Real-time **SHAP KernelExplainer** feature rankings and automated mapping to **MITRE ATT&CK** intrusion phases (*Reconnaissance*, *Initial Access*, *Lateral Movement*, *Command & Control*).
- **Interactive SOC Console:** Full-featured **Streamlit incident response dashboard** with live PCAP ingestion, threat risk gauges, forecast trajectory curves, and JSON report export.

---

## 📂 Project Structure

```text
.
├── configs/                  # Pipeline configuration files
├── data/
│   ├── processed/            # 60s cleaned temporal transition states
│   └── raw/                  # Raw PCAP & CSV records
├── docs/                     # Visualizations & architecture figures
│   └── kstep_forecasting_examples.png
├── models/                   # Saved trained weights (.h5 / .pt)
│   ├── lstm_model.h5         # Flow-only Temporal LSTM (29 features)
│   └── lstm_fused_model.h5   # Multi-modal Fused LSTM (43 features)
├── reports/                  # Detailed metrics & technical documentation
│   ├── kstep_rollout_metrics.json
│   └── technical_report.md
├── src/
│   ├── dashboard/            # Streamlit incident console (app.py)
│   ├── explainability/       # MITRE mapping & SHAP attribution
│   ├── features/             # Flow & packet feature extraction
│   ├── forecasting/          # Autoregressive K-step rollout engine
│   └── models/               # Temporal LSTM architectures
└── tests/                    # Unit & regression test suites
```

---

## ⚡ Quickstart

### 1. Environment Activation
```bash
source myvenv/bin/activate
```

### 2. K-Step Autoregressive Rollout & Visualizations
```bash
python src/forecasting/kstep_rollout.py --k 10 --visualize
```

### 3. MITRE ATT&CK Stage Prediction
```bash
python src/explainability/mitre_stage_mapping.py --prob 0.85 --elapsed 20.0
```

### 4. SHAP Feature Attribution
```bash
python src/explainability/shap_attribution.py --window 584 --k-top 5
```

### 5. Launch Interactive SOC Dashboard
```bash
streamlit run src/dashboard/app.py
```
*(Open `http://localhost:8501` in your browser)*

---

## 📑 Full Documentation
For extensive architectural specifications, empirical tables, and comparative analyses, see:
👉 [reports/technical_report.md](reports/technical_report.md)
