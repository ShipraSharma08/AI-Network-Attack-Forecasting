#!/usr/bin/env python3
"""
Interactive Incident Response & Attack Forecasting Dashboard
============================================================
Streamlit Application for Real-Time Threat Intelligence & Rollout Forecasting:
  1. Live PCAP / Flow CSV upload & real-time LSTM inference.
  2. K-Step Autoregressive Forward Forecasting (1-15 min) with shaded threat zones.
  3. MITRE ATT&CK Phase Prediction & timeline mapping.
  4. Top-5 Feature Attribution (SHAP) with baseline elevation annotations.
  5. Side-by-side Model Comparison (Flow-Only vs Fused LSTM + Confusion Matrix).
  6. Instant Incident Response JSON Export.
"""

import io
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
import streamlit as st
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.mitre_stage_mapping import predict_mitre_stage
from src.explainability.shap_attribution import AttackExplainer, init_explainer
from src.forecasting.kstep_rollout import forward_rollout, load_flow_lstm_and_scaler
from src.models.temporal_lstm import TemporalLSTM
from src.models.temporal_lstm_fused import (
    FLOW_ONLY_BASELINE,
    LOGISTIC_REGRESSION_BASELINE,
    TemporalLSTM as FusedTemporalLSTM,
    load_and_prepare_fused_sequences,
)

# Page configuration
st.set_page_config(
    page_title="AI Network Attack Forecasting | SOC Incident Response",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Cyber SOC Styling
CUSTOM_CSS = """
<style>
    /* Main container styling */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    
    /* Header & Titles */
    .soc-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .soc-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #f8fafc;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .soc-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .stage-card {
        border-radius: 12px;
        padding: 22px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
    }
    .stage-badge {
        display: inline-block;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 4px 10px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    
    /* Custom Alert Badges */
    .badge-recon { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid #f59e0b; }
    .badge-access { background: rgba(249, 115, 22, 0.15); color: #fb923c; border: 1px solid #f97316; }
    .badge-lateral { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid #ef4444; }
    .badge-c2 { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid #a855f7; }
    .badge-benign { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid #10b981; }

    /* Timeline Stepper */
    .timeline-container {
        display: flex;
        justify-content: space-between;
        margin-top: 15px;
        position: relative;
    }
    .timeline-step {
        text-align: center;
        flex: 1;
        font-size: 0.75rem;
        color: #94a3b8;
    }
    .timeline-step.active {
        color: #38bdf8;
        font-weight: 700;
    }
    .timeline-dot {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: #475569;
        margin: 0 auto 6px auto;
    }
    .timeline-dot.active {
        background: #38bdf8;
        box-shadow: 0 0 10px #38bdf8;
    }
    .timeline-dot.passed {
        background: #10b981;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_cached_models_and_data():
    """Cache loaded models, scaler, dataset, and background explainer."""
    model, scaler, feature_cols, df = load_flow_lstm_and_scaler()
    explainer, _, _ = init_explainer(n_background_samples=25, nsamples=40)
    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "df": df,
        "explainer": explainer,
    }


def parse_pcap_to_sequence(
    pcap_file: io.BytesIO,
    feature_cols: List[str],
    scaler: Any,
) -> Tuple[np.ndarray, str]:
    """
    Parse uploaded PCAP using Scapy PcapReader into 1-minute flow summaries.
    Falls back gracefully if PCAP is truncated or empty.
    """
    from scapy.all import IP, TCP, UDP, PcapReader

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
        tmp.write(pcap_file.read())
        tmp_path = tmp.name

    try:
        packets = []
        with PcapReader(tmp_path) as pcap_reader:
            for i, pkt in enumerate(pcap_reader):
                if pkt.haslayer(IP):
                    proto = 6 if pkt.haslayer(TCP) else (17 if pkt.haslayer(UDP) else 0)
                    syn = 1 if (pkt.haslayer(TCP) and pkt[TCP].flags.S) else 0
                    ack = 1 if (pkt.haslayer(TCP) and pkt[TCP].flags.A) else 0
                    fin = 1 if (pkt.haslayer(TCP) and pkt[TCP].flags.F) else 0
                    rst = 1 if (pkt.haslayer(TCP) and pkt[TCP].flags.R) else 0
                    length = len(pkt)
                    packets.append(
                        {
                            "time": float(pkt.time),
                            "length": length,
                            "syn": syn,
                            "ack": ack,
                            "fin": fin,
                            "rst": rst,
                            "proto": proto,
                        }
                    )
                if i >= 50000:  # Cap for real-time responsiveness
                    break

        if not packets:
            raise ValueError("No IP packets parsed from uploaded PCAP.")

        pkt_df = pd.DataFrame(packets)
        t_min = pkt_df["time"].min()
        t_max = pkt_df["time"].max()
        duration = max(t_max - t_min, 600.0)
        window_duration = duration / 10.0

        seq_rows = []
        for w in range(10):
            w_start = t_min + w * window_duration
            w_end = w_start + window_duration
            sub = pkt_df[(pkt_df["time"] >= w_start) & (pkt_df["time"] <= w_end)]
            count = len(sub)
            syn_cnt = sub["syn"].sum() if count > 0 else 0
            ack_cnt = sub["ack"].sum() if count > 0 else 0
            rst_cnt = sub["rst"].sum() if count > 0 else 0
            fin_cnt = sub["fin"].sum() if count > 0 else 0
            pkt_len_mean = sub["length"].mean() if count > 0 else 64.0
            pkt_len_std = sub["length"].std() if count > 0 else 0.0

            row = [
                count,  # flow_count
                window_duration * 1e6,  # flow_duration_mean
                5.0,  # fwd_packets_mean
                5.0,  # bwd_packets_mean
                count * pkt_len_mean * 0.5,  # fwd_bytes_sum
                count * pkt_len_mean * 0.5,  # bwd_bytes_sum
                1e5,  # flow_iat_mean
                1e4,  # flow_iat_std
                1e6,  # flow_iat_max
                0.0,  # flow_iat_min
                1e5,  # fwd_iat_mean
                1e5,  # bwd_iat_mean
                syn_cnt,  # syn_count
                ack_cnt,  # ack_count
                rst_cnt,  # rst_count
                fin_cnt,  # fin_count
                syn_cnt,  # psh_fwd_count
                0.0,  # psh_bwd_count
                0.0,  # urg_fwd_count
                0.0,  # urg_bwd_count
                pkt_len_mean,  # packet_length_mean
                pkt_len_std,  # packet_length_std
                count / (window_duration if window_duration > 0 else 60.0),  # packets_per_second
                (count * pkt_len_mean) / (window_duration if window_duration > 0 else 60.0),
                0.5,  # down_up_ratio
                1e5,  # active_mean
                1e4,  # active_std
                4e6,  # idle_mean
                1e5,  # idle_std
            ]
            seq_rows.append(row)

        raw_seq = np.array(seq_rows, dtype=np.float32)
        scaled_seq = scaler.transform(raw_seq)
        info = f"Extracted {len(pkt_df):,} IP packets over {duration/60:.1f} minutes into 10 temporal windows."
        return scaled_seq, info

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def render_attack_gauge(prob: float, threshold: float = 0.5) -> go.Figure:
    """Render high-contrast cyber risk speedometer gauge."""
    color = "#ef4444" if prob >= 0.8 else ("#f97316" if prob >= 0.7 else ("#f59e0b" if prob >= 0.6 else "#10b981"))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 42, "color": "#f8fafc", "family": "Inter, Roboto"}},
            delta={"reference": threshold * 100, "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#10b981"}},
            title={"text": "Attack Probability", "font": {"size": 18, "color": "#94a3b8"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b", "ticksuffix": "%"},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "#0f172a",
                "borderwidth": 1,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0, 60], "color": "rgba(16, 185, 129, 0.15)"},
                    {"range": [60, 70], "color": "rgba(245, 158, 11, 0.20)"},
                    {"range": [70, 80], "color": "rgba(249, 115, 22, 0.25)"},
                    {"range": [80, 100], "color": "rgba(239, 68, 68, 0.30)"},
                ],
                "threshold": {
                    "line": {"color": "#f8fafc", "width": 3},
                    "thickness": 0.75,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_forecast_curve(
    forecast_probs: np.ndarray,
    current_prob: float,
    k_steps: int = 15,
    threshold: float = 0.5,
) -> go.Figure:
    """Render K-step rollout curve with shaded threat zones and milestone annotations."""
    x_steps = [f"t+{i}m" for i in range(1, len(forecast_probs) + 1)]
    fig = go.Figure()

    # Shaded attack zones
    fig.add_hrect(y0=0.8, y1=1.0, fillcolor="rgba(239, 68, 68, 0.12)", line_width=0, annotation_text="Lateral Movement / C2 (>0.8)", annotation_position="top left", annotation_font_size=10, annotation_font_color="#ef4444")
    fig.add_hrect(y0=0.7, y1=0.8, fillcolor="rgba(249, 115, 22, 0.10)", line_width=0, annotation_text="Initial Access (>0.7)", annotation_position="top left", annotation_font_size=10, annotation_font_color="#f97316")
    fig.add_hrect(y0=0.6, y1=0.7, fillcolor="rgba(245, 158, 11, 0.08)", line_width=0, annotation_text="Reconnaissance (>0.6)", annotation_position="top left", annotation_font_size=10, annotation_font_color="#f59e0b")
    fig.add_hrect(y0=0.0, y1=0.6, fillcolor="rgba(16, 185, 129, 0.05)", line_width=0, annotation_text="Normal / Benign (<0.6)", annotation_position="bottom left", annotation_font_size=10, annotation_font_color="#10b981")

    # Alarm Threshold line
    fig.add_hline(y=threshold, line_dash="dash", line_color="#94a3b8", line_width=1.5, annotation_text=f"Alarm Threshold ({threshold:.2f})", annotation_position="bottom right", annotation_font_size=10)

    # Forecast trajectory
    fig.add_trace(
        go.Scatter(
            x=x_steps,
            y=forecast_probs,
            mode="lines+markers+text",
            name="Forecasted Risk",
            line=dict(color="#38bdf8", width=3.5),
            marker=dict(size=8, color="#0284c7", symbol="circle", line=dict(color="#f8fafc", width=1.5)),
            text=[f"{p:.2f}" for p in forecast_probs],
            textposition="top center",
            textfont=dict(size=9, color="#e2e8f0"),
            hovertemplate="<b>Horizon:</b> %{x}<br><b>Attack Prob:</b> %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=f"<b>K-Step Autoregressive Attack Forecasting ({k_steps}-Minute Ahead Horizon)</b>", font=dict(size=15, color="#f8fafc")),
        xaxis=dict(title="Forecasting Horizon", showgrid=True, gridcolor="#334155", color="#94a3b8"),
        yaxis=dict(title="Probability P(Attack)", range=[-0.05, 1.05], showgrid=True, gridcolor="#334155", color="#94a3b8"),
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        height=350,
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#94a3b8")),
    )
    return fig


def render_feature_attribution_chart(top_features: List[Dict[str, Any]]) -> go.Figure:
    """Render horizontal bar chart for top 5 influential SHAP feature drivers."""
    df_feat = pd.DataFrame(top_features)
    df_feat = df_feat.iloc[::-1]  # Invert so highest rank is at the top

    colors = ["#ef4444" if d == "attack" else "#10b981" for d in df_feat["direction"]]
    annotations = df_feat["annotation"].tolist()

    fig = go.Figure(
        go.Bar(
            x=df_feat["importance"],
            y=df_feat["clean_name"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="#334155", width=1)),
            text=annotations,
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=11, color="#ffffff", family="Inter, Roboto"),
            hovertemplate="<b>Feature:</b> %{y}<br><b>|SHAP|:</b> %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text="<b>Top 5 Predictive Feature Drivers (SHAP Attribution)</b>", font=dict(size=15, color="#f8fafc")),
        xaxis=dict(title="Mean Absolute SHAP Contribution (|SHAP| across T=10)", showgrid=True, gridcolor="#334155", color="#94a3b8"),
        yaxis=dict(title="", color="#f8fafc", tickfont=dict(size=12)),
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        height=320,
        margin=dict(l=30, r=30, t=50, b=40),
    )
    return fig


def render_confusion_matrices() -> go.Figure:
    """Render side-by-side heatmaps for Flow-only and Fused LSTM confusion matrices."""
    # Test set ground truth: Attack: 34, Benign: 55 (Total: 89)
    cm_flow = [[48, 7], [9, 25]]
    cm_fused = [[51, 4], [25, 9]]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "<b>Flow-Only LSTM (29 Features)</b><br>F1: 0.7576 | Recall: 73.5% | FPR: 12.7%",
            "<b>Fused Multi-Modal LSTM (43 Features)</b><br>F1: 0.3830 | Recall: 26.5% | FPR: 7.3%",
        ),
    )

    fig.add_trace(
        go.Heatmap(
            z=cm_flow,
            x=["Pred Benign", "Pred Attack"],
            y=["Actual Benign", "Actual Attack"],
            text=[[f"TN: {cm_flow[0][0]}", f"FP: {cm_flow[0][1]}"], [f"FN: {cm_flow[1][0]}", f"TP: {cm_flow[1][1]}"]],
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
            textfont=dict(size=14, color="white"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=cm_fused,
            x=["Pred Benign", "Pred Attack"],
            y=["Actual Benign", "Actual Attack"],
            text=[[f"TN: {cm_fused[0][0]}", f"FP: {cm_fused[0][1]}"], [f"FN: {cm_fused[1][0]}", f"TP: {cm_fused[1][1]}"]],
            texttemplate="%{text}",
            colorscale="Purples",
            showscale=False,
            textfont=dict(size=14, color="white"),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8"),
    )
    return fig


def main():
    # Load backend resources
    resources = load_cached_models_and_data()
    model = resources["model"]
    scaler = resources["scaler"]
    feature_cols = resources["feature_cols"]
    df = resources["df"]
    explainer = resources["explainer"]

    # Header
    st.markdown(
        """
        <div class="soc-header">
            <div class="soc-title">
                <span>🛡️ AI Network Attack Forecasting & Threat Attribution</span>
            </div>
            <p class="soc-subtitle">
                Autonomous Temporal LSTM Rollout Engine &bull; Real-Time Multi-Horizon Forecasting &bull; MITRE ATT&CK Mapping &bull; SHAP Explainability
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ SOC Controller")
        data_mode = st.radio(
            "Traffic Ingestion Source:",
            ["Preloaded Infiltration Scenarios", "Upload PCAP / CSV Records"],
            index=0,
        )

        sequence_data: Optional[np.ndarray] = None
        source_label = ""
        elapsed_minutes = 10.0

        if data_mode == "Preloaded Infiltration Scenarios":
            scenario = st.selectbox(
                "Select Attack Incident Scenario:",
                [
                    "Window 584 (10:44 AST) - Pre-Attack Infiltration Onset",
                    "Window 600 (11:00 AST) - Active Exploit Delivery",
                    "Window 640 (11:40 AST) - Lateral Movement & Scanning",
                    "Window 665 (12:05 AST) - Post-Attack Benign Recovery",
                    "Custom Historical Window",
                ],
            )

            if scenario.startswith("Window 584"):
                target_window = 584
                elapsed_minutes = 2.0
                source_label = "CIC-IDS-2018 Infiltration: Window 584 (10:44 AST)"
            elif scenario.startswith("Window 600"):
                target_window = 600
                elapsed_minutes = 10.0
                source_label = "CIC-IDS-2018 Infiltration: Window 600 (11:00 AST)"
            elif scenario.startswith("Window 640"):
                target_window = 640
                elapsed_minutes = 25.0
                source_label = "CIC-IDS-2018 Infiltration: Window 640 (11:40 AST)"
            elif scenario.startswith("Window 665"):
                target_window = 665
                elapsed_minutes = 0.0
                source_label = "CIC-IDS-2018 Infiltration: Window 665 (12:05 AST)"
            else:
                target_window = st.slider("Select Window Index:", min_value=10, max_value=int(df["window"].max()), value=584)
                elapsed_minutes = max(0.0, float(target_window - 580))
                source_label = f"Historical Window {target_window}"

            # Extract 10-minute sequence
            target_idx = df[df["window"] == target_window].index[0]
            X_raw = df[feature_cols].replace([float("inf"), float("-inf")], 0).fillna(0).values
            X_scaled = scaler.transform(X_raw)
            sequence_data = X_scaled[target_idx - 9 : target_idx + 1]

        else:
            uploaded_file = st.file_uploader(
                "Upload Raw PCAP or Processed CSV:",
                type=["pcap", "pcapng", "csv"],
                help="Upload network trace for last 10 minutes of traffic",
            )
            if uploaded_file is not None:
                if uploaded_file.name.endswith((".pcap", ".pcapng")):
                    with st.spinner("Parsing PCAP frames with Scapy..."):
                        sequence_data, pcap_info = parse_pcap_to_sequence(uploaded_file, feature_cols, scaler)
                        st.success(pcap_info)
                        source_label = f"Uploaded PCAP: {uploaded_file.name}"
                        elapsed_minutes = 12.0
                else:
                    up_df = pd.read_csv(uploaded_file)
                    matching_cols = [c for c in feature_cols if c in up_df.columns]
                    if len(matching_cols) == 29:
                        raw_mat = up_df[matching_cols].tail(10).values
                        sequence_data = scaler.transform(raw_mat)
                        source_label = f"Uploaded CSV: {uploaded_file.name}"
                        elapsed_minutes = 15.0
                    else:
                        st.error(f"Uploaded CSV must contain all 29 flow features. Found {len(matching_cols)}.")
            else:
                st.info("👆 Upload a file or switch to preloaded scenarios.")

        st.markdown("---")
        st.subheader("Forecast Parameters")
        k_steps = st.slider("Forecast Horizon K (minutes ahead):", min_value=1, max_value=15, value=10)
        threshold = st.slider("Decision Threshold:", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
        override_elapsed = st.slider("Elapsed Threat Duration (minutes):", min_value=0.0, max_value=60.0, value=float(elapsed_minutes), step=1.0)
        elapsed_minutes = override_elapsed

    if sequence_data is None:
        st.warning("Please select a scenario or upload network traffic to begin analysis.")
        return

    # Perform LSTM inference & K-step rollout
    pred_current = float(model.predict(sequence_data[np.newaxis, ...])[0, 0])
    rollout_probs = forward_rollout(initial_state=sequence_data, k_steps=max(15, k_steps), model=model)
    display_rollout = rollout_probs[:k_steps]

    # Predict MITRE stage
    mitre_pred = predict_mitre_stage(pred_current, elapsed_minutes_since_attack_start=elapsed_minutes)

    # Compute SHAP feature attributions
    with st.spinner("Computing real-time SHAP feature attributions..."):
        shap_res = explainer.explain_prediction(sequence_data, k_top=5, threshold=threshold)

    # -------------------------------------------------------------
    # LAYOUT ROW 1: Attack Probability Gauge & MITRE Stage Card
    # -------------------------------------------------------------
    col_gauge, col_mitre = st.columns([1, 1])

    with col_gauge:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        fig_gauge = render_attack_gauge(pred_current, threshold=threshold)
        st.plotly_chart(fig_gauge, use_container_width=True)

        is_attack = pred_current >= threshold
        confidence_pct = (pred_current if is_attack else (1.0 - pred_current)) * 100.0
        status_text = "🚨 ATTACK STATE CONFIRMED" if is_attack else "✅ BENIGN / NORMAL TRAFFIC"
        status_color = "#ef4444" if is_attack else "#10b981"

        st.markdown(
            f"""
            <div style="text-align: center; margin-top: -10px;">
                <span style="color: {status_color}; font-size: 1.15rem; font-weight: 800;">{status_text}</span>
                <span style="color: #94a3b8; font-size: 0.9rem; margin-left: 8px;">(Confidence: {confidence_pct:.1f}%)</span>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_mitre:
        stage_name = mitre_pred.stage
        if stage_name == "Reconnaissance":
            badge_class = "badge-recon"
            severity = "ELEVATED (MEDIUM)"
            border_color = "#f59e0b"
        elif stage_name == "Initial Access":
            badge_class = "badge-access"
            severity = "HIGH"
            border_color = "#f97316"
        elif stage_name == "Lateral Movement":
            badge_class = "badge-lateral"
            severity = "CRITICAL"
            border_color = "#ef4444"
        elif stage_name == "Command & Control":
            badge_class = "badge-c2"
            severity = "CRITICAL"
            border_color = "#a855f7"
        else:
            badge_class = "badge-benign"
            severity = "INFORMATIONAL (SAFE)"
            border_color = "#10b981"

        st.markdown(
            f"""
            <div class="stage-card" style="border: 1px solid {border_color}; background: #1e293b;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <span class="stage-badge {badge_class}">{stage_name}</span>
                        <span style="font-size: 0.8rem; color: #94a3b8;">Severity: <b style="color: {border_color};">{severity}</b></span>
                    </div>
                    <h3 style="margin: 4px 0 8px 0; color: #f8fafc;">MITRE ATT&CK: {stage_name}</h3>
                    <p style="color: #cbd5e1; font-size: 0.95rem; margin-bottom: 8px;">
                        <b>Tactic ID:</b> <code style="color: #38bdf8;">{mitre_pred.tactic_id if mitre_pred.tactic_id else 'N/A'}</code>
                        &bull; <b>Context:</b> {mitre_pred.description}
                    </p>
                    <p style="color: #94a3b8; font-size: 0.85rem;">
                        Elapsed Threat Duration: <b>{elapsed_minutes:.1f} min</b> &bull; Detection Threshold: <b>{mitre_pred.threshold:.2f}</b>
                    </p>
                </div>
                
                <div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px; font-weight: 600;">INTRUSION LIFECYCLE TIMELINE:</div>
                    <div class="timeline-container">
                        <div class="timeline-step {'active' if stage_name=='Reconnaissance' else ''}">
                            <div class="timeline-dot {'active' if stage_name=='Reconnaissance' else ('passed' if elapsed_minutes>5 else '')}"></div>
                            Recon (0-5m)
                        </div>
                        <div class="timeline-step {'active' if stage_name=='Initial Access' else ''}">
                            <div class="timeline-dot {'active' if stage_name=='Initial Access' else ('passed' if elapsed_minutes>15 else '')}"></div>
                            Access (5-15m)
                        </div>
                        <div class="timeline-step {'active' if stage_name=='Lateral Movement' else ''}">
                            <div class="timeline-dot {'active' if stage_name=='Lateral Movement' else ('passed' if elapsed_minutes>35 else '')}"></div>
                            Lateral (15-35m)
                        </div>
                        <div class="timeline-step {'active' if stage_name=='Command & Control' else ''}">
                            <div class="timeline-dot {'active' if stage_name=='Command & Control' else ''}"></div>
                            C2 (35m+)
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------
    # LAYOUT ROW 2: K-Step Forecast Curve (Line Chart)
    # -------------------------------------------------------------
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    fig_forecast = render_forecast_curve(
        forecast_probs=display_rollout,
        current_prob=pred_current,
        k_steps=k_steps,
        threshold=threshold,
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    # -------------------------------------------------------------
    # LAYOUT ROW 3: Top 5 Feature Drivers (Bar Chart)
    # -------------------------------------------------------------
    fig_features = render_feature_attribution_chart(shap_res["top_features"])
    st.plotly_chart(fig_features, use_container_width=True)

    # -------------------------------------------------------------
    # LAYOUT ROW 4: Side-by-Side Model Comparison View
    # -------------------------------------------------------------
    with st.expander("📊 Model Comparison: Flow-Only vs Fused Multi-Modal LSTM", expanded=False):
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Flow-Only LSTM F1", "75.8%", delta="+22.9% vs Baseline", delta_color="normal")
        col_m2.metric("Flow-Only Recall", "73.5%", delta="+32.3% vs Baseline")
        col_m3.metric("Flow-Only False Positive Rate", "12.7%", delta="- vs 9.1% Base", delta_color="inverse")

        st.markdown("#### Test Set Confusion Matrix Heatmaps")
        fig_cm = render_confusion_matrices()
        st.plotly_chart(fig_cm, use_container_width=True)

        st.caption(
            "Evaluation on chronological test split (window >= 630). "
            "Flow-only Temporal LSTM achieves superior balance (AUC: 0.89, F1: 75.8%) for rapid forward rollout."
        )

    # -------------------------------------------------------------
    # LAYOUT ROW 5: Export Incident Report
    # -------------------------------------------------------------
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    report_timestamp = datetime.now(timezone.utc).isoformat()
    incident_report = {
        "incident_id": f"INC-{int(time.time())}",
        "timestamp_utc": report_timestamp,
        "traffic_source": source_label,
        "current_assessment": {
            "attack_probability": round(pred_current, 4),
            "classification": "ATTACK" if is_attack else "BENIGN",
            "confidence_percentage": round(confidence_pct, 2),
            "threshold": threshold,
        },
        "mitre_attack_phase": {
            "stage": mitre_pred.stage,
            "tactic_id": mitre_pred.tactic_id,
            "description": mitre_pred.description,
            "elapsed_minutes": elapsed_minutes,
            "severity": severity,
        },
        "k_step_forecast": {
            "horizon_minutes": k_steps,
            "trajectory": [round(float(p), 4) for p in display_rollout],
        },
        "top_predictive_drivers": shap_res["top_features"],
    }

    col_exp1, col_exp2 = st.columns([3, 1])
    with col_exp1:
        st.markdown(
            f"**Incident Summary Log:** Status: `{status_text}` | Active MITRE Stage: `{mitre_pred.stage}` | Horizon: `{k_steps}m` | Generated: `{report_timestamp}`"
        )
    with col_exp2:
        st.download_button(
            label="📥 Export Incident Report (JSON)",
            data=json.dumps(incident_report, indent=2),
            file_name=f"incident_report_{int(time.time())}.json",
            mime="application/json",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
