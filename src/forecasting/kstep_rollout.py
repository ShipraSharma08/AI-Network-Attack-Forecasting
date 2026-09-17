#!/usr/bin/env python3
"""
K-Step Autoregressive Forward Forecasting Module
================================================
Implements multi-step autoregressive forward forecasting of network attack risk
and state trajectories using the trained flow-only Temporal LSTM.

Key Features:
  1. load_flow_lstm_and_scaler: Loads trained weights from models/lstm_model.h5
     (29 features, T=10) and fits StandardScaler strictly on training states (w <= 609).
  2. forward_rollout: Forward rollout function:
       - pred = model.predict(current_sequence[np.newaxis, ...])
       - attack_prob = pred[0, 0]
       - current_sequence = np.roll(current_sequence, -1, axis=0)
       - current_sequence[-1, :] = current_sequence[-2, :]
  3. evaluate_test_set_auc_pr: Computes AUC-ROC and Precision-Recall metrics
     over horizons K in {1, 5, 10, 15}, printing formatted summary:
     "Horizon=5min: AUC=0.85, Precision@80%=0.72"
  4. visualize_forecasting_curves: Generates publication-grade multi-panel figure
     showing attack probability rising, benign probability dropping, and ROC/PR curves.
     Saves plot to docs/kstep_forecasting_examples.png.
  5. CLI interface:
     python src/forecasting/kstep_rollout.py --k 10 --visualize
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.temporal_lstm import TemporalLSTM

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "state_transitions_clean.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "lstm_model.h5"
DEFAULT_PLOT_PATH = PROJECT_ROOT / "docs" / "kstep_forecasting_examples.png"
DEFAULT_OUTPUT_REPORT = PROJECT_ROOT / "reports" / "kstep_rollout_metrics.json"

BASELINE_METRICS = {
    "precision": 0.7368,
    "recall": 0.4118,
    "f1": 0.5283,
    "fpr": 0.0909,
    "tn": 50,
    "fp": 5,
    "fn": 20,
    "tp": 14,
}


def load_flow_lstm_and_scaler(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    data_path: Union[str, Path] = DEFAULT_DATA_PATH,
    device: str = "cpu",
) -> Tuple[TemporalLSTM, StandardScaler, List[str], pd.DataFrame]:
    """
    Load trained flow-only Temporal LSTM model weights and initialize StandardScaler
    strictly from training transitions (window <= 609) to eliminate data leakage.

    Returns:
      (model, scaler, feature_cols, df)
    """
    model_path = Path(model_path).resolve()
    data_path = Path(data_path).resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Transitions dataset not found: {data_path}")

    # 1. Load transitions data
    df = pd.read_csv(data_path)
    df = df.sort_values("window").reset_index(drop=True)

    feature_cols = [
        c for c in df.columns
        if c.startswith("current_") and c != "current_attack_label"
    ]
    if len(feature_cols) != 29:
        raise ValueError(f"Expected 29 flow features, found {len(feature_cols)}")

    # 2. Fit StandardScaler strictly on training windows (w <= 609)
    train_mask = df["window"] <= 609
    X_train_raw = df.loc[train_mask, feature_cols].replace([float("inf"), float("-inf")], 0).fillna(0).values
    scaler = StandardScaler()
    scaler.fit(X_train_raw)

    # 3. Instantiate TemporalLSTM
    model = TemporalLSTM(input_dim=len(feature_cols), hidden_dim=128, dense_dim=64, dropout_rate=0.3)

    # 4. Load weights from .h5 or .pt
    if model_path.suffix == ".h5" and model_path.exists():
        with h5py.File(model_path, "r") as f:
            state_dict = {name: torch.tensor(f[name][()]) for name in f.keys()}
        model.load_state_dict(state_dict)
    elif model_path.with_suffix(".pt").exists():
        pt_path = model_path.with_suffix(".pt")
        model.load_state_dict(torch.load(pt_path, map_location=device))
    elif model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model file not found at {model_path} or {model_path.with_suffix('.pt')}")

    model = model.to(torch.device(device))
    model.eval()

    return model, scaler, feature_cols, df


def forward_rollout(
    initial_state: np.ndarray,
    k_steps: int,
    model: Any,
) -> np.ndarray:
    """
    Forward rollout function:
    Returns:
        predictions: array of length k_steps
        - predictions[0] = prob(attack at t+1)
        - predictions[1] = prob(attack at t+2)
        - ...
        - predictions[k-1] = prob(attack at t+k)
    """
    outputs = []
    current_sequence = initial_state.copy()  # (10, 29)

    for step in range(k_steps):
        # Predict next state: LSTM(current_sequence) -> attack_prob + next_state
        if hasattr(model, "predict"):
            pred = model.predict(current_sequence[np.newaxis, ...])
        else:
            with torch.no_grad():
                tensor_seq = torch.tensor(current_sequence[np.newaxis, ...], dtype=torch.float32)
                pred = model(tensor_seq).cpu().numpy()

        attack_prob = float(pred[0, 0])  # sigmoid output
        outputs.append(attack_prob)

        # For autoregressive rollout, approximate next state as:
        # - Use last observed state as placeholder (conservative)
        # - Shift window: drop oldest minute, append predicted state
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, :] = current_sequence[-2, :]  # repeat last known state

    return np.array(outputs)


# Exact alias
rollout_forward = forward_rollout


def train_state_transition_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler: StandardScaler,
    alpha: float = 100.0,
) -> Ridge:
    """
    Train an L2-regularized linear transition dynamics model (S_t -> S_{t+1})
    strictly on training states (window <= 609).
    """
    next_cols = [c.replace("current_", "next_") for c in feature_cols]
    train_mask = df["window"] <= 609
    X_cur_raw = df.loc[train_mask, feature_cols].replace([float("inf"), float("-inf")], 0).fillna(0).values
    X_next_raw = df.loc[train_mask, next_cols].replace([float("inf"), float("-inf")], 0).fillna(0).values

    X_cur_scaled = scaler.transform(X_cur_raw)
    X_next_scaled = scaler.transform(X_next_raw)

    transition_model = Ridge(alpha=alpha, random_state=42)
    transition_model.fit(X_cur_scaled, X_next_scaled)

    return transition_model


def rollout_k_steps(
    initial_seq: np.ndarray,
    k_steps: int,
    model: Any,
    strategy: str = "repeat_last",
    transition_model: Optional[Ridge] = None,
    threshold: float = 0.5,
    device: str = "cpu",
) -> Dict[str, Union[List[int], np.ndarray]]:
    """
    Perform autoregressive forward rollout for K steps ahead with metadata.
    """
    if strategy == "repeat_last" or transition_model is None:
        probs = forward_rollout(initial_seq, k_steps=k_steps, model=model)
        preds = (probs >= threshold).astype(int)
        return {
            "horizons": list(range(1, k_steps + 1)),
            "probabilities": probs,
            "predictions": preds,
            "predicted_states": np.repeat(initial_seq[-1:], k_steps, axis=0),
        }

    current_window = initial_seq.copy()
    probabilities = []
    predicted_states = []

    for step in range(1, k_steps + 1):
        if hasattr(model, "predict"):
            pred = model.predict(current_window[np.newaxis, ...])
        else:
            with torch.no_grad():
                dev = torch.device(device)
                tensor_seq = torch.tensor(current_window[np.newaxis, ...], dtype=torch.float32, device=dev)
                pred = model(tensor_seq).cpu().numpy()

        prob = float(pred[0, 0])
        probabilities.append(prob)

        last_state = current_window[-1:]
        next_state_pred = transition_model.predict(last_state)
        predicted_states.append(next_state_pred[0])
        current_window = np.vstack([current_window[1:], next_state_pred])

    probs_arr = np.array(probabilities, dtype=np.float32)
    preds_arr = (probs_arr >= threshold).astype(int)
    states_arr = np.array(predicted_states, dtype=np.float32)

    return {
        "horizons": list(range(1, k_steps + 1)),
        "probabilities": probs_arr,
        "predictions": preds_arr,
        "predicted_states": states_arr,
    }


def evaluate_test_set_auc_pr(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler: StandardScaler,
    model: Any,
    horizons: List[int] = [1, 5, 10, 15],
    split: str = "test",
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate on test set across horizons K in {1, 5, 10, 15}.
    Computes AUC-ROC, Precision-Recall curves, and Precision@80% Recall.
    Outputs: 'Horizon=5min: AUC=0.85, Precision@80%=0.72'
    """
    max_k = max(horizons)
    X_raw = df[feature_cols].replace([float("inf"), float("-inf")], 0).fillna(0).values
    X_scaled = scaler.transform(X_raw)

    if split == "test":
        eval_indices = [i for i in range(len(df)) if df.loc[i, "window"] >= 630]
    else:
        eval_indices = [i for i in range(len(df)) if 610 <= df.loc[i, "window"] <= 629]

    horizon_data = {
        k: {"targets": [], "probs": []}
        for k in horizons
    }

    for idx in eval_indices:
        if idx < 9:
            continue
        init_seq = X_scaled[idx - 9 : idx + 1]
        preds = forward_rollout(initial_state=init_seq, k_steps=max_k, model=model)

        for k in horizons:
            target_idx = idx + k - 1
            if target_idx < len(df):
                target = int(df.loc[target_idx, "next_attack_label"])
                prob = float(preds[k - 1])
                horizon_data[k]["targets"].append(target)
                horizon_data[k]["probs"].append(prob)

    results = {}
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION: AUC-ROC & PRECISION@80% RECALL")
    print("=" * 70)

    for k in horizons:
        y_true = np.array(horizon_data[k]["targets"])
        y_probs = np.array(horizon_data[k]["probs"])

        if len(np.unique(y_true)) > 1:
            auc_roc = float(roc_auc_score(y_true, y_probs))
            precs, recs, _ = precision_recall_curve(y_true, y_probs)
            # Precision where recall >= 0.80
            mask_80 = recs >= 0.80
            prec_at_80 = float(np.max(precs[mask_80])) if np.any(mask_80) else 0.0
            auc_pr = float(auc(recs, precs))
        else:
            auc_roc = 0.0
            prec_at_80 = 0.0
            auc_pr = 0.0

        # Exact required output format
        print(f"Horizon={k}min: AUC={auc_roc:.2f}, Precision@80%={prec_at_80:.2f}")

        # Compute F1 at standard 0.5 threshold
        y_pred = (y_probs >= 0.5).astype(int)
        prec_std = float(precision_score(y_true, y_pred, zero_division=0))
        rec_std = float(recall_score(y_true, y_pred, zero_division=0))
        f1_std = float(f1_score(y_true, y_pred, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        results[k] = {
            "horizon": k,
            "auc": round(auc_roc, 4),
            "precision_at_80_recall": round(prec_at_80, 4),
            "auc_pr": round(auc_pr, 4),
            "precision": round(prec_std, 4),
            "recall": round(rec_std, 4),
            "f1": round(f1_std, 4),
            "fpr": round(fpr, 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "samples": len(y_true),
        }

    print("=" * 70)
    return results


def visualize_forecasting_curves(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler: StandardScaler,
    model: Any,
    save_path: Union[str, Path] = DEFAULT_PLOT_PATH,
    horizons: List[int] = [1, 5, 10, 15],
):
    """
    Visualize forecasting curves and save to docs/kstep_forecasting_examples.png:
      - Sample benign window: show predicted attack probability dropping over 15 minutes
      - Sample attack window: show predicted attack probability rising
      - Precision-Recall curves over horizons K=1, 5, 10, 15
      - ROC curves over horizons K=1, 5, 10, 15
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_path = Path(save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    X_raw = df[feature_cols].replace([float("inf"), float("-inf")], 0).fillna(0).values
    X_scaled = scaler.transform(X_raw)

    # 1. Sample Attack Window (Rising over 15 minutes): Window 584 (10:44 AST, 6 min prior to 10:50 onset)
    idx_attack = df[df["window"] == 584].index[0]
    attack_rollout = forward_rollout(X_scaled[idx_attack - 9 : idx_attack + 1], k_steps=15, model=model)

    # 2. Sample Benign Window (Dropping over 15 minutes): Window 665 (12:05 AST, immediately following attack cessation)
    idx_benign = df[df["window"] == 665].index[0]
    benign_rollout = forward_rollout(X_scaled[idx_benign - 9 : idx_benign + 1], k_steps=15, model=model)

    # 3. Collect test set curves
    test_indices = [i for i in range(len(df)) if df.loc[i, "window"] >= 630]
    horizon_data = {k: {"probs": [], "targets": []} for k in horizons}
    for idx in test_indices:
        preds = forward_rollout(X_scaled[idx - 9 : idx + 1], k_steps=max(horizons), model=model)
        for k in horizons:
            t_idx = idx + k - 1
            if t_idx < len(df):
                horizon_data[k]["probs"].append(preds[k - 1])
                horizon_data[k]["targets"].append(int(df.loc[t_idx, "next_attack_label"]))

    # Setup figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Attack Window (Rising)
    axes[0, 0].plot(range(1, 16), attack_rollout, marker="o", color="#e74c3c", linewidth=2.5, label="Predicted Attack Risk")
    axes[0, 0].axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Alarm Threshold (0.5)")
    axes[0, 0].set_title("Sample Attack Window (Win 584, 10:44 AST): Risk Rising Ahead of Infiltration", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Forecasting Horizon (minutes ahead)")
    axes[0, 0].set_ylabel("Attack Probability P(Attack)")
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Plot 2: Benign Window (Dropping)
    axes[0, 1].plot(range(1, 16), benign_rollout, marker="s", color="#2ecc71", linewidth=2.5, label="Predicted Attack Risk")
    axes[0, 1].axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Alarm Threshold (0.5)")
    axes[0, 1].set_title("Sample Benign Window (Win 665, 12:05 AST): Risk Dropping After Attack", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Forecasting Horizon (minutes ahead)")
    axes[0, 1].set_ylabel("Attack Probability P(Attack)")
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Plot 3: Precision-Recall Curves
    colors = {1: "#3498db", 5: "#9b59b6", 10: "#e67e22", 15: "#e74c3c"}
    for k in horizons:
        y_t = np.array(horizon_data[k]["targets"])
        p = np.array(horizon_data[k]["probs"])
        prec, rec, _ = precision_recall_curve(y_t, p)
        axes[1, 0].plot(rec, prec, label=f"K={k} min", color=colors.get(k, "#333333"), linewidth=2.0)
    axes[1, 0].set_title("Precision-Recall Curves Across Forecast Horizons", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Recall")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Plot 4: ROC Curves
    for k in horizons:
        y_t = np.array(horizon_data[k]["targets"])
        p = np.array(horizon_data[k]["probs"])
        fpr, tpr, _ = roc_curve(y_t, p)
        auc_score = roc_auc_score(y_t, p)
        axes[1, 1].plot(fpr, tpr, label=f"K={k} min (AUC={auc_score:.2f})", color=colors.get(k, "#333333"), linewidth=2.0)
    axes[1, 1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Chance")
    axes[1, 1].set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("False Positive Rate (FPR)")
    axes[1, 1].set_ylabel("True Positive Rate (TPR)")
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"\n[VISUALIZATION] Forecasting curves successfully saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="K-Step Autoregressive Attack Forecasting on Flow-Only LSTM")
    parser.add_argument("--k", type=int, default=10, help="Forecasting rollout horizon K (default: 10)")
    parser.add_argument("--visualize", action="store_true", help="Generate and save visualization plots to docs/kstep_forecasting_examples.png")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Path to state_transitions_clean.csv")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to models/lstm_model.h5")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 15], help="Evaluation horizons K in minutes (default: 1 5 10 15)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (default: 0.5)")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device: 'cpu' or 'cuda'")
    parser.add_argument("--save-metrics", type=Path, default=DEFAULT_OUTPUT_REPORT, help="Path to save metrics JSON")

    args = parser.parse_args()

    print("\n" + "=" * 75)
    print(f"K-STEP AUTOREGRESSIVE NETWORK ATTACK FORECASTING ENGINE (K={args.k})")
    print("=" * 75)

    # 1. Load LSTM model and StandardScaler
    print(f"[1/3] Loading trained Temporal LSTM from {args.model_path} ...")
    model, scaler, feature_cols, df = load_flow_lstm_and_scaler(
        model_path=args.model_path,
        data_path=args.data_path,
        device=args.device,
    )
    print(f"      Model loaded successfully (29 features, T=10)")

    # 2. Evaluate on test set
    print(f"[2/3] Evaluating test set across horizons K={args.horizons} ...")
    eval_results = evaluate_test_set_auc_pr(
        df=df,
        feature_cols=feature_cols,
        scaler=scaler,
        model=model,
        horizons=args.horizons,
        split="test",
    )

    # 3. Visualize if requested
    if args.visualize:
        print("[3/3] Generating forecasting visualization plots...")
        visualize_forecasting_curves(
            df=df,
            feature_cols=feature_cols,
            scaler=scaler,
            model=model,
            save_path=DEFAULT_PLOT_PATH,
            horizons=args.horizons,
        )
    else:
        print("[3/3] Visualization skipped (pass --visualize to generate docs/kstep_forecasting_examples.png).")

    # Save metrics JSON
    if args.save_metrics:
        args.save_metrics.parent.mkdir(parents=True, exist_ok=True)
        report_payload = {
            "model_path": str(args.model_path),
            "data_path": str(args.data_path),
            "k_steps": args.k,
            "threshold": args.threshold,
            "horizons": eval_results,
            "baseline_metrics": BASELINE_METRICS,
        }
        with open(args.save_metrics, "w") as f:
            json.dump(report_payload, f, indent=2)
        print(f"[REPORT] Metrics successfully saved to {args.save_metrics}\n")


if __name__ == "__main__":
    main()
