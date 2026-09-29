#!/usr/bin/env python3
"""
Genuine Autoregressive World-Model Rollout Evaluation
=====================================================
Uses temporal_lstm_worldmodel.py's DUAL-HEAD model:
  - State head: predicts S_hat(t+1) in R^29 (standardized)
  - Attack head: predicts P(attack at t+1)

At each rollout step k:
  1. Feed current window [S(t-9+k)...S(t+k)] through the model
  2. Record attack probability P(attack at t+k+1) from classification head
  3. Record predicted state S_hat(t+k+1) from state-regression head
  4. Slide window: drop oldest, append S_hat(t+k+1) as newest timestep
  5. Compute MSE(k) against ground truth S(t+k+1) if available

Reports BEFORE (old frozen-state method) vs AFTER (genuine state rollout)
for K=1,5,10,15 minute horizons on the test set (window >= 630).
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, auc,
    precision_score, recall_score, f1_score,
    confusion_matrix,
)
import torch
import h5py

PROJECT_ROOT = Path(__file__).resolve().parent
# Walk up until we find src/
while not (PROJECT_ROOT / "src").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.models.temporal_lstm_worldmodel import TemporalLSTMWorldModel

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "state_transitions_clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_worldmodel_model.h5"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "kstep_worldmodel_rollout_metrics.json"

# Old method results from reports/kstep_rollout_metrics.json (frozen-state classifier-only LSTM)
OLD_METHOD_RESULTS = {
    1:  {"auc": 0.8893, "precision_at_80_recall": 0.7632, "auc_pr": 0.7589, "f1": 0.7576, "precision": 0.7812, "recall": 0.7353, "fpr": 0.1273},
    5:  {"auc": 0.7945, "precision_at_80_recall": 0.5714, "auc_pr": 0.6068, "f1": 0.5965, "precision": 0.6296, "recall": 0.5667, "fpr": 0.1818},
    10: {"auc": 0.5847, "precision_at_80_recall": 0.3390, "auc_pr": 0.3547, "f1": 0.4528, "precision": 0.4286, "recall": 0.4800, "fpr": 0.2909},
    15: {"auc": 0.6091, "precision_at_80_recall": 0.3200, "auc_pr": 0.3194, "f1": 0.4255, "precision": 0.3704, "recall": 0.5000, "fpr": 0.3091},
}


def load_worldmodel_and_data(device: str = "cpu"):
    """Load the dual-head world model and prepare data with scaler."""
    print(f"[1/4] Loading dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH).sort_values("window").reset_index(drop=True)

    cur_cols = [c for c in df.columns if c.startswith("current_") and c != "current_attack_label"]
    nxt_cols = [c for c in df.columns if c.startswith("next_") and c not in ["next_attack_label", "next_timestamp"]]

    assert len(cur_cols) == 29, f"Expected 29 features, got {len(cur_cols)}"

    X_raw = df[cur_cols].replace([np.inf, -np.inf], 0).fillna(0).values
    nxt_raw = df[nxt_cols].replace([np.inf, -np.inf], 0).fillna(0).values
    y_all = df["next_attack_label"].values
    windows = df["window"].values

    # Fit scaler on training data only
    train_mask = windows <= 609
    scaler = StandardScaler()
    scaler.fit(X_raw[train_mask])

    X_scaled = scaler.transform(X_raw)
    nxt_scaled = scaler.transform(nxt_raw)

    # Load model
    print(f"[2/4] Loading World Model from {MODEL_PATH} ...")
    model = TemporalLSTMWorldModel(input_dim=29, hidden_dim=128, dense_dim=64, dropout_rate=0.3)

    if MODEL_PATH.exists():
        with h5py.File(MODEL_PATH, "r") as f:
            state_dict = {name: torch.tensor(f[name][()]) for name in f.keys()}
        model.load_state_dict(state_dict)
    elif MODEL_PATH.with_suffix(".pt").exists():
        model.load_state_dict(torch.load(MODEL_PATH.with_suffix(".pt"), map_location=device))
    else:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model.to(torch.device(device))
    model.eval()
    print(f"      Model loaded successfully (dual-head: state + classification)")

    return model, scaler, df, X_scaled, nxt_scaled, nxt_raw, y_all, windows, cur_cols



def load_worldmodel_and_scaler(
    model_path: Union[str, Path] = MODEL_PATH,
    data_path: Union[str, Path] = DATA_PATH,
    device: str = "cpu",
) -> Tuple[TemporalLSTMWorldModel, StandardScaler, List[str], pd.DataFrame]:
    """
    Load trained dual-head Temporal LSTM World Model and fit StandardScaler
    strictly from training transitions (window <= 609) to eliminate data leakage.

    Returns:
      (model, scaler, feature_cols, df)
    """
    model_path = Path(model_path).resolve()
    data_path = Path(data_path).resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Transitions dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    df = df.sort_values("window").reset_index(drop=True)

    feature_cols = [
        c for c in df.columns
        if c.startswith("current_") and c != "current_attack_label"
    ]

    train_mask = df["window"] <= 609
    X_raw = df.loc[train_mask, feature_cols].replace([float("inf"), float("-inf")], 0).fillna(0).values

    scaler = StandardScaler()
    scaler.fit(X_raw)

    model = TemporalLSTMWorldModel(input_dim=len(feature_cols), hidden_dim=128, dense_dim=64, dropout_rate=0.3)

    if model_path.suffix == ".h5" and model_path.exists():
        with h5py.File(model_path, "r") as f:
            state_dict = {name: torch.tensor(f[name][()]) for name in f.keys()}
        model.load_state_dict(state_dict)
    elif model_path.with_suffix(".pt").exists():
        model.load_state_dict(torch.load(model_path.with_suffix(".pt"), map_location=device))
    elif model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"World Model not found at {model_path}")

    model.to(torch.device(device))
    model.eval()

    return model, scaler, feature_cols, df


def forward_rollout(
    initial_state: np.ndarray,
    k_steps: int = 15,
    model: Any = None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Genuine autoregressive K-step World Model rollout.
    At each step:
      1. Predicts P(attack at t+k) and next state S_hat(t+k)
      2. Recursively feeds S_hat(t+k) back into the input sequence
    Returns:
      predictions: 1D numpy array of length k_steps containing attack probabilities.
    """
    probs, _ = genuine_state_rollout(
        initial_window=initial_state,
        k_steps=k_steps,
        model=model,
        device=device,
    )
    return probs


def forward_worldmodel_rollout(
    initial_state: np.ndarray,
    k_steps: int = 15,
    model: Any = None,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genuine autoregressive K-step World Model rollout returning both:
      1. attack_probs: (k_steps,) array
      2. predicted_states: (k_steps, 29) array in standardized space
    """
    return genuine_state_rollout(
        initial_window=initial_state,
        k_steps=k_steps,
        model=model,
        device=device,
    )


def genuine_state_rollout(
    initial_window: np.ndarray,  # (T=10, 29) standardized
    k_steps: int,
    model: TemporalLSTMWorldModel,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genuine autoregressive rollout using the world model's state-regression head.

    At each step:
      1. Forward pass: model(window) -> (state_pred_29, attack_prob)
      2. Record attack_prob
      3. Record state_pred (standardized)
      4. Slide window: drop oldest, append state_pred as newest timestep

    Returns:
      attack_probs: (k_steps,) array of predicted attack probabilities
      state_preds:  (k_steps, 29) array of predicted next states (standardized)
    """
    attack_probs = []
    state_preds = []
    window = initial_window.copy()  # (10, 29)

    for step in range(k_steps):
        x_tensor = torch.tensor(window[np.newaxis, ...], dtype=torch.float32, device=device)
        with torch.no_grad():
            s_pred, a_pred = model(x_tensor)

        attack_prob = float(a_pred[0, 0].cpu().numpy())
        s_hat = s_pred[0].cpu().numpy()  # (29,) standardized next state

        attack_probs.append(attack_prob)
        state_preds.append(s_hat)

        # Slide window: drop oldest timestep, append predicted state
        window = np.roll(window, -1, axis=0)
        window[-1, :] = s_hat  # GENUINE: feed back the predicted 29-feature state

    return np.array(attack_probs), np.array(state_preds)


def frozen_state_rollout(
    initial_window: np.ndarray,
    k_steps: int,
    model: TemporalLSTMWorldModel,
    device: str = "cpu",
) -> np.ndarray:
    """
    Old 'frozen state' rollout for comparison:
    copies the last-known state into each new slot (no state regression feedback).
    """
    attack_probs = []
    window = initial_window.copy()

    for step in range(k_steps):
        x_tensor = torch.tensor(window[np.newaxis, ...], dtype=torch.float32, device=device)
        with torch.no_grad():
            _, a_pred = model(x_tensor)

        attack_prob = float(a_pred[0, 0].cpu().numpy())
        attack_probs.append(attack_prob)

        window = np.roll(window, -1, axis=0)
        window[-1, :] = window[-2, :]  # OLD: copy last known state

    return np.array(attack_probs)


def compute_precision_at_recall(y_true, y_prob, target_recall=0.80):
    """Find the precision at approximately target_recall on the PR curve."""
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    # Find index where recall >= target_recall
    valid = rec_arr >= target_recall
    if valid.any():
        return float(prec_arr[valid][-1])
    return 0.0


def evaluate_horizons(
    model, X_scaled, nxt_scaled, nxt_raw, y_all, windows, df,
    scaler, horizons=[1, 5, 10, 15], seq_len=10, device="cpu",
):
    """
    Evaluate both genuine-state and frozen-state rollouts across horizons.
    Also compute per-step MSE(k) for k=1..max(horizons).
    """
    max_k = max(horizons)
    test_start_indices = []

    # Find valid test starting points (window >= 630 and enough history)
    for i in range(len(df)):
        if windows[i] >= 630 and i >= seq_len - 1:
            test_start_indices.append(i)

    print(f"\n[3/4] Evaluating {len(test_start_indices)} test windows across K={horizons} ...")
    print(f"      Max rollout horizon: {max_k} steps")

    # Storage for horizon-level metrics
    genuine_horizon_data = {k: {"probs": [], "targets": []} for k in horizons}
    frozen_horizon_data = {k: {"probs": [], "targets": []} for k in horizons}

    # Storage for per-step MSE accumulation
    step_mse_accum = {k: [] for k in range(1, max_k + 1)}  # MSE at each rollout step

    for idx in test_start_indices:
        initial_window = X_scaled[idx - seq_len + 1: idx + 1]  # (10, 29)

        # Genuine autoregressive rollout
        genuine_probs, genuine_states = genuine_state_rollout(
            initial_window, max_k, model, device
        )

        # Frozen-state rollout (old method)
        frozen_probs = frozen_state_rollout(
            initial_window, max_k, model, device
        )

        # Collect per-step state prediction error
        for k in range(1, max_k + 1):
            target_idx = idx + k
            if target_idx < len(df):
                # Ground truth next state at t+k (standardized)
                true_next_state = nxt_scaled[target_idx - 1]  # nxt_scaled[i] = S(t+1) for row i
                pred_state = genuine_states[k - 1]
                mse_k = float(np.mean((pred_state - true_next_state) ** 2))
                step_mse_accum[k].append(mse_k)

        # Collect horizon-level attack predictions
        for k in horizons:
            target_idx = idx + k
            if target_idx < len(df):
                true_label = int(y_all[target_idx])
                genuine_horizon_data[k]["probs"].append(genuine_probs[k - 1])
                genuine_horizon_data[k]["targets"].append(true_label)
                frozen_horizon_data[k]["probs"].append(frozen_probs[k - 1])
                frozen_horizon_data[k]["targets"].append(true_label)

    # Compute metrics for each horizon
    results = {"genuine_state_rollout": {}, "frozen_state_rollout": {}, "per_step_mse": {}}

    for method_name, horizon_data in [("genuine_state_rollout", genuine_horizon_data),
                                       ("frozen_state_rollout", frozen_horizon_data)]:
        for k in horizons:
            y_true = np.array(horizon_data[k]["targets"])
            y_prob = np.array(horizon_data[k]["probs"])

            if len(y_true) < 2 or len(np.unique(y_true)) < 2:
                results[method_name][str(k)] = {"error": "insufficient data"}
                continue

            auc_roc = roc_auc_score(y_true, y_prob)
            prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
            auc_pr = auc(rec_arr, prec_arr)
            p_at_80r = compute_precision_at_recall(y_true, y_prob, 0.80)

            y_pred = (y_prob >= 0.5).astype(int)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            results[method_name][str(k)] = {
                "horizon": k,
                "auc_roc": round(auc_roc, 4),
                "auc_pr": round(auc_pr, 4),
                "precision_at_80_recall": round(p_at_80r, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "fpr": round(fpr, 4),
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
                "samples": len(y_true),
            }

    # Per-step MSE
    for k in range(1, max_k + 1):
        if step_mse_accum[k]:
            results["per_step_mse"][str(k)] = {
                "step": k,
                "mean_mse": round(float(np.mean(step_mse_accum[k])), 6),
                "std_mse": round(float(np.std(step_mse_accum[k])), 6),
                "median_mse": round(float(np.median(step_mse_accum[k])), 6),
                "n_samples": len(step_mse_accum[k]),
            }

    return results


def print_comparison_table(results, horizons=[1, 5, 10, 15]):
    """Print formatted BEFORE vs AFTER comparison."""

    print("\n" + "=" * 110)
    print("BEFORE / AFTER COMPARISON: FROZEN-STATE ROLLOUT vs GENUINE WORLD-MODEL STATE ROLLOUT")
    print("=" * 110)
    print(f"{'Horizon':>8} | {'Method':<28} | {'AUC-ROC':>8} | {'AUC-PR':>8} | {'P@80%R':>8} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'FPR':>6} | {'N':>4}")
    print("-" * 110)

    for k in horizons:
        ks = str(k)
        # Old method (from stored results)
        old = OLD_METHOD_RESULTS.get(k, {})
        print(f"{'K=' + str(k) + 'min':>8} | {'OLD: Frozen-State (clf LSTM)':<28} | "
              f"{old.get('auc', 0):>8.4f} | {old.get('auc_pr', 0):>8.4f} | "
              f"{old.get('precision_at_80_recall', 0):>8.4f} | {old.get('precision', 0):>6.4f} | "
              f"{old.get('recall', 0):>6.4f} | {old.get('f1', 0):>6.4f} | {old.get('fpr', 0):>6.4f} | {'—':>4}")

        # Frozen state (new model, old method)
        frozen = results["frozen_state_rollout"].get(ks, {})
        if "error" not in frozen:
            print(f"{'':>8} | {'NEW-A: Frozen-State (WM)':<28} | "
                  f"{frozen.get('auc_roc', 0):>8.4f} | {frozen.get('auc_pr', 0):>8.4f} | "
                  f"{frozen.get('precision_at_80_recall', 0):>8.4f} | {frozen.get('precision', 0):>6.4f} | "
                  f"{frozen.get('recall', 0):>6.4f} | {frozen.get('f1', 0):>6.4f} | {frozen.get('fpr', 0):>6.4f} | "
                  f"{frozen.get('samples', 0):>4}")

        # Genuine state rollout
        genuine = results["genuine_state_rollout"].get(ks, {})
        if "error" not in genuine:
            print(f"{'':>8} | {'NEW-B: Genuine State (WM)':<28} | "
                  f"{genuine.get('auc_roc', 0):>8.4f} | {genuine.get('auc_pr', 0):>8.4f} | "
                  f"{genuine.get('precision_at_80_recall', 0):>8.4f} | {genuine.get('precision', 0):>6.4f} | "
                  f"{genuine.get('recall', 0):>6.4f} | {genuine.get('f1', 0):>6.4f} | {genuine.get('fpr', 0):>6.4f} | "
                  f"{genuine.get('samples', 0):>4}")

        print("-" * 110)

    print("=" * 110)


def print_mse_compounding_table(results, max_k=15):
    """Print per-step MSE table showing error compounding."""
    mse_data = results.get("per_step_mse", {})
    if not mse_data:
        print("\n[WARNING] No per-step MSE data available.")
        return

    print("\n" + "=" * 80)
    print("PER-STEP STATE PREDICTION ERROR: MSE(k) COMPOUNDING ANALYSIS")
    print("(Does prediction error grow as rollout horizon k increases?)")
    print("=" * 80)
    print(f"{'Step k':>7} | {'Mean MSE':>12} | {'Std MSE':>12} | {'Median MSE':>12} | {'N Samples':>10}")
    print("-" * 80)

    for k in range(1, max_k + 1):
        ks = str(k)
        if ks in mse_data:
            d = mse_data[ks]
            print(f"{k:>7} | {d['mean_mse']:>12.6f} | {d['std_mse']:>12.6f} | {d['median_mse']:>12.6f} | {d['n_samples']:>10}")

    print("=" * 80)

    # Summary
    k1 = mse_data.get("1", {}).get("mean_mse", 0)
    k15 = mse_data.get(str(max_k), {}).get("mean_mse", 0)
    if k1 > 0:
        ratio = k15 / k1
        print(f"\n[COMPOUNDING ANALYSIS] MSE at k=1: {k1:.6f}, MSE at k={max_k}: {k15:.6f}")
        print(f"                      Compounding ratio (k={max_k}/k=1): {ratio:.2f}x")
        if ratio > 2.0:
            print(f"                      [EXPECTED] Error compounds significantly — typical of autoregressive world models.")
        else:
            print(f"                      Error growth is moderate — model dynamics are relatively stable.")


def main():
    print("\n" + "=" * 80)
    print("GENUINE AUTOREGRESSIVE WORLD-MODEL ROLLOUT EVALUATION")
    print("Using temporal_lstm_worldmodel.py dual-head (state + classification)")
    print("=" * 80)

    device = "cpu"

    model, scaler, df, X_scaled, nxt_scaled, nxt_raw, y_all, windows, cur_cols = \
        load_worldmodel_and_data(device)

    horizons = [1, 5, 10, 15]

    results = evaluate_horizons(
        model, X_scaled, nxt_scaled, nxt_raw, y_all, windows, df,
        scaler, horizons=horizons, seq_len=10, device=device,
    )

    print_comparison_table(results, horizons)
    print_mse_compounding_table(results, max_k=15)

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[REPORT] Metrics saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
