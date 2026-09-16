#!/usr/bin/env python3
"""
SHAP Feature Attribution Module for Network Attack Forecasting
==============================================================
Provides model explainability for the flow-only Temporal LSTM using SHAP values:
  1. Loads trained LSTM and training sequences (w <= 609).
  2. Extracts the LSTM's last hidden layer as surrogate model.
  3. Uses KernelExplainer(model_predict, background_data) to compute Shapley values.
  4. Aggregates SHAP contributions across time steps T=10 for each of the 29 flow features.
  5. Ranks top-K features driving the attack or benign prediction.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.kstep_rollout import load_flow_lstm_and_scaler
from src.models.temporal_lstm import TemporalLSTM

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "state_transitions_clean.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "lstm_model.h5"


class LSTMSurrogateModel(nn.Module):
    """
    Extracts the LSTM's last hidden layer representation and subsequent dense layers
    as an explainer-compatible surrogate model.
    """

    def __init__(self, full_model: TemporalLSTM):
        super().__init__()
        self.lstm = full_model.lstm
        self.dropout1 = full_model.dropout1
        self.dense1 = full_model.dense1
        self.relu = full_model.relu
        self.dropout2 = full_model.dropout2
        self.dense2 = full_model.dense2
        self.sigmoid = full_model.sigmoid

    def extract_last_hidden(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Extract the last hidden state h_T from LSTM."""
        lstm_out, _ = self.lstm(x_seq)
        return lstm_out[:, -1, :]

    def predict_from_hidden(self, h_last: torch.Tensor) -> torch.Tensor:
        """Forward pass from last hidden representation to attack probability."""
        out = self.dropout1(h_last)
        out = self.relu(self.dense1(out))
        out = self.dropout2(out)
        out = self.sigmoid(self.dense2(out))
        return out

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        h_last = self.extract_last_hidden(x_seq)
        return self.predict_from_hidden(h_last)


class AttackExplainer:
    """
    Wrapper for SHAP KernelExplainer on the Temporal LSTM network forecasting model.
    """

    def __init__(
        self,
        model: TemporalLSTM,
        scaler: StandardScaler,
        feature_cols: List[str],
        background_sequences: np.ndarray,
        baseline_stats: Optional[pd.DataFrame] = None,
        nsamples: int = 50,
        device: str = "cpu",
    ):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.device = torch.device(device)
        self.nsamples = nsamples
        self.surrogate = LSTMSurrogateModel(model).to(self.device)
        self.surrogate.eval()

        # Background data: shape (N, 10, 29) -> flattened (N, 290)
        self.n_bg = len(background_sequences)
        self.bg_flat = background_sequences.reshape(self.n_bg, -1)

        # Precompute baseline feature means and stds for human-readable annotations
        self.baseline_stats = baseline_stats

        # Initialize KernelExplainer
        def model_predict(x_flat: np.ndarray) -> np.ndarray:
            """Evaluates surrogate model given flattened or 3D sequence arrays."""
            if x_flat.ndim == 1:
                x_flat = x_flat[np.newaxis, :]
            batch_size = x_flat.shape[0]
            x_seq = x_flat.reshape(batch_size, 10, len(feature_cols))
            with torch.no_grad():
                tensor_x = torch.tensor(x_seq, dtype=torch.float32, device=self.device)
                preds = self.surrogate(tensor_x).cpu().numpy()
            return preds.flatten()

        self.model_predict_fn = model_predict
        # Use a compact background subset for efficiency
        bg_subset = self.bg_flat[: min(25, len(self.bg_flat))]
        self.explainer = shap.KernelExplainer(self.model_predict_fn, bg_subset)

    def explain_prediction(
        self,
        sequence: np.ndarray,
        k_top: int = 5,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Explain the model's prediction for a 10-minute input sequence.

        Args:
            sequence: (10, 29) numpy array of scaled or unscaled features.
            k_top: Number of top influential features to return.
            threshold: Decision boundary threshold.

        Returns:
            dict with top-k features driving attack/benign prediction.
        """
        seq_arr = np.array(sequence, dtype=np.float32)
        if seq_arr.shape != (10, len(self.feature_cols)):
            raise ValueError(
                f"Expected sequence shape (10, {len(self.feature_cols)}), got {seq_arr.shape}"
            )

        # Compute raw model prediction
        attack_prob = float(self.model_predict_fn(seq_arr.reshape(1, -1))[0])
        is_attack = bool(attack_prob >= threshold)
        confidence = float(attack_prob if is_attack else (1.0 - attack_prob))

        # Flatten for KernelExplainer
        flat_seq = seq_arr.reshape(1, -1)
        raw_shap = self.explainer.shap_values(flat_seq, nsamples=self.nsamples, silent=True)

        if isinstance(raw_shap, list):
            shap_flat = np.array(raw_shap[0]).flatten()
        else:
            shap_flat = np.array(raw_shap).flatten()

        # Reshape to (10, 29) [time, feature]
        shap_grid_time_feat = shap_flat.reshape(10, len(self.feature_cols))
        # Transpose to (29, 10) [feature, time] so that .mean(axis=1) averages over time
        shap_values = shap_grid_time_feat.T

        # Aggregate across time steps (29 flow features)
        feature_importance = np.abs(shap_values).mean(axis=1)  # avg over time
        feature_sum = np.abs(shap_values).sum(axis=1)  # sum over time
        top_indices = np.argsort(feature_importance)[-k_top:][::-1]

        # Build feature driver details
        top_features = []
        for rank, idx in enumerate(top_indices, 1):
            feat_name = self.feature_cols[idx]
            clean_name = feat_name.replace("current_", "")
            imp_val = float(feature_importance[idx])
            sum_val = float(feature_sum[idx])
            mean_shap_dir = float(shap_values[idx].mean())  # positive pushes to attack

            # Baseline comparison if available
            raw_seq_val = seq_arr[:, idx]
            pct_above_baseline = 0.0
            annotation = f"{clean_name} driving {'attack' if mean_shap_dir > 0 else 'benign'} risk (SHAP: {imp_val:.4f})"

            if self.baseline_stats is not None and feat_name in self.baseline_stats:
                b_mean = self.baseline_stats.loc["mean", feat_name]
                b_std = self.baseline_stats.loc["std", feat_name]
                curr_mean = float(np.mean(raw_seq_val))
                if abs(b_mean) > 1e-6:
                    pct_above_baseline = ((curr_mean - b_mean) / abs(b_mean)) * 100.0
                    annotation = f"{clean_name} elevated {pct_above_baseline:+.1f}% above baseline"
                elif b_std > 1e-6:
                    z_score = (curr_mean - b_mean) / b_std
                    annotation = f"{clean_name} z-score {z_score:+.2f} vs baseline"

            top_features.append(
                {
                    "rank": rank,
                    "feature": feat_name,
                    "clean_name": clean_name,
                    "importance": round(imp_val, 4),
                    "sum_importance": round(sum_val, 4),
                    "direction": "attack" if mean_shap_dir >= 0 else "benign",
                    "mean_shap": round(mean_shap_dir, 4),
                    "percent_above_baseline": round(pct_above_baseline, 1),
                    "annotation": annotation,
                }
            )

        return {
            "prediction": round(attack_prob, 4),
            "is_attack": is_attack,
            "confidence": round(confidence, 4),
            "threshold": threshold,
            "top_features": top_features,
            "top_indices": top_indices.tolist(),
            "feature_importance": [round(float(v), 4) for v in feature_importance],
            "feature_names": self.feature_cols,
            "shap_values": shap_values,  # (29, 10) [feature, time]
        }


def init_explainer(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    data_path: Union[str, Path] = DEFAULT_DATA_PATH,
    n_background_samples: int = 30,
    nsamples: int = 60,
    device: str = "cpu",
) -> Tuple[AttackExplainer, pd.DataFrame, List[str]]:
    """
    Initialize the AttackExplainer with trained weights and non-leaking background data.
    """
    model, scaler, feature_cols, df = load_flow_lstm_and_scaler(
        model_path=model_path, data_path=data_path, device=device
    )

    train_mask = df["window"] <= 609
    train_df = df[train_mask].reset_index(drop=True)

    X_train_raw = train_df[feature_cols].replace([float("inf"), float("-inf")], 0).fillna(0).values
    X_train_scaled = scaler.transform(X_train_raw)

    # Compute baseline statistics
    raw_df_feats = train_df[feature_cols].replace([float("inf"), float("-inf")], 0).fillna(0)
    baseline_stats = pd.DataFrame(
        {
            "mean": raw_df_feats.mean(),
            "std": raw_df_feats.std(),
        }
    ).T

    # Sample representative background sequences
    bg_seqs = []
    step = max(1, (len(X_train_scaled) - 10) // n_background_samples)
    for i in range(10, len(X_train_scaled), step):
        bg_seqs.append(X_train_scaled[i - 10 : i])
        if len(bg_seqs) >= n_background_samples:
            break

    bg_seqs_arr = np.array(bg_seqs, dtype=np.float32)

    explainer = AttackExplainer(
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        background_sequences=bg_seqs_arr,
        baseline_stats=baseline_stats,
        nsamples=nsamples,
        device=device,
    )

    return explainer, df, feature_cols


def main():
    parser = argparse.ArgumentParser(description="SHAP Attribution for Network Attack Forecasting")
    parser.add_argument("--window", type=int, default=584, help="Window index to explain (default: 584 attack onset)")
    parser.add_argument("--k-top", type=int, default=5, help="Number of top features to report")
    parser.add_argument("--nsamples", type=int, default=50, help="SHAP evaluation samples")
    args = parser.parse_args()

    print("\n" + "=" * 75)
    print(f"SHAP FEATURE ATTRIBUTION ENGINE (Window {args.window})")
    print("=" * 75)

    explainer, df, feature_cols = init_explainer(nsamples=args.nsamples)

    # Extract test window sequence
    target_rows = df[df["window"] == args.window]
    if target_rows.empty:
        print(f"Window {args.window} not found, defaulting to index 640.")
        idx = 640
    else:
        idx = target_rows.index[0]

    X_raw = df[feature_cols].replace([float("inf"), float("-inf")], 0).fillna(0).values
    X_scaled = explainer.scaler.transform(X_raw)
    seq = X_scaled[idx - 9 : idx + 1]

    explanation = explainer.explain_prediction(seq, k_top=args.k_top)

    print(f"Predicted Attack Probability: {explanation['prediction']:.4f}")
    print(f"Classification:               {'ATTACK' if explanation['is_attack'] else 'BENIGN'} (Confidence: {explanation['confidence']:.2f})")
    print("\nTop Features Driving Prediction:")
    print("-" * 75)
    for feat in explanation["top_features"]:
        print(f"  {feat['rank']}. {feat['clean_name']:<30} | SHAP Importance: {feat['importance']:.4f} | {feat['annotation']}")
    print("-" * 75 + "\n")


if __name__ == "__main__":
    main()
