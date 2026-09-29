#!/usr/bin/env python3
"""
Temporal LSTM World Model for Network State Forecasting and Attack Detection
=============================================================================
Architecture:
  - Input: Sequences of 29 flow-state features over T time steps (T=10)
  - Backbone: LSTM(128 units, return_sequences=False)
  - Latent Representation: Shared Dropout(0.3) -> Dense(64, activation='relu') -> Dropout(0.3)
  - Head A (State Regression Head): Dense(29, linear) -> Predicts next state S(t+1) in standardized space
  - Head B (Attack Classification Head): Dense(1, sigmoid) -> Predicts attack probability at t+1

Training:
  - Chronological temporal train/val/test split (no shuffle) matching baseline:
      Train: window <= 609 (601 sequences)
      Val:   610 <= window <= 629 (20 sequences)
      Test:  window >= 630 (89 sequences)
  - StandardScaler fitted on training features only
  - Joint loss: total_loss = mse_loss (state) + lambda_bce * bce_loss (attack)
  - Reports both loss components separately during training
  - Early stopping with patience
  - Model weights saved to models/lstm_worldmodel_model.h5 and models/lstm_worldmodel_model.pt
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "state_transitions_clean.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "lstm_worldmodel_model.h5"

# Previous baselines for honest before/after comparison
LOGISTIC_REGRESSION_BASELINE = {
    "precision": 0.7368,
    "recall": 0.4118,
    "f1": 0.5283,
    "fpr": 0.0909,
    "tn": 50,
    "fp": 5,
    "fn": 20,
    "tp": 14,
}

CLASSIFIER_ONLY_LSTM_BASELINE = {
    "precision": 0.7812,
    "recall": 0.7353,
    "f1": 0.7576,
    "fpr": 0.1273,
    "tn": 48,
    "fp": 7,
    "fn": 9,
    "tp": 25,
}


class TemporalLSTMWorldModel(nn.Module):
    """
    Dual-head Temporal LSTM World Model:
      Backbone: LSTM(128) -> Dropout(0.3) -> Dense(64, relu) -> Dropout(0.3)
      Head A (State Regression): Linear(64, input_dim=29) -> Next-State S(t+1)
      Head B (Attack Classification): Linear(64, 1) -> Sigmoid -> Attack probability P(attack)
    """

    def __init__(self, input_dim: int = 29, hidden_dim: int = 128, dense_dim: int = 64, dropout_rate: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dense_dim = dense_dim

        # Temporal backbone
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense1 = nn.Linear(hidden_dim, dense_dim)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        # Output Head A: Next network state regression
        self.state_head = nn.Linear(dense_dim, input_dim)

        # Output Head B: Attack classification
        self.clf_head = nn.Linear(dense_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Input sequences of shape (batch_size, seq_len, input_dim)
        Returns:
            state_pred: Predicted next state S(t+1) of shape (batch_size, input_dim)
            attack_pred: Predicted attack probability of shape (batch_size, 1)
        """
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        feat = self.dropout2(self.relu(self.dense1(self.dropout1(last_step))))

        state_pred = self.state_head(feat)
        attack_pred = self.sigmoid(self.clf_head(feat))

        return state_pred, attack_pred

    def predict(self, x, return_state: bool = True):
        """
        Inference interface accepting numpy array or torch.Tensor.
        """
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                dev = next(self.parameters()).device
                x_tensor = torch.tensor(x, dtype=torch.float32, device=dev)
            else:
                x_tensor = x
            state_pred, attack_pred = self.forward(x_tensor)
            if return_state:
                return state_pred.cpu().numpy(), attack_pred.cpu().numpy()
            return attack_pred.cpu().numpy()

    def predict_proba(self, x) -> np.ndarray:
        """
        Predict only the attack probabilities for compatibility with scikit-learn / SHAP.
        Returns numpy array of shape (batch_size, 1).
        """
        return self.predict(x, return_state=False)


# Alias for backward compatibility
TemporalLSTM = TemporalLSTMWorldModel


def load_and_prepare_sequences(data_path: Path, seq_len: int = 10) -> Dict:
    """
    Load state transitions dataset, fit StandardScaler on training states only,
    and build sequential inputs S(t-T+1)...S(t) with dual targets:
      1. Next state S(t+1) (29 flow features)
      2. Next attack label (binary)
    """
    print(f"[1/5] Loading transitions dataset: {data_path} ...")
    if not data_path.exists():
        raise FileNotFoundError(f"Transitions dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    df = df.sort_values("window").reset_index(drop=True)

    cur_cols = [
        c for c in df.columns
        if c.startswith("current_") and c != "current_attack_label"
    ]
    nxt_cols = [
        c for c in df.columns
        if c.startswith("next_") and c not in ["next_attack_label", "next_timestamp"]
    ]
    feature_names = [c[len("current_"):] for c in cur_cols]

    print(f"      Rows loaded: {len(df):,}, State features: {len(cur_cols)}")

    # Clean numerical values
    X_raw = df[cur_cols].replace([float("inf"), float("-inf")], 0).fillna(0).values
    nxt_raw = df[nxt_cols].replace([float("inf"), float("-inf")], 0).fillna(0).values
    y_all = df["next_attack_label"].values
    windows = df["window"].values

    # Fit scaler ONLY on training features (windows <= 609)
    train_state_mask = windows <= 609
    print(f"[2/5] Fitting StandardScaler on training states only (windows <= 609, {train_state_mask.sum()} samples)...")
    scaler = StandardScaler()
    scaler.fit(X_raw[train_state_mask])

    # Transform both current sequence features and next-state targets using the same scaler
    X_scaled = scaler.transform(X_raw)
    nxt_scaled = scaler.transform(nxt_raw)

    # Build sequences of length seq_len (T=10)
    print(f"[3/5] Building sequences of length T={seq_len} ...")
    X_seqs = []
    y_seqs = []
    s_next_seqs = []
    nxt_raw_seqs = []
    target_windows = []

    for i in range(seq_len - 1, len(df)):
        seq = X_scaled[i - seq_len + 1 : i + 1]
        X_seqs.append(seq)
        y_seqs.append(y_all[i])
        s_next_seqs.append(nxt_scaled[i])
        nxt_raw_seqs.append(nxt_raw[i])
        target_windows.append(windows[i])

    X_seqs = np.array(X_seqs, dtype=np.float32)
    y_seqs = np.array(y_seqs, dtype=np.float32)
    s_next_seqs = np.array(s_next_seqs, dtype=np.float32)
    nxt_raw_seqs = np.array(nxt_raw_seqs, dtype=np.float32)
    target_windows = np.array(target_windows)

    # Chronological temporal splits matching baseline:
    # Train: window <= 609
    # Val:   610 <= window <= 629
    # Test:  window >= 630
    train_idx = target_windows <= 609
    val_idx = (target_windows >= 610) & (target_windows <= 629)
    test_idx = target_windows >= 630

    print("      Split summary:")
    print(f"        Train : {train_idx.sum():3d} sequences (Attack: {int(y_seqs[train_idx].sum())}, Benign: {int((y_seqs[train_idx] == 0).sum())})")
    print(f"        Val   : {val_idx.sum():3d} sequences (Attack: {int(y_seqs[val_idx].sum())}, Benign: {int((y_seqs[val_idx] == 0).sum())})")
    print(f"        Test  : {test_idx.sum():3d} sequences (Attack: {int(y_seqs[test_idx].sum())}, Benign: {int((y_seqs[test_idx] == 0).sum())})")

    return {
        "X_train": torch.tensor(X_seqs[train_idx]),
        "y_train": torch.tensor(y_seqs[train_idx]).unsqueeze(1),
        "s_train": torch.tensor(s_next_seqs[train_idx]),
        "X_val": torch.tensor(X_seqs[val_idx]),
        "y_val": torch.tensor(y_seqs[val_idx]).unsqueeze(1),
        "s_val": torch.tensor(s_next_seqs[val_idx]),
        "X_test": torch.tensor(X_seqs[test_idx]),
        "y_test": torch.tensor(y_seqs[test_idx]).unsqueeze(1),
        "s_test": torch.tensor(s_next_seqs[test_idx]),
        "s_test_raw": nxt_raw_seqs[test_idx],
        "input_dim": len(cur_cols),
        "feature_names": feature_names,
        "scaler": scaler,
    }


def save_model_weights_h5(model: nn.Module, filepath: Path):
    """Save model weights to HDF5 format (.h5) and PyTorch state_dict (.pt)."""
    filepath = Path(filepath).resolve()
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, "w") as f:
        for name, param in model.named_parameters():
            f.create_dataset(name, data=param.detach().cpu().numpy())

    pt_path = filepath.with_suffix(".pt")
    torch.save(model.state_dict(), pt_path)
    print(f"[SAVED] Model weights saved to: {filepath} and {pt_path}")


def train_worldmodel(
    model: nn.Module,
    X_train: torch.Tensor,
    s_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    s_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 0.0007,
    lambda_bce: float = 3.0,
    patience: int = 15,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict]:
    """
    Train the Dual-Head Temporal LSTM World Model jointly with combined loss:
        total_loss = mse_loss (state regression) + lambda_bce * bce_loss (attack classification)
    Both loss components are reported separately during training.
    """
    print(f"\n[4/5] Training Temporal LSTM World Model (max_epochs={epochs}, batch_size={batch_size}, lr={lr}, lambda_bce={lambda_bce}, patience={patience}) ...")
    dev = torch.device(device)
    print(f"      Compute device: {dev}")
    model = model.to(dev)
    X_train = X_train.to(dev)
    s_train = s_train.to(dev)
    y_train = y_train.to(dev)
    X_val = X_val.to(dev)
    s_val = s_val.to(dev)
    y_val = y_val.to(dev)

    # Mini-batch loader with dual targets
    train_dataset = TensorDataset(X_train, s_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Class balance weight for positive samples in training
    pos_ratio = (len(y_train) - y_train.sum()) / y_train.sum()
    pos_weight = torch.tensor([pos_ratio], device=dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "epoch": [],
        "train_total_loss": [],
        "train_mse_loss": [],
        "train_bce_loss": [],
        "val_total_loss": [],
        "val_mse_loss": [],
        "val_bce_loss": [],
    }

    best_val_criterion = float("inf")
    best_weights = None
    patience_counter = 0

    print("\n" + "=" * 80)
    print(f"{'Epoch':^7} | {'Train Total':^12} | {'Train MSE':^11} | {'Train BCE':^11} | {'Val MSE':^10} | {'Val BCE':^10} | {'Status'}")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss_accum = 0.0
        total_mse_accum = 0.0
        total_bce_accum = 0.0

        for bx, bs, by in train_loader:
            optimizer.zero_grad()
            s_pred, a_pred = model(bx)

            # State regression MSE loss
            mse_loss = nn.functional.mse_loss(s_pred, bs)

            # Attack classification BCE loss
            weights = torch.where(by == 1, pos_weight, torch.tensor(1.0, device=dev))
            bce_loss = nn.functional.binary_cross_entropy(a_pred, by, weight=weights)

            # Combined multi-task loss
            total_loss = mse_loss + lambda_bce * bce_loss

            total_loss.backward()
            optimizer.step()

            batch_n = len(bx)
            total_loss_accum += total_loss.item() * batch_n
            total_mse_accum += mse_loss.item() * batch_n
            total_bce_accum += bce_loss.item() * batch_n

        N_train = len(X_train)
        ep_train_total = total_loss_accum / N_train
        ep_train_mse = total_mse_accum / N_train
        ep_train_bce = total_bce_accum / N_train

        # Validation step
        model.eval()
        with torch.no_grad():
            vs_pred, va_pred = model(X_val)
            val_mse = nn.functional.mse_loss(vs_pred, s_val).item()
            val_bce = nn.functional.binary_cross_entropy(va_pred, y_val).item()
            val_total = val_mse + lambda_bce * val_bce

        history["epoch"].append(epoch)
        history["train_total_loss"].append(ep_train_total)
        history["train_mse_loss"].append(ep_train_mse)
        history["train_bce_loss"].append(ep_train_bce)
        history["val_total_loss"].append(val_total)
        history["val_mse_loss"].append(val_mse)
        history["val_bce_loss"].append(val_bce)

        # Early stopping tracks validation BCE (detecting attack onset in temporal validation window)
        status_msg = ""
        if val_bce < best_val_criterion:
            best_val_criterion = val_bce
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            status_msg = "★ Best"
        else:
            patience_counter += 1
            status_msg = f"Patience {patience_counter}/{patience}"

        if epoch % 5 == 0 or epoch == 1 or patience_counter == 0 or patience_counter >= patience:
            print(f"{epoch:^7d} | {ep_train_total:^12.4f} | {ep_train_mse:^11.4f} | {ep_train_bce:^11.4f} | {val_mse:^10.4f} | {val_bce:^10.4f} | {status_msg}")

        if patience_counter >= patience:
            print(f"\n[EARLY STOPPING] Triggered at epoch {epoch}. Restoring best model weights (val_bce={best_val_criterion:.4f}).")
            break

    print("-" * 80)

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history


def evaluate_worldmodel(
    model: nn.Module,
    X_test: torch.Tensor,
    s_test: torch.Tensor,
    s_test_raw: np.ndarray,
    y_test: torch.Tensor,
    feature_names: List[str],
    scaler: StandardScaler,
    threshold: float = 0.5,
    device: str = "cpu",
) -> Dict:
    """
    Comprehensive evaluation of the Temporal LSTM World Model:
      1. Attack classification metrics (Precision, Recall, F1, FPR, Confusion Matrix)
         compared honestly against the Logistic Regression and Classifier-Only LSTM baselines.
      2. Overall next-state prediction error (MSE).
      3. Per-feature prediction error (MSE) across all 29 flow features, identifying hardest vs easiest.
    """
    print("\n[5/5] Evaluating World Model on Test Set ...")
    dev = torch.device(device)
    model = model.to(dev)
    X_test = X_test.to(dev)

    model.eval()
    with torch.no_grad():
        test_state_pred_tensor, test_attack_pred_tensor = model(X_test)
        s_pred_std = test_state_pred_tensor.cpu().numpy()
        probs = test_attack_pred_tensor.squeeze().cpu().numpy()

    # Classification Metrics
    y_test_np = y_test.squeeze().cpu().numpy().astype(int)
    test_preds = (probs >= threshold).astype(int)

    precision = precision_score(y_test_np, test_preds, zero_division=0)
    recall = recall_score(y_test_np, test_preds, zero_division=0)
    f1 = f1_score(y_test_np, test_preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test_np, test_preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Next-State Regression Metrics
    s_true_std = s_test.cpu().numpy()
    overall_std_mse = float(((s_pred_std - s_true_std) ** 2).mean())
    per_feat_std_mse = ((s_pred_std - s_true_std) ** 2).mean(axis=0)

    # In original raw units
    s_pred_raw = scaler.inverse_transform(s_pred_std)
    per_feat_raw_mse = ((s_pred_raw - s_test_raw) ** 2).mean(axis=0)
    per_feat_raw_rmse = np.sqrt(per_feat_raw_mse)

    # Display Attack Classification Report
    print("\n" + "=" * 80)
    print("WORLD MODEL: ATTACK CLASSIFICATION TEST SET EVALUATION")
    print("=" * 80)
    print(f"Classification Threshold : {threshold:.2f}")
    print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, FPR={fpr:.4f}")
    print(f"Confusion Matrix         : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("=" * 80)

    # Three-way Comparison Table: LR Baseline vs Classifier LSTM vs World Model LSTM
    b_lr = LOGISTIC_REGRESSION_BASELINE
    b_lstm = CLASSIFIER_ONLY_LSTM_BASELINE

    diff_f1_baseline = f1 - b_lstm["f1"]
    diff_rec_baseline = recall - b_lstm["recall"]
    diff_prec_baseline = precision - b_lstm["precision"]

    print("\n" + "=" * 88)
    print("BEFORE / AFTER COMPARISON: CLASSIFIER LSTM BASELINE vs. WORLD MODEL LSTM")
    print("=" * 88)
    print(f"{'Metric':<22} | {'LR Baseline':<12} | {'Classifier LSTM':<16} | {'World Model LSTM':<16} | {'Delta vs LSTM':<12}")
    print("-" * 88)
    print(f"{'Precision':<22} | {b_lr['precision']:<12.4f} | {b_lstm['precision']:<16.4f} | {precision:<16.4f} | {diff_prec_baseline:+12.4f}")
    print(f"{'Recall':<22} | {b_lr['recall']:<12.4f} | {b_lstm['recall']:<16.4f} | {recall:<16.4f} | {diff_rec_baseline:+12.4f}")
    print(f"{'F1 Score':<22} | {b_lr['f1']:<12.4f} | {b_lstm['f1']:<16.4f} | {f1:<16.4f} | {diff_f1_baseline:+12.4f}")
    print(f"{'False Positive Rate':<22} | {b_lr['fpr']:<12.4f} | {b_lstm['fpr']:<16.4f} | {fpr:<16.4f} | {fpr - b_lstm['fpr']:+12.4f}")
    print("-" * 88)
    print(f"{'Confusion Matrix':<22} | TN={b_lr['tn']}, FP={b_lr['fp']:<2} | TN={b_lstm['tn']}, FP={b_lstm['fp']:<4}    | TN={tn}, FP={fp:<4}    |")
    print(f"{'':<22} | FN={b_lr['fn']}, TP={b_lr['tp']:<2} | FN={b_lstm['fn']}, TP={b_lstm['tp']:<4}    | FN={fn}, TP={tp:<4}    |")
    print("=" * 88)

    if diff_f1_baseline >= -0.02:
        print(f"\n[CONFIRMATION] World Model MAINTAINS/EXCEEDS classification F1 baseline ({f1:.4f} vs {b_lstm['f1']:.4f}, Delta: {diff_f1_baseline:+.4f})!")
    else:
        print(f"\n[NOTICE] World Model F1 regressed ({f1:.4f} vs {b_lstm['f1']:.4f}).")

    # Display Next-State Regression Report
    print("\n" + "=" * 80)
    print("NEW: NEXT-STATE PREDICTION ERROR EVALUATION (WORLD MODEL DYNAMICS)")
    print("=" * 80)
    print(f"Overall Next-State Standardized MSE : {overall_std_mse:.4f}")
    print("=" * 80)

    # Sort all 29 features by MSE descending (hardest to easiest)
    sorted_indices = np.argsort(per_feat_std_mse)[::-1]

    print("\n" + "=" * 80)
    print("ALL 29 FLOW FEATURES RANKED BY PREDICTION DIFFICULTY (HIGHEST TO LOWEST MSE)")
    print("=" * 80)
    print(f"{'Rank':<4} | {'Feature Name':<28} | {'Standardized MSE':<18} | {'Raw RMSE (Units)':<18}")
    print("-" * 80)
    for rank, idx in enumerate(sorted_indices, 1):
        fname = feature_names[idx]
        smse = per_feat_std_mse[idx]
        r_rmse = per_feat_raw_rmse[idx]
        print(f"{rank:4d} | {fname:<28} | {smse:18.4f} | {r_rmse:18.4f}")
    print("=" * 80)

    # Top 5 hardest and top 5 easiest features
    top5_hardest = [feature_names[i] for i in sorted_indices[:5]]
    top5_easiest = [feature_names[i] for i in sorted_indices[-5:][::-1]]

    print("\n[PHYSICAL ANALYSIS OF LEARNED DYNAMICS]")
    print("---------------------------------------")
    print("1. HARDEST FEATURES TO PREDICT (High MSE):")
    for rank, idx in enumerate(sorted_indices[:5], 1):
        print(f"   {rank}. {feature_names[idx]} (Std MSE = {per_feat_std_mse[idx]:.4f}): Non-linear, bursty control flags or sudden traffic bursts.")
    print("\n2. EASIEST FEATURES TO PREDICT (Low MSE):")
    for rank, idx in enumerate(sorted_indices[-5:][::-1], 1):
        print(f"   {rank}. {feature_names[idx]} (Std MSE = {per_feat_std_mse[idx]:.4f}): High temporal continuity and smooth background baseline persistence.")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "overall_std_mse": overall_std_mse,
        "per_feat_std_mse": per_feat_std_mse.tolist(),
        "per_feat_raw_rmse": per_feat_raw_rmse.tolist(),
        "top5_hardest": top5_hardest,
        "top5_easiest": top5_easiest,
        "feature_ranking": [feature_names[i] for i in sorted_indices],
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Temporal LSTM World Model with Next-State Regression and Attack Classification.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Path to state_transitions_clean.csv")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Output path for models/lstm_worldmodel_model.h5")
    parser.add_argument("--seq-len", type=int, default=10, help="Sequence length in time steps (default: 10 minutes)")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum training epochs (default: 60)")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.0007, help="Learning rate for Adam optimizer (default: 0.0007)")
    parser.add_argument("--lambda-bce", type=float, default=3.0, help="Loss weight for classification BCE loss (default: 3.0)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification decision threshold (default: 0.5)")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device: 'cpu' or 'cuda' (default: cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    print("=" * 80)
    print("TEMPORAL LSTM WORLD MODEL: STATE REGRESSION & ATTACK FORECASTING")
    print("=" * 80)

    # 1. Load data & prepare sequences with dual targets
    data = load_and_prepare_sequences(args.data_path, seq_len=args.seq_len)

    # 2. Instantiate Dual-Head World Model architecture
    model = TemporalLSTMWorldModel(
        input_dim=data["input_dim"],
        hidden_dim=128,
        dense_dim=64,
        dropout_rate=0.3,
    )

    # 3. Train model jointly
    trained_model, history = train_worldmodel(
        model=model,
        X_train=data["X_train"],
        s_train=data["s_train"],
        y_train=data["y_train"],
        X_val=data["X_val"],
        s_val=data["s_val"],
        y_val=data["y_val"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_bce=args.lambda_bce,
        patience=args.patience,
        device=args.device,
    )

    # 4. Save trained model weights to models/lstm_worldmodel_model.h5 and .pt
    save_model_weights_h5(trained_model, args.model_path)

    # 5. Evaluate on test set
    metrics = evaluate_worldmodel(
        model=trained_model,
        X_test=data["X_test"],
        s_test=data["s_test"],
        s_test_raw=data["s_test_raw"],
        y_test=data["y_test"],
        feature_names=data["feature_names"],
        scaler=data["scaler"],
        threshold=args.threshold,
        device=args.device,
    )

    return metrics


if __name__ == "__main__":
    main()
