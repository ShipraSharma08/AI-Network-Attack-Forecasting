#!/usr/bin/env python3
"""
Multi-Modal Fused Temporal LSTM Network State Forecasting Model
===============================================================
Architecture:
  - Input: Sequences of 43 fused state features (29 flow + 14 packet) over T time steps (T=10)
  - LSTM(128 units, return_sequences=False)
  - Dropout(0.3)
  - Dense(64, activation='relu')
  - Dropout(0.3)
  - Dense(1, activation='sigmoid')  # Binary classification (attack/benign)

Data & Training:
  - Loads data/processed/fused_flow_packet_states.csv
  - Chronological non-shuffled train/val/test splits matching temporal boundaries:
      * Train: timestamp <= 2018-02-28 11:09:00
      * Val:   2018-02-28 11:10:00 <= timestamp <= 2018-02-28 11:29:00
      * Test:  timestamp >= 2018-02-28 11:30:00
  - StandardScaler fit on training features only
  - Adam (lr=0.001), balanced binary cross-entropy loss
  - Early stopping on validation loss
  - Model weights saved to models/lstm_fused_model.h5
"""

import argparse
import os
import sys
from pathlib import Path

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

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "fused_flow_packet_states.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "lstm_fused_model.h5"

# Reference baselines for comparison
FLOW_ONLY_BASELINE = {
    "precision": 0.7812,
    "recall": 0.7353,
    "f1": 0.7576,
    "fpr": 0.1273,
    "tn": 48,
    "fp": 7,
    "fn": 9,
    "tp": 25,
}

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


class TemporalLSTM(nn.Module):
    """
    Temporal LSTM for fused network state dynamics:
      LSTM(128) -> Dropout(0.3) -> Dense(64, relu) -> Dropout(0.3) -> Dense(1, sigmoid)
    """

    def __init__(self, input_dim=43, hidden_dim=128, dense_dim=64, dropout_rate=0.3):
        super().__init__()
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
        self.dense2 = nn.Linear(dense_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        out = self.dropout1(last_step)
        out = self.relu(self.dense1(out))
        out = self.dropout2(out)
        out = self.sigmoid(self.dense2(out))
        return out


def load_and_prepare_fused_sequences(data_path: Path, seq_len: int = 10):
    """
    Load fused flow+packet states, extract target S(t) -> Y(t+1),
    scale training features only, and construct sequences of length seq_len.
    """
    print(f"[1/5] Loading fused dataset from: {data_path} ...")
    if not data_path.exists():
        raise FileNotFoundError(f"Fused dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 1-step lookahead target: S(t) -> Y(t+1)
    df["next_attack_label"] = df["attack_label"].shift(-1)
    df = df.iloc[:-1].copy()

    non_feature_cols = {"timestamp", "window", "elapsed_seconds", "attack_label", "next_attack_label"}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    flow_features = [c for c in feature_cols if not c.startswith("unique_") and c not in [
        "packet_count", "ttl_mean", "ttl_std", "tcp_window_mean", "tcp_window_std",
        "payload_mean", "payload_std", "payload_sum", "payload_max", "fragmented_packets"
    ]]
    packet_features = [c for c in feature_cols if c not in flow_features]

    print(f"      Rows: {len(df):,}")
    print(f"      Unified Features : {len(feature_cols)} ({len(flow_features)} flow + {len(packet_features)} packet)")

    # Clean numerical values
    X_raw = df[feature_cols].replace([float("inf"), float("-inf")], 0.0).fillna(0.0).values
    y_all = df["next_attack_label"].values
    ts = df["timestamp"].values

    # Chronological split masks matching the project's temporal boundaries:
    # Train : timestamp <= 2018-02-28 11:09:00
    # Val   : 2018-02-28 11:10:00 <= timestamp <= 2018-02-28 11:29:00
    # Test  : timestamp >= 2018-02-28 11:30:00
    train_split_time = np.datetime64("2018-02-28 11:09:00")
    val_start_time = np.datetime64("2018-02-28 11:10:00")
    val_end_time = np.datetime64("2018-02-28 11:29:00")
    test_start_time = np.datetime64("2018-02-28 11:30:00")

    train_state_mask = ts <= train_split_time
    print(f"[2/5] Fitting StandardScaler on training states only ({train_state_mask.sum()} samples)...")
    scaler = StandardScaler()
    scaler.fit(X_raw[train_state_mask])
    X_scaled = scaler.transform(X_raw)

    # Build sequences of length seq_len (T=10)
    print(f"[3/5] Building sequences of length T={seq_len} ...")
    X_seqs = []
    y_seqs = []
    split_types = []

    for i in range(seq_len - 1, len(df)):
        seq = X_scaled[i - seq_len + 1 : i + 1]
        X_seqs.append(seq)
        y_seqs.append(y_all[i])
        current_time = ts[i]

        if current_time <= train_split_time:
            split_types.append("train")
        elif val_start_time <= current_time <= val_end_time:
            split_types.append("val")
        elif current_time >= test_start_time:
            split_types.append("test")
        else:
            split_types.append("other")

    X_seqs = np.array(X_seqs, dtype=np.float32)
    y_seqs = np.array(y_seqs, dtype=np.float32)
    split_types = np.array(split_types)

    train_idx = split_types == "train"
    val_idx = split_types == "val"
    test_idx = split_types == "test"

    print("      Split summary:")
    print(f"        Train : {train_idx.sum():3d} sequences (Attack: {int(y_seqs[train_idx].sum())}, Benign: {int((y_seqs[train_idx] == 0).sum())})")
    print(f"        Val   : {val_idx.sum():3d} sequences (Attack: {int(y_seqs[val_idx].sum())}, Benign: {int((y_seqs[val_idx] == 0).sum())})")
    print(f"        Test  : {test_idx.sum():3d} sequences (Attack: {int(y_seqs[test_idx].sum())}, Benign: {int((y_seqs[test_idx] == 0).sum())})")

    return {
        "X_train": torch.tensor(X_seqs[train_idx]),
        "y_train": torch.tensor(y_seqs[train_idx]).unsqueeze(1),
        "X_val": torch.tensor(X_seqs[val_idx]),
        "y_val": torch.tensor(y_seqs[val_idx]).unsqueeze(1),
        "X_test": torch.tensor(X_seqs[test_idx]),
        "y_test": torch.tensor(y_seqs[test_idx]).unsqueeze(1),
        "input_dim": len(feature_cols),
        "flow_count": len(flow_features),
        "packet_count": len(packet_features),
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
    print(f"[SAVED] Fused model weights saved to: {filepath} and {pt_path}")


def train_fused_lstm(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 0.001,
    patience: int = 15,
    device: str = "cpu",
):
    """Train the Fused LSTM with Adam and early stopping."""
    print(f"\n[4/5] Training Fused Temporal LSTM (max_epochs={epochs}, batch_size={batch_size}, lr={lr}, patience={patience}) ...")
    dev = torch.device(device)
    print(f"      Compute device: {dev}")
    model = model.to(dev)
    X_train = X_train.to(dev)
    y_train = y_train.to(dev)
    X_val = X_val.to(dev)
    y_val = y_val.to(dev)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    pos_ratio = (len(y_train) - y_train.sum()) / y_train.sum()
    pos_weight = torch.tensor([pos_ratio], device=dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    print("\n----- Training Progress -----")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^12} | {'Status'}")
    print("-" * 45)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for bx, by in train_loader:
            optimizer.zero_grad()
            preds = model(bx)
            weights = torch.where(by == 1, pos_weight, torch.tensor(1.0, device=dev))
            loss = nn.functional.binary_cross_entropy(preds, by, weight=weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)

        epoch_train_loss = total_loss / len(X_train)

        # Validation step
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            epoch_val_loss = nn.functional.binary_cross_entropy(val_preds, y_val).item()

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

        status_msg = ""
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            status_msg = "★ Best"
        else:
            patience_counter += 1
            status_msg = f"Patience {patience_counter}/{patience}"

        if epoch % 5 == 0 or epoch == 1 or patience_counter == 0 or patience_counter >= patience:
            print(f"{epoch:^7d} | {epoch_train_loss:^12.4f} | {epoch_val_loss:^12.4f} | {status_msg}")

        if patience_counter >= patience:
            print(f"\n[EARLY STOPPING] Triggered at epoch {epoch}. Restoring best model weights (val_loss={best_val_loss:.4f}).")
            break

    print("-" * 45)

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history


def evaluate_fused_lstm(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor, threshold: float = 0.5, device: str = "cpu"):
    """Evaluate on the test set and perform 3-way comparative evaluation."""
    print("\n[5/5] Evaluating Fused LSTM on Test Set ...")
    dev = torch.device(device)
    model = model.to(dev)
    X_test = X_test.to(dev)

    model.eval()
    with torch.no_grad():
        test_probs = model(X_test).squeeze().cpu().numpy()

    y_test_np = y_test.squeeze().cpu().numpy().astype(int)
    test_preds = (test_probs >= threshold).astype(int)

    precision = precision_score(y_test_np, test_preds, zero_division=0)
    recall = recall_score(y_test_np, test_preds, zero_division=0)
    f1 = f1_score(y_test_np, test_preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_test_np, test_preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n" + "=" * 70)
    print("FUSED LSTM TEST SET EVALUATION RESULTS (43 FEATURES)")
    print("=" * 70)
    print(f"Classification Threshold : {threshold:.2f}")
    print(f"Fused LSTM: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, FPR={fpr:.4f}")
    print(f"Confusion Matrix         : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("=" * 70)

    # Comparison against Flow-Only LSTM and Logistic Regression
    fo_prec = FLOW_ONLY_BASELINE["precision"]
    fo_rec = FLOW_ONLY_BASELINE["recall"]
    fo_f1 = FLOW_ONLY_BASELINE["f1"]
    fo_fpr = FLOW_ONLY_BASELINE["fpr"]

    diff_f1_flow = f1 - fo_f1
    pct_imp_flow = (diff_f1_flow / fo_f1) * 100

    diff_f1_lr = f1 - LOGISTIC_REGRESSION_BASELINE["f1"]
    pct_imp_lr = (diff_f1_lr / LOGISTIC_REGRESSION_BASELINE["f1"]) * 100

    print("\n" + "=" * 75)
    print("COMPARATIVE EVALUATION: BASELINE vs. FLOW-ONLY vs. FUSED LSTM")
    print("=" * 75)
    print(f"{'Model':<20} | {'Features':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'FPR':<8}")
    print("-" * 75)
    print(f"{'Logistic Regression':<20} | {'29 (flow)':<10} | {LOGISTIC_REGRESSION_BASELINE['precision']:<10.4f} | {LOGISTIC_REGRESSION_BASELINE['recall']:<10.4f} | {LOGISTIC_REGRESSION_BASELINE['f1']:<10.4f} | {LOGISTIC_REGRESSION_BASELINE['fpr']:<8.4f}")
    print(f"{'Flow-Only LSTM':<20} | {'29 (flow)':<10} | {fo_prec:<10.4f} | {fo_rec:<10.4f} | {fo_f1:<10.4f} | {fo_fpr:<8.4f}")
    print(f"{'Fused LSTM (Multi)':<20} | {'43 (fused)':<10} | {precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f} | {fpr:<8.4f}")
    print("=" * 75)

    print("\n[REQUESTED SUMMARY]")
    print(f"Fused LSTM: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, FPR={fpr:.4f}")
    print(f"Improvement over flow-only: {pct_imp_flow:+.2f}%")
    print(f"Improvement over logistic baseline: {pct_imp_lr:+.2f}%")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "pct_imp_flow": pct_imp_flow,
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Fused Multi-Modal Temporal LSTM.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Path to fused_flow_packet_states.csv")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Output path for models/lstm_fused_model.h5")
    parser.add_argument("--seq-len", type=int, default=10, help="Sequence length in time steps (default: 10 minutes)")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum training epochs (default: 60)")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer (default: 0.001)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification decision threshold (default: 0.5)")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device: 'cpu' or 'cuda' (default: cpu)")
    parser.add_argument("--seed", type=int, default=7, help="Random seed (default: 7)")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    print("=" * 70)
    print("FUSED MULTI-MODAL TEMPORAL LSTM PIPELINE (43 FEATURES)")
    print("=" * 70)

    # 1. Load data & prepare sequences
    data = load_and_prepare_fused_sequences(args.data_path, seq_len=args.seq_len)

    # 2. Instantiate architecture with 43 input features
    model = TemporalLSTM(
        input_dim=data["input_dim"],
        hidden_dim=128,
        dense_dim=64,
        dropout_rate=0.3,
    )

    # 3. Train model
    trained_model, history = train_fused_lstm(
        model=model,
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
    )

    # 4. Save trained model weights to models/lstm_fused_model.h5
    save_model_weights_h5(trained_model, args.model_path)

    # 5. Evaluate on test set & compare with baseline and flow-only
    evaluate_fused_lstm(
        model=trained_model,
        X_test=data["X_test"],
        y_test=data["y_test"],
        threshold=args.threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
