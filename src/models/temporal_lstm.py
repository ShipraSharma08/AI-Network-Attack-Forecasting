#!/usr/bin/env python3
"""
Temporal LSTM Network State Forecasting Model
=============================================
Architecture:
  - Input: Sequences of 29 flow-state features over T time steps (T=10)
  - LSTM(128 units, return_sequences=False)
  - Dropout(0.3)
  - Dense(64, activation='relu')
  - Dropout(0.3)
  - Dense(1, activation='sigmoid')  # Binary classification (attack/benign)

Training:
  - Chronological temporal train/val/test split (no shuffling)
  - StandardScaler fitted on training features only
  - Adam (lr=0.001), binary cross-entropy loss with class balance handling
  - Early stopping on validation loss
  - Model weights saved to models/lstm_model.h5
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

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "state_transitions_clean.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "lstm_model.h5"

# Logistic regression baseline reference values
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


class TemporalLSTM(nn.Module):
    """
    Temporal LSTM model for learning network state dynamics:
      LSTM(128) -> Dropout(0.3) -> Dense(64, relu) -> Dropout(0.3) -> Dense(1, sigmoid)
    """

    def __init__(self, input_dim=29, hidden_dim=128, dense_dim=64, dropout_rate=0.3):
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
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # return_sequences=False: extract output of the final time step
        last_step = lstm_out[:, -1, :]
        out = self.dropout1(last_step)
        out = self.relu(self.dense1(out))
        out = self.dropout2(out)
        out = self.sigmoid(self.dense2(out))
        return out

    def predict(self, x):
        """
        Keras-compatible prediction interface accepting numpy array or tensor.
        Returns numpy array of predictions of shape (batch_size, 1).
        """
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                dev = next(self.parameters()).device
                x_tensor = torch.tensor(x, dtype=torch.float32, device=dev)
            else:
                x_tensor = x
            out = self.forward(x_tensor)
            return out.cpu().numpy()


def load_and_prepare_sequences(data_path: Path, seq_len: int = 10):
    """
    Load state transitions, scale training features only, and construct sequences.
    """
    print(f"[1/5] Loading transitions dataset: {data_path} ...")
    if not data_path.exists():
        raise FileNotFoundError(f"Transitions dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    df = df.sort_values("window").reset_index(drop=True)

    feature_cols = [
        c for c in df.columns
        if c.startswith("current_") and c != "current_attack_label"
    ]
    print(f"      Rows loaded: {len(df):,}, Features: {len(feature_cols)}")

    # Clean numerical values
    X_raw = df[feature_cols].replace([float("inf"), float("-inf")], 0).fillna(0).values
    y_all = df["next_attack_label"].values
    windows = df["window"].values

    # Fit scaler ONLY on training features (windows <= 609)
    train_state_mask = windows <= 609
    print(f"[2/5] Fitting StandardScaler on training states only (windows <= 609, {train_state_mask.sum()} samples)...")
    scaler = StandardScaler()
    scaler.fit(X_raw[train_state_mask])
    X_scaled = scaler.transform(X_raw)

    # Build sequences of length seq_len (T=10)
    print(f"[3/5] Building sequences of length T={seq_len} ...")
    X_seqs = []
    y_seqs = []
    target_windows = []

    for i in range(seq_len - 1, len(df)):
        seq = X_scaled[i - seq_len + 1 : i + 1]
        X_seqs.append(seq)
        y_seqs.append(y_all[i])
        target_windows.append(windows[i])

    X_seqs = np.array(X_seqs, dtype=np.float32)
    y_seqs = np.array(y_seqs, dtype=np.float32)
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
        "X_val": torch.tensor(X_seqs[val_idx]),
        "y_val": torch.tensor(y_seqs[val_idx]).unsqueeze(1),
        "X_test": torch.tensor(X_seqs[test_idx]),
        "y_test": torch.tensor(y_seqs[test_idx]).unsqueeze(1),
        "input_dim": len(feature_cols),
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


def train_lstm(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 0.001,
    patience: int = 15,
    device: str = "cpu",
):
    """
    Train the LSTM with Adam, balanced binary cross-entropy, and early stopping.
    """
    print(f"\n[4/5] Training Temporal LSTM (max_epochs={epochs}, batch_size={batch_size}, lr={lr}, patience={patience}) ...")
    dev = torch.device(device)
    print(f"      Compute device: {dev}")
    model = model.to(dev)
    X_train = X_train.to(dev)
    y_train = y_train.to(dev)
    X_val = X_val.to(dev)
    y_val = y_val.to(dev)

    # Mini-batch loader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Class balance weight for positive samples in training
    pos_ratio = (len(y_train) - y_train.sum()) / y_train.sum()
    pos_weight = torch.tensor([pos_ratio], device=dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
    }

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


def evaluate_lstm(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor, threshold: float = 0.5, device: str = "cpu"):
    """
    Evaluate trained LSTM on test set and compare with logistic regression baseline.
    """
    print("\n[5/5] Evaluating on Test Set ...")
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
    print("LSTM TEST SET EVALUATION RESULTS")
    print("=" * 70)
    print(f"Classification Threshold : {threshold:.2f}")
    print(f"LSTM: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, FPR={fpr:.4f}")
    print(f"Confusion Matrix         : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("=" * 70)

    # Comparison with logistic regression baseline
    b_prec = BASELINE_METRICS["precision"]
    b_rec = BASELINE_METRICS["recall"]
    b_f1 = BASELINE_METRICS["f1"]
    b_fpr = BASELINE_METRICS["fpr"]

    diff_prec = precision - b_prec
    diff_rec = recall - b_rec
    diff_f1 = f1 - b_f1
    diff_fpr = fpr - b_fpr

    print("\n" + "=" * 70)
    print("COMPARISON: LOGISTIC REGRESSION BASELINE vs. TEMPORAL LSTM")
    print("=" * 70)
    print(f"{'Metric':<22} | {'Baseline (LR)':<15} | {'Temporal LSTM':<15} | {'Delta':<12}")
    print("-" * 70)
    print(f"{'Precision':<22} | {b_prec:<15.4f} | {precision:<15.4f} | {diff_prec:+12.4f} ({diff_prec / b_prec * 100:+.1f}%)")
    print(f"{'Recall':<22} | {b_rec:<15.4f} | {recall:<15.4f} | {diff_rec:+12.4f} ({diff_rec / b_rec * 100:+.1f}%)")
    print(f"{'F1 Score':<22} | {b_f1:<15.4f} | {f1:<15.4f} | {diff_f1:+12.4f} ({diff_f1 / b_f1 * 100:+.1f}%)")
    print(f"{'False Positive Rate':<22} | {b_fpr:<15.4f} | {fpr:<15.4f} | {diff_fpr:+12.4f}")
    print("-" * 70)
    print(f"{'Confusion Matrix':<22} | TN={BASELINE_METRICS['tn']}, FP={BASELINE_METRICS['fp']}    | TN={tn}, FP={fp}    |")
    print(f"{'':<22} | FN={BASELINE_METRICS['fn']}, TP={BASELINE_METRICS['tp']}    | FN={fn}, TP={tp}    |")
    print("=" * 70)

    outperforms = f1 > b_f1
    if outperforms:
        print(f"\n[CONFIRMATION] LSTM OUTPERFORMS the baseline by {diff_f1:+.4f} F1 score ({diff_f1 / b_f1 * 100:+.1f}% improvement)!")
    else:
        print(f"\n[NOTICE] LSTM did not exceed the baseline F1 score ({f1:.4f} vs {b_f1:.4f}).")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "outperforms": outperforms,
    }


def print_loss_curve(history: dict):
    """Print an ASCII visualization of the training & validation loss curves."""
    print("\n----- Loss Curve Summary -----")
    epochs = history["epoch"]
    t_loss = history["train_loss"]
    v_loss = history["val_loss"]

    step = max(1, len(epochs) // 10)
    sample_indices = list(range(0, len(epochs), step))
    if (len(epochs) - 1) not in sample_indices:
        sample_indices.append(len(epochs) - 1)

    print(f"{'Epoch':^6} | {'Train Loss':^12} | {'Val Loss':^12} | {'Visual Progression'}")
    print("-" * 65)
    max_loss = max(max(t_loss), max(v_loss))
    scale = 25.0 / max_loss if max_loss > 0 else 1.0

    for idx in sample_indices:
        ep = epochs[idx]
        tl = t_loss[idx]
        vl = v_loss[idx]
        t_bar = "#" * int(tl * scale)
        v_bar = "*" * int(vl * scale)
        print(f"{ep:^6d} | {tl:^12.4f} | {vl:^12.4f} | T: {t_bar:<15} V: {v_bar}")
    print("Legend: T (#) = Train Loss, V (*) = Val Loss")
    print("------------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Temporal LSTM for network state attack forecasting.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Path to state_transitions_clean.csv")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Output path for models/lstm_model.h5")
    parser.add_argument("--seq-len", type=int, default=10, help="Sequence length in time steps (default: 10 minutes)")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum training epochs (default: 60)")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer (default: 0.001)")
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

    print("=" * 70)
    print("TEMPORAL LSTM STATE FORECASTING PIPELINE")
    print("=" * 70)

    # 1. Load data & prepare sequences
    data = load_and_prepare_sequences(args.data_path, seq_len=args.seq_len)

    # 2. Instantiate architecture
    model = TemporalLSTM(
        input_dim=data["input_dim"],
        hidden_dim=128,
        dense_dim=64,
        dropout_rate=0.3,
    )

    # 3. Train model
    trained_model, history = train_lstm(
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

    # 4. Save trained model weights to models/lstm_model.h5
    save_model_weights_h5(trained_model, args.model_path)

    # 5. Print training history loss curve
    print_loss_curve(history)

    # 6. Evaluate on test set & compare with baseline
    metrics = evaluate_lstm(
        model=trained_model,
        X_test=data["X_test"],
        y_test=data["y_test"],
        threshold=args.threshold,
        device=args.device,
    )

    sys.exit(0 if metrics["outperforms"] else 1)


if __name__ == "__main__":
    main()
