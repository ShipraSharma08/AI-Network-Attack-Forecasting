#!/usr/bin/env python3
"""
Out-of-Distribution Specificity Test: Thursday-01-03-2018
==========================================================
Evaluates the ALREADY-TRAINED flow-only LSTM (models/lstm_model.h5) on
CIC-IDS2018 Thursday-01-03-2018 without any retraining.

GROUND TRUTH CORRECTION (2026-09-16):
  The Thursday CSV covers 01:00-12:59 UTC. ALL windows are treated as BENIGN because:
    (a) Raw CSV "Infilteration" labels (min=02:00, max=10:55) are the SAME
        CICFlowMeter annotation artifact already rejected for Wednesday (where raw
        labels started at 01:42, 9 hours before the documented 10:50 attack).
    (b) The officially documented Thursday attack window (14:00-15:37, UNB IDS-2018)
        falls entirely OUTSIDE this file's 01:00-12:59 capture range.
    (c) Only ONE Thursday-01-03-2018 file exists in the S3 bucket listing --
        NO confirmed separate file covers the 14:00-15:37 period.

  This evaluation therefore measures SPECIFICITY / FALSE-POSITIVE RATE ONLY.
  It answers: does the Wednesday-trained LSTM generate false alarms on a full
  day of confirmed benign Thursday traffic it has never seen?

Usage:
    python src/evaluation/multiday_validation.py
    python src/evaluation/multiday_validation.py --threshold 0.4 --plot
    python src/evaluation/multiday_validation.py --report-path reports/multiday_validation.json
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.temporal_lstm import TemporalLSTM
from src.forecasting.kstep_rollout import load_flow_lstm_and_scaler

THURSDAY_CSV = (
    PROJECT_ROOT / "data" / "raw"
    / "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
)
WEDNESDAY_TRANSITIONS = (
    PROJECT_ROOT / "data" / "processed" / "state_transitions_clean.csv"
)
MODEL_PATH          = PROJECT_ROOT / "models" / "lstm_model.h5"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "multiday_validation.json"

ATTACK_WINDOW_MISMATCH_NOTE = (
    "Attack window mismatch: this file's traffic (01:00-12:59) does not overlap "
    "the documented Thursday infiltration window (14:00-15:37). "
    "This evaluation measures false-positive rate on unseen benign traffic only, "
    "not attack recall."
)

RAW_LABEL_FINDINGS = {
    "infilteration_flow_count": 93063,
    "infilteration_min_timestamp": "2018-03-01 02:00:00",
    "infilteration_max_timestamp": "2018-03-01 10:54:59",
    "documented_attack_start": "2018-03-01 14:00:00",
    "documented_attack_end": "2018-03-01 15:37:00",
    "verdict": (
        "Raw Infilteration labels (02:00-10:55) are the same CICFlowMeter annotation "
        "artifact already rejected for Wednesday (labels started at 01:42, "
        "9h before documented 10:50 attack). Labels overridden to 0 for all windows."
    ),
}

WINDOW_SECONDS = 60
SEQ_LEN        = 10

WEDNESDAY_RESULTS = {
    "fpr": 0.0364,
    "attack_windows": 34,
    "benign_windows": 55,
    "total_windows": 89,
}

FEATURE_COLUMNS = [
    "Flow Duration",
    "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags",
    "SYN Flag Cnt", "ACK Flag Cnt", "RST Flag Cnt", "FIN Flag Cnt",
    "Pkt Len Mean", "Pkt Len Std",
    "Flow Pkts/s", "Flow Byts/s",
    "Down/Up Ratio",
    "Active Mean", "Active Std",
    "Idle Mean", "Idle Std",
]


def build_thursday_states(csv_path: Path) -> pd.DataFrame:
    """Load Thursday CSV, aggregate to 60s windows, set ALL labels = 0 (benign)."""
    print(f"\n[1/4] Loading Thursday flow CSV: {csv_path}")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Thursday CSV not found: {csv_path}\n"
            f"Run: python scripts/download_thursday_flows.py"
        )

    usecols = ["Timestamp", "Label"] + FEATURE_COLUMNS
    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    print(f"      Raw rows loaded: {len(df):,}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).copy()
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    raw_label_dist = df["Label"].str.strip().value_counts()
    print(f"      Valid rows: {len(df):,}  |  "
          f"Time range: {df['Timestamp'].min()} -> {df['Timestamp'].max()}")
    print(f"      Raw CSV label distribution (NOT used as ground truth):")
    for lbl, cnt in raw_label_dist.items():
        print(f"        {lbl}: {cnt:,}")

    print(f"\n      GROUND TRUTH OVERRIDE: attack_label = 0 for ALL windows")
    print(f"      Reason: raw 'Infilteration' labels (02:00-10:55) are a")
    print(f"        CICFlowMeter artifact identical to Wednesday's (01:42,")
    print(f"        9h before the actual attack). Documented attack window")
    print(f"        (14:00-15:37) is OUTSIDE this file's capture range.")
    print(f"      -> {ATTACK_WINDOW_MISMATCH_NOTE}")

    df["window_start"] = df["Timestamp"].dt.floor(f"{WINDOW_SECONDS}s")
    grouped = df.groupby("window_start")

    states = grouped.size().rename("flow_count").to_frame()

    agg_map = {
        "flow_duration_mean": ("Flow Duration",   "mean"),
        "fwd_packets_mean":   ("Tot Fwd Pkts",    "mean"),
        "bwd_packets_mean":   ("Tot Bwd Pkts",    "mean"),
        "fwd_bytes_sum":      ("TotLen Fwd Pkts", "sum"),
        "bwd_bytes_sum":      ("TotLen Bwd Pkts", "sum"),
        "flow_iat_mean":      ("Flow IAT Mean",   "mean"),
        "flow_iat_std":       ("Flow IAT Std",    "mean"),
        "flow_iat_max":       ("Flow IAT Max",    "max"),
        "flow_iat_min":       ("Flow IAT Min",    "min"),
        "fwd_iat_mean":       ("Fwd IAT Mean",    "mean"),
        "bwd_iat_mean":       ("Bwd IAT Mean",    "mean"),
        "syn_count":          ("SYN Flag Cnt",    "sum"),
        "ack_count":          ("ACK Flag Cnt",    "sum"),
        "rst_count":          ("RST Flag Cnt",    "sum"),
        "fin_count":          ("FIN Flag Cnt",    "sum"),
        "psh_fwd_count":      ("Fwd PSH Flags",   "sum"),
        "psh_bwd_count":      ("Bwd PSH Flags",   "sum"),
        "urg_fwd_count":      ("Fwd URG Flags",   "sum"),
        "urg_bwd_count":      ("Bwd URG Flags",   "sum"),
        "packet_length_mean": ("Pkt Len Mean",    "mean"),
        "packet_length_std":  ("Pkt Len Std",     "mean"),
        "packets_per_second": ("Flow Pkts/s",     "mean"),
        "bytes_per_second":   ("Flow Byts/s",     "mean"),
        "down_up_ratio":      ("Down/Up Ratio",   "mean"),
        "active_mean":        ("Active Mean",     "mean"),
        "active_std":         ("Active Std",      "mean"),
        "idle_mean":          ("Idle Mean",       "mean"),
        "idle_std":           ("Idle Std",        "mean"),
    }
    for col_name, (src, func) in agg_map.items():
        states[col_name] = getattr(grouped[src], func)()

    start = df["Timestamp"].min().floor(f"{WINDOW_SECONDS}s")
    end   = df["Timestamp"].max().floor(f"{WINDOW_SECONDS}s")
    complete_index = pd.date_range(start=start, end=end, freq=f"{WINDOW_SECONDS}s")
    states = states.reindex(complete_index)
    states.index.name = "timestamp"

    zero_features = [
        "flow_count", "fwd_bytes_sum", "bwd_bytes_sum",
        "syn_count", "ack_count", "rst_count", "fin_count",
        "psh_fwd_count", "psh_bwd_count", "urg_fwd_count", "urg_bwd_count",
    ]
    states[zero_features] = states[zero_features].fillna(0)
    states["window"] = np.arange(len(states))
    states["attack_label"] = 0  # ALL BENIGN
    states["elapsed_seconds"] = (states.index - states.index[0]).total_seconds()
    states = states.reset_index()

    print(f"\n      Total 60s windows: {len(states):,}  (all labeled benign)")
    return states


def build_thursday_sequences(states, scaler, feature_cols, seq_len=SEQ_LEN):
    print(f"\n[2/4] Building sequences (T={seq_len}, features={len(feature_cols)})...")
    raw_cols = [c.replace("current_", "") for c in feature_cols]
    missing = [c for c in raw_cols if c not in states.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing[:5]}")

    X_raw = (
        states[raw_cols].replace([float("inf"), float("-inf")], 0)
        .fillna(0).values.astype(np.float32)
    )
    X_scaled = scaler.transform(X_raw)

    X_seqs, seq_timestamps = [], []
    for i in range(seq_len - 1, len(states)):
        X_seqs.append(X_scaled[i - seq_len + 1 : i + 1])
        seq_timestamps.append(states["timestamp"].iloc[i])

    X_seqs = np.array(X_seqs, dtype=np.float32)
    seq_timestamps = pd.Series(seq_timestamps).reset_index(drop=True)
    print(f"      Sequences built: {len(X_seqs):,} (all benign ground truth)")
    return X_seqs, seq_timestamps


def run_specificity_inference(model, X_seqs, seq_timestamps, threshold=0.5, device="cpu"):
    """Compute FPR, specificity, FP list on all-benign Thursday sequences."""
    print(f"\n[3/4] Running LSTM inference (threshold={threshold})...")

    model.eval()
    dev = torch.device(device)
    model = model.to(dev)
    with torch.no_grad():
        probs = model(torch.tensor(X_seqs, dtype=torch.float32, device=dev)
                      ).squeeze().cpu().numpy()

    n_total = len(probs)
    preds   = (probs >= threshold).astype(int)
    fp_mask = preds == 1

    fp_count    = int(fp_mask.sum())
    tn_count    = n_total - fp_count
    fpr         = fp_count / n_total if n_total > 0 else 0.0
    specificity = 1.0 - fpr
    mean_prob   = float(np.mean(probs))
    max_prob    = float(np.max(probs))

    fp_details = [
        {"timestamp": str(seq_timestamps.iloc[i]), "predicted_prob": round(float(probs[i]), 5)}
        for i in range(n_total) if fp_mask[i]
    ]

    print(f"\n      -- Specificity Results (all-benign ground truth) --")
    print(f"      Total benign windows evaluated : {n_total:,}")
    print(f"      True negatives (correct)       : {tn_count:,}")
    print(f"      False positives                : {fp_count}")
    print(f"      False positive rate (FPR)      : {fpr:.4f}  ({fpr*100:.2f}%)")
    print(f"      Specificity (1 - FPR)          : {specificity:.4f}  ({specificity*100:.2f}%)")
    print(f"      Mean predicted attack prob     : {mean_prob:.4f}")
    print(f"      Max predicted attack prob      : {max_prob:.4f}")

    if fp_details:
        print(f"\n      False-positive windows:")
        for fp in fp_details:
            print(f"        {fp['timestamp']}   prob={fp['predicted_prob']:.4f}")
    else:
        print(f"\n      Zero false positives.")

    return {
        "threshold": threshold,
        "total_benign_windows": n_total,
        "true_negatives": tn_count,
        "false_positives": fp_count,
        "fpr": round(fpr, 6),
        "specificity": round(specificity, 6),
        "mean_predicted_prob": round(mean_prob, 6),
        "max_predicted_prob": round(max_prob, 6),
        "fp_windows": fp_details,
        "probs": probs,
    }


def print_report(results):
    r = results
    w = WEDNESDAY_RESULTS

    print("\n" + "=" * 80)
    print("THURSDAY-01-03-2018  OOD SPECIFICITY EVALUATION  (CORRECTED)")
    print("=" * 80)
    print(f"\n  NOTE: {ATTACK_WINDOW_MISMATCH_NOTE}\n")
    print("─" * 80)
    print(f"  {'Metric':<42} {'Value':>15}")
    print("─" * 80)
    print(f"  {'Total benign windows evaluated':<42} {r['total_benign_windows']:>15,}")
    print(f"  {'True negatives (correct)':<42} {r['true_negatives']:>15,}")
    print(f"  {'False positives':<42} {r['false_positives']:>15}")
    print(f"  {'False positive rate (FPR)':<42} {r['fpr']:>15.4f}  ({r['fpr']*100:.2f}%)")
    print(f"  {'Specificity (TNR = 1 - FPR)':<42} {r['specificity']:>15.4f}  ({r['specificity']*100:.2f}%)")
    print(f"  {'Mean predicted attack probability':<42} {r['mean_predicted_prob']:>15.4f}")
    print(f"  {'Max predicted attack probability':<42} {r['max_predicted_prob']:>15.4f}")
    print(f"  {'Decision threshold':<42} {r['threshold']:>15.2f}")
    print("─" * 80)
    print(f"  {'REFERENCE: Wednesday in-distribution FPR':<42} {w['fpr']:>15.4f}  ({w['fpr']*100:.2f}%)")
    fpr_delta = r['fpr'] - w['fpr']
    print(f"  {'Thursday FPR delta vs Wednesday':<42} {fpr_delta:>+15.4f}")
    print("─" * 80)
    print(f"\n  False-positive windows ({r['false_positives']} total):")
    if r["fp_windows"]:
        for fp in r["fp_windows"]:
            print(f"    {fp['timestamp']}  prob={fp['predicted_prob']:.4f}")
        if len(r["fp_windows"]) > 1:
            fp_times = pd.to_datetime([fp["timestamp"] for fp in r["fp_windows"]])
            gaps = pd.Series(fp_times.sort_values()).diff().dropna().dt.total_seconds()
            pattern = "CLUSTERED (within 5 min)" if (len(gaps) > 0 and gaps.min() <= 300) else "SCATTERED"
            print(f"    Pattern: {pattern}")
    else:
        print("    None.")
    print("\n" + "=" * 80)
    print("  DROPPED METRICS (undefined — no positive class in ground truth)")
    print("  Precision, Recall, F1, AUC-ROC: not computable / not meaningful.")
    print("=" * 80)


def plot_results(results, states, seq_len=SEQ_LEN):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not available.")
        return

    probs = results["probs"]
    timestamps = states["timestamp"].iloc[seq_len - 1:].reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Thursday-01-03-2018 Specificity Evaluation (All-Benign Ground Truth)",
        fontsize=13, fontweight="bold",
    )

    axes[0].plot(timestamps, probs, color="#4F9FFF", lw=1.0, alpha=0.8,
                 label="Predicted prob")
    axes[0].axhline(0.5, color="red", linestyle="--", lw=1.2, label="Threshold")
    fp_mask = probs >= 0.5
    if fp_mask.any():
        axes[0].scatter(timestamps[fp_mask], probs[fp_mask],
                        color="red", s=40, zorder=5, label=f"FP ({fp_mask.sum()})")
    axes[0].set_ylabel("Attack Probability")
    axes[0].set_xlabel("Time (UTC)")
    axes[0].set_title("Predicted Probability Timeline")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    fig.autofmt_xdate()

    axes[1].hist(probs, bins=50, color="#4F9FFF", alpha=0.85, edgecolor="white")
    axes[1].axvline(0.5, color="red", linestyle="--", lw=1.5, label="Threshold")
    axes[1].set_xlabel("Predicted Attack Probability")
    axes[1].set_ylabel("Window Count")
    axes[1].set_title("Probability Distribution (all benign)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    out_dir = PROJECT_ROOT / "docs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "thursday_ood_evaluation.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n      Plot saved -> {out_path}")


def save_results(results, states, report_path, args):
    probs = results["probs"]
    timestamps = states["timestamp"].iloc[SEQ_LEN - 1:].reset_index(drop=True)
    stride = max(1, len(probs) // 720)
    trace = [
        {
            "timestamp": str(timestamps.iloc[i]),
            "attack_prob": round(float(probs[i]), 5),
            "true_label": 0,
            "predicted": int(probs[i] >= results["threshold"]),
        }
        for i in range(0, len(probs), stride)
    ]

    report = {
        "meta": {
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "model_file": str(MODEL_PATH),
            "scaler_source": str(WEDNESDAY_TRANSITIONS),
            "scaler_train_window": "<= 609 (Wednesday train split, no retraining)",
            "thursday_csv": str(args.thursday_csv),
            "threshold": results["threshold"],
            "seq_len": SEQ_LEN,
        },
        "ground_truth_note": ATTACK_WINDOW_MISMATCH_NOTE,
        "raw_label_findings": RAW_LABEL_FINDINGS,
        "thursday_file": {
            "date": "2018-03-01",
            "capture_range": "01:00-12:59 UTC",
            "documented_attack_window": {
                "start": "2018-03-01 14:00:00",
                "end": "2018-03-01 15:37:00",
                "status": "OUTSIDE file coverage -- no verified attack traffic in this file",
            },
            "s3_listing_note": (
                "Only ONE Thursday-01-03-2018 file found in S3 listing. "
                "No separate afternoon file confirmed."
            ),
            "attack_scenario": "Infiltration from inside -- Dropbox download + Nmap portscan",
            "total_windows": len(states),
            "ground_truth_attack_windows": 0,
            "ground_truth_benign_windows": len(states),
        },
        "wednesday_reference": {
            "date": "2018-02-28",
            "split": "test (windows >= 630)",
            "fpr": WEDNESDAY_RESULTS["fpr"],
            "note": "In-distribution FPR for comparison only",
        },
        "specificity_results": {
            "total_benign_windows": results["total_benign_windows"],
            "true_negatives": results["true_negatives"],
            "false_positives": results["false_positives"],
            "fpr": results["fpr"],
            "specificity": results["specificity"],
            "mean_predicted_prob": results["mean_predicted_prob"],
            "max_predicted_prob": results["max_predicted_prob"],
            "wednesday_fpr_delta": round(results["fpr"] - WEDNESDAY_RESULTS["fpr"], 6),
        },
        "dropped_metrics": {
            "precision": "undefined -- no positive class in ground truth",
            "recall":    "undefined -- no positive class in ground truth",
            "f1":        "undefined -- no positive class in ground truth",
            "auc_roc":   "undefined -- no positive class in ground truth",
        },
        "false_positive_windows": results["fp_windows"],
        "probability_trace": trace,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    size_kb = report_path.stat().st_size / 1024
    print(f"\n[saved] {report_path}  ({size_kb:.1f} KB, {len(trace)} trace points)")


def main():
    parser = argparse.ArgumentParser(
        description="Corrected specificity evaluation on Thursday-01-03-2018 (all-benign)"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--thursday-csv", type=Path, default=THURSDAY_CSV)
    parser.add_argument("--transitions-csv", type=Path, default=WEDNESDAY_TRANSITIONS)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    print("=" * 80)
    print("OUT-OF-DISTRIBUTION SPECIFICITY TEST (CORRECTED — all-benign ground truth)")
    print("Wednesday LSTM -> Thursday-01-03-2018")
    print("=" * 80)

    states = build_thursday_states(args.thursday_csv)

    print(f"\n[loading] Model + Wednesday-fitted scaler...")
    model, scaler, feature_cols, _ = load_flow_lstm_and_scaler(
        model_path=MODEL_PATH,
        data_path=args.transitions_csv,
        device=args.device,
    )
    print(f"          {MODEL_PATH.name}  |  {len(feature_cols)} features  |  no retraining")

    X_seqs, seq_timestamps = build_thursday_sequences(
        states, scaler, feature_cols, seq_len=args.seq_len
    )

    results = run_specificity_inference(
        model, X_seqs, seq_timestamps,
        threshold=args.threshold, device=args.device,
    )

    print_report(results)

    if args.plot:
        plot_results(results, states, seq_len=args.seq_len)

    save_results(results, states, args.report_path, args)

    print("\n" + "=" * 80)
    print(f"DONE  |  Report: {args.report_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
