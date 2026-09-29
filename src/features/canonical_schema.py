"""
Canonical Network Flow State Feature Schema & Validation Module
===============================================================
Authoritative Single Source of Truth for the 29 Flow-Level Network State Features
used across training, evaluation, PCAP parsing, and dashboard inference.
"""

from typing import List, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CANONICAL_29_FLOW_FEATURES: Tuple[str, ...] = (
    "flow_count",
    "flow_duration_mean",
    "fwd_packets_mean",
    "bwd_packets_mean",
    "fwd_bytes_sum",
    "bwd_bytes_sum",
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_max",
    "flow_iat_min",
    "fwd_iat_mean",
    "bwd_iat_mean",
    "syn_count",
    "ack_count",
    "rst_count",
    "fin_count",
    "psh_fwd_count",
    "psh_bwd_count",
    "urg_fwd_count",
    "urg_bwd_count",
    "packet_length_mean",
    "packet_length_std",
    "packets_per_second",
    "bytes_per_second",
    "down_up_ratio",
    "active_mean",
    "active_std",
    "idle_mean",
    "idle_std",
)

NUM_CANONICAL_FEATURES: int = len(CANONICAL_29_FLOW_FEATURES)
assert NUM_CANONICAL_FEATURES == 29, f"Expected 29 features, got {NUM_CANONICAL_FEATURES}"

FEATURE_METADATA: Dict[str, Dict[str, str]] = {
    "flow_count": {"unit": "flows/window", "desc": "Total active flows in 60s window"},
    "flow_duration_mean": {"unit": "microseconds", "desc": "Mean flow duration"},
    "fwd_packets_mean": {"unit": "packets/flow", "desc": "Mean forward packets per flow"},
    "bwd_packets_mean": {"unit": "packets/flow", "desc": "Mean backward packets per flow"},
    "fwd_bytes_sum": {"unit": "bytes", "desc": "Total forward payload & header bytes"},
    "bwd_bytes_sum": {"unit": "bytes", "desc": "Total backward payload & header bytes"},
    "flow_iat_mean": {"unit": "microseconds", "desc": "Mean flow inter-arrival time"},
    "flow_iat_std": {"unit": "microseconds", "desc": "Std dev of flow inter-arrival time"},
    "flow_iat_max": {"unit": "microseconds", "desc": "Maximum flow inter-arrival time"},
    "flow_iat_min": {"unit": "microseconds", "desc": "Minimum flow inter-arrival time"},
    "fwd_iat_mean": {"unit": "microseconds", "desc": "Mean forward packet inter-arrival time"},
    "bwd_iat_mean": {"unit": "microseconds", "desc": "Mean backward packet inter-arrival time"},
    "syn_count": {"unit": "flags", "desc": "Total SYN flags observed (scanning indicator)"},
    "ack_count": {"unit": "flags", "desc": "Total ACK flags observed (session handshake/C2)"},
    "rst_count": {"unit": "flags", "desc": "Total RST connection resets (aborted sweeps)"},
    "fin_count": {"unit": "flags", "desc": "Total FIN teardowns"},
    "psh_fwd_count": {"unit": "flags", "desc": "Forward PSH flags (payload delivery/exploit)"},
    "psh_bwd_count": {"unit": "flags", "desc": "Backward PSH flags"},
    "urg_fwd_count": {"unit": "flags", "desc": "Forward URG flags"},
    "urg_bwd_count": {"unit": "flags", "desc": "Backward URG flags"},
    "packet_length_mean": {"unit": "bytes", "desc": "Mean packet size"},
    "packet_length_std": {"unit": "bytes", "desc": "Std dev of packet size"},
    "packets_per_second": {"unit": "packets/sec", "desc": "Mean flow packet throughput"},
    "bytes_per_second": {"unit": "bytes/sec", "desc": "Mean flow byte throughput"},
    "down_up_ratio": {"unit": "ratio", "desc": "Download to upload packet ratio"},
    "active_mean": {"unit": "microseconds", "desc": "Mean active burst period before idle"},
    "active_std": {"unit": "microseconds", "desc": "Std dev of active burst period"},
    "idle_mean": {"unit": "microseconds", "desc": "Mean idle period between bursts"},
    "idle_std": {"unit": "microseconds", "desc": "Std dev of idle period"},
}


def assert_schema_compliance(columns: List[str], allow_prefixed: bool = False) -> None:
    """
    Validate that a given list of feature column names exactly matches
    the canonical 29 features in exact order.
    If allow_prefixed is True, strips 'current_' or 'next_' prefixes before comparison.
    Raises ValueError with descriptive diagnosis if any discrepancy exists.
    """
    if len(columns) != NUM_CANONICAL_FEATURES:
        raise ValueError(
            f"Feature count mismatch: Expected {NUM_CANONICAL_FEATURES} features, "
            f"but received {len(columns)}."
        )

    for idx, (expected, actual) in enumerate(zip(CANONICAL_29_FLOW_FEATURES, columns)):
        check_actual = actual
        if allow_prefixed:
            for prefix in ("current_", "next_"):
                if check_actual.startswith(prefix):
                    check_actual = check_actual[len(prefix):]
                    break
        if expected != check_actual:
            raise ValueError(
                f"Feature ordering mismatch at position {idx}: "
                f"expected '{expected}', got '{actual}'."
            )


def clean_feature_matrix(
    X: Union[np.ndarray, pd.DataFrame]
) -> np.ndarray:
    """
    Clean numerical matrix:
      - convert to float32
      - replace inf and -inf with 0.0
      - fill NaN with 0.0
    """
    if isinstance(X, pd.DataFrame):
        arr = X.values
    else:
        arr = np.array(X)

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return arr


def fit_canonical_scaler(
    df: pd.DataFrame,
    train_window_cutoff: int = 609,
) -> StandardScaler:
    """
    Fit a StandardScaler strictly on the chronological training split (window <= 609).
    Guarantees zero future-information temporal leakage.
    """
    train_df = df[df["window"] <= train_window_cutoff]
    missing = [c for c in CANONICAL_29_FLOW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing canonical features: {missing}")

    X_train = clean_feature_matrix(train_df[list(CANONICAL_29_FLOW_FEATURES)])
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler
