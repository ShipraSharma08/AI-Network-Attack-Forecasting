import pandas as pd
import numpy as np
from pathlib import Path


FLOW_FILE = "data/raw/Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv"
PACKET_FILE = "data/processed/infiltration_packet_features.csv"
OUTPUT_FILE = "data/processed/network_states.csv"

WINDOW_SECONDS = 60


def build_flow_states():
    print("[1/3] Loading flow data...")

    columns = [
        "Timestamp",
        "Flow Duration",
        "Tot Fwd Pkts",
        "Tot Bwd Pkts",
        "TotLen Fwd Pkts",
        "TotLen Bwd Pkts",
        "Flow IAT Mean",
        "Flow IAT Std",
        "Flow IAT Max",
        "Flow IAT Min",
        "Fwd IAT Mean",
        "Bwd IAT Mean",
        "Fwd PSH Flags",
        "Bwd PSH Flags",
        "Fwd URG Flags",
        "Bwd URG Flags",
        "SYN Flag Cnt",
        "ACK Flag Cnt",
        "RST Flag Cnt",
        "FIN Flag Cnt",
        "Pkt Len Mean",
        "Pkt Len Std",
        "Flow Pkts/s",
        "Flow Byts/s",
        "Down/Up Ratio",
        "Active Mean",
        "Active Std",
        "Idle Mean",
        "Idle Std",
        "Label",
    ]

    df = pd.read_csv(
        FLOW_FILE,
        usecols=columns,
        low_memory=False,
    )

    df = df[df["Label"].isin(["Benign", "Infilteration"])].copy()

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        dayfirst=True,
        errors="coerce",
    )

    df = df.dropna(subset=["Timestamp"])

    df["elapsed_sec"] = (
        df["Timestamp"] - df["Timestamp"].min()
    ).dt.total_seconds()

    df["window"] = (
        df["elapsed_sec"] // WINDOW_SECONDS
    ).astype(int)

    numeric_columns = [
        c for c in columns
        if c not in ["Timestamp", "Label"]
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    states = df.groupby("window").agg(
        flow_count=("Label", "size"),
        flow_duration_mean=("Flow Duration", "mean"),
        fwd_packets_mean=("Tot Fwd Pkts", "mean"),
        bwd_packets_mean=("Tot Bwd Pkts", "mean"),
        fwd_bytes_sum=("TotLen Fwd Pkts", "sum"),
        bwd_bytes_sum=("TotLen Bwd Pkts", "sum"),
        flow_iat_mean=("Flow IAT Mean", "mean"),
        flow_iat_std=("Flow IAT Std", "mean"),
        flow_iat_max=("Flow IAT Max", "max"),
        flow_iat_min=("Flow IAT Min", "min"),
        fwd_iat_mean=("Fwd IAT Mean", "mean"),
        bwd_iat_mean=("Bwd IAT Mean", "mean"),
        syn_count=("SYN Flag Cnt", "sum"),
        ack_count=("ACK Flag Cnt", "sum"),
        rst_count=("RST Flag Cnt", "sum"),
        fin_count=("FIN Flag Cnt", "sum"),
        psh_fwd_count=("Fwd PSH Flags", "sum"),
        psh_bwd_count=("Bwd PSH Flags", "sum"),
        urg_fwd_count=("Fwd URG Flags", "sum"),
        urg_bwd_count=("Bwd URG Flags", "sum"),
        packet_length_mean=("Pkt Len Mean", "mean"),
        packet_length_std=("Pkt Len Std", "mean"),
        packets_per_second=("Flow Pkts/s", "mean"),
        bytes_per_second=("Flow Byts/s", "mean"),
        down_up_ratio=("Down/Up Ratio", "mean"),
        active_mean=("Active Mean", "mean"),
        active_std=("Active Std", "mean"),
        idle_mean=("Idle Mean", "mean"),
        idle_std=("Idle Std", "mean"),
    )

    states["infiltration_label"] = ((states.index * WINDOW_SECONDS >= 35400) & (states.index * WINDOW_SECONDS < 39900)).astype(int)

    return states


def build_packet_states():
    print("[2/3] Loading packet data...")

    df = pd.read_csv(
        PACKET_FILE,
        low_memory=False,
    )

    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="coerce",
    )

    df = df.dropna(subset=["timestamp"]).copy()

    df["elapsed_sec"] = (
        df["timestamp"] - df["timestamp"].min()
    )

    df["window"] = (
        df["elapsed_sec"] // WINDOW_SECONDS
    ).astype(int)

    numeric_columns = [
        "ttl",
        "tcp_window",
        "payload_bytes",
        "ip_fragment",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    states = df.groupby("window").agg(
        packet_count=("timestamp", "size"),
        unique_src_ips=("src_ip", "nunique"),
        unique_dst_ips=("dst_ip", "nunique"),
        unique_src_ports=("src_port", "nunique"),
        unique_dst_ports=("dst_port", "nunique"),
        ttl_mean=("ttl", "mean"),
        ttl_std=("ttl", "std"),
        tcp_window_mean=("tcp_window", "mean"),
        tcp_window_std=("tcp_window", "std"),
        payload_mean=("payload_bytes", "mean"),
        payload_sum=("payload_bytes", "sum"),
        payload_max=("payload_bytes", "max"),
        fragmented_packets=("ip_fragment", "sum"),
    )

    return states


def main():
    Path("data/processed").mkdir(
        parents=True,
        exist_ok=True,
    )

    flow_states = build_flow_states()
    packet_states = build_packet_states()

    print("[3/3] Combining flow and packet states...")

    states = flow_states.join(
        packet_states,
        how="left",
    )

    states = states.reset_index()

    states["window_start_sec"] = (
        states["window"] * WINDOW_SECONDS
    )

    states = states.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    states.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Network state construction complete.")
    print("Rows:", len(states))
    print("Columns:", len(states.columns))
    print("Output:", OUTPUT_FILE)
    print("Window size:", WINDOW_SECONDS)
    print(
        "Infilteration windows:",
        int(states["infiltration_label"].sum()),
    )


if __name__ == "__main__":
    main()       
