import pandas as pd
import numpy as np
from pathlib import Path


FLOW_FILE = "data/raw/Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv"
OUTPUT_FILE = "data/processed/flow_network_states.csv"

WINDOW_SECONDS = 60

# Official CIC-IDS2018 infiltration window for 28-Feb-2018
ATTACK_START = pd.Timestamp("2018-02-28 10:50:00")
ATTACK_END = pd.Timestamp("2018-02-28 12:05:00")


FEATURE_COLUMNS = [
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
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",

    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",

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
]


def main():

    print("[1/4] Loading flow dataset...")

    usecols = ["Timestamp"] + FEATURE_COLUMNS

    df = pd.read_csv(
        FLOW_FILE,
        usecols=usecols,
        low_memory=False
    )

    print(f"Loaded rows: {len(df):,}")


    print("[2/4] Cleaning timestamps and numeric features...")

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        dayfirst=True,
        errors="coerce"
    )

    df = df.dropna(subset=["Timestamp"]).copy()

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    print(f"Valid timestamp rows: {len(df):,}")
    print(f"Start: {df['Timestamp'].min()}")
    print(f"End:   {df['Timestamp'].max()}")


    print("[3/4] Building complete 60-second network states...")

    # Align every flow to a real clock-minute boundary.
    df["window_start"] = df["Timestamp"].dt.floor(
        f"{WINDOW_SECONDS}s"
    )

    # Aggregate flow behaviour inside every minute.
    grouped = df.groupby("window_start")

    states = grouped.size().rename("flow_count").to_frame()

    states["flow_duration_mean"] = grouped[
        "Flow Duration"
    ].mean()

    states["fwd_packets_mean"] = grouped[
        "Tot Fwd Pkts"
    ].mean()

    states["bwd_packets_mean"] = grouped[
        "Tot Bwd Pkts"
    ].mean()

    states["fwd_bytes_sum"] = grouped[
        "TotLen Fwd Pkts"
    ].sum()

    states["bwd_bytes_sum"] = grouped[
        "TotLen Bwd Pkts"
    ].sum()

    states["flow_iat_mean"] = grouped[
        "Flow IAT Mean"
    ].mean()

    states["flow_iat_std"] = grouped[
        "Flow IAT Std"
    ].mean()

    states["flow_iat_max"] = grouped[
        "Flow IAT Max"
    ].max()

    states["flow_iat_min"] = grouped[
        "Flow IAT Min"
    ].min()

    states["fwd_iat_mean"] = grouped[
        "Fwd IAT Mean"
    ].mean()

    states["bwd_iat_mean"] = grouped[
        "Bwd IAT Mean"
    ].mean()

    states["syn_count"] = grouped[
        "SYN Flag Cnt"
    ].sum()

    states["ack_count"] = grouped[
        "ACK Flag Cnt"
    ].sum()

    states["rst_count"] = grouped[
        "RST Flag Cnt"
    ].sum()

    states["fin_count"] = grouped[
        "FIN Flag Cnt"
    ].sum()

    states["psh_fwd_count"] = grouped[
        "Fwd PSH Flags"
    ].sum()

    states["psh_bwd_count"] = grouped[
        "Bwd PSH Flags"
    ].sum()

    states["urg_fwd_count"] = grouped[
        "Fwd URG Flags"
    ].sum()

    states["urg_bwd_count"] = grouped[
        "Bwd URG Flags"
    ].sum()

    states["packet_length_mean"] = grouped[
        "Pkt Len Mean"
    ].mean()

    states["packet_length_std"] = grouped[
        "Pkt Len Std"
    ].mean()

    states["packets_per_second"] = grouped[
        "Flow Pkts/s"
    ].mean()

    states["bytes_per_second"] = grouped[
        "Flow Byts/s"
    ].mean()

    states["down_up_ratio"] = grouped[
        "Down/Up Ratio"
    ].mean()

    states["active_mean"] = grouped[
        "Active Mean"
    ].mean()

    states["active_std"] = grouped[
        "Active Std"
    ].mean()

    states["idle_mean"] = grouped[
        "Idle Mean"
    ].mean()

    states["idle_std"] = grouped[
        "Idle Std"
    ].mean()


    # ---------------------------------------------------------
    # IMPORTANT:
    # Reindex to a COMPLETE one-minute timeline.
    # This creates states even when no flows occurred.
    # ---------------------------------------------------------

    start = df["Timestamp"].min().floor(
        f"{WINDOW_SECONDS}s"
    )

    end = df["Timestamp"].max().floor(
        f"{WINDOW_SECONDS}s"
    )

    complete_index = pd.date_range(
        start=start,
        end=end,
        freq=f"{WINDOW_SECONDS}s"
    )

    states = states.reindex(complete_index)

    states.index.name = "timestamp"


    # Traffic counters are genuinely zero when
    # no flow occurred in that window.
    zero_features = [
        "flow_count",
        "fwd_bytes_sum",
        "bwd_bytes_sum",
        "syn_count",
        "ack_count",
        "rst_count",
        "fin_count",
        "psh_fwd_count",
        "psh_bwd_count",
        "urg_fwd_count",
        "urg_bwd_count",
    ]

    states[zero_features] = states[
        zero_features
    ].fillna(0)


    # Add sequential window index.
    states["window"] = np.arange(len(states))


    # Official attack target.
    states["attack_label"] = (
        (states.index >= ATTACK_START)
        & (states.index < ATTACK_END)
    ).astype(int)


    # Relative time from beginning of dataset.
    states["elapsed_seconds"] = (
        states.index - states.index[0]
    ).total_seconds()


    # Put identifiers first.
    states = states.reset_index()

    first_columns = [
        "timestamp",
        "window",
        "elapsed_seconds",
        "attack_label",
    ]

    other_columns = [
        c for c in states.columns
        if c not in first_columns
    ]

    states = states[
        first_columns + other_columns
    ]


    print("[4/4] Saving temporal network states...")

    Path("data/processed").mkdir(
        parents=True,
        exist_ok=True
    )

    states.to_csv(
        OUTPUT_FILE,
        index=False
    )


    print()
    print("=" * 60)
    print("TEMPORAL NETWORK STATE CONSTRUCTION COMPLETE")
    print("=" * 60)
    print(f"Rows:             {len(states):,}")
    print(f"Columns:          {len(states.columns)}")
    print(f"Start:            {states['timestamp'].min()}")
    print(f"End:              {states['timestamp'].max()}")
    print(
        f"Attack windows:   "
        f"{states['attack_label'].sum():,}"
    )
    print(
        f"Benign windows:   "
        f"{(states['attack_label'] == 0).sum():,}"
    )
    print(f"Output:           {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
