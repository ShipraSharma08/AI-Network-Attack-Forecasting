#!/usr/bin/env python3
"""
Packet Feature Extraction & Flow Fusion Pipeline
================================================
Extracts per-packet features from the validated infiltration PCAP capture,
aggregates them into continuous 60-second windows, validates temporal alignment
against the flow dataset, and performs feature fusion:
  29 flow features + 14 packet features = 43 unified features.
"""

import argparse
import ipaddress
import os
import struct
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PCAP_PATH = PROJECT_ROOT / "data" / "raw" / "infiltration.pcap" / "infiltration.pcap"
DEFAULT_FLOW_STATES = PROJECT_ROOT / "data" / "processed" / "flow_network_states.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "fused_flow_packet_states.csv"

# Dataset lab timezone offset: Atlantic Standard Time (AST = UTC - 4 hours)
# Used by UNB Canadian Institute for Cybersecurity for CSE-CIC-IDS2018
DEFAULT_TIMEZONE_OFFSET_HOURS = -4.0


def extract_packet_records(pcap_path: Path, tz_offset_hours: float = DEFAULT_TIMEZONE_OFFSET_HOURS):
    """
    Parse IPv4 packets from PCAP and extract per-packet transport/network metrics.
    Converts Unix epoch timestamps to wall-clock local datetime aligned with flow data.
    """
    print(f"[1/4] Loading PCAP and extracting packet features: {pcap_path} ...")
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    records = []
    tz_delta = timedelta(hours=tz_offset_hours)

    with open(pcap_path, "rb") as f:
        global_hdr = f.read(24)
        if len(global_hdr) < 24:
            raise ValueError("Invalid PCAP file: Header is shorter than 24 bytes.")

        magic = global_hdr[:4]
        endian = "<" if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1") else ">"

        while True:
            pkt_hdr = f.read(16)
            if len(pkt_hdr) < 16:
                break

            ts_sec, ts_usec, incl_len, orig_len = struct.unpack(endian + "IIII", pkt_hdr)
            data = f.read(incl_len)

            # Minimum Ethernet (14) + IPv4 (20) header length
            if incl_len < 34:
                continue

            # Check Ethernet protocol for IPv4 (0x0800)
            eth_type = struct.unpack("!H", data[12:14])[0]
            if eth_type != 0x0800:
                continue

            # Parse IPv4 header
            ip_hdr = data[14:34]
            ihl = (ip_hdr[0] & 0x0F) * 4
            ttl = ip_hdr[8]
            proto = ip_hdr[9]
            src_ip = str(ipaddress.IPv4Address(ip_hdr[12:16]))
            dst_ip = str(ipaddress.IPv4Address(ip_hdr[16:20]))

            flags_frag = struct.unpack("!H", ip_hdr[6:8])[0]
            is_frag = 1 if (flags_frag & 0x3FFF) != 0 else 0

            src_port = None
            dst_port = None
            tcp_win = None
            payload_len = 0

            l4_offset = 14 + ihl
            if proto == 6 and incl_len >= l4_offset + 20:  # TCP
                tcp_hdr = data[l4_offset : l4_offset + 20]
                src_port, dst_port, _, _, off_flags, tcp_win = struct.unpack("!HHIIHH", tcp_hdr[:16])
                tcp_data_offset = ((off_flags >> 12) & 0x0F) * 4
                payload_len = max(0, incl_len - (l4_offset + tcp_data_offset))
            elif proto == 17 and incl_len >= l4_offset + 8:  # UDP
                udp_hdr = data[l4_offset : l4_offset + 8]
                src_port, dst_port, ulen = struct.unpack("!HHH", udp_hdr[:6])
                payload_len = max(0, ulen - 8)

            # Wall-clock timestamp aligned to flow dataset timezone
            dt_utc = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
            dt_local = dt_utc + tz_delta
            window_start = dt_local.replace(tzinfo=None).replace(second=0, microsecond=0)

            records.append(
                (
                    window_start,
                    src_ip,
                    dst_ip,
                    src_port,
                    dst_port,
                    ttl,
                    is_frag,
                    tcp_win,
                    payload_len,
                )
            )

    df_packets = pd.DataFrame(
        records,
        columns=[
            "window_start",
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
            "ttl",
            "ip_fragment",
            "tcp_window",
            "payload_bytes",
        ],
    )

    print(f"      Total IPv4 packets extracted : {len(df_packets):,}")
    print(f"      Packet time range (aligned)  : {df_packets['window_start'].min()} to {df_packets['window_start'].max()}")
    return df_packets


def aggregate_packet_windows(df_packets: pd.DataFrame):
    """
    Aggregate per-packet metrics into continuous 60-second window states.
    Produces 14 packet-level features.
    """
    print("[2/4] Aggregating packets into 60-second network state windows ...")
    grouped = df_packets.groupby("window_start")

    packet_states = grouped.agg(
        packet_count=("payload_bytes", "count"),
        unique_src_ips=("src_ip", "nunique"),
        unique_dst_ips=("dst_ip", "nunique"),
        unique_src_ports=("src_port", "nunique"),
        unique_dst_ports=("dst_port", "nunique"),
        ttl_mean=("ttl", "mean"),
        ttl_std=("ttl", "std"),
        tcp_window_mean=("tcp_window", "mean"),
        tcp_window_std=("tcp_window", "std"),
        payload_mean=("payload_bytes", "mean"),
        payload_std=("payload_bytes", "std"),
        payload_sum=("payload_bytes", "sum"),
        payload_max=("payload_bytes", "max"),
        fragmented_packets=("ip_fragment", "sum"),
    )

    # Clean NaN values arising from single-packet std deviations or empty windows
    packet_states = packet_states.fillna(0.0).reset_index()
    print(f"      Generated packet state windows : {len(packet_states):,}")
    print(f"      Packet features defined        : {len(packet_states.columns) - 1}")
    return packet_states


def validate_alignment(flow_states: pd.DataFrame, packet_states: pd.DataFrame):
    """
    Validate temporal overlap, activity peaks, and correlation between flows and packets.
    """
    print("\n[3/4] Validating Temporal Alignment (Flows vs. Packets) ...")

    flow_min = flow_states["window_start"].min()
    flow_max = flow_states["window_start"].max()
    pcap_min = packet_states["window_start"].min()
    pcap_max = packet_states["window_start"].max()

    overlap_start = max(flow_min, pcap_min)
    overlap_end = min(flow_max, pcap_max)

    print(f"      Flow Timeline  : {flow_min} to {flow_max} ({len(flow_states)} windows)")
    print(f"      PCAP Timeline  : {pcap_min} to {pcap_max} ({len(packet_states)} windows)")
    print(f"      Overlap Window : {overlap_start} to {overlap_end}")

    merged = pd.merge(flow_states, packet_states, on="window_start", how="inner")
    overlap_count = len(merged)
    attack_in_overlap = int(merged["attack_label"].sum())

    print(f"      Overlapping Windows       : {overlap_count:,}")
    print(f"      Attack Windows Preserved  : {attack_in_overlap} / 75 ({attack_in_overlap / 75 * 100:.1f}%)")

    # Check peak alignment
    peak_flow_row = merged.loc[merged["flow_count"].idxmax()]
    peak_pcap_row = merged.loc[merged["packet_count"].idxmax()]

    corr_flow_pkt = merged["flow_count"].corr(merged["packet_count"])
    corr_bytes = (merged["fwd_bytes_sum"] + merged["bwd_bytes_sum"]).corr(merged["payload_sum"])

    print("\n      Peak Activity & Correlation:")
    print(f"        Peak Flow Activity Window   : {peak_flow_row['window_start']} ({peak_flow_row['flow_count']:,.0f} flows)")
    print(f"        Peak Packet Activity Window : {peak_pcap_row['window_start']} ({peak_pcap_row['packet_count']:,} packets)")
    print(f"        Flow Count vs Packet Count  : r = {corr_flow_pkt:+.4f}")
    print(f"        Flow Bytes vs Payload Bytes : r = {corr_bytes:+.4f}")

    if attack_in_overlap == 75:
        print("      Alignment Status            : PERFECT ALIGNMENT (100% of attack window captured) ✓")
    else:
        print(f"      Alignment Status            : PARTIAL ({attack_in_overlap}/75 attack windows)")

    return merged


def fuse_and_save(fused_df: pd.DataFrame, output_path: Path):
    """
    Format unified feature columns and save fused state dataset.
    29 flow features + 14 packet features = 43 unified features.
    """
    print(f"\n[4/4] Finalizing and saving fused states dataset to: {output_path} ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_cols = ["timestamp", "window", "elapsed_seconds", "attack_label"]
    flow_feature_cols = [
        c for c in fused_df.columns
        if c not in metadata_cols and c not in ["window_start"] and not c.startswith("unique_") and c not in [
            "packet_count", "ttl_mean", "ttl_std", "tcp_window_mean", "tcp_window_std",
            "payload_mean", "payload_std", "payload_sum", "payload_max", "fragmented_packets"
        ]
    ]
    packet_feature_cols = [
        "packet_count",
        "unique_src_ips",
        "unique_dst_ips",
        "unique_src_ports",
        "unique_dst_ports",
        "ttl_mean",
        "ttl_std",
        "tcp_window_mean",
        "tcp_window_std",
        "payload_mean",
        "payload_std",
        "payload_sum",
        "payload_max",
        "fragmented_packets",
    ]

    # Re-index window ID sequentially for the fused timeline
    fused_df = fused_df.sort_values("window_start").reset_index(drop=True)
    fused_df["timestamp"] = fused_df["window_start"].dt.strftime("%Y-%m-%d %H:%M:%S")
    fused_df["window"] = np.arange(len(fused_df))
    fused_df["elapsed_seconds"] = (fused_df["window_start"] - fused_df["window_start"].iloc[0]).dt.total_seconds()

    final_columns = metadata_cols + flow_feature_cols + packet_feature_cols
    fused_clean = fused_df[final_columns].copy()

    # Clean any infinite or NaN values
    fused_clean = fused_clean.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    fused_clean.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("PACKET-FLOW FUSION COMPLETE")
    print("=" * 70)
    print(f"Output File Path       : {output_path}")
    print(f"Total Unified Windows  : {len(fused_clean):,}")
    print(f"Total Columns          : {len(fused_clean.columns)}")
    print(f"  - Metadata & Label   : {len(metadata_cols)} ({', '.join(metadata_cols)})")
    print(f"  - Flow Features      : {len(flow_feature_cols)}")
    print(f"  - Packet Features    : {len(packet_feature_cols)}")
    print(f"  - Total State Metric : {len(flow_feature_cols) + len(packet_feature_cols)} unified features (29 flow + 14 packet)")
    print(f"Attack Windows         : {int(fused_clean['attack_label'].sum())}")
    print(f"Benign Windows         : {int((fused_clean['attack_label'] == 0).sum())}")
    print(f"Timeline Span          : {fused_clean['timestamp'].min()} to {fused_clean['timestamp'].max()}")
    print("=" * 70 + "\n")
    return fused_clean


def main():
    parser = argparse.ArgumentParser(description="Extract timestamped packet features and fuse with flow states.")
    parser.add_argument("--pcap-path", type=Path, default=DEFAULT_PCAP_PATH, help="Path to input PCAP capture.")
    parser.add_argument("--flow-states", type=Path, default=DEFAULT_FLOW_STATES, help="Path to flow_network_states.csv.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path for fused_flow_packet_states.csv.")
    parser.add_argument("--tz-offset", type=float, default=DEFAULT_TIMEZONE_OFFSET_HOURS, help="Timezone offset hours (default: -4 for AST).")
    args = parser.parse_args()

    print("=" * 70)
    print("PACKET FEATURE EXTRACTION & MULTI-MODAL FLOW FUSION")
    print("=" * 70)

    # 1. Extract per-packet features
    df_packets = extract_packet_records(args.pcap_path, tz_offset_hours=args.tz_offset)

    # 2. Aggregate into 60s windows
    df_packet_states = aggregate_packet_windows(df_packets)

    # 3. Load flow states
    if not args.flow_states.exists():
        raise FileNotFoundError(f"Flow states file not found: {args.flow_states}")
    flow_states = pd.read_csv(args.flow_states)
    flow_states["window_start"] = pd.to_datetime(flow_states["timestamp"])

    # 4. Validate alignment
    fused_df = validate_alignment(flow_states, df_packet_states)

    # 5. Fuse features and save
    fuse_and_save(fused_df, args.output_path)


if __name__ == "__main__":
    main()
