import csv
import os
import sys
from datetime import datetime, timezone
from scapy.all import IP, TCP, UDP, PcapReader, rdpcap


def _parse_to_utc_datetime(val):
    """Normalize string, pd.Timestamp, or datetime to a timezone-naive UTC datetime object."""
    if isinstance(val, str):
        val = datetime.fromisoformat(val.strip())
    if hasattr(val, "to_pydatetime"):
        val = val.to_pydatetime()
    if val.tzinfo is not None:
        val = val.astimezone(timezone.utc).replace(tzinfo=None)
    return val


def validate_pcap_window(
    pcap_path,
    attack_start="2018-02-28 10:50:00",
    attack_end="2018-02-28 12:05:00",
    raise_on_error=True,
):
    """
    Validate that a PCAP file's timestamps match the official attack window.

    1. Load the PCAP file using Scapy: scapy.all.rdpcap(pcap_path)
    2. Extract the first and last packet timestamps (pkt.time, Unix epoch)
    3. Convert epoch timestamps to wall-clock UTC datetime
    4. Assert that:
       - First packet timestamp >= attack_start (2018-02-28 10:50:00)
       - Last packet timestamp <= attack_end (2018-02-28 12:05:00)
       - If either assertion fails, raise AssertionError with a detailed message
         showing the actual PCAP range vs. expected range
    5. Log the PCAP timestamp bounds and packet count to console
    6. Return a dict:
       {
         "pcap_valid": True/False,
         "first_packet_time": datetime,
         "last_packet_time": datetime,
         "packet_count": int,
         "message": str
       }
    """
    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    start_dt = _parse_to_utc_datetime(attack_start)
    end_dt = _parse_to_utc_datetime(attack_end)

    print(f"[1/3] Loading PCAP file with Scapy: {pcap_path} ...")
    packets = rdpcap(pcap_path)
    packet_count = len(packets)

    if packet_count == 0:
        raise ValueError(f"PCAP file {pcap_path} contains 0 packets.")

    print(f"[2/3] Extracting packet timestamps for {packet_count:,} packets ...")
    first_epoch = float(packets[0].time)
    last_epoch = float(packets[-1].time)

    first_packet_time = datetime.fromtimestamp(first_epoch, tz=timezone.utc).replace(tzinfo=None)
    last_packet_time = datetime.fromtimestamp(last_epoch, tz=timezone.utc).replace(tzinfo=None)

    # 5. Log the PCAP timestamp bounds and packet count to console
    print(f"\n===== PCAP TIMESTAMP BOUNDS =====")
    print(f"File Path:         {pcap_path}")
    print(f"Total Packets:     {packet_count:,}")
    print(f"First Packet UTC:  {first_packet_time} (Epoch: {first_epoch})")
    print(f"Last Packet UTC:   {last_packet_time} (Epoch: {last_epoch})")
    print(f"Expected Window:   [{start_dt} to {end_dt}]")
    print(f"=================================\n")

    # 4. Assert bounds
    valid_start = first_packet_time >= start_dt
    valid_end = last_packet_time <= end_dt
    pcap_valid = valid_start and valid_end

    if not pcap_valid:
        reasons = []
        if not valid_start:
            reasons.append(
                f"First packet ({first_packet_time}) is earlier than expected start ({start_dt})"
            )
        if not valid_end:
            reasons.append(
                f"Last packet ({last_packet_time}) is later than expected end ({end_dt})"
            )
        error_msg = (
            f"PCAP window validation failed! {'; '.join(reasons)}. "
            f"Actual PCAP range: [{first_packet_time} to {last_packet_time}], "
            f"Expected attack range: [{start_dt} to {end_dt}]."
        )
        if raise_on_error:
            raise AssertionError(error_msg)
        else:
            return {
                "pcap_valid": False,
                "first_packet_time": first_packet_time,
                "last_packet_time": last_packet_time,
                "packet_count": packet_count,
                "message": error_msg,
            }

    success_msg = (
        f"PCAP timestamps are within the expected attack window. "
        f"Actual range: [{first_packet_time} to {last_packet_time}], "
        f"Packet count: {packet_count:,}."
    )
    print(f"[PCAP VALIDATION] {success_msg}")

    # 6. Return validation dictionary
    return {
        "pcap_valid": True,
        "first_packet_time": first_packet_time,
        "last_packet_time": last_packet_time,
        "packet_count": packet_count,
        "message": success_msg,
    }


def extract_pcap_to_csv(pcap_path, output_path):
    with PcapReader(pcap_path) as packets, open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "src_ip",
                "dst_ip",
                "protocol",
                "src_port",
                "dst_port",
                "ttl",
                "ip_fragment",
                "tcp_window",
                "tcp_flags",
                "payload_bytes",
            ],
        )
        writer.writeheader()

        for pkt in packets:
            if IP not in pkt:
                continue

            row = {
                "timestamp": float(pkt.time),
                "src_ip": pkt[IP].src,
                "dst_ip": pkt[IP].dst,
                "protocol": int(pkt[IP].proto),
                "src_port": None,
                "dst_port": None,
                "ttl": int(pkt[IP].ttl),
                "ip_fragment": int(pkt[IP].flags.MF or pkt[IP].frag),
                "tcp_window": None,
                "tcp_flags": None,
                "payload_bytes": 0,
            }

            if TCP in pkt:
                row["src_port"] = int(pkt[TCP].sport)
                row["dst_port"] = int(pkt[TCP].dport)
                row["tcp_window"] = int(pkt[TCP].window)
                row["tcp_flags"] = str(pkt[TCP].flags)
                row["payload_bytes"] = len(bytes(pkt[TCP].payload))

            elif UDP in pkt:
                row["src_port"] = int(pkt[UDP].sport)
                row["dst_port"] = int(pkt[UDP].dport)
                row["payload_bytes"] = len(bytes(pkt[UDP].payload))

            writer.writerow(row)


if __name__ == "__main__":
    pcap_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/infiltration.pcap"
    attack_start = "2018-02-28 10:50:00"
    attack_end = "2018-02-28 12:05:00"

    print("=" * 60)
    print("PCAP WINDOW VALIDATION RUNNER")
    print("=" * 60)
    print(f"Target PCAP:     {pcap_path}")
    print(f"Attack Start:    {attack_start}")
    print(f"Attack End:      {attack_end}")
    print("=" * 60)

    try:
        result = validate_pcap_window(
            pcap_path=pcap_path,
            attack_start=attack_start,
            attack_end=attack_end,
            raise_on_error=True,
        )
        print("\nValidation Result Dict:")
        for k, v in result.items():
            print(f"  {k}: {v}")
        sys.exit(0)
    except AssertionError as ae:
        print(f"\n[VALIDATION FAILED - AssertionError]: {ae}")
        sys.exit(1)
    except FileNotFoundError as fnf:
        print(f"\n[FILE NOT FOUND]: {fnf}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        sys.exit(1)
