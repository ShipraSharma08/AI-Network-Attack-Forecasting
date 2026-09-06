from scapy.all import PcapReader, IP, TCP, UDP
import csv


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
    extract_pcap_to_csv(
        "/tmp/infiltration.pcap",
        "data/processed/infiltration_packet_features.csv",
    )
    print("Packet feature extraction complete.")
