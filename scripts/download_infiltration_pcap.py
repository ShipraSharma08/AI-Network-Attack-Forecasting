#!/usr/bin/env python3
"""
Targeted Infiltration PCAP Downloader
====================================
Downloads only the specific Infiltration victim host PCAP
(172.31.69.13, ~67.7 MB compressed / ~87.8 MB uncompressed) directly from the
official CSE-CIC-IDS2018 AWS S3 archive via HTTP byte-range request.
Avoids downloading the full 53.25 GB pcap.zip archive.
"""

import os
import sys
import urllib.request
import zlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TARGET_DIR = PROJECT_ROOT / "data" / "raw" / "infiltration.pcap"
TARGET_PCAP = TARGET_DIR / "infiltration.pcap"

S3_URL = (
    "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/"
    "Original+Network+Traffic+and+Log+data/Wednesday-28-02-2018/pcap.zip"
)

# Byte range for pcap/capEC2AMAZ-O4EL3NG-172.31.69.13 compressed stream
RANGE_START = 31149712886
COMPRESSED_SIZE = 70988531
RANGE_END = RANGE_START + COMPRESSED_SIZE - 1


def download_and_extract_pcap(output_path: Path):
    """Stream and decompress the specific member PCAP using HTTP Range requests."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp")

    headers = {
        "Range": f"bytes={RANGE_START}-{RANGE_END}",
        "User-Agent": "Mozilla/5.0 (AI-Network-Attack-Forecasting)",
    }
    req = urllib.request.Request(S3_URL, headers=headers)

    print("=" * 70)
    print("TARGETED INFILTRATION PCAP DOWNLOADER (CSE-CIC-IDS2018)")
    print("=" * 70)
    print(f"Source URL       : {S3_URL}")
    print(f"Target Member    : pcap/capEC2AMAZ-O4EL3NG-172.31.69.13 (Victim IP: 172.31.69.13)")
    print(f"Byte Range       : {RANGE_START:,} - {RANGE_END:,} ({COMPRESSED_SIZE / (1024**2):.2f} MB)")
    print(f"Destination Path : {output_path}")
    print("-" * 70)
    print("Connecting to AWS S3 ca-central-1 endpoint...")

    with urllib.request.urlopen(req) as response:
        if response.status not in (200, 206):
            raise RuntimeError(f"Unexpected HTTP status {response.status}: {response.reason}")

        print(f"HTTP Status: {response.status} (Partial Content). Streaming and decompressing...")
        decompressor = zlib.decompressobj(-15)
        downloaded = 0

        with open(temp_path, "wb") as f_out:
            while True:
                chunk = response.read(1024 * 512)
                if not chunk:
                    break
                downloaded += len(chunk)
                f_out.write(decompressor.decompress(chunk))
                pct = (downloaded / COMPRESSED_SIZE) * 100
                mb = downloaded / (1024**2)
                print(f"\rProgress: {pct:5.1f}% ({mb:6.2f} MB / {COMPRESSED_SIZE / (1024**2):.2f} MB)", end="", flush=True)

            f_out.write(decompressor.flush())

    print()
    if temp_path.exists():
        temp_path.replace(output_path)

    uncompressed_size = output_path.stat().st_size
    print(f"[COMPLETE] Extracted raw PCAP successfully ({uncompressed_size / (1024**2):.2f} MB uncompressed).")
    print("=" * 70)
    return output_path


def main():
    try:
        pcap_path = download_and_extract_pcap(TARGET_PCAP)
    except Exception as exc:
        print(f"\n[DOWNLOAD FAILED] {exc}")
        print("Note: If network connection failed, ensure outbound HTTPS to AWS S3 ca-central-1 is allowed.")
        sys.exit(1)

    # Validate downloaded PCAP window using packet_features
    try:
        from src.features.packet_features import validate_pcap_window
        print("\n[VALIDATION] Running validate_pcap_window() on downloaded PCAP...")
        result = validate_pcap_window(str(pcap_path))
        print(f"[STATUS] {result.get('message')}")
        sys.exit(0)
    except Exception as err:
        print(f"[VALIDATION NOTICE] {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
