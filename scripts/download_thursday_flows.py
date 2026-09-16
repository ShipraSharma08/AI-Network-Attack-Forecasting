#!/usr/bin/env python3
"""
Download Thursday-01-03-2018 Flow CSV from CIC-IDS2018 S3 Bucket
=================================================================
Steps:
  1. List all CSVs in s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/
  2. Find the Thursday-01-03-2018 CSV filename from the actual listing
     (don't hardcode or guess — confirm from S3 first)
  3. Download ONLY that file to data/raw/
  4. Print file size and row count

Usage:
    python scripts/download_thursday_flows.py [--output-dir data/raw]
"""

import argparse
import subprocess
import sys
from pathlib import Path

S3_BASE = "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/"
REGION = "ca-central-1"


def list_s3_files(prefix: str) -> list[str]:
    """List all objects under the given S3 prefix (no-sign-request)."""
    cmd = [
        "aws", "s3", "ls",
        "--no-sign-request",
        "--region", REGION,
        prefix,
    ]
    print(f"[1/4] Listing S3 bucket contents:")
    print(f"      {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] aws s3 ls failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    print(f"\n      Found {len(lines)} entries:")
    for line in lines:
        print(f"        {line}")

    # Extract filenames only (last whitespace-delimited token per line)
    filenames = [l.split()[-1] for l in lines if l]
    return filenames


def find_thursday_csv(filenames: list[str]) -> str:
    """
    Identify the Thursday-01-03-2018 CSV from the listing.
    Tries multiple common naming conventions used by CIC-IDS2018.
    """
    candidates = [
        f for f in filenames
        if "thursday" in f.lower() or "01-03-2018" in f or "03-01-2018" in f
    ]
    if not candidates:
        print(
            "\n[WARNING] Could not auto-match 'Thursday' in filenames.\n"
            "          Full listing:\n" + "\n".join(f"  {f}" for f in filenames),
            file=sys.stderr,
        )
        # Fallback: prompt user to pick
        print("\nEnter the exact filename from the listing above: ", end="")
        chosen = input().strip()
        if not chosen:
            print("[ERROR] No filename provided.", file=sys.stderr)
            sys.exit(1)
        return chosen

    if len(candidates) > 1:
        print(f"\n[WARNING] Multiple Thursday matches found: {candidates}")
        print("          Using first match: ", candidates[0])

    return candidates[0]


def download_file(s3_key: str, dest_dir: Path) -> Path:
    """Download one file from S3 to dest_dir using aws s3 cp."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / s3_key
    s3_uri = S3_BASE + s3_key

    print(f"\n[3/4] Downloading:")
    print(f"      Source : {s3_uri}")
    print(f"      Dest   : {dest_path}")

    cmd = [
        "aws", "s3", "cp",
        "--no-sign-request",
        "--region", REGION,
        s3_uri,
        str(dest_path),
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Download failed.", file=sys.stderr)
        sys.exit(1)

    return dest_path


def report_file(dest_path: Path):
    """Print file size and row count."""
    size_bytes = dest_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    print(f"\n[4/4] Download complete. Verifying...")
    print(f"      File  : {dest_path}")
    print(f"      Size  : {size_mb:.1f} MB ({size_bytes:,} bytes)")

    # Count rows efficiently without loading entire CSV into RAM
    try:
        import subprocess as sp
        row_count_result = sp.run(["wc", "-l", str(dest_path)], capture_output=True, text=True)
        if row_count_result.returncode == 0:
            total_lines = int(row_count_result.stdout.strip().split()[0])
            # Subtract 1 for header row
            data_rows = total_lines - 1
            print(f"      Rows  : {data_rows:,} (header excluded)")
        else:
            # Fallback: partial pandas read just for shape
            import pandas as pd
            df = pd.read_csv(dest_path, nrows=5)
            print(f"      Columns: {len(df.columns)}")
            print(f"      (Full row count: use wc -l for large files)")
    except Exception as e:
        print(f"      (Row count failed: {e})")


def main():
    parser = argparse.ArgumentParser(description="Download Thursday-01-03-2018 CIC-IDS2018 flow CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Destination directory for the downloaded CSV (default: data/raw)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("CIC-IDS2018 THURSDAY FLOW CSV DOWNLOADER")
    print("=" * 70)

    # Step 1: List all files in the S3 prefix
    filenames = list_s3_files(S3_BASE)

    if not filenames:
        print("[ERROR] No files returned from S3 listing. Check AWS CLI installation.", file=sys.stderr)
        sys.exit(1)

    # Step 2: Find the Thursday CSV
    print(f"\n[2/4] Identifying Thursday-01-03-2018 CSV from {len(filenames)} entries...")
    thursday_csv = find_thursday_csv(filenames)
    print(f"      ✓ Matched: {thursday_csv}")

    # Step 3: Download
    dest_path = download_file(thursday_csv, args.output_dir)

    # Step 4: Report
    report_file(dest_path)

    print("\n" + "=" * 70)
    print(f"DONE. Thursday CSV saved to: {dest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
