#!/usr/bin/env python3
"""
Automated PCAP Discovery, Setup, and Validation Script
======================================================
This script searches local directories for CIC-IDS2018 Wednesday Infiltration
PCAP files, links/copies the discovered file to the target location, and executes
strict timestamp window validation against the ground-truth attack period.
"""

import argparse
import fnmatch
import os
import shutil
import sys
from pathlib import Path

# Ensure project root is in sys.path so src imports work reliably
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.features.packet_features import validate_pcap_window
except ImportError as err:
    print(f"[ERROR] Unable to import validate_pcap_window from src.features.packet_features: {err}")
    sys.exit(1)


TARGET_PCAP_DIR = PROJECT_ROOT / "data" / "raw" / "infiltration.pcap"
TARGET_PCAP_FILE = TARGET_PCAP_DIR / "infiltration.pcap"

DEFAULT_SEARCH_LOCATIONS = [
    PROJECT_ROOT / "data" / "raw",
    Path.home() / "datasets",
    Path("/tmp"),
    Path.home() / "Downloads",
]

PCAP_PATTERNS = [
    "*infiltration*.pcap",
    "*Wednesday*.pcap",
    "*28-02*.pcap",
]

ATTACK_START = "2018-02-28 10:50:00"
ATTACK_END = "2018-02-28 12:05:00"


def format_bytes(size_bytes: int) -> str:
    """Format bytes to human readable string (KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KB ({size_bytes:,} bytes)"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.2f} MB ({size_bytes:,} bytes)"
    else:
        return f"{size_bytes / (1024**3):.2f} GB ({size_bytes:,} bytes)"


def search_local_pcaps(search_dirs=None, patterns=None):
    """
    Search designated directories for PCAP files matching specific filename patterns.
    """
    if search_dirs is None:
        search_dirs = DEFAULT_SEARCH_LOCATIONS
    if patterns is None:
        patterns = PCAP_PATTERNS

    found_files = []
    seen_paths = set()

    for directory in search_dirs:
        dir_path = Path(directory).expanduser().resolve()
        if not dir_path.exists() or not dir_path.is_dir():
            continue

        try:
            for root, _, files in os.walk(dir_path, followlinks=True):
                for filename in files:
                    file_path = Path(root) / filename
                    file_lower = filename.lower()

                    for pattern in patterns:
                        if fnmatch.fnmatch(file_lower, pattern.lower()):
                            resolved = file_path.resolve()
                            if resolved not in seen_paths and file_path.is_file():
                                seen_paths.add(resolved)
                                try:
                                    size = file_path.stat().st_size
                                except OSError:
                                    size = 0
                                found_files.append((file_path, size))
                            break
        except PermissionError:
            continue

    return found_files


def print_missing_diagnostics():
    """Print diagnostic guidance when no valid PCAP file is found."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC NOTICE: NO INFILTRATION PCAP FILE FOUND LOCALLY")
    print("=" * 70)
    print(f"Target file location : {TARGET_PCAP_FILE}")
    print(f"Expected timestamp   : {ATTACK_START} to {ATTACK_END} (UTC)")
    print(f"Expected file size   : ~500 MB - 2 GB")
    print("-" * 70)
    print("INSTRUCTIONS TO OBTAIN THE PCAP FILE:")
    print("  a) Ask your project teammate to share the file via Git LFS / push")
    print("     or provide a direct cloud drive download link (Google Drive / S3).")
    print("  b) Download from the official CSE-CIC-IDS2018 dataset repository:")
    print("     - Dataset: Wednesday-28-02-2018 (Infiltration attack)")
    print("     - Once obtained, place the file at:")
    print(f"       {TARGET_PCAP_FILE}")
    print("       or pass the path directly: python scripts/setup_pcap.py --pcap /path/to/file.pcap")
    print("-" * 70)
    print("STATUS:")
    print("  PCAP validation PENDING. Flow pipeline ready. Packet fusion deferred until PCAP obtained.")
    print("=" * 70 + "\n")


def setup_and_validate(pcap_source: Path, use_copy: bool = False):
    """
    Link or copy the found PCAP file to the target location, then validate.
    """
    pcap_source = Path(pcap_source).resolve()
    if not pcap_source.exists() or not pcap_source.is_file():
        print(f"[ERROR] Source PCAP file does not exist: {pcap_source}")
        return False

    TARGET_PCAP_DIR.mkdir(parents=True, exist_ok=True)

    # If source is already at target path, skip copying
    if pcap_source.resolve() == TARGET_PCAP_FILE.resolve():
        print(f"[SETUP] PCAP file is already in target location: {TARGET_PCAP_FILE}")
    else:
        # Remove existing target if it is a broken symlink or stale file
        if TARGET_PCAP_FILE.is_symlink() or TARGET_PCAP_FILE.exists():
            try:
                TARGET_PCAP_FILE.unlink()
            except OSError as e:
                print(f"[WARN] Could not remove existing target {TARGET_PCAP_FILE}: {e}")

        if use_copy:
            print(f"[SETUP] Copying PCAP from:\n  {pcap_source}\n  -> {TARGET_PCAP_FILE} ...")
            shutil.copy2(pcap_source, TARGET_PCAP_FILE)
        else:
            print(f"[SETUP] Symlinking PCAP from:\n  {pcap_source}\n  -> {TARGET_PCAP_FILE} ...")
            try:
                TARGET_PCAP_FILE.symlink_to(pcap_source)
            except OSError as err:
                print(f"[WARN] Symlink failed ({err}). Falling back to copying file...")
                shutil.copy2(pcap_source, TARGET_PCAP_FILE)

    print(f"[SETUP] PCAP configured at: {TARGET_PCAP_FILE}")
    print("\n[VALIDATION] Invoking validate_pcap_window()...")

    try:
        result = validate_pcap_window(
            pcap_path=str(TARGET_PCAP_FILE),
            attack_start=ATTACK_START,
            attack_end=ATTACK_END,
            raise_on_error=True,
        )
        print("\n" + "=" * 70)
        print("PCAP VALIDATION SUCCEEDED")
        print("=" * 70)
        print(f"Status           : VALID (True)")
        print(f"Packets Scanned  : {result.get('packet_count', 0):,}")
        print(f"First Packet UTC : {result.get('first_packet_time')}")
        print(f"Last Packet UTC  : {result.get('last_packet_time')}")
        print(f"Details          : {result.get('message')}")
        print("=" * 70 + "\n")
        return True
    except AssertionError as ae:
        print("\n" + "=" * 70)
        print("PCAP VALIDATION FAILED (Out-of-Window / Mismatch)")
        print("=" * 70)
        print(f"Reason: {ae}")
        print("=" * 70 + "\n")
        return False
    except Exception as exc:
        print("\n" + "=" * 70)
        print(f"PCAP VALIDATION ERROR: {exc}")
        print("=" * 70 + "\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Automated PCAP Discovery, Setup, and Validation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  # 1. Automatic local discovery across standard directories:
  python scripts/setup_pcap.py

  # 2. Specify an exact PCAP file explicitly:
  python scripts/setup_pcap.py --pcap /path/to/wednesday_infiltration.pcap

  # 3. Add custom search directories:
  python scripts/setup_pcap.py --search-dir /media/external/datasets

  # 4. Copy file instead of creating a symlink:
  python scripts/setup_pcap.py --pcap ~/datasets/infiltration.pcap --copy
        """,
    )
    parser.add_argument(
        "--pcap",
        type=str,
        default=None,
        help="Direct path to an infiltration PCAP file to link and validate.",
    )
    parser.add_argument(
        "--search-dir",
        action="append",
        default=[],
        help="Additional directory to search for PCAP files.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy the PCAP file instead of symlinking (useful if transferring across volumes).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("AUTOMATED PCAP DISCOVERY & VALIDATION PIPELINE")
    print("=" * 70)

    # Case A: User explicitly passed a PCAP path
    if args.pcap:
        candidate_path = Path(args.pcap).expanduser().resolve()
        print(f"[DISCOVERY] Manually specified PCAP: {candidate_path}")
        if not candidate_path.exists():
            print(f"[ERROR] Specified PCAP file does not exist: {candidate_path}")
            sys.exit(1)
        valid = setup_and_validate(candidate_path, use_copy=args.copy)
        sys.exit(0 if valid else 1)

    # Case B: Search local directories
    search_locations = DEFAULT_SEARCH_LOCATIONS + [Path(d) for d in args.search_dir]
    print("[DISCOVERY] Scanning common local directories for Infiltration PCAP files:")
    for loc in search_locations:
        print(f"  - {loc}")

    print(f"[DISCOVERY] Matching patterns: {', '.join(PCAP_PATTERNS)}")
    found = search_local_pcaps(search_dirs=search_locations)

    if not found:
        print("\n[DISCOVERY] No matching PCAP files found in scanned directories.")
        print_missing_diagnostics()
        sys.exit(1)

    print(f"\n[DISCOVERY] Found {len(found)} candidate PCAP file(s):")
    for idx, (path, size) in enumerate(found, 1):
        print(f"  [{idx}] {path} ({format_bytes(size)})")

    # Select the first candidate (or largest if multiple)
    # Prefer files that actually reside at target location or have 'infiltration'
    found.sort(key=lambda x: (x[0] == TARGET_PCAP_FILE, "infiltration" in x[0].name.lower(), x[1]), reverse=True)
    selected_pcap, selected_size = found[0]

    print(f"\n[DISCOVERY] Selected candidate for setup: {selected_pcap} ({format_bytes(selected_size)})")
    valid = setup_and_validate(selected_pcap, use_copy=args.copy)
    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
