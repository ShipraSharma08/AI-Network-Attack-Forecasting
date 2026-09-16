#!/usr/bin/env python3
"""
Tamper-Evident Evidence Chain for Attack Prediction Audit Logging
=================================================================
Implements a cryptographically linked block chain for tamper-evident logging
of network attack predictions from the Temporal LSTM forecasting engine.

Block Structure:
  - index             : int — Block sequence number (0 = genesis)
  - timestamp         : ISO 8601 UTC string
  - window_id         : int — 60-second state window this prediction is for
  - attack_probability: float — Predicted attack risk (0.0–1.0)
  - mitre_stage       : str — Mapped MITRE ATT&CK phase
  - mitre_tactic_id   : str — MITRE tactic identifier (e.g. "TA0043")
  - top_shap_features : list of {feature, contribution} dicts
  - model_version     : str — Model identifier (e.g. "lstm_flow_only_v1")
  - prev_hash         : str — SHA-256 hex digest of the previous block
  - hash              : str — SHA-256 hex digest of this block's content

Chain Guarantees:
  - Deterministic hashing: canonical JSON with sort_keys=True, no whitespace
  - Backward linkage: each block embeds its predecessor's hash
  - Integrity verification: recompute & compare every block hash + chain link
  - Tamper detection: identify which blocks are corrupted and why
"""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "state_transitions_clean.csv"
DEFAULT_CHAIN_PATH = PROJECT_ROOT / "data" / "processed" / "evidence_chain.json"
DEFAULT_MODEL_VERSION = "lstm_flow_only_v1"
GENESIS_PREV_HASH = "0" * 64


# ---------------------------------------------------------------------------
# Block dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvidenceBlock:
    """
    A single immutable evidence record in the tamper-evident chain.
    """
    index: int
    timestamp: str
    window_id: int
    attack_probability: float
    mitre_stage: str
    mitre_tactic_id: Optional[str]
    top_shap_features: List[Dict[str, Any]]
    model_version: str
    prev_hash: str
    hash: str = ""  # Computed and assigned after block is fully built

    def to_dict(self) -> Dict[str, Any]:
        """Serialize block to a plain dict (JSON-safe)."""
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "window_id": self.window_id,
            "attack_probability": self.attack_probability,
            "mitre_stage": self.mitre_stage,
            "mitre_tactic_id": self.mitre_tactic_id,
            "top_shap_features": self.top_shap_features,
            "model_version": self.model_version,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceBlock":
        return cls(**data)


# ---------------------------------------------------------------------------
# Tamper verification result
# ---------------------------------------------------------------------------

@dataclass
class TamperedBlock:
    """Record of a single chain integrity failure."""
    index: int
    window_id: int
    reason: str          # "hash_mismatch" | "broken_chain_link"
    stored_hash: str
    expected_hash: Optional[str] = None
    stored_prev_hash: Optional[str] = None
    actual_prev_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "window_id": self.window_id,
            "reason": self.reason,
            "stored_hash": self.stored_hash,
            "expected_hash": self.expected_hash,
            "stored_prev_hash": self.stored_prev_hash,
            "actual_prev_hash": self.actual_prev_hash,
        }


# ---------------------------------------------------------------------------
# EvidenceChain
# ---------------------------------------------------------------------------

class EvidenceChain:
    """
    Tamper-evident, cryptographically linked chain of attack prediction records.

    Usage:
        chain = EvidenceChain()
        chain.add_block({
            "window_id": 590,
            "attack_probability": 0.87,
            "mitre_stage": "Reconnaissance",
            "mitre_tactic_id": "TA0043",
            "top_shap_features": [{"feature": "syn_count", "contribution": 0.032}],
            "model_version": "lstm_flow_only_v1",
        })
        assert chain.verify_chain()
        chain.save("data/processed/evidence_chain.json")
    """

    def __init__(self, model_version: str = DEFAULT_MODEL_VERSION):
        self.model_version = model_version
        self.chain: List[EvidenceBlock] = []
        self._create_genesis_block()

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    @staticmethod
    def compute_hash(block_data: Dict[str, Any]) -> str:
        """
        Compute the SHA-256 hex digest of a block.

        The input dict must NOT contain the 'hash' field — it should be the
        block content as it was when the hash was originally computed.
        Serialization is deterministic: sort_keys=True, no whitespace separators.

        Args:
            block_data: block dict without the 'hash' key.

        Returns:
            64-character lowercase hex SHA-256 digest.
        """
        # Exclude 'hash' field for deterministic computation
        payload = {k: v for k, v in block_data.items() if k != "hash"}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Genesis
    # ------------------------------------------------------------------

    def _create_genesis_block(self) -> None:
        """Append the genesis block (index=0, prev_hash='000...000')."""
        ts = datetime.now(timezone.utc).isoformat()
        genesis_data: Dict[str, Any] = {
            "index": 0,
            "timestamp": ts,
            "window_id": -1,
            "attack_probability": 0.0,
            "mitre_stage": "GENESIS",
            "mitre_tactic_id": None,
            "top_shap_features": [],
            "model_version": self.model_version,
            "prev_hash": GENESIS_PREV_HASH,
        }
        genesis_hash = self.compute_hash(genesis_data)
        genesis_block = EvidenceBlock(hash=genesis_hash, **genesis_data)
        self.chain.append(genesis_block)

    # ------------------------------------------------------------------
    # Block addition
    # ------------------------------------------------------------------

    def add_block(self, prediction_data: Dict[str, Any]) -> EvidenceBlock:
        """
        Append a new prediction evidence block to the chain.

        Required keys in prediction_data:
            - window_id          : int
            - attack_probability : float
            - mitre_stage        : str
            - mitre_tactic_id    : str | None
            - top_shap_features  : list[dict] with keys {feature, contribution}

        Optional keys:
            - timestamp    : ISO 8601 string (defaults to UTC now)
            - model_version: str (defaults to chain default)

        Returns:
            The newly appended EvidenceBlock.
        """
        prev_block = self.chain[-1]

        block_data: Dict[str, Any] = {
            "index": len(self.chain),
            "timestamp": prediction_data.get(
                "timestamp", datetime.now(timezone.utc).isoformat()
            ),
            "window_id": int(prediction_data["window_id"]),
            "attack_probability": float(prediction_data["attack_probability"]),
            "mitre_stage": str(prediction_data.get("mitre_stage", "Unknown")),
            "mitre_tactic_id": prediction_data.get("mitre_tactic_id"),
            "top_shap_features": list(prediction_data.get("top_shap_features", [])),
            "model_version": prediction_data.get("model_version", self.model_version),
            "prev_hash": prev_block.hash,
        }

        block_hash = self.compute_hash(block_data)
        block = EvidenceBlock(hash=block_hash, **block_data)
        self.chain.append(block)
        return block

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_chain(self) -> bool:
        """
        Walk the entire chain and verify every block's integrity.

        Checks per block:
          (a) Recomputed hash matches stored hash.
          (b) stored prev_hash matches the previous block's stored hash.

        Returns:
            True if every block passes both checks, False otherwise.
        """
        return len(self.get_tampered_blocks()) == 0

    def get_tampered_blocks(self) -> List[TamperedBlock]:
        """
        Identify all blocks that fail integrity verification.

        Returns:
            List of TamperedBlock records — one per corrupted block.
            Each record includes the block index, reason, and diagnostic
            hash values for forensic analysis.
        """
        failures: List[TamperedBlock] = []

        for i, block in enumerate(self.chain):
            block_dict = block.to_dict()
            expected_hash = self.compute_hash(block_dict)  # 'hash' key excluded automatically

            # Check (a): Does recomputed hash match what's stored?
            if expected_hash != block.hash:
                failures.append(
                    TamperedBlock(
                        index=i,
                        window_id=block.window_id,
                        reason="hash_mismatch",
                        stored_hash=block.hash,
                        expected_hash=expected_hash,
                    )
                )

            # Check (b): Does prev_hash match actual previous block's hash?
            if i > 0:
                actual_prev_hash = self.chain[i - 1].hash
                if block.prev_hash != actual_prev_hash:
                    failures.append(
                        TamperedBlock(
                            index=i,
                            window_id=block.window_id,
                            reason="broken_chain_link",
                            stored_hash=block.hash,
                            stored_prev_hash=block.prev_hash,
                            actual_prev_hash=actual_prev_hash,
                        )
                    )

        return failures

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """
        Persist the chain to a JSON file.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        chain_data = {
            "model_version": self.model_version,
            "block_count": len(self.chain),
            "chain": [block.to_dict() for block in self.chain],
        }
        with open(path, "w") as f:
            json.dump(chain_data, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EvidenceChain":
        """
        Restore a chain from a JSON file.

        The restored chain bypasses genesis creation — blocks are loaded directly.

        Args:
            path: Source JSON file path.

        Returns:
            Populated EvidenceChain instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Evidence chain not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        instance = object.__new__(cls)
        instance.model_version = data.get("model_version", DEFAULT_MODEL_VERSION)
        instance.chain = [EvidenceBlock.from_dict(b) for b in data["chain"]]
        return instance

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return number of blocks including genesis."""
        return len(self.chain)

    def __getitem__(self, index: int) -> EvidenceBlock:
        return self.chain[index]

    def summary(self) -> Dict[str, Any]:
        """Return a compact summary of the chain's state."""
        tampered = self.get_tampered_blocks()
        attack_blocks = [b for b in self.chain if b.mitre_stage != "GENESIS" and b.attack_probability >= 0.5]
        return {
            "total_blocks": len(self.chain),
            "prediction_blocks": len(self.chain) - 1,  # Exclude genesis
            "attack_blocks": len(attack_blocks),
            "chain_valid": len(tampered) == 0,
            "tampered_count": len(tampered),
            "tampered_indices": [t.index for t in tampered],
            "first_block_hash": self.chain[0].hash,
            "last_block_hash": self.chain[-1].hash if self.chain else None,
        }


# ---------------------------------------------------------------------------
# CLI Demonstration
# ---------------------------------------------------------------------------

def _build_chain_from_csv(
    data_path: Path,
    chain: EvidenceChain,
    elapsed_start_minutes: float = 0.0,
    verbose: bool = False,
) -> EvidenceChain:
    """Build chain entries from the 75 known attack-window rows (590–664)."""
    import pandas as pd
    from src.explainability.mitre_stage_mapping import predict_mitre_stage

    df = pd.read_csv(data_path).sort_values("window").reset_index(drop=True)
    target = df[(df["window"] >= 590) & (df["window"] <= 664)].reset_index(drop=True)

    print(f"  Building evidence blocks for {len(target)} prediction windows (590–664)...")

    # Placeholder SHAP features derived from top-correlated flow columns
    PLACEHOLDER_SHAP = [
        {"feature": "syn_count",           "contribution": 0.0321},
        {"feature": "packet_length_mean",  "contribution": 0.0195},
        {"feature": "ack_count",           "contribution": 0.0183},
        {"feature": "fwd_bytes_sum",       "contribution": 0.0144},
        {"feature": "packets_per_second",  "contribution": 0.0098},
    ]

    for i, row in target.iterrows():
        window_id = int(row["window"])
        elapsed = elapsed_start_minutes + float(i)

        # Use attack_label as a proxy for probability (in a real pipeline this
        # comes from forward_rollout); attenuate slightly so it stays in [0, 1]
        attack_prob = 0.92 if int(row.get("next_attack_label", row.get("current_attack_label", 1))) == 1 else 0.31
        # Perturb probability slightly per window to make blocks unique
        attack_prob = min(0.99, max(0.01, attack_prob + (window_id % 7 - 3) * 0.01))

        mitre_pred = predict_mitre_stage(attack_prob, elapsed)

        chain.add_block({
            "window_id": window_id,
            "attack_probability": round(attack_prob, 4),
            "mitre_stage": mitre_pred.stage,
            "mitre_tactic_id": mitre_pred.tactic_id,
            "top_shap_features": PLACEHOLDER_SHAP,
            "model_version": DEFAULT_MODEL_VERSION,
        })

        if verbose and (i + 1) % 10 == 0:
            print(f"    → Block {i + 1}/75: window={window_id}, "
                  f"prob={attack_prob:.2f}, stage={mitre_pred.stage}")

    return chain


def main():
    parser = argparse.ArgumentParser(
        description="Tamper-Evident Evidence Chain for Attack Prediction Audit Logging"
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--chain-path", type=Path, default=DEFAULT_CHAIN_PATH)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    print("\n" + "=" * 75)
    print("TAMPER-EVIDENT ATTACK PREDICTION EVIDENCE CHAIN")
    print("=" * 75)

    # ------------------------------------------------------------------ #
    # STEP 1: Build chain from 75 attack-window predictions               #
    # ------------------------------------------------------------------ #
    print("\n[1/4] Building EvidenceChain from 75 attack prediction windows...")
    chain = EvidenceChain(model_version=DEFAULT_MODEL_VERSION)
    _build_chain_from_csv(args.data_path, chain, elapsed_start_minutes=5.0, verbose=args.verbose)

    print(f"  ✓ Chain built: {len(chain)} blocks total (1 genesis + 75 prediction blocks)")

    # ------------------------------------------------------------------ #
    # STEP 2: Save to evidence_chain.json                                 #
    # ------------------------------------------------------------------ #
    print(f"\n[2/4] Saving chain to {args.chain_path} ...")
    chain.save(args.chain_path)
    saved_size_kb = args.chain_path.stat().st_size / 1024
    print(f"  ✓ Saved: {saved_size_kb:.1f} KB")

    # ------------------------------------------------------------------ #
    # STEP 3: Verify the original chain                                   #
    # ------------------------------------------------------------------ #
    print("\n[3/4] Verifying chain integrity...")
    chain_loaded = EvidenceChain.load(args.chain_path)
    is_valid = chain_loaded.verify_chain()
    tampered = chain_loaded.get_tampered_blocks()
    summary = chain_loaded.summary()

    status_icon = "✅" if is_valid else "❌"
    print(f"  {status_icon} Chain integrity: {'VALID' if is_valid else 'INVALID'} "
          f"({summary['prediction_blocks']} blocks, {summary['tampered_count']} tampered)")
    if tampered:
        for t in tampered:
            print(f"       └─ Block {t.index} (window {t.window_id}): {t.reason}")

    # ------------------------------------------------------------------ #
    # STEP 4: Tamper demo — mutate block[10], re-verify                   #
    # ------------------------------------------------------------------ #
    print("\n[4/4] TAMPER DEMONSTRATION: Mutating block[10].attack_probability without rehashing...")
    original_prob = chain_loaded.chain[10].attack_probability
    original_hash = chain_loaded.chain[10].hash

    # Direct mutation — deliberately bypass hash recomputation
    chain_loaded.chain[10].attack_probability = 0.0  # forge benign!

    print(f"  Original  block[10] attack_probability: {original_prob:.4f}")
    print(f"  Forged    block[10] attack_probability: 0.0000")
    print(f"  Stored hash (unchanged):               {original_hash[:24]}...")

    # Re-verify — should detect tampering
    tampered_after = chain_loaded.get_tampered_blocks()
    is_valid_after = len(tampered_after) == 0
    tampered_indices = [t.index for t in tampered_after]
    tampered_reasons  = {t.index: t.reason for t in tampered_after}

    print(f"\n  Chain integrity after tamper:")
    print(f"  {'✅' if is_valid_after else '❌'} VALID: {is_valid_after} | "
          f"Tampered blocks detected: {len(tampered_after)}")
    for t in tampered_after:
        print(f"       └─ Block {t.index} (window {chain_loaded.chain[t.index].window_id}): "
              f"{t.reason}")
        if t.expected_hash:
            print(f"             Stored hash  : {t.stored_hash[:24]}...")
            print(f"             Expected hash: {t.expected_hash[:24]}...")

    # Print final summary
    print("\n" + "─" * 75)
    print("SUMMARY")
    print("─" * 75)
    print(f"  Blocks logged : {summary['prediction_blocks']}")
    print(f"  Attack blocks : {summary['attack_blocks']}")
    print(f"  Genesis hash  : {chain_loaded.chain[0].hash[:32]}...")
    print(f"  Last block    : {chain_loaded.chain[-1].hash[:32]}...")
    print(f"  Chain saved   : {args.chain_path}")
    print("─" * 75 + "\n")


if __name__ == "__main__":
    main()
