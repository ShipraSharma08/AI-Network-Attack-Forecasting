"""Unit tests for tamper-evident EvidenceChain."""

import json
import tempfile
import unittest
from pathlib import Path

from src.blockchain.evidence_chain import EvidenceBlock, EvidenceChain, TamperedBlock

SAMPLE_PREDICTION = {
    "window_id": 590,
    "attack_probability": 0.87,
    "mitre_stage": "Reconnaissance",
    "mitre_tactic_id": "TA0043",
    "top_shap_features": [
        {"feature": "syn_count", "contribution": 0.032},
        {"feature": "ack_count", "contribution": 0.019},
    ],
    "model_version": "lstm_flow_only_v1",
}


class TestEvidenceChain(unittest.TestCase):

    def setUp(self):
        self.chain = EvidenceChain()

    # ------------------------------------------------------------------
    # Genesis block
    # ------------------------------------------------------------------

    def test_genesis_block_exists(self):
        self.assertEqual(len(self.chain), 1)
        genesis = self.chain[0]
        self.assertEqual(genesis.index, 0)
        self.assertEqual(genesis.window_id, -1)
        self.assertEqual(genesis.prev_hash, "0" * 64)
        self.assertEqual(genesis.mitre_stage, "GENESIS")
        self.assertEqual(len(genesis.hash), 64)

    def test_genesis_hash_is_valid(self):
        genesis = self.chain[0]
        recomputed = EvidenceChain.compute_hash(genesis.to_dict())
        self.assertEqual(recomputed, genesis.hash)

    # ------------------------------------------------------------------
    # add_block
    # ------------------------------------------------------------------

    def test_add_block_increments_index(self):
        b1 = self.chain.add_block(SAMPLE_PREDICTION)
        b2 = self.chain.add_block({**SAMPLE_PREDICTION, "window_id": 591})
        self.assertEqual(b1.index, 1)
        self.assertEqual(b2.index, 2)
        self.assertEqual(len(self.chain), 3)  # genesis + 2 blocks

    def test_add_block_links_prev_hash(self):
        b1 = self.chain.add_block(SAMPLE_PREDICTION)
        b2 = self.chain.add_block({**SAMPLE_PREDICTION, "window_id": 591})
        # b1.prev_hash == genesis.hash
        self.assertEqual(b1.prev_hash, self.chain[0].hash)
        # b2.prev_hash == b1.hash
        self.assertEqual(b2.prev_hash, b1.hash)

    def test_add_block_hash_is_deterministic(self):
        b = self.chain.add_block(SAMPLE_PREDICTION)
        recomputed = EvidenceChain.compute_hash(b.to_dict())
        self.assertEqual(recomputed, b.hash)

    # ------------------------------------------------------------------
    # compute_hash
    # ------------------------------------------------------------------

    def test_compute_hash_excludes_hash_field(self):
        """Hash of block without 'hash' field should equal hash of block with 'hash' field."""
        b = self.chain.add_block(SAMPLE_PREDICTION)
        d = b.to_dict()
        # With 'hash' key present → should be excluded internally
        h1 = EvidenceChain.compute_hash(d)
        # Without 'hash' key
        d_no_hash = {k: v for k, v in d.items() if k != "hash"}
        h2 = EvidenceChain.compute_hash(d_no_hash)
        self.assertEqual(h1, h2)

    def test_compute_hash_is_64_chars(self):
        b = self.chain.add_block(SAMPLE_PREDICTION)
        h = EvidenceChain.compute_hash(b.to_dict())
        self.assertEqual(len(h), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in h))

    # ------------------------------------------------------------------
    # verify_chain / get_tampered_blocks
    # ------------------------------------------------------------------

    def test_fresh_chain_is_valid(self):
        for i in range(5):
            self.chain.add_block({**SAMPLE_PREDICTION, "window_id": 590 + i})
        self.assertTrue(self.chain.verify_chain())
        self.assertEqual(self.chain.get_tampered_blocks(), [])

    def test_hash_mismatch_detected(self):
        self.chain.add_block(SAMPLE_PREDICTION)
        # Tamper: mutate data without updating hash
        self.chain.chain[1].attack_probability = 0.0
        tampered = self.chain.get_tampered_blocks()
        self.assertFalse(self.chain.verify_chain())
        self.assertEqual(len(tampered), 1)
        self.assertEqual(tampered[0].index, 1)
        self.assertEqual(tampered[0].reason, "hash_mismatch")

    def test_broken_chain_link_detected(self):
        self.chain.add_block(SAMPLE_PREDICTION)
        self.chain.add_block({**SAMPLE_PREDICTION, "window_id": 591})
        # Corrupt block[1].hash directly → block[2].prev_hash will no longer match
        self.chain.chain[1].hash = "a" * 64
        tampered = self.chain.get_tampered_blocks()
        # block[1] has hash_mismatch (stored vs recomputed)
        # block[2] has broken_chain_link (prev_hash != block[1].hash)
        reasons = {t.reason for t in tampered}
        self.assertIn("hash_mismatch", reasons)
        self.assertIn("broken_chain_link", reasons)

    def test_tampered_block_has_correct_index(self):
        for i in range(10):
            self.chain.add_block({**SAMPLE_PREDICTION, "window_id": 590 + i})
        # Tamper block at index 5
        self.chain.chain[5].attack_probability = 0.999
        tampered = self.chain.get_tampered_blocks()
        tampered_indices = [t.index for t in tampered]
        self.assertIn(5, tampered_indices)

    # ------------------------------------------------------------------
    # save / load roundtrip
    # ------------------------------------------------------------------

    def test_save_and_load_roundtrip(self):
        for i in range(5):
            self.chain.add_block({**SAMPLE_PREDICTION, "window_id": 590 + i})
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        self.chain.save(tmp_path)
        loaded = EvidenceChain.load(tmp_path)
        tmp_path.unlink()

        self.assertEqual(len(loaded), len(self.chain))
        self.assertTrue(loaded.verify_chain())
        # Hashes must match exactly
        for orig, restored in zip(self.chain.chain, loaded.chain):
            self.assertEqual(orig.hash, restored.hash)
            self.assertEqual(orig.window_id, restored.window_id)

    def test_load_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            EvidenceChain.load("/tmp/nonexistent_chain_999.json")

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    def test_summary_fields(self):
        for i in range(3):
            self.chain.add_block({**SAMPLE_PREDICTION, "window_id": 590 + i})
        s = self.chain.summary()
        self.assertEqual(s["total_blocks"], 4)  # genesis + 3
        self.assertEqual(s["prediction_blocks"], 3)
        self.assertTrue(s["chain_valid"])
        self.assertEqual(s["tampered_count"], 0)


if __name__ == "__main__":
    unittest.main()
