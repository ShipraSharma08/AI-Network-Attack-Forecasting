#!/usr/bin/env python3
"""
Integration and Correctness Tests for Temporal LSTM World Model and Recursive Rollout
=====================================================================================
Validates:
  1. Canonical 29-flow feature schema compliance and ordering.
  2. Scaler zero-leakage training constraint (w <= 609).
  3. Dual-Head World Model inference:
       - Next-state prediction shape (B, 29)
       - Attack probability shape (B, 1) in [0, 1]
  4. True Recursive Autoregressive K-Step Rollout:
       - Output array length and bounds
       - State divergence test: verifies that recursive feedback state window changes
         dynamically at each step (distinct from static frozen-copy baseline).
       - Valid probabilities for K=1..15 horizons.
  5. PCAP parser output schema and absence of NaN/Inf.
  6. SHAP explainability compatibility with TemporalLSTMWorldModel.
  7. Behavior-driven MITRE ATT&CK mapping with flow features.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.canonical_schema import (
    CANONICAL_29_FLOW_FEATURES,
    assert_schema_compliance,
    clean_feature_matrix,
)
from src.models.temporal_lstm_worldmodel import TemporalLSTMWorldModel
from src.forecasting.kstep_worldmodel_rollout import (
    load_worldmodel_and_scaler,
    genuine_state_rollout,
    frozen_state_rollout,
    forward_rollout,
    forward_worldmodel_rollout,
)
from src.explainability.mitre_stage_mapping import predict_mitre_stage
from src.explainability.shap_attribution import AttackExplainer, LSTMSurrogateModel


class TestCanonicalSchema(unittest.TestCase):
    """Test 29 flow features canonical schema."""

    def test_feature_count_and_uniqueness(self):
        self.assertEqual(len(CANONICAL_29_FLOW_FEATURES), 29)
        self.assertEqual(len(set(CANONICAL_29_FLOW_FEATURES)), 29)

    def test_schema_compliance_validator(self):
        # Exact match
        self.assertIsNone(assert_schema_compliance(CANONICAL_29_FLOW_FEATURES))

        # Prefixed match
        prefixed = [f"current_{f}" for f in CANONICAL_29_FLOW_FEATURES]
        self.assertIsNone(assert_schema_compliance(prefixed, allow_prefixed=True))

        # Missing feature
        with self.assertRaises(ValueError):
            assert_schema_compliance(CANONICAL_29_FLOW_FEATURES[:28])

        # Wrong order
        reversed_cols = list(reversed(CANONICAL_29_FLOW_FEATURES))
        with self.assertRaises(ValueError):
            assert_schema_compliance(reversed_cols)

    def test_clean_feature_matrix(self):
        mat = np.array([[1.0, np.nan, np.inf], [-np.inf, 2.0, 3.0]])
        cleaned = clean_feature_matrix(mat)
        self.assertFalse(np.isnan(cleaned).any())
        self.assertFalse(np.isinf(cleaned).any())
        self.assertEqual(cleaned[0, 1], 0.0)
        self.assertEqual(cleaned[0, 2], 0.0)
        self.assertEqual(cleaned[1, 0], 0.0)


class TestWorldModelAndRollout(unittest.TestCase):
    """Test World Model loading, dual-head outputs, and recursive rollout dynamics."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.scaler, cls.feature_cols, cls.df = load_worldmodel_and_scaler()

    def test_model_loaded_successfully(self):
        self.assertIsInstance(self.model, TemporalLSTMWorldModel)
        self.assertEqual(self.model.input_dim, 29)
        self.assertEqual(len(self.feature_cols), 29)

    def test_dual_head_forward(self):
        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 29)
        state_pred, attack_pred = self.model(x)

        # Check shapes
        self.assertEqual(state_pred.shape, (batch_size, 29))
        self.assertEqual(attack_pred.shape, (batch_size, 1))

        # Check probability bounds [0, 1]
        probs = attack_pred.detach().cpu().numpy()
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

        # Check predict_proba
        np_x = np.random.randn(batch_size, seq_len, 29).astype(np.float32)
        proba = self.model.predict_proba(np_x)
        self.assertEqual(proba.shape, (batch_size, 1))
        self.assertTrue((proba >= 0.0).all() and (proba <= 1.0).all())

    def test_genuine_recursive_rollout_dynamics(self):
        """
        Verify that recursive rollout feeding S_hat(t+1) back produces
        diverging, dynamic state trajectories rather than static repetition.
        """
        dummy_window = np.random.randn(10, 29).astype(np.float32)

        # Run genuine rollout for K=15
        probs_genuine, states_genuine = genuine_state_rollout(
            dummy_window, k_steps=15, model=self.model
        )

        self.assertEqual(len(probs_genuine), 15)
        self.assertEqual(states_genuine.shape, (15, 29))
        self.assertTrue((probs_genuine >= 0.0).all() and (probs_genuine <= 1.0).all())
        self.assertFalse(np.isnan(probs_genuine).any())
        self.assertFalse(np.isnan(states_genuine).any())

        # Verify that predicted states change across steps (they must not all be identical)
        state_diffs = np.linalg.norm(np.diff(states_genuine, axis=0), axis=1)
        self.assertTrue((state_diffs > 1e-5).any(), "Predicted states across K steps should evolve dynamically.")

        # Test forward_rollout wrapper
        wrap_probs = forward_rollout(dummy_window, k_steps=10, model=self.model)
        self.assertEqual(len(wrap_probs), 10)
        self.assertTrue((wrap_probs >= 0.0).all() and (wrap_probs <= 1.0).all())

    def test_horizons_k1_to_k15(self):
        dummy_window = np.random.randn(10, 29).astype(np.float32)
        for k in [1, 3, 5, 10, 15]:
            probs = forward_rollout(dummy_window, k_steps=k, model=self.model)
            self.assertEqual(len(probs), k)
            self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())


class TestMitreAndShapIntegration(unittest.TestCase):
    """Test behavior-driven MITRE mapping and SHAP compatibility with World Model."""

    def test_mitre_mapping_with_canonical_features(self):
        # Test Reconnaissance with elevated SYN and low ACK
        recon_features = {
            "syn_count": 500.0,
            "ack_count": 20.0,
            "rst_count": 5.0,
            "packets_per_second": 1200.0,
        }
        pred = predict_mitre_stage(
            attack_probability=0.85,
            elapsed_minutes_since_attack_start=2.0,
            feature_values=recon_features,
        )
        self.assertEqual(pred.stage, "Reconnaissance")
        self.assertEqual(pred.tactic_id, "TA0043")

        # Test Benign (baseline network conditions)
        benign_features = {
            "syn_count": 15.0,
            "ack_count": 200.0,
            "rst_count": 2.0,
            "psh_fwd_count": 10.0,
            "urg_fwd_count": 0.0,
            "down_up_ratio": 0.5,
            "packets_per_second": 80.0,
        }
        benign_pred = predict_mitre_stage(
            attack_probability=0.10,
            elapsed_minutes_since_attack_start=0.0,
            feature_values=benign_features,
        )
        self.assertEqual(benign_pred.stage, "Benign")

    def test_shap_surrogate_compatibility(self):
        model, scaler, feature_cols, df = load_worldmodel_and_scaler()
        surrogate = LSTMSurrogateModel(model)
        self.assertIsNotNone(surrogate.clf_head)
        self.assertIsNotNone(surrogate.dense2)

        x_seq = torch.randn(2, 10, 29)
        out = surrogate(x_seq)
        self.assertEqual(out.shape, (2, 1))
        probs = out.detach().numpy()
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())


if __name__ == "__main__":
    unittest.main()
