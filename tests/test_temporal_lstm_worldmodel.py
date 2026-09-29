"""Unit tests for TemporalLSTMWorldModel dual-head architecture."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models.temporal_lstm_worldmodel import (
    TemporalLSTMWorldModel,
    save_model_weights_h5,
)


class TestTemporalLSTMWorldModel(unittest.TestCase):
    def setUp(self):
        self.input_dim = 29
        self.seq_len = 10
        self.batch_size = 4
        self.model = TemporalLSTMWorldModel(
            input_dim=self.input_dim,
            hidden_dim=32,
            dense_dim=16,
            dropout_rate=0.2,
        )

    def test_forward_output_shapes(self):
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        state_pred, attack_pred = self.model(x)
        self.assertEqual(state_pred.shape, (self.batch_size, self.input_dim))
        self.assertEqual(attack_pred.shape, (self.batch_size, 1))

    def test_attack_probabilities_bounded(self):
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        _, attack_pred = self.model(x)
        probs = attack_pred.detach().numpy()
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))

    def test_predict_interface_numpy_and_tensor(self):
        x_np = np.random.randn(self.batch_size, self.seq_len, self.input_dim).astype(np.float32)
        s_pred, a_pred = self.model.predict(x_np, return_state=True)
        self.assertIsInstance(s_pred, np.ndarray)
        self.assertIsInstance(a_pred, np.ndarray)
        self.assertEqual(s_pred.shape, (self.batch_size, self.input_dim))
        self.assertEqual(a_pred.shape, (self.batch_size, 1))

        a_only = self.model.predict(x_np, return_state=False)
        self.assertIsInstance(a_only, np.ndarray)
        self.assertEqual(a_only.shape, (self.batch_size, 1))

    def test_joint_loss_gradient_flow(self):
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        s_target = torch.randn(self.batch_size, self.input_dim)
        y_target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        self.model.train()
        s_pred, a_pred = self.model(x)
        mse_loss = nn.functional.mse_loss(s_pred, s_target)
        bce_loss = nn.functional.binary_cross_entropy(a_pred, y_target)
        total_loss = mse_loss + 3.0 * bce_loss
        total_loss.backward()
        self.assertIsNotNone(self.model.lstm.weight_ih_l0.grad)
        self.assertIsNotNone(self.model.dense1.weight.grad)
        self.assertIsNotNone(self.model.state_head.weight.grad)
        self.assertIsNotNone(self.model.clf_head.weight.grad)

    def test_weights_saving_h5_and_pt(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_model.h5"
            save_model_weights_h5(self.model, save_path)
            self.assertTrue(save_path.exists())
            self.assertTrue(save_path.with_suffix(".pt").exists())

            loaded_model = TemporalLSTMWorldModel(
                input_dim=self.input_dim,
                hidden_dim=32,
                dense_dim=16,
                dropout_rate=0.2,
            )
            loaded_model.load_state_dict(torch.load(save_path.with_suffix(".pt"), weights_only=True))
            self.model.eval()
            loaded_model.eval()

            x = torch.randn(2, self.seq_len, self.input_dim)
            with torch.no_grad():
                orig_s, orig_a = self.model(x)
                load_s, load_a = loaded_model(x)
            np.testing.assert_allclose(orig_s.numpy(), load_s.numpy(), rtol=1e-5)
            np.testing.assert_allclose(orig_a.numpy(), load_a.numpy(), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
