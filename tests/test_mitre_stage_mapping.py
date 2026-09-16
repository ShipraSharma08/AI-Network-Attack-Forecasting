"""Unit tests for MITRE ATT&CK stage mapping."""

import unittest
from src.explainability.mitre_stage_mapping import (
    MitreStagePrediction,
    predict_mitre_stage,
    map_trajectory_to_mitre_stages,
)


class TestMitreStageMapping(unittest.TestCase):
    def test_reconnaissance_stage(self):
        res = predict_mitre_stage(attack_probability=0.75, elapsed_minutes_since_attack_start=3.0)
        self.assertEqual(res.stage, "Reconnaissance")
        self.assertEqual(res.confidence, 0.75)
        self.assertIn("probe", res.description)
        self.assertEqual(res.tactic_id, "TA0043")

        # Sub-threshold
        sub = predict_mitre_stage(attack_probability=0.55, elapsed_minutes_since_attack_start=3.0)
        self.assertEqual(sub.stage, "Benign")

    def test_initial_access_stage(self):
        res = predict_mitre_stage(attack_probability=0.82, elapsed_minutes_since_attack_start=10.0)
        self.assertEqual(res.stage, "Initial Access")
        self.assertEqual(res.confidence, 0.82)
        self.assertIn("exploit", res.description)
        self.assertEqual(res.tactic_id, "TA0001")

        # Sub-threshold
        sub = predict_mitre_stage(attack_probability=0.68, elapsed_minutes_since_attack_start=10.0)
        self.assertEqual(sub.stage, "Benign")

    def test_lateral_movement_stage(self):
        res = predict_mitre_stage(attack_probability=0.85, elapsed_minutes_since_attack_start=25.0)
        self.assertEqual(res.stage, "Lateral Movement")
        self.assertEqual(res.confidence, 0.85)
        self.assertIn("172.31.69.12/14/24", res.description)
        self.assertEqual(res.tactic_id, "TA0008")

        # Sub-threshold
        sub = predict_mitre_stage(attack_probability=0.78, elapsed_minutes_since_attack_start=25.0)
        self.assertEqual(sub.stage, "Benign")

    def test_command_and_control_stage(self):
        res = predict_mitre_stage(attack_probability=0.72, elapsed_minutes_since_attack_start=40.0)
        self.assertEqual(res.stage, "Command & Control")
        self.assertEqual(res.confidence, 0.72)
        self.assertIn("persistence", res.description)
        self.assertEqual(res.tactic_id, "TA0011")

        # Sub-threshold
        sub = predict_mitre_stage(attack_probability=0.65, elapsed_minutes_since_attack_start=40.0)
        self.assertEqual(sub.stage, "Benign")

    def test_unpacking_and_dict_compatibility(self):
        res = predict_mitre_stage(attack_probability=0.9, elapsed_minutes_since_attack_start=2.0)
        stage, conf = res
        self.assertEqual(stage, "Reconnaissance")
        self.assertEqual(conf, 0.9)
        self.assertEqual(res["stage"], "Reconnaissance")
        self.assertEqual(res.stage, "Reconnaissance")

    def test_trajectory_mapping(self):
        probs = [0.65, 0.75, 0.85, 0.75]
        trajectory = map_trajectory_to_mitre_stages(
            probabilities=probs,
            start_elapsed_minutes=2.0,
            step_minutes=10.0,
        )
        self.assertEqual(len(trajectory), 4)
        self.assertEqual(trajectory[0].stage, "Reconnaissance")
        self.assertEqual(trajectory[1].stage, "Initial Access")
        self.assertEqual(trajectory[2].stage, "Lateral Movement")
        self.assertEqual(trajectory[3].stage, "Benign")


if __name__ == "__main__":
    unittest.main()
