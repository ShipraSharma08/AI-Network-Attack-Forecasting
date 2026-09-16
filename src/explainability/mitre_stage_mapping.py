#!/usr/bin/env python3
"""
MITRE ATT&CK Stage Mapping Module
=================================
Maps predicted network attack states and probability trajectories to
corresponding MITRE ATT&CK tactics and operational stages based on
elapsed intrusion progression.

Logic:
  - 0-5 min  + prob > 0.6 -> "Reconnaissance" (initial probe, TA0043)
  - 5-15 min + prob > 0.7 -> "Initial Access" (exploit delivery, TA0001)
  - 15-35 min + prob > 0.8 -> "Lateral Movement" (scanning 172.31.69.12/14/24, TA0008)
  - 35+ min  + prob > 0.7 -> "Command & Control" (persistence, TA0011)

Outputs:
  Predicted attack stage with confidence, descriptions, and MITRE tactic identifiers.
"""

import argparse
import json
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union


class MitreStagePrediction(dict):
    """
    Structured representation of a predicted MITRE ATT&CK stage.
    Supports dictionary access, attribute access, JSON serialization,
    and tuple unpacking: (stage, confidence).
    """

    def __init__(
        self,
        stage: str,
        confidence: float,
        description: str,
        tactic_id: Optional[str] = None,
        elapsed_minutes: float = 0.0,
        attack_probability: float = 0.0,
        threshold: float = 0.5,
    ):
        data = {
            "stage": stage,
            "confidence": round(float(confidence), 4),
            "description": description,
            "tactic_id": tactic_id,
            "elapsed_minutes": round(float(elapsed_minutes), 2),
            "attack_probability": round(float(attack_probability), 4),
            "threshold": round(float(threshold), 4),
        }
        super().__init__(data)
        self.__dict__.update(data)

    def __iter__(self) -> Iterator[Any]:
        # Allows: stage, confidence = predict_mitre_stage(...)
        return iter((self["stage"], self["confidence"]))

    def __repr__(self) -> str:
        return (
            f"MitreStagePrediction(stage='{self['stage']}', "
            f"confidence={self['confidence']:.2f}, "
            f"description='{self['description']}', "
            f"tactic_id='{self['tactic_id']}')"
        )


def predict_mitre_stage(
    attack_probability: float,
    elapsed_minutes_since_attack_start: float,
) -> MitreStagePrediction:
    """
    Predict the MITRE ATT&CK phase given the forecast attack probability
    and elapsed intrusion progression time in minutes.

    Rule Matrix:
      - 0 to 5 min   & prob > 0.6 -> Reconnaissance (initial probe)
      - 5 to 15 min  & prob > 0.7 -> Initial Access (exploit delivery)
      - 15 to 35 min & prob > 0.8 -> Lateral Movement (scanning 172.31.69.12/14/24)
      - 35+ min      & prob > 0.7 -> Command & Control (persistence)
      - Otherwise                 -> Benign / Sub-threshold

    Args:
        attack_probability: Predicted probability of attack, in range [0.0, 1.0].
        elapsed_minutes_since_attack_start: Minutes elapsed since attack started (>= 0).

    Returns:
        MitreStagePrediction containing:
            - stage: e.g. "Reconnaissance", "Initial Access", "Lateral Movement", "Command & Control"
            - confidence: Confidence score for the stage
            - description: Detailed tactical context
            - tactic_id: MITRE ATT&CK tactic ID (e.g. TA0043)
            - elapsed_minutes: Input elapsed duration
            - attack_probability: Input probability
            - threshold: Probability threshold used for this time window
    """
    prob = float(attack_probability)
    elapsed = max(0.0, float(elapsed_minutes_since_attack_start))

    if elapsed <= 5.0:
        threshold = 0.6
        if prob > threshold:
            return MitreStagePrediction(
                stage="Reconnaissance",
                confidence=prob,
                description="initial probe",
                tactic_id="TA0043",
                elapsed_minutes=elapsed,
                attack_probability=prob,
                threshold=threshold,
            )
    elif elapsed <= 15.0:
        threshold = 0.7
        if prob > threshold:
            return MitreStagePrediction(
                stage="Initial Access",
                confidence=prob,
                description="exploit delivery",
                tactic_id="TA0001",
                elapsed_minutes=elapsed,
                attack_probability=prob,
                threshold=threshold,
            )
    elif elapsed <= 35.0:
        threshold = 0.8
        if prob > threshold:
            return MitreStagePrediction(
                stage="Lateral Movement",
                confidence=prob,
                description="scanning 172.31.69.12/14/24",
                tactic_id="TA0008",
                elapsed_minutes=elapsed,
                attack_probability=prob,
                threshold=threshold,
            )
    else:  # elapsed > 35.0
        threshold = 0.7
        if prob > threshold:
            return MitreStagePrediction(
                stage="Command & Control",
                confidence=prob,
                description="persistence",
                tactic_id="TA0011",
                elapsed_minutes=elapsed,
                attack_probability=prob,
                threshold=threshold,
            )

    # If probability does not exceed required threshold for the current elapsed time window
    sub_threshold_conf = 1.0 - prob if prob < 0.5 else prob
    return MitreStagePrediction(
        stage="Benign",
        confidence=round(sub_threshold_conf, 4),
        description="attack probability below threshold for active stage detection",
        tactic_id=None,
        elapsed_minutes=elapsed,
        attack_probability=prob,
        threshold=threshold,
    )


def map_trajectory_to_mitre_stages(
    probabilities: List[float],
    start_elapsed_minutes: float = 0.0,
    step_minutes: float = 1.0,
) -> List[MitreStagePrediction]:
    """
    Map an entire forward rollout probability sequence to a trajectory of MITRE stages.
    """
    trajectory = []
    for idx, prob in enumerate(probabilities):
        t = start_elapsed_minutes + (idx * step_minutes)
        pred = predict_mitre_stage(prob, t)
        trajectory.append(pred)
    return trajectory


def parse_args():
    parser = argparse.ArgumentParser(
        description="Map predicted attack state & elapsed duration to MITRE ATT&CK phase."
    )
    parser.add_argument(
        "--prob",
        type=float,
        default=0.75,
        help="Predicted attack probability [0.0 - 1.0] (default: 0.75)",
    )
    parser.add_argument(
        "--elapsed",
        type=float,
        default=10.0,
        help="Elapsed minutes since attack start (default: 10.0)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON format",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run comprehensive demonstration over all MITRE phases",
    )
    return parser.parse_args()


def run_demo():
    print("=" * 80)
    print("MITRE ATT&CK STAGE MAPPING DEMONSTRATION")
    print("=" * 80)

    test_cases = [
        (0.85, 2.0, "0-5 min + prob > 0.6"),
        (0.50, 2.0, "0-5 min + prob <= 0.6 (Sub-threshold)"),
        (0.75, 10.0, "5-15 min + prob > 0.7"),
        (0.65, 10.0, "5-15 min + prob <= 0.7 (Sub-threshold)"),
        (0.88, 25.0, "15-35 min + prob > 0.8"),
        (0.75, 25.0, "15-35 min + prob <= 0.8 (Sub-threshold)"),
        (0.92, 45.0, "35+ min + prob > 0.7"),
        (0.60, 45.0, "35+ min + prob <= 0.7 (Sub-threshold)"),
    ]

    for prob, elapsed, scenario in test_cases:
        res = predict_mitre_stage(prob, elapsed)
        print(f"Scenario: {scenario:<38} -> Elapsed: {elapsed:4.1f}m | Prob: {prob:4.2f}")
        print(f"   => Stage:      {res.stage}")
        print(f"      Confidence: {res.confidence:.2f}")
        print(f"      Tactic ID:  {res.tactic_id}")
        print(f"      Desc:       {res.description}")
        print("-" * 80)


def main():
    args = parse_args()

    if args.demo:
        run_demo()
        return

    result = predict_mitre_stage(
        attack_probability=args.prob,
        elapsed_minutes_since_attack_start=args.elapsed,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 60)
        print("MITRE ATT&CK STAGE PREDICTION")
        print("=" * 60)
        print(f"Input Attack Probability: {args.prob:.4f}")
        print(f"Elapsed Time:             {args.elapsed:.1f} minutes")
        print(f"Predicted Stage:          {result.stage}")
        print(f"Confidence:               {result.confidence:.2f}")
        if result.tactic_id:
            print(f"MITRE Tactic ID:          {result.tactic_id}")
        print(f"Operational Context:      {result.description}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
