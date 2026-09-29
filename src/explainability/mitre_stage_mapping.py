#!/usr/bin/env python3
"""
MITRE ATT&CK Stage Mapping — Behavior-Driven Classifier
=========================================================
Replaces the probability+time heuristic with a flow-feature behavioral
classifier while preserving the original heuristic as a fallback.

Behavioral rules (using the 29 flow features available from the flow-only model):

  Reconnaissance (TA0043):
    syn_count elevated vs. dataset baseline, low ack_count relative to
    syn_count (incomplete handshakes = scanning signature).

  Initial Access (TA0001):
    psh_fwd_count or urg_fwd_count spike (payload delivery), asymmetric
    down_up_ratio.

  Lateral Movement (TA0008):
    Sustained elevated syn_count over multiple consecutive windows (not just
    one spike), rst_count elevated (failed connection attempts).

  Command & Control (TA0011):
    Sustained ack_count with LOW variance in flow_iat_mean (regular
    polling/beaconing pattern), stable low packets_per_second.

When no behavioral threshold is crossed, the original probability+time
heuristic is used as a fallback (avoids "Benign" false negatives on
borderline cases).
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


# ---------------------------------------------------------------------------
# Behavioral thresholds calibrated from data/processed/state_transitions_clean.csv
#
# Baselines (benign population):
#   syn_count:     mean=50,  p90=107, p95=127
#   ack_count:     mean=184, p50=199
#   rst_count:     mean=176, p90=364, p95=457
#   psh_fwd_count: mean=50,  p90=107, p95=127
#   urg_fwd_count: mean=1.3, p90=0,   p95=8
#   down_up_ratio: mean=0.50
#   flow_iat_std:  mean=979k, p90=1.6M
#   packets_per_s: mean=1334
# ---------------------------------------------------------------------------

# Reconnaissance: elevated SYN with low ACK/SYN ratio (incomplete handshakes)
_RECON_SYN_THRESHOLD = 100.0          # above benign p90 (~107)
_RECON_SYN_ACK_RATIO_CEIL = 0.35      # benign mean ~0.21, attack ~0.27

# Initial Access: PSH/URG spike + asymmetric traffic ratio
_IA_PSH_FWD_THRESHOLD = 100.0         # above benign p90
_IA_URG_FWD_THRESHOLD = 8.0           # above benign p95 (most benign = 0)
_IA_DOWN_UP_RATIO_THRESHOLD = 0.60    # benign mean ~0.50, attack mean ~0.65

# Lateral Movement: sustained SYN + elevated RST (requires history)
_LM_SYN_SUSTAINED_THRESHOLD = 80.0    # must be elevated for >=2 consecutive windows
_LM_RST_THRESHOLD = 300.0             # attack median ~309 vs benign median ~165

# Command & Control: sustained ACK + low IAT variance (beaconing)
_C2_ACK_THRESHOLD = 300.0             # attack mean ~361 vs benign mean ~184
_C2_IAT_STD_LOW_THRESHOLD = 1_200_000.0  # low variance = regular polling
_C2_FLOW_COUNT_THRESHOLD = 800.0      # elevated flow count (attack mean ~1686)


def _get_feat(features, name, default=0.0):
    """Retrieve a feature, stripping 'current_' prefix if needed."""
    if name in features:
        return float(features[name])
    prefixed = "current_" + name
    if prefixed in features:
        return float(features[prefixed])
    return default


def _classify_behavioral(features, attack_probability, elapsed_minutes, prior_windows=None):
    """
    Attempt behavior-driven MITRE stage classification from flow features.

    Returns a MitreStagePrediction if a behavioral signature matches,
    or None if no behavioral threshold is crossed (caller should fall
    back to the probability+time heuristic).
    """
    prob = float(attack_probability)
    elapsed = max(0.0, float(elapsed_minutes))

    syn = _get_feat(features, "syn_count")
    ack = _get_feat(features, "ack_count")
    rst = _get_feat(features, "rst_count")
    psh_fwd = _get_feat(features, "psh_fwd_count")
    urg_fwd = _get_feat(features, "urg_fwd_count")
    down_up = _get_feat(features, "down_up_ratio")
    iat_std = _get_feat(features, "flow_iat_std")
    flow_count = _get_feat(features, "flow_count")

    syn_ack_ratio = syn / (ack + 1.0)
    prior = prior_windows or []

    # ------------------------------------------------------------------
    # Rule 1: Command & Control (TA0011)
    #   Sustained high ACK with LOW IAT variance = regular beaconing.
    #   Checked first because C2 is the most operationally critical stage
    #   and typically appears later in an attack chain.
    # ------------------------------------------------------------------
    if ack > _C2_ACK_THRESHOLD and iat_std < _C2_IAT_STD_LOW_THRESHOLD:
        sustained_c2 = sum(
            1
            for pw in prior[-3:]
            if _get_feat(pw, "ack_count") > _C2_ACK_THRESHOLD * 0.8
            and _get_feat(pw, "flow_iat_std") < _C2_IAT_STD_LOW_THRESHOLD * 1.2
        )
        if sustained_c2 >= 2 or (len(prior) == 0 and flow_count > _C2_FLOW_COUNT_THRESHOLD):
            confidence = min(0.95, 0.70 + 0.10 * sustained_c2 + 0.05 * (prob / 1.0))
            return MitreStagePrediction(
                stage="Command & Control",
                confidence=confidence,
                description=(
                    "beaconing: ack=%.0f, iat_std=%.0f "
                    "(sustained %d prior windows)" % (ack, iat_std, sustained_c2)
                ),
                tactic_id="TA0011",
                elapsed_minutes=elapsed,
                attack_probability=prob,
                threshold=0.5,
            )

    # ------------------------------------------------------------------
    # Rule 2: Lateral Movement (TA0008)
    #   Sustained elevated SYN over >=2 consecutive windows + elevated RST
    #   (failed connection attempts to new hosts).
    # ------------------------------------------------------------------
    if syn > _LM_SYN_SUSTAINED_THRESHOLD and rst > _LM_RST_THRESHOLD:
        sustained_syn = sum(
            1
            for pw in prior[-3:]
            if _get_feat(pw, "syn_count") > _LM_SYN_SUSTAINED_THRESHOLD
        )
        if sustained_syn >= 1:
            confidence = min(0.95, 0.65 + 0.10 * sustained_syn + 0.05 * (prob / 1.0))
            return MitreStagePrediction(
                stage="Lateral Movement",
                confidence=confidence,
                description=(
                    "sustained scanning: syn=%.0f (sustained %d windows), "
                    "rst=%.0f failed connections" % (syn, sustained_syn + 1, rst)
                ),
                tactic_id="TA0008",
                elapsed_minutes=elapsed,
                attack_probability=prob,
                threshold=0.5,
            )

    # ------------------------------------------------------------------
    # Rule 3: Initial Access (TA0001)
    #   PSH/URG spike (payload delivery) + asymmetric traffic.
    # ------------------------------------------------------------------
    if (psh_fwd > _IA_PSH_FWD_THRESHOLD or urg_fwd > _IA_URG_FWD_THRESHOLD) and \
       down_up > _IA_DOWN_UP_RATIO_THRESHOLD:
        trigger = []
        if psh_fwd > _IA_PSH_FWD_THRESHOLD:
            trigger.append("psh_fwd=%.0f" % psh_fwd)
        if urg_fwd > _IA_URG_FWD_THRESHOLD:
            trigger.append("urg_fwd=%.0f" % urg_fwd)
        trigger.append("down_up=%.3f" % down_up)
        confidence = min(0.95, 0.60 + 0.15 * (prob / 1.0) + 0.05 * (urg_fwd / 30.0))
        return MitreStagePrediction(
            stage="Initial Access",
            confidence=confidence,
            description="payload delivery: %s" % ", ".join(trigger),
            tactic_id="TA0001",
            elapsed_minutes=elapsed,
            attack_probability=prob,
            threshold=0.5,
        )

    # ------------------------------------------------------------------
    # Rule 4: Reconnaissance (TA0043)
    #   Elevated SYN with low SYN/ACK completion ratio (scanning).
    # ------------------------------------------------------------------
    if syn > _RECON_SYN_THRESHOLD and syn_ack_ratio > _RECON_SYN_ACK_RATIO_CEIL:
        confidence = min(0.95, 0.55 + 0.20 * (syn / 200.0) + 0.10 * (prob / 1.0))
        return MitreStagePrediction(
            stage="Reconnaissance",
            description=(
                "port scanning: syn=%.0f, syn/ack_ratio=%.3f "
                "(incomplete handshakes)" % (syn, syn_ack_ratio)
            ),
            confidence=confidence,
            tactic_id="TA0043",
            elapsed_minutes=elapsed,
            attack_probability=prob,
            threshold=0.5,
        )

    # No behavioral signature matched.
    return None


def _heuristic_fallback(attack_probability, elapsed_minutes):
    """
    Original probability+time heuristic (preserved as fallback).

    Rule Matrix:
      - 0 to 5 min   & prob > 0.6 -> Reconnaissance
      - 5 to 15 min  & prob > 0.7 -> Initial Access
      - 15 to 35 min & prob > 0.8 -> Lateral Movement
      - 35+ min      & prob > 0.7 -> Command & Control
      - Otherwise                 -> Benign / Sub-threshold
    """
    prob = float(attack_probability)
    elapsed = max(0.0, float(elapsed_minutes))

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

    # If probability does not exceed required threshold
    threshold_val = 0.5
    if elapsed <= 5.0:
        threshold_val = 0.6
    elif elapsed <= 15.0:
        threshold_val = 0.7
    elif elapsed <= 35.0:
        threshold_val = 0.8
    else:
        threshold_val = 0.7

    sub_threshold_conf = 1.0 - prob if prob < 0.5 else prob
    return MitreStagePrediction(
        stage="Benign",
        confidence=round(sub_threshold_conf, 4),
        description="attack probability below threshold for active stage detection",
        tactic_id=None,
        elapsed_minutes=elapsed,
        attack_probability=prob,
        threshold=threshold_val,
    )


def predict_mitre_stage(
    attack_probability,
    elapsed_minutes_since_attack_start,
    feature_values=None,
    prior_windows=None,
):
    """
    Predict the MITRE ATT&CK phase, using behavior-driven classification
    when flow features are available, falling back to the probability+time
    heuristic otherwise.

    Args:
        attack_probability: Predicted probability of attack, in range [0.0, 1.0].
        elapsed_minutes_since_attack_start: Minutes elapsed since attack started (>= 0).
        feature_values: Optional dict of the 29 flow features for the current window.
            Keys can be bare names (e.g. 'syn_count') or prefixed ('current_syn_count').
        prior_windows: Optional list of feature dicts for preceding windows
            (most recent last), used for sustained-pattern detection.

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

    # --- Behavior-driven classification (when features are available) ---
    if feature_values is not None:
        behavioral_result = _classify_behavioral(
            features=feature_values,
            attack_probability=prob,
            elapsed_minutes=elapsed,
            prior_windows=prior_windows,
        )
        if behavioral_result is not None:
            return behavioral_result

    # --- Fallback: original probability+time heuristic ---
    return _heuristic_fallback(prob, elapsed)


def map_trajectory_to_mitre_stages(
    probabilities,
    start_elapsed_minutes=0.0,
    step_minutes=1.0,
    feature_sequence=None,
):
    """
    Map an entire forward rollout probability sequence to a trajectory of MITRE stages.
    """
    trajectory = []
    for idx, prob in enumerate(probabilities):
        t = start_elapsed_minutes + (idx * step_minutes)
        feats = feature_sequence[idx] if feature_sequence else None
        prior = feature_sequence[:idx] if feature_sequence and idx > 0 else None
        pred = predict_mitre_stage(prob, t, feature_values=feats, prior_windows=prior)
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
        print("Scenario: %-38s -> Elapsed: %4.1fm | Prob: %4.2f" % (scenario, elapsed, prob))
        print("   => Stage:      %s" % res.stage)
        print("      Confidence: %.2f" % res.confidence)
        print("      Tactic ID:  %s" % res.tactic_id)
        print("      Desc:       %s" % res.description)
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
        print("Input Attack Probability: %.4f" % args.prob)
        print("Elapsed Time:             %.1f minutes" % args.elapsed)
        print("Predicted Stage:          %s" % result.stage)
        print("Confidence:               %.2f" % result.confidence)
        if result.tactic_id:
            print("MITRE Tactic ID:          %s" % result.tactic_id)
        print("Operational Context:      %s" % result.description)
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
