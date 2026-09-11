import pandas as pd
from pathlib import Path


INPUT_FILE = "data/processed/flow_network_states.csv"
OUTPUT_FILE = "data/processed/state_transitions.csv"


def main():

    print("[1/3] Loading network states...")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamp"]
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"States loaded: {len(df):,}")


    print("[2/3] Building S(t) -> S(t+1) transitions...")

    # Features that describe the current network state.
    # Labels and identifiers are kept separate to prevent leakage.
    excluded = {
        "timestamp",
        "window",
        "elapsed_seconds",
        "attack_label",
    }

    feature_columns = [
        c for c in df.columns
        if c not in excluded
    ]

    current = df[feature_columns].copy()
    current.columns = [
        f"current_{c}" for c in current.columns
    ]

    next_state = df[feature_columns].shift(-1).copy()
    next_state.columns = [
        f"next_{c}" for c in next_state.columns
    ]

    transitions = pd.concat(
        [
            df[["timestamp", "window", "attack_label"]].rename(
                columns={"attack_label": "current_attack_label"}
            ),
            current,
            next_state,
            df["attack_label"].shift(-1).rename(
                "next_attack_label"
            ),
        ],
        axis=1,
    )

    # Last state has no S(t+1), therefore remove it.
    transitions = transitions.iloc[:-1].copy()

    # Verify actual one-minute transitions.
    transitions["next_timestamp"] = (
        df["timestamp"].shift(-1).iloc[:-1].values
    )

    transitions["time_delta_seconds"] = (
        transitions["next_timestamp"]
        - transitions["timestamp"]
    ).dt.total_seconds()

    valid = transitions["time_delta_seconds"] == 60

    print(f"Valid 60-second transitions: {valid.sum():,}")
    print(f"Invalid transitions: {(~valid).sum():,}")


    # Keep only continuous temporal transitions.
    transitions = transitions[valid].copy()

    # Remove helper column after validation.
    transitions = transitions.drop(
        columns=["time_delta_seconds"]
    )


    print("[3/3] Saving transition dataset...")

    Path("data/processed").mkdir(
        parents=True,
        exist_ok=True
    )

    transitions.to_csv(
        OUTPUT_FILE,
        index=False
    )


    print()
    print("=" * 60)
    print("STATE TRANSITION DATASET COMPLETE")
    print("=" * 60)
    print(f"Transitions:        {len(transitions):,}")
    print(f"Current features:   {len(feature_columns)}")
    print(
        f"Attack transitions: "
        f"{int(transitions['next_attack_label'].sum()):,}"
    )
    print(
        f"Benign transitions: "
        f"{int((transitions['next_attack_label'] == 0).sum()):,}"
    )
    print(f"Output:             {OUTPUT_FILE}")
    print()
    print("Transition definition:")
    print("S(t) -> S(t+1)")
    print("=" * 60)


if __name__ == "__main__":
    main()
