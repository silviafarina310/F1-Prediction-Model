"""
Turns the f1_results_2022_2026.csv file into a features dataset for modeling.

Every rolling/cumulative feature is shifted by one race before being
computed, so a row never sees information from its own race or any
race that hasn't happened yet. 
"""

import pandas as pd

IN_PATH = "f1_results_2022_2026.csv"
OUT_PATH = "f1_features_2022_2026.csv"

FINISH_STATUSES = {"Finished", "Lapped", "+1 Lap", "+2 Laps", "+3 Laps"}

def build_features(df: pd.DataFrame):
    df = df.sort_values(["season", "round", "position"]).reset_index(drop=True)

    df["dnf"] = (~df["status"].isin(FINISH_STATUSES)).astype(int)

    for window in(3, 5):
        df[f"driver_form_{window}"] = (
            df.groupby("driver_id")["position"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"constructor_form_{window}"] = (
            df.groupby("constructor_id")["position"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
    
    df["driver_points_to_date"] = (
        df.groupby(["season", "driver_id"])["points"]
        .transform(lambda s: s.shift(1).cumsum()).fillna(0)
    )
    df["constructor_points_to_date"] = (
        df.groupby(["season", "constructor_id"])["points"]
        .transform(lambda s: s.shift(1).cumsum().fillna(0))
    )

    # This driver's average finish at this specific circuit, prior visits only.
    # NaN for a driver's first-ever visit to a circuit -- left as NaN on purpose,
    # since LightGBM/XGBoost handle missing values natively and a fillna here
    # would just be inventing a number for "no information yet".
    df["driver_circuit_avg_pos"] = (
        df.groupby(["driver_id", "circuit_id"])["position"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )

    df["driver_dnf_rate_5"] = (
        df.groupby("driver_id")["dnf"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )
 
    return df


def main():
    df = pd.read_csv(IN_PATH)
    features = build_features(df)
    features.to_csv(OUT_PATH, index=False)
    print(f"Saved {features.shape[0]} rows x {features.shape[1]} columns to {OUT_PATH}")


if __name__ == "__main__":
    main()