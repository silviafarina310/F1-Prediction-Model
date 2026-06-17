"""
predict_race_3.py

Predicts the full finishing order for a race using the approach validated
via walk-forward testing: a blend of grid position and LightGBM Ranker model trained on engineered form/history features.

Two modes:
  (default)   Use SEASON/ROUND's grid positions straight from the historical
              results CSV: useful for backtesting against a race we
              already know the real outcome of.
  --live      Fetch real qualifying results for SEASON/ROUND from the
              Jolpica API instead: use this right after qualifying finishes
              for a race that hasn't been run yet.

Either way, the model is trained only on races strictly before SEASON/ROUND,
so a backtest never gets to peek at its own future.

Usage:
    python predict_race_3.py --season 2026 --round 7
    python predict_race_3.py --season 2026 --round 9 --live
"""

import argparse
import requests
import pandas as pd
import lightgbm as lgb

RESULTS_PATH = "f1_results_2022_2026.csv"
GRID_WEIGHT = 0.4  # validated blend weight: 40% grid, 60% model
FINISH_STATUSES = {"Finished", "Lapped", "+1 Lap", "+2 Laps", "+3 Laps"}
NUMERIC_FEATURES = [
    "grid",
    "driver_form_3", "driver_form_5",
    "constructor_form_3", "constructor_form_5",
    "driver_points_to_date", "constructor_points_to_date",
    "driver_circuit_avg_pos",
    "driver_dnf_rate_5",
]
CATEGORICAL_FEATURES = ["track_type", "downforce_level"]  # Categorical features for LightGBM

def fetch_qualifying_live(season: int, round_: int) -> list[dict]:
    url = f"https://api.jolpi.ca/ergast/f1/{season}/{round_}/qualifying/"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    races = resp.json()["MRData"]["RaceTable"]["Races"]
    if not races:
        raise SystemExit(f"No qualifying data found yet for season {season} round {round_}.")
    race = races[0]
    circuit = race["Circuit"]
    rows = []
    for q in race["QualifyingResults"]:
        driver, constructor = q["Driver"], q["Constructor"]
        rows.append({
            "season": season, "round": round_, "race_name": race["raceName"],
            "race_date": race["date"], "circuit_id": circuit["circuitId"],
            "circuit_name": circuit["circuitName"], "country": circuit["Location"]["country"],
            "latitude": float(circuit["Location"]["lat"]), "longitude": float(circuit["Location"]["long"]),
            "driver_id": driver["driverId"], "driver_code": driver.get("code"),
            "given_name": driver["givenName"], "family_name": driver["familyName"],
            "dob": driver["dateOfBirth"], "nationality": driver["nationality"],
            "constructor_id": constructor["constructorId"], "constructor_name": constructor["name"],
            "grid": int(q["position"]),
            "position": None, "position_text": None, "points": None, "laps": None, "status": None,
        })
    return rows


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["season", "round", "grid"]).reset_index(drop=True)
    df["dnf"] = df["status"].apply(lambda s: int(s not in FINISH_STATUSES) if pd.notna(s) else 0)

    for window in (3, 5):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    history = pd.read_csv(RESULTS_PATH)

    from features import build_features
    history = build_features(history)

    if args.live:
        target_rows = pd.DataFrame(fetch_qualifying_live(args.season, args.round))
        # Add features to live data
        target_rows = build_features(target_rows)
        combined = pd.concat([history, target_rows], ignore_index=True)
    else:
        combined = history.copy()

    before_mask = (combined["season"] < args.season) | (
        (combined["season"] == args.season) & (combined["round"] < args.round)
    )
    
    train = combined[before_mask].copy()
    target = combined[(combined["season"] == args.season) & (combined["round"] == args.round)].copy()

    if target.empty:
        raise SystemExit(f"No data found for {args.season} round {args.round}.")

    fill = train[NUMERIC_FEATURES].mean()

    train_groups = train.groupby(['season', 'round']).size().values
    train_labels = 25 - train["position"]
    
    model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", importance_type="gain")

    model.fit(
        X=train[NUMERIC_FEATURES + CATEGORICAL_FEATURES].fillna(0),
        y=train_labels,
        group=train_groups,
        categorical_feature=CATEGORICAL_FEATURES,
    )

    target["model_pred"] = model.predict(target[NUMERIC_FEATURES + CATEGORICAL_FEATURES].fillna(0))
    target["model_pred_pos"] = target["model_pred"].rank(ascending=False)
    
    #Blend with grid (40% grid, 60% model)
    target["blend_score"] = GRID_WEIGHT * target["grid"] + (1 - GRID_WEIGHT) * target["model_pred_pos"]
    target = target.sort_values("blend_score").reset_index(drop=True)
    target["predicted_position"] = range(1, len(target) + 1)

    print(f"\nPredicted finishing order for {args.season} round {args.round}:\n")
    
    # Define the base columns we definitely have
    cols = ["predicted_position", "driver_code", "family_name", "constructor_name", "grid"]
    
    # Only try to add 'position' if it exists in the dataframe AND is not full of NaNs (for live races)
    if "position" in target.columns and target["position"].notna().any():
        cols.append("position")
    
    # Print the dataframe using only the columns we confirmed exist
    print(target[cols].to_string(index=False))

if __name__ == "__main__":
    main()