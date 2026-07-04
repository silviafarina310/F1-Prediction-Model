"""
predict_race_stacking.py

Predicts the full finishing order for a race using a Stacking Regressor:
Combines a Ridge Regressor and a regularized Random Forest Regressor,
blended with the starting grid position.
"""

import argparse
import requests
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor

RESULTS_PATH = "f1_results_2022_2026.csv"
GRID_WEIGHT = 0.4  # 40% grid, 60% stacking model prediction
FINISH_STATUSES = {"Finished", "Lapped", "+1 Lap", "+2 Laps", "+3 Laps"}
NUMERIC_FEATURES = [
    "grid",
    "driver_form_3", "driver_form_5",
    "constructor_form_3", "constructor_form_5",
    "driver_points_to_date", "constructor_points_to_date",
    "driver_circuit_avg_pos",
    "driver_dnf_rate_5",
]

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

def predict_race(history: pd.DataFrame, season: int, round_: int, live: bool = False) -> pd.DataFrame:
    if live:
        target_rows = pd.DataFrame(fetch_qualifying_live(season, round_))
        # Build features on history ONLY — never contaminate with live NaN rows
        history_feat = build_features(history.copy())
        # For live target, grab each driver's most recent feature values from history
        latest = (
            history_feat.sort_values(["season", "round"])
            .groupby("driver_id")
            .last()
            .reset_index()
        )
        # Merge live grid onto those features
        target = target_rows.merge(
            latest[["driver_id"] + NUMERIC_FEATURES],
            on="driver_id", how="left"
        )
        # Override grid with the actual qualifying position
        target["grid"] = target_rows["grid"].values
        train = history_feat[history_feat["position"].notna()].copy()
    else:
        combined = build_features(history.copy())
        before_mask = (combined["season"] < season) | (
            (combined["season"] == season) & (combined["round"] < round_)
        )
        train = combined[before_mask & combined["position"].notna()].copy()
        target = combined[
            (combined["season"] == season) & (combined["round"] == round_)
        ].copy()
        if target.empty:
            raise ValueError(
                f"No data found for season {season} round {round_}. Use live=True for an upcoming race."
            )

    fill = train[NUMERIC_FEATURES].mean()
    X_train = train[NUMERIC_FEATURES].fillna(fill).fillna(0)
    y_train = train["position"]
    X_target = target[NUMERIC_FEATURES].fillna(fill).fillna(0)
    X_target = target[NUMERIC_FEATURES].fillna(fill).fillna(0)

    # Define base estimators tailored to small data sizes
    base_estimators = [
        ('ridge', Ridge(alpha=5.0)),
        ('rf', RandomForestRegressor(n_estimators=50, max_depth=4, min_samples_leaf=3, random_state=42))
    ]
    
    # Define a clean linear meta-learner to prevent final stage overfitting
    meta_learner = Ridge(alpha=1.0)

    # Initialize Stacking Regressor with 5-fold internal cross-validation
    stacking_model = StackingRegressor(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    
    # Fit and Predict
    stacking_model.fit(X_train, y_train)
    target["model_pred"] = stacking_model.predict(X_target)
    
    # Blend model outputs back with the direct grid constraints
    target["blend_score"] = GRID_WEIGHT * target["grid"] + (1 - GRID_WEIGHT) * target["model_pred"]
    target = target.sort_values("blend_score").reset_index(drop=True)
    target["predicted_position"] = range(1, len(target) + 1)
    
    return target

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--live", action="store_true", help="Fetch live qualifying from Jolpica API")
    args = parser.parse_args()

    history = pd.read_csv(RESULTS_PATH)
    target = predict_race(history, args.season, args.round, live=args.live)

    print(f"\nPredicted finishing order -- {target['race_name'].iloc[0]} ({args.season} round {args.round}):\n")
    display_df = target.copy()
    cols = ["predicted_position", "driver_code", "family_name", "constructor_name", "grid"]
    if display_df["position"].notna().any():
        display_df = display_df.rename(columns={"position": "actual_position"})
        display_df["actual_position"] = display_df["actual_position"].astype(int)
        cols.append("actual_position")
    print(display_df[cols].to_string(index=False))

if __name__ == "__main__":
    main()