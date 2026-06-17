import streamlit as st
import pandas as pd
import lightgbm as lgb
from predict_race_3 import (
    build_features, 
    NUMERIC_FEATURES, 
    CATEGORICAL_FEATURES, 
    GRID_WEIGHT, 
    fetch_qualifying_live
)

st.set_page_config(layout="wide")
st.title("🏎️ F1 Prediction Dashboard")

# 1. Sidebar Controls
st.sidebar.header("Configuration")
is_live = st.sidebar.checkbox("Live Prediction (Fetch from API)", value=False)
season = st.sidebar.selectbox("Season", [2026, 2025, 2024], index=0)
round_ = st.sidebar.number_input("Round", min_value=1, max_value=24, value=7)

# 2. Data Loading & Prediction Logic
@st.cache_data
def get_historical_data():
    return pd.read_csv("f1_results_2022_2026.csv")

def run_prediction():
    history = get_historical_data()
    # Ensure build_features is actually adding the columns
    history = build_features(history) 
    
    # Check if columns were added
    if 'track_type' not in history.columns:
        st.error("Error: 'track_type' missing from DataFrame. Check features.py.")
        return None
    
    if is_live:
        try:
            target_rows = pd.DataFrame(fetch_qualifying_live(season, round_))
            target_rows = build_features(target_rows)
            combined = pd.concat([history, target_rows], ignore_index=True)
        except Exception as e:
            st.error(f"Error fetching live data: {e}")
            return None
    else:
        combined = history.copy()

    # Split Data
    before_mask = (combined["season"] < season) | ((combined["season"] == season) & (combined["round"] < round_))
    train = combined[before_mask].copy()
    target = combined[(combined["season"] == season) & (combined["round"] == round_)].copy()

    if target.empty:
        st.warning("No data found for this selection.")
        return None

    # Train Model
    fill = train[NUMERIC_FEATURES].mean()
    model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", importance_type="gain")
    model.fit(
        X=train[NUMERIC_FEATURES + CATEGORICAL_FEATURES].fillna(0),
        y=25 - train["position"],
        group=train.groupby(['season', 'round']).size().values,
        categorical_feature=CATEGORICAL_FEATURES,
    )

    # Predict
    target["model_pred"] = model.predict(target[NUMERIC_FEATURES + CATEGORICAL_FEATURES].fillna(0))
    target["model_pred_pos"] = target["model_pred"].rank(ascending=False)
    target["blend_score"] = GRID_WEIGHT * target["grid"] + (1 - GRID_WEIGHT) * target["model_pred_pos"]
    target = target.sort_values("blend_score").reset_index(drop=True)
    target["predicted_position"] = range(1, len(target) + 1)
    
    return target

if st.button("Run Analysis"):
    results = run_prediction()
    if results is not None:
        # Show only relevant columns for the table
        display_cols = ["predicted_position", "driver_code", "constructor_name", "grid"]
        
        # Add actual position if it exists (i.e., not live mode)
        if "position" in results.columns and results["position"].notna().any():
            display_cols.insert(1, "position")
            results = results.rename(columns={"position": "actual_pos"})
            
        st.subheader(f"Prediction for Round {round_}")
        st.dataframe(results[display_cols], use_container_width=True)