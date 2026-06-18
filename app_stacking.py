import streamlit as st
import pandas as pd

# Import from your new stacking regressor script
from predict_race_stacking import predict_race, RESULTS_PATH

st.set_page_config(page_title="F1 Race Predictor (Stacking)", layout="wide")
st.title("🏎️ F1 Finishing Order Predictor")
st.caption("Grid position blended with an ensemble Stacking Regressor (Ridge + Random Forest), optimized for the 2022-2026 regulations.")

st.sidebar.header("Configuration")
is_live = st.sidebar.checkbox("Live prediction (fetch qualifying from API)", value=False)
season = st.sidebar.selectbox("Season", [2026, 2025, 2024, 2023, 2022], index=0)
round_ = st.sidebar.number_input("Round", min_value=1, max_value=24, value=7)

if is_live:
    st.sidebar.info("Fetches the latest qualifying results live from the Jolpica API. Use this for a race that hasn't run yet.")
else:
    st.sidebar.info("Uses grid positions already in the historical CSV. Good for backtesting a race we already know the result of.")


@st.cache_data
def get_historical_data() -> pd.DataFrame:
    return pd.read_csv(RESULTS_PATH)


if st.button("Run Prediction", type="primary"):
    history = get_historical_data()

    try:
        with st.spinner("Training ensemble meta-learner and predicting..."):
            results = predict_race(history, season, round_, live=is_live)
    except ValueError as e:
        st.warning(str(e))
        results = None
    except Exception as e:
        st.error(f"Error fetching live data or running ensemble: {e}")
        results = None

    if results is not None:
        st.subheader(f"{results['race_name'].iloc[0]} — {season} Round {round_}")

        display_cols = ["predicted_position", "driver_code", "family_name", "constructor_name", "grid"]
        has_actual = results["position"].notna().any()
        if has_actual:
            results = results.rename(columns={"position": "actual_position"})
            results["actual_position"] = results["actual_position"].astype(int)
            display_cols.append("actual_position")

        st.dataframe(
            results[display_cols].rename(columns={
                "predicted_position": "Pred.", "driver_code": "Driver",
                "family_name": "Name", "constructor_name": "Team",
                "grid": "Grid", "actual_position": "Actual",
            }),
            use_container_width=True, hide_index=True,
        )

        if has_actual:
            hits = (results["predicted_position"] == results["actual_position"]).sum()
            st.caption(f"{hits} of {len(results)} drivers predicted in their exact finishing position.")
else:
    st.info("Configure a season and round in the sidebar, then click **Run Prediction**.")