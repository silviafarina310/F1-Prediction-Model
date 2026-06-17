import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
import lightgbm as lgb
from features import build_features
from predict_race_3 import NUMERIC_FEATURES, CATEGORICAL_FEATURES

def evaluate_model():
    df = pd.read_csv("f1_results_2022_2026.csv")
    df = build_features(df)
    
    # We only evaluate on races where we have a known finishing position
    eval_races = df[df["position"].notna() & (df["season"] >= 2023)].groupby(["season", "round"])
    
    ndcg_scores = []
    
    for (season, round_), race_data in eval_races:
        # Train on all data BEFORE this race
        train = df[((df["season"] < season) | ((df["season"] == season) & (df["round"] < round_)))]
        test = race_data
        
        if train.empty or test.empty: continue
        
        features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        fill = train[NUMERIC_FEATURES].mean()
        
        model = lgb.LGBMRanker(objective="lambdarank")
        model.fit(
            X=train[features].fillna(0),
            y=25 - train["position"],
            group=train.groupby(['season', 'round']).size().values,
            categorical_feature=CATEGORICAL_FEATURES
        )
      
        preds = model.predict(test[features].fillna(0))
        
        # NDCG calculation
        # y_true needs to be a 2D array: [relevance_scores]
        true_relevance = np.array([25 - test["position"].values])
        pred_scores = np.array([preds])
        
        score = ndcg_score(true_relevance, pred_scores)
        ndcg_scores.append(score)
        print(f"Race {season} R{round_}: NDCG = {score:.4f}")

    print(f"\nAverage NDCG over {len(ndcg_scores)} races: {np.mean(ndcg_scores):.4f}")

if __name__ == "__main__":
    evaluate_model()