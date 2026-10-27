"""
Research Gap 5: Explainable Predictions & Distinction from Causality
===================================================================
Addresses literature gaps across:
- Amadavadi et al. (2024, ICOSEC): Calls for model interpretability in diamond valuation.
- Randhawa et al. (2025, Springer CCIS): Notes lack of explainability in non-linear pricing models.
- Madena & Pandey (2025, IJRP): Emphasizes predictive transparency.

Methodology:
- Computes exact TreeSHAP values on the validated XGBoost regression model.
- Analyzes Global Feature Importance (mean |SHAP| values).
- Generates Local Explanations for 3 representative diamond tiers:
    1. Tier 1: Small/Budget diamond (< 0.5 ct)
    2. Tier 2: Mid-tier Commercial diamond (0.9 - 1.2 ct)
    3. Tier 3: High-value Luxury diamond (> 2.0 ct)
- Formally contrasts:
    (A) Genuine SHAP values (Shapley game-theoretic marginal contributions)
    (B) App Heuristic (1-at-a-time deviation from fixed reference diamond: 1.0ct Ideal G VS2)
    (C) True Causal Effects (interventional counterfactuals).
"""

import json
import os
import sys

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import shap

RANDOM_SEED = 42

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, "..", "..", "data"))

def load_data():
    path = os.path.join(DATA_DIR, "diamonds (1).csv")
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    valid_mask = ~((df["x"] <= 0) | (df["y"] <= 0) | (df["z"] <= 0))
    df_clean = df[valid_mask].copy().reset_index(drop=True)
    feat_cols = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]
    return train_test_split(df_clean[feat_cols], df_clean["price"], test_size=0.2, random_state=RANDOM_SEED)

def main():
    print("=" * 70)
    print("EXPERIMENT 5: SHAP EXPLAINABILITY & CAUSAL DISTINCTION")
    print("=" * 70)

    X_train, X_test, y_train, y_test = load_data()

    cat_cols = ["cut", "color", "clarity"]
    num_cols = ["carat", "depth", "table", "x", "y", "z"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)
    feature_names = list(preprocessor.get_feature_names_out())

    print("Fitting XGBoost model for SHAP analysis ...")
    model = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.06,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train_enc, y_train)

    # Use TreeExplainer
    print("Computing TreeSHAP values on representative sample of test set (N=1,000) ...")
    explainer = shap.TreeExplainer(model)
    sample_indices = np.random.RandomState(RANDOM_SEED).choice(len(X_test), size=1000, replace=False)
    X_sample_enc = X_test_enc[sample_indices]
    X_sample_raw = X_test.iloc[sample_indices].reset_index(drop=True)
    y_sample = y_test.iloc[sample_indices].reset_index(drop=True)

    shap_values = explainer.shap_values(X_sample_enc)
    base_value = float(explainer.expected_value)

    # 1. Global Feature Importance (Mean |SHAP|)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    global_importance_df = pd.DataFrame({
        "Encoded_Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap,
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    # Aggregate back to raw feature categories
    raw_importance = {}
    for raw_col in ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]:
        matching_cols = [c for c in feature_names if raw_col in c]
        raw_importance[raw_col] = float(global_importance_df[global_importance_df["Encoded_Feature"].isin(matching_cols)]["Mean_Abs_SHAP"].sum())

    raw_importance_df = pd.DataFrame(list(raw_importance.items()), columns=["Raw_Feature", "Aggregated_SHAP_Impact"]).sort_values(by="Aggregated_SHAP_Impact", ascending=False).reset_index(drop=True)

    print("\n--- Global Feature Importance (Aggregated SHAP Impact in $) ---")
    print(raw_importance_df.to_string(index=False))

    # 2. Local Explanations for 3 Characteristic Case Studies
    # Case 1: Budget Diamond (< 0.5 ct)
    # Case 2: Mid-tier Diamond (0.9 - 1.1 ct)
    # Case 3: Luxury Diamond (> 2.0 ct)
    budget_idx = np.where(X_sample_raw["carat"] <= 0.4)[0][0]
    mid_idx = np.where((X_sample_raw["carat"] >= 0.95) & (X_sample_raw["carat"] <= 1.05))[0][0]
    luxury_idx = np.where(X_sample_raw["carat"] >= 2.0)[0][0]

    case_studies = [
        {"tier": "Budget Diamond (Carat <= 0.40)", "index": int(budget_idx)},
        {"tier": "Mid-tier Diamond (Carat ~ 1.00)", "index": int(mid_idx)},
        {"tier": "Luxury Diamond (Carat >= 2.00)", "index": int(luxury_idx)},
    ]

    case_study_outputs = []
    for cs in case_studies:
        idx = cs["index"]
        raw_row = X_sample_raw.iloc[idx].to_dict()
        actual_price = float(y_sample.iloc[idx])
        pred_price = float(model.predict(X_sample_enc[idx:idx+1])[0])
        shap_row = shap_values[idx]

        top_factors = pd.DataFrame({
            "Feature": feature_names,
            "SHAP_Value": shap_row,
        }).sort_values(by="SHAP_Value", key=abs, ascending=False).head(5).to_dict(orient="records")

        case_study_outputs.append({
            "tier": cs["tier"],
            "attributes": raw_row,
            "actual_price": actual_price,
            "predicted_price": round(pred_price, 2),
            "base_value": round(base_value, 2),
            "top_5_shap_drivers": top_factors,
        })

    # Save exports
    global_importance_df.to_csv(os.path.join(OUTPUT_DIR, "shap_global_importance_encoded.csv"), index=False)
    raw_importance_df.to_csv(os.path.join(OUTPUT_DIR, "shap_global_importance_raw.csv"), index=False)

    full_output = {
        "base_expected_value": base_value,
        "global_importance_raw": raw_importance_df.to_dict(orient="records"),
        "case_studies": case_study_outputs,
        "methodology_note": {
            "shap_definition": "SHAP (SHapley Additive exPlanations) computes exact game-theoretic Shapley values distributing the difference between model output and expected value.",
            "app_comparison_distinction": "The application's 'Explain My Prediction' module computes a one-at-a-time difference from an Ideal 1.0 ct G VS2 baseline diamond. While intuitive for non-technical buyers, it is a heuristic comparison and must not be confused with mathematically balanced TreeSHAP values.",
            "causal_disclaimer": "Neither SHAP nor reference-difference values constitute physical or market causation. Changing a diamond's grading on paper does not causally alter its market clearing value without joint gemological consideration.",
        },
    }

    with open(os.path.join(OUTPUT_DIR, "shap_explainability_results.json"), "w") as f:
        json.dump(full_output, f, indent=2)

    print("\n" + "=" * 70)
    print("SHAP EXPLAINABILITY ANALYSIS COMPLETED")
    print(f"Results written to: {os.path.join(OUTPUT_DIR, 'shap_explainability_results.json')}")
    print("=" * 70)

if __name__ == "__main__":
    main()

