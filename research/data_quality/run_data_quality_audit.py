"""
Research Gap 1: Data Quality, Integrity, and Robustness Experiment
=================================================================
Investigates:
1. Physically invalid dimensions (x, y, z = 0) in the Standard Dataset.
2. Identical feature profile collisions with price divergence.
3. Impact of removing invalid dimensions on CatBoost and XGBoost regression metrics.
4. Detailed Dataset missingness, L/W ratio discrepancies, and Type distribution.
"""

import json
import os
import sys

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Ensure reproducibility
RANDOM_SEED = 42

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, "..", "..", "data"))

def audit_standard_dataset():
    print("=" * 70)
    print("AUDITING STANDARD DATASET (data/diamonds (1).csv)")
    print("=" * 70)
    path = os.path.join(DATA_DIR, "diamonds (1).csv")
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    total_rows = len(df)
    exact_duplicates = df.duplicated().sum()

    # Zero dimensions check
    zero_x = (df["x"] <= 0).sum()
    zero_y = (df["y"] <= 0).sum()
    zero_z = (df["z"] <= 0).sum()
    zero_any = ((df["x"] <= 0) | (df["y"] <= 0) | (df["z"] <= 0)).sum()
    zero_df = df[(df["x"] <= 0) | (df["y"] <= 0) | (df["z"] <= 0)]

    # Feature profile collisions
    feat_cols = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]
    dup_profiles = df[df.duplicated(subset=feat_cols, keep=False)]
    unique_profiles = df.duplicated(subset=feat_cols, keep="first").sum()

    # Price variance within identical profiles
    profile_groups = df.groupby(feat_cols)["price"].agg(["count", "min", "max", "std"]).reset_index()
    multi_profiles = profile_groups[profile_groups["count"] > 1]
    varying_profiles = multi_profiles[multi_profiles["std"] > 0]
    max_price_spread = (multi_profiles["max"] - multi_profiles["min"]).max()

    print(f"Total raw rows: {total_rows}")
    print(f"Exact row duplicates: {exact_duplicates}")
    print(f"Physically impossible zero dimensions: {zero_any} records")
    print(f"  - x <= 0: {zero_x}")
    print(f"  - y <= 0: {zero_y}")
    print(f"  - z <= 0: {zero_z}")
    print(f"Rows sharing identical 9-feature profiles: {len(dup_profiles)} rows ({len(multi_profiles)} unique profiles)")
    print(f"Profiles with price divergence: {len(varying_profiles)} (Max spread: ${max_price_spread:.2f})")

    # Save anomaly records
    zero_df.to_csv(os.path.join(OUTPUT_DIR, "standard_zero_dimensions.csv"), index=False)
    varying_profiles.head(50).to_csv(os.path.join(OUTPUT_DIR, "standard_divergent_profiles_sample.csv"), index=False)

    return df, {
        "total_rows": int(total_rows),
        "zero_dimensions_count": int(zero_any),
        "clean_rows": int(total_rows - zero_any),
        "repeated_profile_rows": int(len(dup_profiles)),
        "unique_repeated_profiles": int(len(multi_profiles)),
        "profiles_with_price_divergence": int(len(varying_profiles)),
        "max_price_spread": float(max_price_spread),
    }

def run_dimension_cleaning_experiment(df_raw):
    print("\n" + "=" * 70)
    print("CONTROLLED EXPERIMENT: IMPACT OF ZERO-DIMENSION REMOVAL")
    print("=" * 70)
    feat_cols = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]

    # Valid rows index
    valid_mask = ~((df_raw["x"] <= 0) | (df_raw["y"] <= 0) | (df_raw["z"] <= 0))
    df_clean = df_raw[valid_mask].copy().reset_index(drop=True)

    # We evaluate BOTH models on strictly identical held-out test records (valid rows only)
    # Split clean data first into 80% train pool and 20% test
    X_clean = df_clean[feat_cols]
    y_clean = df_clean["price"]
    X_train_clean, X_test, y_train_clean, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=RANDOM_SEED
    )

    # For the "uncleaned" training set, add back the 20 zero-dimension records into training
    zero_records = df_raw[~valid_mask][feat_cols]
    zero_prices = df_raw[~valid_mask]["price"]
    X_train_uncleaned = pd.concat([X_train_clean, zero_records], axis=0).reset_index(drop=True)
    y_train_uncleaned = pd.concat([y_train_clean, zero_prices], axis=0).reset_index(drop=True)

    cat_cols = ["cut", "color", "clarity"]
    num_cols = ["carat", "depth", "table", "x", "y", "z"]

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    # --- Config A: Trained WITH zero-dimension anomalies ---
    X_tr_A_enc = preprocessor.fit_transform(X_train_uncleaned)
    X_te_enc = preprocessor.transform(X_test)

    xgb_model_A = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    xgb_model_A.fit(X_tr_A_enc, y_train_uncleaned)
    preds_A = xgb_model_A.predict(X_te_enc)

    mae_A = mean_absolute_error(y_test, preds_A)
    rmse_A = np.sqrt(mean_squared_error(y_test, preds_A))
    r2_A = r2_score(y_test, preds_A)

    # --- Config B: Trained WITHOUT zero-dimension anomalies (Cleaned) ---
    preprocessor_B = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    X_tr_B_enc = preprocessor_B.fit_transform(X_train_clean)
    X_te_B_enc = preprocessor_B.transform(X_test)

    xgb_model_B = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    xgb_model_B.fit(X_tr_B_enc, y_train_clean)
    preds_B = xgb_model_B.predict(X_te_B_enc)

    mae_B = mean_absolute_error(y_test, preds_B)
    rmse_B = np.sqrt(mean_squared_error(y_test, preds_B))
    r2_B = r2_score(y_test, preds_B)

    # Also check prediction on the zero-dimension records themselves
    zero_preds_A = xgb_model_A.predict(preprocessor.transform(zero_records))
    zero_preds_B = xgb_model_B.predict(preprocessor_B.transform(zero_records))
    zero_mae_A = mean_absolute_error(zero_prices, zero_preds_A)
    zero_mae_B = mean_absolute_error(zero_prices, zero_preds_B)

    print("Config A (Zero-dims in Training, N_train=43,156):")
    print(f"  Test MAE:  ${mae_A:.2f}")
    print(f"  Test RMSE: ${rmse_A:.2f}")
    print(f"  Test R²:   {r2_A:.6f}")
    print("\nConfig B (Zero-dims Excluded, N_train=43,136):")
    print(f"  Test MAE:  ${mae_B:.2f}")
    print(f"  Test RMSE: ${rmse_B:.2f}")
    print(f"  Test R²:   {r2_B:.6f}")
    print(f"\nDelta on clean test set: MAE Delta = ${mae_B - mae_A:.2f}, RMSE Delta = ${rmse_B - rmse_A:.2f}")
    print(f"Error on invalid zero records when evaluated: Config A MAE = ${zero_mae_A:.2f}, Config B MAE = ${zero_mae_B:.2f}")

    return {
        "config_A_uncleaned": {"train_size": len(X_train_uncleaned), "mae": float(mae_A), "rmse": float(rmse_A), "r2": float(r2_A)},
        "config_B_cleaned": {"train_size": len(X_train_clean), "mae": float(mae_B), "rmse": float(rmse_B), "r2": float(r2_B)},
        "clean_test_size": len(X_test),
        "zero_record_error": {"config_A_mae": float(zero_mae_A), "config_B_mae": float(zero_mae_B)},
    }

def audit_detailed_dataset():
    print("\n" + "=" * 70)
    print("AUDITING DETAILED DATASET (data/diamonds (cleaned).csv)")
    print("=" * 70)
    path = os.path.join(DATA_DIR, "diamonds (cleaned).csv")
    df = pd.read_csv(path)

    total_rows = len(df)
    missing_summary = df.isnull().sum()
    missing_pct = (df.isnull().sum() / total_rows * 100).round(2)
    missing_table = pd.DataFrame({"Missing_Count": missing_summary, "Missing_Percent": missing_pct})

    # Length/Width Ratio Check
    valid_dims = df[df["Length"].notna() & df["Width"].notna() & (df["Width"] > 0)].copy()
    valid_dims["computed_ratio"] = valid_dims["Length"] / valid_dims["Width"]
    valid_dims["ratio_diff"] = (valid_dims["computed_ratio"] - valid_dims["Length/Width Ratio"]).abs()
    lw_discrepancies = valid_dims[valid_dims["ratio_diff"] > 0.01]

    # Type Breakdown & Price Stats
    type_stats = df.groupby("Type")["Price"].agg(["count", "mean", "std", "median", "min", "max"]).reset_index()

    print(f"Total records: {total_rows}")
    print(f"Columns with >50% missing values: Cut ({missing_pct['Cut']}%), Culet ({missing_pct['Culet']}%), Fluorescence ({missing_pct['Fluorescence']}%)")
    print(f"Records with L/W ratio discrepancies (>0.01): {len(lw_discrepancies)} ({len(lw_discrepancies)/total_rows*100:.1f}%)")
    print(f"Maximum L/W discrepancy: {valid_dims['ratio_diff'].max():.4f}")
    print("\nPrice statistics by Type:")
    print(type_stats.to_string(index=False))

    # Save exports
    missing_table.to_csv(os.path.join(OUTPUT_DIR, "detailed_missing_summary.csv"))
    lw_discrepancies.head(100).to_csv(os.path.join(OUTPUT_DIR, "detailed_lw_discrepancies_sample.csv"), index=False)
    type_stats.to_csv(os.path.join(OUTPUT_DIR, "detailed_type_price_summary.csv"), index=False)

    return {
        "total_rows": int(total_rows),
        "missing_cut_pct": float(missing_pct["Cut"]),
        "missing_culet_pct": float(missing_pct["Culet"]),
        "missing_fluorescence_pct": float(missing_pct["Fluorescence"]),
        "lw_ratio_discrepancy_count": int(len(lw_discrepancies)),
        "max_lw_ratio_discrepancy": float(valid_dims["ratio_diff"].max()),
        "type_breakdown": type_stats.to_dict(orient="records"),
    }

def main():
    df_standard, standard_metrics = audit_standard_dataset()
    exp_metrics = run_dimension_cleaning_experiment(df_standard)
    detailed_metrics = audit_detailed_dataset()

    full_results = {
        "standard_dataset_audit": standard_metrics,
        "dimension_cleaning_experiment": exp_metrics,
        "detailed_dataset_audit": detailed_metrics,
    }

    with open(os.path.join(OUTPUT_DIR, "data_quality_results.json"), "w") as f:
        json.dump(full_results, f, indent=2)

    print("\n" + "=" * 70)
    print("SUCCESS: Data quality audit & experiment completed.")
    print(f"Results written to: {os.path.join(OUTPUT_DIR, 'data_quality_results.json')}")
    print("=" * 70)

if __name__ == "__main__":
    main()
