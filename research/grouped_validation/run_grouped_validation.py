"""
Research Gap 2: Generalization & Grouped Profile Validation Experiment
=====================================================================
Investigates whether identical diamond feature profiles present across random
train/test partitions inflate reported performance.

Directly provides TWO paired, compatible evaluation protocols:
1. PAIRED HOLDOUT PROTOCOL:
   - Method A1: Random 80/20 Holdout (leaks 117 duplicate profiles)
   - Method A2: Grouped 80/20 Holdout (GroupShuffleSplit, 0% profile leakage)
2. PAIRED 5-FOLD CROSS-VALIDATION PROTOCOL:
   - Method B1: Random 5-Fold Cross-Validation (Standard KFold, profile leakage across all folds)
   - Method B2: Grouped 5-Fold Cross-Validation (GroupKFold, strictly 0% profile leakage across all folds)
"""

import json
import os
import sys

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from catboost import CatBoostRegressor

RANDOM_SEED = 42

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, "..", "..", "data"))

def get_clean_data():
    path = os.path.join(DATA_DIR, "diamonds (1).csv")
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Exclude physically impossible zero dimensions
    valid_mask = ~((df["x"] <= 0) | (df["y"] <= 0) | (df["z"] <= 0))
    df_clean = df[valid_mask].copy().reset_index(drop=True)

    feat_cols = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]
    
    # Construct exact 9-feature profile group ID
    df_clean["profile_str"] = (
        df_clean["carat"].astype(str) + "_" +
        df_clean["cut"] + "_" +
        df_clean["color"] + "_" +
        df_clean["clarity"] + "_" +
        df_clean["depth"].astype(str) + "_" +
        df_clean["table"].astype(str) + "_" +
        df_clean["x"].astype(str) + "_" +
        df_clean["y"].astype(str) + "_" +
        df_clean["z"].astype(str)
    )
    df_clean["profile_id"] = pd.factorize(df_clean["profile_str"])[0]

    return df_clean, feat_cols

def build_preprocessor():
    cat_cols = ["cut", "color", "clarity"]
    num_cols = ["carat", "depth", "table", "x", "y", "z"]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

def train_and_eval(X_train, y_train, X_test, y_test):
    prep = build_preprocessor()
    X_train_enc = prep.fit_transform(X_train)
    X_test_enc = prep.transform(X_test)

    model_xgb = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.07,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model_xgb.fit(X_train_enc, y_train)
    pred_xgb = model_xgb.predict(X_test_enc)

    model_cat = CatBoostRegressor(
        iterations=500,
        learning_rate=0.07,
        depth=6,
        cat_features=["cut", "color", "clarity"],
        random_seed=RANDOM_SEED,
        verbose=0,
    )
    model_cat.fit(X_train, y_train)
    pred_cat = model_cat.predict(X_test)

    # 55/45 Ensemble
    pred_ens = 0.55 * pred_cat + 0.45 * pred_xgb

    metrics = {
        "xgb": {
            "mae": float(mean_absolute_error(y_test, pred_xgb)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred_xgb))),
            "r2": float(r2_score(y_test, pred_xgb)),
        },
        "catboost": {
            "mae": float(mean_absolute_error(y_test, pred_cat)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred_cat))),
            "r2": float(r2_score(y_test, pred_cat)),
        },
        "ensemble": {
            "mae": float(mean_absolute_error(y_test, pred_ens)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred_ens))),
            "r2": float(r2_score(y_test, pred_ens)),
        },
    }
    return metrics

def main():
    print("=" * 70)
    print("EXPERIMENT 2: PAIRED RANDOM VS GROUPED VALIDATION BENCHMARK")
    print("=" * 70)

    df, feat_cols = get_clean_data()
    total_records = len(df)
    unique_profiles = df["profile_id"].nunique()
    print(f"Total physically valid records: {total_records}")
    print(f"Unique 9-feature profiles: {unique_profiles}")

    # ====================================================
    # PROTOCOL 1: PAIRED 80/20 HOLDOUT EVALUATION
    # ====================================================
    print("\n" + "=" * 50)
    print("PROTOCOL 1: PAIRED 80/20 HOLDOUT COMPARISON")
    print("=" * 50)

    # A1: Random Holdout
    train_idx_A1, test_idx_A1 = train_test_split(
        np.arange(total_records), test_size=0.2, random_state=RANDOM_SEED
    )
    tr_p_A1 = set(df.iloc[train_idx_A1]["profile_id"])
    te_p_A1 = set(df.iloc[test_idx_A1]["profile_id"])
    overlap_A1 = tr_p_A1.intersection(te_p_A1)
    leaked_A1 = df.iloc[test_idx_A1]["profile_id"].isin(overlap_A1).sum()

    print("Fitting A1: Random Holdout ...")
    metrics_A1 = train_and_eval(
        df.iloc[train_idx_A1][feat_cols], df.iloc[train_idx_A1]["price"],
        df.iloc[test_idx_A1][feat_cols], df.iloc[test_idx_A1]["price"]
    )
    print(f"  A1 (Random Holdout): Train N={len(train_idx_A1)}, Test N={len(test_idx_A1)}, Leaked Rows={leaked_A1}")
    print(f"  A1 Results: MAE=${metrics_A1['ensemble']['mae']:.2f}, RMSE=${metrics_A1['ensemble']['rmse']:.2f}, R²={metrics_A1['ensemble']['r2']:.6f}")

    # A2: Grouped Holdout
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx_A2, test_idx_A2 = next(gss.split(df, groups=df["profile_id"]))
    tr_p_A2 = set(df.iloc[train_idx_A2]["profile_id"])
    te_p_A2 = set(df.iloc[test_idx_A2]["profile_id"])
    overlap_A2 = tr_p_A2.intersection(te_p_A2)
    leaked_A2 = df.iloc[test_idx_A2]["profile_id"].isin(overlap_A2).sum()
    assert len(overlap_A2) == 0, "Leakage in GroupShuffleSplit!"

    print("Fitting A2: Grouped Holdout ...")
    metrics_A2 = train_and_eval(
        df.iloc[train_idx_A2][feat_cols], df.iloc[train_idx_A2]["price"],
        df.iloc[test_idx_A2][feat_cols], df.iloc[test_idx_A2]["price"]
    )
    print(f"  A2 (Grouped Holdout): Train N={len(train_idx_A2)}, Test N={len(test_idx_A2)}, Leaked Rows={leaked_A2} (0% leakage)")
    print(f"  A2 Results: MAE=${metrics_A2['ensemble']['mae']:.2f}, RMSE=${metrics_A2['ensemble']['rmse']:.2f}, R²={metrics_A2['ensemble']['r2']:.6f}")

    # ====================================================
    # PROTOCOL 2: PAIRED 5-FOLD CROSS-VALIDATION
    # ====================================================
    print("\n" + "=" * 50)
    print("PROTOCOL 2: PAIRED 5-FOLD CROSS-VALIDATION")
    print("=" * 50)

    # B1: Random 5-Fold CV
    print("Running B1: Random 5-Fold Cross-Validation ...")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    results_B1 = []
    leaked_folds_B1 = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(df), 1):
        tr_prof = set(df.iloc[tr_idx]["profile_id"])
        val_prof = set(df.iloc[val_idx]["profile_id"])
        leaked_count = df.iloc[val_idx]["profile_id"].isin(tr_prof.intersection(val_prof)).sum()
        leaked_folds_B1.append(int(leaked_count))

        m_fold = train_and_eval(
            df.iloc[tr_idx][feat_cols], df.iloc[tr_idx]["price"],
            df.iloc[val_idx][feat_cols], df.iloc[val_idx]["price"]
        )
        results_B1.append(m_fold["ensemble"])
        print(f"  B1 Fold {fold}: Leaked={leaked_count} | MAE=${m_fold['ensemble']['mae']:.2f}, RMSE=${m_fold['ensemble']['rmse']:.2f}, R²={m_fold['ensemble']['r2']:.6f}")

    b1_mae = np.mean([f["mae"] for f in results_B1])
    b1_mae_std = np.std([f["mae"] for f in results_B1])
    b1_rmse = np.mean([f["rmse"] for f in results_B1])
    b1_rmse_std = np.std([f["rmse"] for f in results_B1])
    b1_r2 = np.mean([f["r2"] for f in results_B1])
    b1_r2_std = np.std([f["r2"] for f in results_B1])
    total_leaked_B1 = sum(leaked_folds_B1)

    print(f"B1 Random 5-Fold Mean: MAE=${b1_mae:.2f} (±{b1_mae_std:.2f}), RMSE=${b1_rmse:.2f} (±{b1_rmse_std:.2f}), R²={b1_r2:.6f} (Total leaked={total_leaked_B1})")

    # B2: Grouped 5-Fold CV
    print("\nRunning B2: Grouped 5-Fold Cross-Validation ...")
    gkf = GroupKFold(n_splits=5)
    results_B2 = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(df, groups=df["profile_id"]), 1):
        tr_prof = set(df.iloc[tr_idx]["profile_id"])
        val_prof = set(df.iloc[val_idx]["profile_id"])
        assert len(tr_prof.intersection(val_prof)) == 0, f"Leakage in B2 Fold {fold}!"

        m_fold = train_and_eval(
            df.iloc[tr_idx][feat_cols], df.iloc[tr_idx]["price"],
            df.iloc[val_idx][feat_cols], df.iloc[val_idx]["price"]
        )
        results_B2.append(m_fold["ensemble"])
        print(f"  B2 Fold {fold}: Leaked=0 | MAE=${m_fold['ensemble']['mae']:.2f}, RMSE=${m_fold['ensemble']['rmse']:.2f}, R²={m_fold['ensemble']['r2']:.6f}")

    b2_mae = np.mean([f["mae"] for f in results_B2])
    b2_mae_std = np.std([f["mae"] for f in results_B2])
    b2_rmse = np.mean([f["rmse"] for f in results_B2])
    b2_rmse_std = np.std([f["rmse"] for f in results_B2])
    b2_r2 = np.mean([f["r2"] for f in results_B2])
    b2_r2_std = np.std([f["r2"] for f in results_B2])

    print(f"B2 Grouped 5-Fold Mean: MAE=${b2_mae:.2f} (±{b2_mae_std:.2f}), RMSE=${b2_rmse:.2f} (±{b2_rmse_std:.2f}), R²={b2_r2:.6f} (Strictly 0 leakage)")

    # ====================================================
    # FINAL RECONCILED PAIRED COMPARISON TABLE
    # ====================================================
    comparison_table = pd.DataFrame([
        {
            "Protocol": "Holdout Comparison",
            "Methodology": "A1. Random 80/20 Holdout",
            "Train_N": len(train_idx_A1),
            "Test_N": len(test_idx_A1),
            "Leaked_Test_Rows": leaked_A1,
            "Ensemble_MAE": round(metrics_A1["ensemble"]["mae"], 2),
            "Ensemble_RMSE": round(metrics_A1["ensemble"]["rmse"], 2),
            "Ensemble_R2": round(metrics_A1["ensemble"]["r2"], 6),
        },
        {
            "Protocol": "Holdout Comparison",
            "Methodology": "A2. Grouped 80/20 Holdout",
            "Train_N": len(train_idx_A2),
            "Test_N": len(test_idx_A2),
            "Leaked_Test_Rows": 0,
            "Ensemble_MAE": round(metrics_A2["ensemble"]["mae"], 2),
            "Ensemble_RMSE": round(metrics_A2["ensemble"]["rmse"], 2),
            "Ensemble_R2": round(metrics_A2["ensemble"]["r2"], 6),
        },
        {
            "Protocol": "Cross-Validation Comparison",
            "Methodology": "B1. Random 5-Fold CV (Mean)",
            "Train_N": int(total_records * 0.8),
            "Test_N": int(total_records * 0.2),
            "Leaked_Test_Rows": total_leaked_B1,
            "Ensemble_MAE": round(b1_mae, 2),
            "Ensemble_RMSE": round(b1_rmse, 2),
            "Ensemble_R2": round(b1_r2, 6),
        },
        {
            "Protocol": "Cross-Validation Comparison",
            "Methodology": "B2. Grouped 5-Fold CV (Mean)",
            "Train_N": int(total_records * 0.8),
            "Test_N": int(total_records * 0.2),
            "Leaked_Test_Rows": 0,
            "Ensemble_MAE": round(b2_mae, 2),
            "Ensemble_RMSE": round(b2_rmse, 2),
            "Ensemble_R2": round(b2_r2, 6),
        },
    ])

    print("\n" + "=" * 70)
    print("FINAL PAIRED VALIDATION COMPARISON TABLE")
    print("=" * 70)
    print(comparison_table.to_string(index=False))

    comparison_table.to_csv(os.path.join(OUTPUT_DIR, "validation_method_comparison.csv"), index=False)

    full_output = {
        "metadata": {
            "total_records": total_records,
            "unique_profiles": unique_profiles,
            "random_seed": RANDOM_SEED,
        },
        "holdout_protocol": {
            "A1_random_holdout": {"metrics": metrics_A1, "leaked_rows": int(leaked_A1)},
            "A2_grouped_holdout": {"metrics": metrics_A2, "leaked_rows": 0},
            "holdout_delta_mae": round(metrics_A2["ensemble"]["mae"] - metrics_A1["ensemble"]["mae"], 2),
            "holdout_delta_rmse": round(metrics_A2["ensemble"]["rmse"] - metrics_A1["ensemble"]["rmse"], 2),
        },
        "cross_validation_protocol": {
            "B1_random_5fold_cv": {
                "mean_mae": float(b1_mae), "std_mae": float(b1_mae_std),
                "mean_rmse": float(b1_rmse), "std_rmse": float(b1_rmse_std),
                "mean_r2": float(b1_r2), "std_r2": float(b1_r2_std),
                "total_leaked_rows": int(total_leaked_B1),
                "folds": results_B1,
            },
            "B2_grouped_5fold_cv": {
                "mean_mae": float(b2_mae), "std_mae": float(b2_mae_std),
                "mean_rmse": float(b2_rmse), "std_rmse": float(b2_rmse_std),
                "mean_r2": float(b2_r2), "std_r2": float(b2_r2_std),
                "total_leaked_rows": 0,
                "folds": results_B2,
            },
            "cv_delta_mae": round(b2_mae - b1_mae, 2),
            "cv_delta_rmse": round(b2_rmse - b1_rmse, 2),
        },
    }

    with open(os.path.join(OUTPUT_DIR, "grouped_validation_results.json"), "w") as f:
        json.dump(full_output, f, indent=2)

    print(f"\nResults successfully written to: {os.path.join(OUTPUT_DIR, 'grouped_validation_results.json')}")

if __name__ == "__main__":
    main()
