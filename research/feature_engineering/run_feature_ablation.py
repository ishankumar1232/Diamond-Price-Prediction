"""
Research Gap 4: Domain Feature Engineering & Interaction Ablation
=================================================================
Rigorous Training-Only Feature Selection & Untouched Test Set Evaluation.

Methodology complying with strict ML audit guidelines:
1. Split 80% Train (N=43,136) and 20% Test (N=10,784) with seed 42.
   The 20% test set is held out untouched during all feature selection.
2. Inside the training set, perform 5-Fold Cross-Validation across candidate feature sets:
   - Config 0: Baseline 9 standard features (carat, cut, color, clarity, depth, table, x, y, z)
   - Config 1: Baseline + Physical Volume (x * y * z)
   - Config 2: Baseline + Carat-Quality Interactions (carat x cut, carat x color, carat x clarity)
   - Config 3: Baseline + Carat x Clarity Only
   - Config 4: Baseline + Geometry (Volume, Aspect Ratio, Depth Ratio)
   - Config 5: Full Engineered Set (All 7 candidate features)
3. Select the best performing feature set strictly based on Training 5-Fold CV MAE.
4. Fit the winning configuration on the full training set and evaluate exactly ONCE
   on the untouched 20% test set alongside the baseline for unbiased evaluation.
"""

import json
import os
import sys

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb

RANDOM_SEED = 42

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, "..", "..", "data"))

def load_clean_data():
    path = os.path.join(DATA_DIR, "diamonds (1).csv")
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    valid_mask = ~((df["x"] <= 0) | (df["y"] <= 0) | (df["z"] <= 0))
    df_clean = df[valid_mask].copy().reset_index(drop=True)
    return df_clean

def add_engineered_features(df):
    df_feat = df.copy()

    # Ordinal mappings for interaction terms
    cut_map = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
    color_map = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}
    clarity_map = {"I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8}

    df_feat["cut_num"] = df_feat["cut"].map(cut_map).astype(float)
    df_feat["color_num"] = df_feat["color"].map(color_map).astype(float)
    df_feat["clarity_num"] = df_feat["clarity"].map(clarity_map).astype(float)

    # Domain interaction features
    df_feat["carat_x_cut"] = df_feat["carat"] * df_feat["cut_num"]
    df_feat["carat_x_color"] = df_feat["carat"] * df_feat["color_num"]
    df_feat["carat_x_clarity"] = df_feat["carat"] * df_feat["clarity_num"]

    # Geometric & physical features
    df_feat["volume"] = df_feat["x"] * df_feat["y"] * df_feat["z"]
    df_feat["carat_density"] = df_feat["carat"] / (df_feat["volume"] + 1e-5)
    df_feat["aspect_ratio"] = df_feat["x"] / (df_feat["y"] + 1e-5)
    df_feat["depth_ratio"] = df_feat["z"] / (df_feat["x"] + 1e-5)

    return df_feat

def evaluate_single_fit(X_tr, y_tr, X_val, y_val, cat_cols, num_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    X_tr_enc = preprocessor.fit_transform(X_tr)
    X_val_enc = preprocessor.transform(X_val)

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.07,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_tr_enc, y_tr)
    preds = model.predict(X_val_enc)

    mae = float(mean_absolute_error(y_val, preds))
    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    r2 = float(r2_score(y_val, preds))
    return mae, rmse, r2

def cv_evaluate_feature_set(train_df, cat_cols, num_cols, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    maes, rmses, r2s = [], [], []

    feature_cols = cat_cols + num_cols
    X = train_df[feature_cols]
    y = train_df["price"]

    for tr_idx, val_idx in kf.split(train_df):
        X_tr_f, y_tr_f = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val_f, y_val_f = X.iloc[val_idx], y.iloc[val_idx]

        m, r, r2_val = evaluate_single_fit(X_tr_f, y_tr_f, X_val_f, y_val_f, cat_cols, num_cols)
        maes.append(m)
        rmses.append(r)
        r2s.append(r2_val)

    return {
        "mean_mae": float(np.mean(maes)),
        "std_mae": float(np.std(maes)),
        "mean_rmse": float(np.mean(rmses)),
        "std_rmse": float(np.std(rmses)),
        "mean_r2": float(np.mean(r2s)),
        "std_r2": float(np.std(r2s)),
    }

def main():
    print("=" * 70)
    print("EXPERIMENT 4: RIGOROUS FEATURE ENGINEERING & INTERACTION ABLATION")
    print("=" * 70)

    df_raw = load_clean_data()
    df_all = add_engineered_features(df_raw)

    # 1. Strict Train / Test Split (Untouched Test Set)
    train_df, test_df = train_test_split(df_all, test_size=0.2, random_state=RANDOM_SEED)
    print(f"Total Records: {len(df_all)}")
    print(f"Training Partition (used for CV feature selection): N = {len(train_df)}")
    print(f"Held-out Test Partition (untouched during selection): N = {len(test_df)}")

    cat_base = ["cut", "color", "clarity"]
    num_base = ["carat", "depth", "table", "x", "y", "z"]

    configs = {
        "Config 0: Baseline (Standard 9 Features)": {
            "cat": cat_base,
            "num": num_base,
        },
        "Config 1: Baseline + Physical Volume (x*y*z)": {
            "cat": cat_base,
            "num": num_base + ["volume"],
        },
        "Config 2: Baseline + Carat-Quality Interactions": {
            "cat": cat_base,
            "num": num_base + ["carat_x_cut", "carat_x_color", "carat_x_clarity"],
        },
        "Config 3: Baseline + Carat x Clarity Only": {
            "cat": cat_base,
            "num": num_base + ["carat_x_clarity"],
        },
        "Config 4: Baseline + Geometry (Volume, Aspect, Depth Ratio)": {
            "cat": cat_base,
            "num": num_base + ["volume", "aspect_ratio", "depth_ratio"],
        },
        "Config 5: Full Engineered Set (All Candidate Features)": {
            "cat": cat_base,
            "num": num_base + [
                "volume", "carat_density", "aspect_ratio", "depth_ratio",
                "carat_x_cut", "carat_x_color", "carat_x_clarity",
            ],
        },
    }

    # 2. Cross-Validation on Training Split Only
    print("\n" + "=" * 50)
    print("STAGE 1: 5-FOLD CROSS-VALIDATION ON TRAINING DATA ONLY")
    print("=" * 50)

    cv_results = []
    base_cv_mae = None

    for name, cfg in configs.items():
        cat_c = cfg["cat"]
        num_c = cfg["num"]
        total_feats = len(cat_c) + len(num_c)
        print(f"Running 5-fold CV on Training split for: {name} ({total_feats} features) ...")

        cv_metrics = cv_evaluate_feature_set(train_df, cat_c, num_c, n_splits=5)
        mean_mae = cv_metrics["mean_mae"]

        if base_cv_mae is None:
            base_cv_mae = mean_mae
            delta_cv_mae = 0.0
        else:
            delta_cv_mae = mean_mae - base_cv_mae

        print(f"  CV Train MAE: ${mean_mae:.2f} (±{cv_metrics['std_mae']:.2f}) | Δ vs Baseline: ${delta_cv_mae:+.2f}")

        cv_results.append({
            "Configuration": name,
            "Total_Features": total_feats,
            "Train_CV_MAE": round(mean_mae, 2),
            "Train_CV_MAE_Std": round(cv_metrics["std_mae"], 2),
            "Train_CV_RMSE": round(cv_metrics["mean_rmse"], 2),
            "Train_CV_R2": round(cv_metrics["mean_r2"], 6),
            "Delta_CV_MAE": round(delta_cv_mae, 2),
            "Decision": "Baseline" if delta_cv_mae == 0 else ("Candidate Improvement" if delta_cv_mae < -2.0 else "Marginal / Redundant"),
            "cat": cat_c,
            "num": num_c,
        })

    cv_df = pd.DataFrame(cv_results)
    print("\n" + "=" * 70)
    print("TRAINING-ONLY 5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 70)
    print(cv_df[["Configuration", "Total_Features", "Train_CV_MAE", "Train_CV_RMSE", "Train_CV_R2", "Delta_CV_MAE", "Decision"]].to_string(index=False))

    # Identify Winning Configuration strictly from Training CV
    best_config_row = cv_df.sort_values("Train_CV_MAE").iloc[0]
    best_config_name = best_config_row["Configuration"]
    print(f"\nWinning Configuration Selected based on Training CV MAE: {best_config_name}")

    # 3. Final Evaluation on Untouched Test Set
    print("\n" + "=" * 50)
    print("STAGE 2: FINAL EVALUATION ON UNTOUCHED TEST SET (N = 10,784)")
    print("=" * 50)

    # A. Baseline on untouched test set
    base_cfg = configs["Config 0: Baseline (Standard 9 Features)"]
    base_test_mae, base_test_rmse, base_test_r2 = evaluate_single_fit(
        train_df[base_cfg["cat"] + base_cfg["num"]], train_df["price"],
        test_df[base_cfg["cat"] + base_cfg["num"]], test_df["price"],
        base_cfg["cat"], base_cfg["num"]
    )
    print(f"Baseline (Config 0) Untouched Test Set Evaluation:")
    print(f"  Test MAE: ${base_test_mae:.2f} | Test RMSE: ${base_test_rmse:.2f} | Test R²: {base_test_r2:.6f}")

    # B. Selected Winner on untouched test set
    win_cat = best_config_row["cat"]
    win_num = best_config_row["num"]
    win_test_mae, win_test_rmse, win_test_r2 = evaluate_single_fit(
        train_df[win_cat + win_num], train_df["price"],
        test_df[win_cat + win_num], test_df["price"],
        win_cat, win_num
    )
    test_delta_mae = win_test_mae - base_test_mae
    test_delta_rmse = win_test_rmse - base_test_rmse

    print(f"\nSelected Winner ({best_config_name}) Untouched Test Set Evaluation:")
    print(f"  Test MAE: ${win_test_mae:.2f} (Δ ${test_delta_mae:+.2f}) | Test RMSE: ${win_test_rmse:.2f} (Δ ${test_delta_rmse:+.2f}) | Test R²: {win_test_r2:.6f}")

    # Also evaluate Config 2 (Carat-Quality Interactions) directly on test set for academic comparison
    cfg2 = configs["Config 2: Baseline + Carat-Quality Interactions"]
    c2_test_mae, c2_test_rmse, c2_test_r2 = evaluate_single_fit(
        train_df[cfg2["cat"] + cfg2["num"]], train_df["price"],
        test_df[cfg2["cat"] + cfg2["num"]], test_df["price"],
        cfg2["cat"], cfg2["num"]
    )

    final_comparison_table = pd.DataFrame([
        {
            "Configuration": "Config 0: Baseline (Standard 9 Features)",
            "Selection_Stage": "Baseline",
            "Train_CV_MAE": round(cv_df.loc[cv_df['Configuration'] == 'Config 0: Baseline (Standard 9 Features)', 'Train_CV_MAE'].values[0], 2),
            "Untouched_Test_MAE": round(base_test_mae, 2),
            "Untouched_Test_RMSE": round(base_test_rmse, 2),
            "Untouched_Test_R2": round(base_test_r2, 6),
            "Test_Delta_MAE": 0.0,
        },
        {
            "Configuration": "Config 2: Baseline + Carat-Quality Interactions",
            "Selection_Stage": "Candidate (Domain Literature)",
            "Train_CV_MAE": round(cv_df.loc[cv_df['Configuration'] == 'Config 2: Baseline + Carat-Quality Interactions', 'Train_CV_MAE'].values[0], 2),
            "Untouched_Test_MAE": round(c2_test_mae, 2),
            "Untouched_Test_RMSE": round(c2_test_rmse, 2),
            "Untouched_Test_R2": round(c2_test_r2, 6),
            "Test_Delta_MAE": round(c2_test_mae - base_test_mae, 2),
        },
        {
            "Configuration": f"Winner: {best_config_name}",
            "Selection_Stage": "Selected by Training CV",
            "Train_CV_MAE": round(best_config_row["Train_CV_MAE"], 2),
            "Untouched_Test_MAE": round(win_test_mae, 2),
            "Untouched_Test_RMSE": round(win_test_rmse, 2),
            "Untouched_Test_R2": round(win_test_r2, 6),
            "Test_Delta_MAE": round(test_delta_mae, 2),
        },
    ])

    print("\n" + "=" * 70)
    print("FINAL AUDITED RECONCILIATION TABLE")
    print("=" * 70)
    print(final_comparison_table.to_string(index=False))

    # Save CSV and JSON
    cv_export_df = cv_df.drop(columns=["cat", "num"])
    cv_export_df.to_csv(os.path.join(OUTPUT_DIR, "feature_selection_cv_summary.csv"), index=False)
    final_comparison_table.to_csv(os.path.join(OUTPUT_DIR, "feature_ablation_final_test.csv"), index=False)

    full_results = {
        "metadata": {
            "total_records": len(df_all),
            "train_size": len(train_df),
            "test_size": len(test_df),
            "random_seed": RANDOM_SEED,
            "selection_criterion": "Lowest 5-Fold Cross-Validation MAE on Training Partition Only",
        },
        "training_cv_results": cv_export_df.to_dict(orient="records"),
        "selected_configuration": {
            "name": best_config_name,
            "total_features": int(best_config_row["Total_Features"]),
            "train_cv_mae": float(best_config_row["Train_CV_MAE"]),
            "untouched_test_mae": float(win_test_mae),
            "untouched_test_rmse": float(win_test_rmse),
            "untouched_test_r2": float(win_test_r2),
            "test_delta_mae_vs_baseline": float(test_delta_mae),
        },
        "baseline_evaluation": {
            "untouched_test_mae": float(base_test_mae),
            "untouched_test_rmse": float(base_test_rmse),
            "untouched_test_r2": float(base_test_r2),
        },
    }

    with open(os.path.join(OUTPUT_DIR, "feature_ablation_results.json"), "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\nAll audited feature ablation results written to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
