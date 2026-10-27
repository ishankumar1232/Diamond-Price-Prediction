"""
Research Gap 3: Fair Multi-Model Comparison & Benchmarking
=========================================================
Addresses literature gaps across:
- Kigo et al. (2023, Sci. Rep.): Supervised ML benchmark on diamond pricing
- Kamila et al. (2024, CogITo): Comparison of KNN, MLP, and Tree models
- Amadavadi et al. (2024, ICOSEC): Ensemble model comparison (RF, Extra Trees, GBR, XGB)
- Randhawa et al. (2025, Springer CCIS): Benchmarking tree models against linear regression
- Acharya & Senapati (2024, Springer): Comparison of data mining algorithms

Methodology:
- Strictly identical 80/20 train/test split (Seed 42, N_train=43,136, N_test=10,784).
- Preprocessing fit ONLY on training data (One-Hot for categories, StandardScaler for linear models).
- 5-Fold Cross Validation performed on training data only.
- Final test evaluation performed once on untouched test set.
- Records MAE, RMSE, R², and fit time.
"""

import json
import os
import sys
import time

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Candidate models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
import xgboost as xgb
from catboost import CatBoostRegressor

RANDOM_SEED = 42

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, "..", "..", "data"))

def load_data():
    path = os.path.join(DATA_DIR, "diamonds (1).csv")
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Exclude physically impossible zero dimensions
    valid_mask = ~((df["x"] <= 0) | (df["y"] <= 0) | (df["z"] <= 0))
    df_clean = df[valid_mask].copy().reset_index(drop=True)

    feat_cols = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]
    X = df_clean[feat_cols]
    y = df_clean["price"]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

def main():
    print("=" * 70)
    print("EXPERIMENT 3: FAIR MULTI-MODEL BENCHMARKING")
    print("=" * 70)

    X_train, X_test, y_train, y_test = load_data()
    print(f"Dataset: N_train={len(X_train)}, N_test={len(X_test)}")

    cat_cols = ["cut", "color", "clarity"]
    num_cols = ["carat", "depth", "table", "x", "y", "z"]

    # Preprocessors
    prep_tree = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    prep_linear = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    # Pre-transform training and test sets
    X_train_tree = prep_tree.fit_transform(X_train)
    X_test_tree = prep_tree.transform(X_test)

    X_train_linear = prep_linear.fit_transform(X_train)
    X_test_linear = prep_linear.transform(X_test)

    # Models definition
    models = {
        "Linear Regression": (LinearRegression(), X_train_linear, X_test_linear),
        "Ridge Regression (alpha=1.0)": (Ridge(alpha=1.0, random_state=RANDOM_SEED), X_train_linear, X_test_linear),
        "Decision Tree Regressor": (DecisionTreeRegressor(max_depth=15, min_samples_leaf=5, random_state=RANDOM_SEED), X_train_tree, X_test_tree),
        "Random Forest Regressor (200 trees)": (RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_SEED), X_train_tree, X_test_tree),
        "Extra Trees Regressor (200 trees)": (ExtraTreesRegressor(n_estimators=200, max_depth=20, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_SEED), X_train_tree, X_test_tree),
        "Gradient Boosting Regressor": (GradientBoostingRegressor(n_estimators=300, learning_rate=0.08, max_depth=5, random_state=RANDOM_SEED), X_train_tree, X_test_tree),
        "XGBoost Regressor": (xgb.XGBRegressor(n_estimators=600, learning_rate=0.06, max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=RANDOM_SEED), X_train_tree, X_test_tree),
    }

    results = []
    trained_models = {}

    for name, (model, X_tr, X_te) in models.items():
        print(f"\nEvaluating: {name} ...")
        t0 = time.time()
        
        # 5-fold CV on training set
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv_scores = cross_val_score(model, X_tr, y_train, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1)
        cv_mae = -np.mean(cv_scores)

        # Fit on full training set
        model.fit(X_tr, y_train)
        fit_time = time.time() - t0

        # Evaluate on untouched test set
        preds = model.predict(X_te)
        test_mae = mean_absolute_error(y_test, preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, preds))
        test_r2 = r2_score(y_test, preds)

        trained_models[name] = preds

        results.append({
            "Model": name,
            "CV_Train_MAE": round(cv_mae, 2),
            "Test_MAE": round(test_mae, 2),
            "Test_RMSE": round(test_rmse, 2),
            "Test_R2": round(test_r2, 6),
            "Fit_Time_Sec": round(fit_time, 2),
        })
        print(f"  CV MAE: ${cv_mae:.2f} | Test MAE: ${test_mae:.2f} | Test RMSE: ${test_rmse:.2f} | Test R²: {test_r2:.6f} | Time: {fit_time:.1f}s")

    # Native CatBoost Regressor
    print("\nEvaluating: CatBoost Regressor ...")
    t0 = time.time()
    cat_model = CatBoostRegressor(
        iterations=600,
        learning_rate=0.06,
        depth=6,
        cat_features=cat_cols,
        random_seed=RANDOM_SEED,
        verbose=0,
    )
    cat_model.fit(X_train, y_train)
    fit_time = time.time() - t0
    cat_preds = cat_model.predict(X_test)
    cat_mae = mean_absolute_error(y_test, cat_preds)
    cat_rmse = np.sqrt(mean_squared_error(y_test, cat_preds))
    cat_r2 = r2_score(y_test, cat_preds)
    trained_models["CatBoost Regressor"] = cat_preds

    results.append({
        "Model": "CatBoost Regressor",
        "CV_Train_MAE": None,
        "Test_MAE": round(cat_mae, 2),
        "Test_RMSE": round(cat_rmse, 2),
        "Test_R2": round(cat_r2, 6),
        "Fit_Time_Sec": round(fit_time, 2),
    })
    print(f"  Test MAE: ${cat_mae:.2f} | Test RMSE: ${cat_rmse:.2f} | Test R²: {cat_r2:.6f} | Time: {fit_time:.1f}s")

    # CatBoost + XGBoost Ensemble
    print("\nEvaluating: CatBoost + XGBoost Ensemble (0.55 / 0.45) ...")
    ens_preds = 0.55 * cat_preds + 0.45 * trained_models["XGBoost Regressor"]
    ens_mae = mean_absolute_error(y_test, ens_preds)
    ens_rmse = np.sqrt(mean_squared_error(y_test, ens_preds))
    ens_r2 = r2_score(y_test, ens_preds)

    results.append({
        "Model": "CatBoost + XGBoost Ensemble (0.55 / 0.45)",
        "CV_Train_MAE": None,
        "Test_MAE": round(ens_mae, 2),
        "Test_RMSE": round(ens_rmse, 2),
        "Test_R2": round(ens_r2, 6),
        "Fit_Time_Sec": round(fit_time + results[-2]["Fit_Time_Sec"], 2),
    })
    print(f"  Test MAE: ${ens_mae:.2f} | Test RMSE: ${ens_rmse:.2f} | Test R²: {ens_r2:.6f}")

    results_df = pd.DataFrame(results).sort_values(by="Test_MAE")
    print("\n" + "=" * 70)
    print("FINAL MODEL BENCHMARK TABLE (Ranked by Test MAE)")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # Save exports
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_benchmark_comparison.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "model_benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

