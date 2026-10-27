"""
Controlled Ablation Experiment: Investigating Missingness and Selection Bias in Diamond Origin Classification
==========================================================================================================
Addresses User Audit Point 5:
Compares classification performance across feature subsets to isolate whether high nominal
classification accuracy (GIA vs Lab-Grown) is driven by:
1. Full Feature Set (including missingness signals in Culet and Fluorescence)
2. Ablated Feature Set (removing Culet and Fluorescence which have >70% missingness)
3. Core 4Cs & Physical Dimensions Only
4. Carat Weight Alone (evaluating pure inventory selection bias)
5. 1-Carat Bracket Controlled Evaluation (0.95 - 1.05 ct, controlling for carat weight)
"""

import json
import os
import sys

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

RANDOM_SEED = 42

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, "..", "..", "data"))

def main():
    print("=" * 70)
    print("EXPERIMENT: ORIGIN CLASSIFIER CONTROLLED ABLATION & BIAS AUDIT")
    print("=" * 70)

    csv_path = os.path.join(DATA_DIR, "diamonds (cleaned).csv")
    df = pd.read_csv(csv_path)

    # Exclude 5 rows with missing core properties
    df_clean = df.dropna(subset=["Carat Weight", "Shape", "Color", "Clarity", "Price"]).reset_index(drop=True)
    print(f"Total dataset rows: {len(df)}")
    print(f"Records with valid core attributes: {len(df_clean)}")
    print("Type breakdown in valid records:")
    print(df_clean["Type"].value_counts())

    # Target: 1 = Lab-Grown (GIA Lab-Grown or IGI Lab-Grown), 0 = GIA (traditional natural proxy)
    y = (df_clean["Type"] != "GIA").astype(int)

    train_df, test_df, y_train, y_test = train_test_split(
        df_clean, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    results = []

    def evaluate_configuration(config_name, cat_cols, num_cols, train_data, test_data, y_tr, y_te, note=""):
        transformers = []
        if cat_cols:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
            ])
            transformers.append(("cat", cat_pipe, cat_cols))
        if num_cols:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ])
            transformers.append(("num", num_pipe, num_cols))

        preprocessor = ColumnTransformer(transformers=transformers)
        clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1)
        pipeline = Pipeline([
            ("prep", preprocessor),
            ("clf", clf),
        ])

        features = cat_cols + num_cols
        pipeline.fit(train_data[features], y_tr)
        preds = pipeline.predict(test_data[features])
        probs = pipeline.predict_proba(test_data[features])[:, 1]

        acc = float(accuracy_score(y_te, preds))
        f1 = float(f1_score(y_te, preds))
        roc = float(roc_auc_score(y_te, probs))
        cm = confusion_matrix(y_te, preds).tolist()

        print(f"\nConfiguration: {config_name}")
        print(f"  Test Accuracy: {acc * 100:.2f}% | F1: {f1:.4f} | ROC-AUC: {roc:.4f}")
        print(f"  Confusion Matrix: {cm}")

        results.append({
            "Configuration": config_name,
            "Features_Count": len(features),
            "Test_Accuracy": round(acc, 4),
            "Test_F1_Score": round(f1, 4),
            "Test_ROC_AUC": round(roc, 4),
            "Confusion_Matrix": cm,
            "Evaluation_Sample": len(y_te),
            "Scientific_Note": note,
        })
        return pipeline

    # Config 1: Full Feature Set (includes Culet & Fluorescence missingness)
    cat_all = ["Shape", "Cut", "Color", "Clarity", "Polish", "Symmetry", "Girdle", "Culet", "Fluorescence"]
    num_all = ["Carat Weight", "Length/Width Ratio", "Depth %", "Table %", "Length", "Width", "Height"]
    evaluate_configuration(
        "Config 1: Full Feature Set (All Available Attributes)",
        cat_all, num_all, train_df, test_df, y_train, y_test,
        note="Includes Culet (70.8% null) and Fluorescence (81.2% null) missingness flags."
    )

    # Config 2: Ablated Set (Remove Culet and Fluorescence)
    cat_no_missing = ["Shape", "Cut", "Color", "Clarity", "Polish", "Symmetry", "Girdle"]
    evaluate_configuration(
        "Config 2: Missingness-Prone Features Removed",
        cat_no_missing, num_all, train_df, test_df, y_train, y_test,
        note="Culet and Fluorescence excluded; tests sensitivity to scraper reporting gaps."
    )

    # Config 3: Standard 4Cs and Physical Dimensions Only
    cat_4c = ["Shape", "Cut", "Color", "Clarity"]
    num_4c = ["Carat Weight", "Length", "Width", "Height", "Depth %", "Table %"]
    evaluate_configuration(
        "Config 3: Standard 4Cs & Dimensions Only",
        cat_4c, num_4c, train_df, test_df, y_train, y_test,
        note="Excludes finishing grades (Polish, Symmetry, Girdle); retains only fundamental 4Cs and geometry."
    )

    # Config 4: Carat Weight Alone
    evaluate_configuration(
        "Config 4: Carat Weight Alone",
        [], ["Carat Weight"], train_df, test_df, y_train, y_test,
        note="Demonstrates how much classification power stems purely from inventory selection bias (GIA mean 1.06 ct vs Lab mean 1.42 ct)."
    )

    # Config 5: Controlled Carat Bracket Evaluation (0.95 - 1.05 ct)
    df_1ct = df_clean[(df_clean["Carat Weight"] >= 0.95) & (df_clean["Carat Weight"] <= 1.05)].reset_index(drop=True)
    y_1ct = (df_1ct["Type"] != "GIA").astype(int)
    tr_1ct, te_1ct, y_tr_1ct, y_te_1ct = train_test_split(
        df_1ct, y_1ct, test_size=0.25, random_state=RANDOM_SEED, stratify=y_1ct
    )
    evaluate_configuration(
        "Config 5: Matched 1.0-Carat Bracket (Core 4Cs Only)",
        cat_4c, ["Length", "Width", "Height", "Depth %", "Table %"], tr_1ct, te_1ct, y_tr_1ct, y_te_1ct,
        note="Controls for carat weight by evaluating solely within the 0.95-1.05 ct window (N=3,757 diamonds)."
    )

    summary_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("CLASSIFIER ABLATION SUMMARY")
    print("=" * 70)
    print(summary_df[["Configuration", "Features_Count", "Test_Accuracy", "Test_F1_Score", "Test_ROC_AUC", "Evaluation_Sample"]].to_string(index=False))

    summary_df.to_csv(os.path.join(OUTPUT_DIR, "origin_classifier_ablation.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "origin_classifier_ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAblation results written to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

