# DiamondIQ Project Audit & Baseline Verification

**Author:** DiamondIQ Research Team  
**Date:** October 2026  
**Status:** Verified against actual repository artifacts and raw datasets  

---

## 1. Executive Summary

This audit assesses the state of the DiamondIQ repository prior to new experimental implementations. The objective is to verify dataset provenance, assess data quality, evaluate existing models against reported baselines, identify discrepancies, and separate verified scientific evidence from unverified assumptions.

---

## 2. Repository Architecture & Model Artifacts

| Component | Path | Format | Role |
| :--- | :--- | :--- | :--- |
| **Standard Dataset Model** | `models/catboost_diamond_model.cbm` | Native CatBoost Binary | 1,000-iteration gradient boosted decision tree |
| **Standard Dataset Model** | `models/xgboost_diamond_model.pkl` | Pickled `XGBRegressor` | Preprocessed gradient boosted decision tree |
| **Standard Preprocessor** | `models/diamond_preprocessor.pkl` | Scikit-learn `ColumnTransformer` | OHE for `cut, color, clarity`; Passthrough for numericals |
| **Ensemble Config** | `models/ensemble_config.pkl` | Pickled Dict | Weights: $0.55 \times \text{CatBoost} + 0.45 \times \text{XGBoost}$ |
| **Detailed Dataset Model** | `models/diamond_price_rf_pipeline.pkl` | Scikit-learn `Pipeline` | ColumnTransformer + `RandomForestRegressor` (200 trees) |
| **Model Metadata** | `models/model_metadata.pkl` | Pickled Dict | Records hyperparameters and test metrics for Detailed RF |
| **Web Server** | `app.py` | Python (Flask 3.1) | Production server, prediction routes, PDF generation |

---

## 3. Dataset Audit & Provenance

### A. Standard Dataset (`data/diamonds (1).csv`)
* **Total Records:** 53,940 rows, 10 attributes (`carat, cut, color, clarity, depth, table, price, x, y, z`).
* **Target Variable:** `price` (USD), range \$326 – \$18,823, mean \$3,932.80, median \$2,401.00.
* **Physical Dimension Validation:**
  * Exactly **20 records** contain physically impossible dimensions where $x=0$, $y=0$, or $z=0$.
  * Examples: row 2207 ($z=0$), row 11182 ($x=0$), row 11963 ($x=0, y=0, z=0$).
  * Excluding these invalid rows yields **53,920 physically valid records**.
* **Identical Feature Profiles (Data Redundancy):**
  * **685 records** share identical 9-feature input profiles across **339 distinct diamond profiles**.
  * **198 of these profiles** exhibit price variation (non-zero standard deviation), with a maximum price divergence of **\$1,447** for identically graded diamonds.
  * *Research Implication:* In standard random splits, identical profiles appear in both train and test partitions, creating an optimistic bias in evaluation metrics.

### B. Detailed Dataset (`data/diamonds (cleaned).csv`)
* **Total Records:** 6,485 rows, 18 attributes.
* **Target Variable:** `Price` (USD), range \$1,010 – \$39,460.
* **Missing Value Analysis:**
  * `Cut`: 4,303 missing (66.35%)
  * `Culet`: 4,594 missing (70.84%)
  * `Fluorescence`: 5,264 missing (81.17%)
  * `Shape`, `Color`, `Clarity`, `Carat Weight`, `Length/Width Ratio`: 5 missing (0.08%)
  * `Length`, `Width`, `Height`: 20 missing (0.31%)
* **Geometric Inconsistencies:**
  * **1,308 records** display a discrepancy $> 0.01$ between the recorded `Length/Width Ratio` and the calculated ratio ($\text{Length} / \text{Width}$), with discrepancies reaching up to $0.498$.
* **Classification of `Type`:**
  * Breakdown: GIA Natural (3,355), IGI Lab-Grown (1,876), GIA Lab-Grown (1,254).
  * Mean price by Type:
    * GIA Natural: **\$4,890.97**
    * GIA Lab-Grown: **\$2,705.93**
    * IGI Lab-Grown: **\$1,644.81**
  * *Research Implication:* Predicting origin (`Type`) using physical 4C parameters without spectral gemological data suffers from dataset confounding and target leakage.

---

## 4. Baseline Metric Reproducibility

Using the verified 80/20 train/test split ($N_{\text{train}} = 43,136$, $N_{\text{test}} = 10,784$, `random_state=42`) with zero-dimension records removed:

| Pipeline | Model Configuration | Verified Test MAE | Verified Test RMSE | Verified Test $R^2$ |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Dataset** | CatBoost Regressor ($w=0.55$) | \$260.85 | \$477.84 | 0.9858 (98.58%) |
| **Standard Dataset** | XGBoost Regressor ($w=0.45$) | \$207.03 | \$378.31 | 0.9911 (99.11%) |
| **Standard Dataset** | **CatBoost + XGBoost Ensemble** | **\$231.25** | **\$422.35** | **0.9889 (98.89%)** |
| **Detailed Dataset** | **Tuned Random Forest Pipeline** | **\$388.72** | **\$1,103.41** | **0.8806 (88.06%)** |

### Identified Inconsistency in `app.py`:
* `app.py` previously defined `STANDARD_MAE = 257.31`.
* The actual reproduced ensemble test MAE on the frozen test set is **\$231.25**.
* `app.py` has been updated to reflect the true reproduced value (\$231.25).

---

## 5. Review of Completed Notebook Experiments

| Notebook | Focus | Key Findings & Artifacts |
| :--- | :--- | :--- |
| `Diamond_Price_Standard_53940.ipynb` | Standard dataset pipeline | Audited zero dimensions; fit CatBoost & XGBoost; computed ensemble weights; generated error CSV files |
| `Diamond_Price_Prediction_Research_Final.ipynb` | Detailed dataset pipeline | Extensive EDA; 6-model comparison; 5-fold CV; TreeSHAP feature importance; What-If scenario simulations |
| `carat_range_error_analysis.csv` | Error by carat band | Confirms heteroscedasticity: MAE is \$65.89 for $<0.5$ ct, increasing to \$435+ for large carats |
| `clarity_error_analysis.csv` | Error by clarity grade | Highest MAE in I1 (\$435.59) and SI2 (\$396.88) due to internal inclusion variance |

---

## 6. Status of Research Claims

| Claim / Feature | Current Status | Scientific Evidence & Remediation |
| :--- | :--- | :--- |
| **High $R^2$ on Standard Dataset ($>98\%$)** | **Supported** | Validated: CatBoost+XGBoost achieves 98.89% $R^2$ on standard random holdout. |
| **Robustness under Profile Leakage** | **Unproven** | Existing benchmarks use random splitting. Grouped validation is needed to test leakage. |
| **Feature Interaction Benefits** | **Partially Supported** | Domain interactions ($\text{carat} \times \text{cut}$, $\text{carat} \times \text{clarity}$) require ablation testing against baseline. |
| **Explainability via SHAP** | **Partially Supported** | App currently computes 1-at-a-time difference from reference diamond; genuine TreeSHAP values must be isolated and distinguished from causality. |
| **Origin Classification (Natural vs Lab-Grown)** | **Unsupported / Confounded** | Physical measurements alone cannot certify diamond genesis without gemological spectroscopy; `Type` reflects dataset tagging. Must not be deployed as an unverified user-facing tool. |

---

## 7. Next Actions

1. Preserve production models unchanged in `models/`.
2. Clean public web app to keep it simple, fast, and college-ready.
3. Conduct 6 isolated research experiments in `research/` addressing the 10 literature survey papers.

