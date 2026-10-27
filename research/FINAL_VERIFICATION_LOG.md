# DiamondIQ Final Evidence Verification Log & Audit Trail

**Project:** DiamondIQ — Diamond Price Prediction  
**Date of Audit:** October 2026  
**Auditor:** Senior ML Research Engineer / DiamondIQ Mentorship Audit  
**Operating Environment:** Python 3.13, Windows 11, Scikit-Learn 1.9.1, XGBoost 3.4.1, CatBoost 1.2.10, SHAP 0.52.0, Pandas 3.0.6, Flask 3.1.3  

---

## 1. Executive Summary & Verification Methodology

Every numerical claim, metric table, split count, and gemological reference in `DIAMONDIQ_RESEARCH_REPORT.md` was subjected to an independent audit against the underlying datasets, Jupyter notebooks, serialized model artifacts, and test scripts. 

### Core Audit Principles Enforced:
1. **Zero Silent Substitutions:** Never conflate distinct evaluation protocols (e.g., holdout vs. cross-validation; training error vs. test error; exact-row deduplication vs. input-profile grouping).
2. **Provenance Traceability:** Every metric must trace directly to a reproducible execution command, input CSV row count, and serialized JSON/CSV artifact on disk.
3. **Gemological Reality vs. Tabular Artifacts:** Clearly delineate what is physically grounded in diamond science (crystal lattice properties, spectroscopy requirements) versus what is an artifact of web-scraped inventory tables (selection bias, reporting missingness).
4. **Preservation of Public Production Code:** The public DiamondIQ application (`app.py`, `templates/`, `static/`, and `models/`) was preserved untouched.

---

## 2. Step-by-Step Script Execution & Output Cross-Check

### Audit 2.1: Production Model Artifact Verification
* **Command:** Python script executing `models/catboost_diamond_model.cbm`, `models/xgboost_diamond_model.pkl`, and `models/diamond_preprocessor.pkl`.
* **Dataset:** `data/diamonds (1).csv`
* **Finding:** The saved models were trained in `notebooks/Diamond_Price_Prediction_Final_Model.ipynb` on an exact-duplicate-dropped split ($N_{\text{total}} = 53,794, N_{\text{train}} = 43,035, N_{\text{test}} = 10,759$, seed 42):
  * **Verified Held-Out Test Performance:** CatBoost $\text{MAE} = \$266.81, \text{RMSE} = \$497.26$; XGBoost $\text{MAE} = \$258.77, \text{RMSE} = \$508.66$; **Ensemble (0.55/0.45) $\text{MAE} = \$257.31, \text{RMSE} = \$492.73, R^2 = 0.984071$**.
  * **Profile Sharing Disclosure:** While exact 10-feature row overlap was 0.00% (no duplicate rows crossed the split), exactly **70 test rows (0.65%)** share identical 9-feature input profiles with training rows. This evaluation is therefore a documented held-out test split, but is **not strictly profile-disjoint**.
  * **The \$231.25 Leakage Finding:** When evaluated on a dimension-cleaned split ($N = 53,920$) without prior duplicate removal, index shifting caused **8,407 test rows (77.96% of the test partition) to overlap with the model's training data**. The lower MAE (\$231.25) was an artifact of severe index-shift data leakage.
* **Audit Action:** Excluded \$231.25 from independent test claims and relabeled it as an index-shifted evaluation artifact in Section 5.2. Verified \$257.31 as the true documented held-out test MAE.

---

### Audit 2.2: Data Quality & Invalid Dimension Audit
* **Script:** `research/data_quality/run_data_quality_audit.py`
* **Output Artifacts:** `data_quality_results.json`, `standard_zero_dimensions.csv`
* **Verified Values:**
  * Raw dataset rows: **53,940**.
  * Exactly **20 records** have non-positive dimensions ($x=0$: 8, $y=0$: 7, $z=0$: 20).
  * Valid clean rows: **53,920**.
  * Distinct 9-feature input profiles: **53,576**.
  * Repeated profile rows: **685** across **340 unique specifications** (198 with price variance, max spread \$1,447).
  * Impact of zero-dimension cleaning on held-out clean test records ($N = 10,784$):
    * Uncleaned Model A: Test $\text{MAE} = \$282.73, \text{RMSE} = \$551.29, R^2 = 0.981043$.
    * Cleaned Model B: Test $\text{MAE} = \$281.61, \text{RMSE} = \$548.45, R^2 = 0.981237$.
    * Difference: $\Delta \text{MAE} = -\$1.12, \Delta \text{RMSE} = -\$2.84, \Delta R^2 = +0.000194$.
  * **Correction on Invalid Record Evaluation:** Earlier reported \$215.96 MAE was a training memorization error because all 20 records were in Config A's training data. An honest 50/50 held-out evaluation ($N_{\text{test, zero}} = 10$) showed Model A test MAE = **\$1,542.45** and Model B test MAE = **\$2,527.22**.
  * **Application Response Verification:** Tested `client.post('/predict', data={'x': '0.0', ...})`. Returns **HTTP 200 OK** with inline validation feedback banner (corrected report from HTTP 400 claim).

---

### Audit 2.3: Paired Grouped Profile Validation Benchmark
* **Script:** `research/grouped_validation/run_grouped_validation.py`
* **Output Artifacts:** `grouped_validation_results.json`, `validation_method_comparison.csv`
* **Verified Values:**
  * **Overlap Distinction:**
    * Notebook split ($N = 53,794$): Exact 10-feature row overlap = 0 (0.00%); Identical 9-feature input profile overlap = 70 (0.65%).
    * Standard clean split ($N = 53,920$): Exact 10-feature row overlap = 46 (0.43%); Identical 9-feature input profile overlap = 117 (1.08%).
    * Grouped split (`GroupShuffleSplit`): Exact 10-feature row overlap = 0 (0.00%); Identical 9-feature input profile overlap = 0 (0.00%).
  * **Protocol 1 (Holdout Comparison):**
    * Method A1 (Random 80/20, 117 leaked rows): $\text{MAE} = \$273.44, \text{RMSE} = \$517.60, R^2 = 0.983289$.
    * Method A2 (Grouped 80/20, 0 leaked rows): $\text{MAE} = \$272.05, \text{RMSE} = \$528.19, R^2 = 0.982094$.
    * Difference: $\Delta \text{MAE} = -\$1.39, \Delta \text{RMSE} = +\$10.59, \Delta R^2 = -0.001195$.
  * **Protocol 2 (5-Fold Cross-Validation Comparison):**
    * Method B1 (Random 5-Fold, 555 leaked rows): Mean $\text{MAE} = \$278.37 \pm 4.85, \text{RMSE} = \$532.99 \pm 9.71, R^2 = 0.982123$.
    * Method B2 (Grouped 5-Fold, 0 leaked rows): Mean $\text{MAE} = \$278.87 \pm 3.92, \text{RMSE} = \$534.74 \pm 13.92, R^2 = 0.982000$.
    * Difference: $\Delta \text{MAE} = +\$0.50, \Delta \text{RMSE} = +\$1.75, \Delta R^2 = -0.000123$.

---

### Audit 2.4: Multi-Model Regression Benchmark
* **Script:** `research/model_comparison/run_model_comparison.py`
* **Output Artifacts:** `model_benchmark_results.json`, `model_benchmark_comparison.csv`
* **Verified Values ($N_{\text{train}} = 43,136, N_{\text{test}} = 10,784$, standardized 600 iterations/trees):**
  * CatBoost + XGBoost Ensemble: Test $\text{MAE} = \$273.00, \text{RMSE} = \$519.07, R^2 = 0.9832$
  * Extra Trees (200 trees): CV MAE \$285.89; Test $\text{MAE} = \$274.52, \text{RMSE} = \$546.72, R^2 = 0.9814$ (31.0 s)
  * CatBoost (600 iters): Test $\text{MAE} = \$277.29, \text{RMSE} = \$520.01, R^2 = 0.9831$ (61.2 s)
  * XGBoost (600 trees): CV MAE \$296.90; Test $\text{MAE} = \$287.09, \text{RMSE} = \$555.65, R^2 = 0.9807$
  * Random Forest (200 trees): CV MAE \$303.94; Test $\text{MAE} = \$287.72, \text{RMSE} = \$594.36, R^2 = 0.9780$ (29.4 s)
  * Gradient Boosting: CV MAE \$319.63; Test $\text{MAE} = \$309.93, \text{RMSE} = \$598.65, R^2 = 0.9776$
  * Decision Tree: CV MAE \$372.73; Test $\text{MAE} = \$357.59, \text{RMSE} = \$735.68, R^2 = 0.9662$ (5.3 s)
  * Ridge Regression ($\alpha=1.0$): CV MAE \$739.02; Test $\text{MAE} = \$725.79, \text{RMSE} = \$1,135.08, R^2 = 0.9196$
  * Multiple Linear Regression: CV MAE \$739.60; Test $\text{MAE} = \$726.30, \text{RMSE} = \$1,135.94, R^2 = 0.9195$

---

### Audit 2.5: Feature Engineering & Interaction Ablation
* **Script:** `research/feature_engineering/run_feature_ablation.py`
* **Output Artifacts:** `feature_ablation_results.json`, `feature_selection_cv_summary.csv`
* **Verified Values:**
  * **Stage 1 (Training-Only 5-Fold CV on $N = 43,136$):**
    * Config 0 (Baseline 9 feats): $\text{CV MAE} = \$298.40 \pm 4.03$
    * Config 1 (Baseline + Volume): $\text{CV MAE} = \$297.70$ ($\Delta = -\$0.70$, redundant)
    * Config 2 (Carat-Quality Interactions): $\text{CV MAE} = \$271.47$ ($\Delta = -\$26.93$, strong improvement)
    * Config 3 (Carat $\times$ Clarity Only): $\text{CV MAE} = \$277.95$ ($\Delta = -\$20.45$)
    * Config 4 (Geometry Ratios): $\text{CV MAE} = \$290.27$ ($\Delta = -\$8.13$)
    * Config 5 (Full Engineered Set): $\text{CV MAE} = \$261.37 \pm 3.85$ ($\Delta = -\$37.03$, **Selected Winner**)
  * **Stage 2 (Single Evaluation on Untouched Random Test, $N = 10,784$):**
    * Baseline (Config 0): Test $\text{MAE} = \$289.12, \text{RMSE} = \$558.03, R^2 = 0.980577$
    * Full Engineered Set (Config 5): Test $\text{MAE} = \$255.96, \text{RMSE} = \$503.78, R^2 = 0.984170$ ($\Delta = -\$33.16$ MAE, $-\$54.25$ RMSE)
  * **Grouped Validation Verification (0% profile leakage, $N_{\text{test}} = 10,783$):**
    * Baseline (Config 0): Grouped Test $\text{MAE} = \$283.04, \text{RMSE} = \$568.76, R^2 = 0.979237$
    * Carat-Quality Interactions (Config 2): Grouped Test $\text{MAE} = \$258.39, \text{RMSE} = \$518.18$ ($\Delta = -\$24.65$ MAE, $-\$50.58$ RMSE)
    * Full Engineered Set (Config 5): Grouped Test $\text{MAE} = \$253.88, \text{RMSE} = \$513.42$ ($\Delta = -\$29.16$ MAE, $-\$55.34$ RMSE)

---

### Audit 2.6: TreeSHAP Attribution & Local Explanations
* **Script:** `research/explainability/run_shap_analysis.py`
* **Output Artifacts:** `shap_explainability_results.json`, `shap_global_importance_raw.csv`
* **Verified Values ($N = 1,000$ test diamonds):**
  * Base Expected Value: **\$3,926.04**
  * Global Mean Absolute SHAP: $y$ (\$2,791.90, 42.5%), Carat (\$1,185.46, 18.0%), Clarity (\$1,135.67, 17.3%), Color (\$769.56, 11.7%), $z$ (\$348.35, 5.3%), $x$ (\$186.47, 2.8%), Cut (\$89.59, 1.4%), Depth (\$40.99, 0.6%), Table (\$19.34, 0.3%).
  * Case Study 1 (Budget, 0.31 ct, Very Good, I, SI1): Actual \$523, Pred \$473.61 ($y$: $-\$1,545.02$, carat: $-\$1,002.93$, color I: $-\$490.05$).
  * Case Study 2 (Mid-tier, 1.00 ct, Premium, G, VS2): Actual \$6,048, Pred \$5,793.39 (carat: $+\$1,018.61$, clarity VS2: $+\$726.98$, color G: $-\$386.48$).
  * Case Study 3 (Luxury, 2.02 ct, Good, I, SI2): Actual \$12,927, Pred \$13,318.68 ($y$: $+\$6,549.59$, carat: $+\$4,066.72$, color I: $-\$1,189.37$).

---

### Audit 2.7: Detailed Dataset Audit & Classifier Controlled Ablation
* **Script:** `research/market_analysis/run_origin_analysis.py` and `research/market_analysis/run_origin_ablation.py`
* **Output Artifacts:** `origin_classifier_ablation_results.json`, `origin_classifier_ablation.csv`, `type_price_statistics.csv`
* **Verified Values:**
  * Raw dataset rows: **6,485** (`GIA` = 3,355; `IGI Lab-Grown` = 1,876; `GIA Lab-Grown` = 1,254).
  * Carat-valid rows: **6,480** (`GIA` = 3,352; `IGI Lab-Grown` = 1,874; `GIA Lab-Grown` = 1,254).
  * Missing value rates: Cut (66.35%), Culet (70.84%), Fluorescence (81.17%). Discrepancies in Length/Width ratio > 0.01: 1,308 rows (20.17%).
  * Observed listing price differences across carat brackets:
    * 0.75 - 1.25 ct: GIA \$4,197.72 vs. Lab \$1,160.50 (72.35% difference)
    * 1.25 - 1.75 ct: GIA \$10,252.22 vs. Lab \$2,117.14 (79.35% difference)
    * 1.75 - 2.50 ct: GIA \$24,885.31 vs. Lab \$4,145.98 (83.34% difference)
    * > 2.50 ct: GIA \$36,856.00 vs. Lab \$7,130.20 (80.65% difference)
  * **Classifier Controlled Ablation Results ($N_{\text{test}} = 1,296$):**
    * Config 1 (Full Features): Accuracy = **98.15%**, $F_1$ = **0.9806**, ROC-AUC = **0.9991**
    * Config 2 (Missingness dropped: Culet & Fluorescence excluded): Accuracy = **97.07%**, $F_1$ = **0.9693**
    * Config 3 (Standard 4Cs & Dimensions only): Accuracy = **96.45%**, $F_1$ = **0.9631**
    * Config 4 (Carat Weight Alone): Accuracy = **82.64%**, $F_1$ = **0.8221**, ROC-AUC = **0.8904** (**Proves massive selection bias**)
    * Config 5 (Matched 1.0-ct bracket, 0.95–1.05 ct, Core 4Cs only): Accuracy = **94.89%**, $F_1$ = **0.8943**

---

## 3. Comprehensive Summary: Verified, Corrected, and Unreproduced Claims

| Category | Claim / Topic | Status | Evidence & Resolution |
| :--- | :--- | :---: | :--- |
| **Verified** | Production Ensemble Documented Held-Out Performance | **VERIFIED** | $\text{MAE} = \$257.31, \text{RMSE} = \$492.73, R^2 = 0.984071$ on documented held-out test partition ($N=10,759$, 0% exact row overlap, disclosing 70 shared 9-feature input profiles / 0.65%). |
| **Verified** | Geometric Cleaning Data Impact | **VERIFIED** | 53,940 raw $\rightarrow$ 20 zero-dim rows $\rightarrow$ 53,920 clean; improves clean test RMSE by \$2.84 (\$548.45 vs \$551.29). |
| **Verified** | Feature Interactions under Grouped Split | **VERIFIED** | Config 5 improves Grouped Test MAE by **-\$29.16** (\$253.88 vs \$283.04) and RMSE by **-\$55.34** with 0% profile overlap. |
| **Verified** | Origin Classification Selection Bias | **VERIFIED** | Carat weight alone predicts dataset tags with **82.64% accuracy** due to scraper inventory bias (GIA mean 1.06 ct vs Lab mean 1.42 ct). |
| **Corrected** | $231.25 Production Metric | **CORRECTED** | Identified as index-shift data leakage (8,407 test rows overlapped with training data). Removed from independent reporting; verified independent MAE is **\$257.31**. |
| **Corrected** | Exact-Row vs. Input-Profile Overlap | **CORRECTED** | Clarified that deduplication eliminated exact 10-feature duplicates, but left 70 test rows (0.65%) with identical 9-feature input profiles. Zero leakage was only achieved under grouped validation. |
| **Corrected** | Invalid Records $215.96 Metric | **CORRECTED** | Relabeled from "test MAE" to training memorization error. Documented honest held-out test error (\$1,542.45 to \$2,527.22). |
| **Corrected** | Backend Status Code for Invalid Inputs | **CORRECTED** | Corrected claim from HTTP 400 to **HTTP 200 OK with inline validation banner**, matching actual Flask behavior. |
| **Corrected** | Labeling GIA Listings as "Natural" | **CORRECTED** | Removed unverified claims of "natural origin." Relabeled strictly as observed commercial listing categories (`GIA`, `GIA Lab-Grown`, `IGI Lab-Grown`). |
| **Corrected** | Generalizing Grouped Validation Scope | **CORRECTED** | Qualified findings to this specific dataset (~1.08% duplicate rate; holdout RMSE +10.59, MAE -1.39; CV RMSE +1.75), explicitly preventing over-generalization to all literature. |
| **Unreproduced** | Physical Natural Genesis for GIA listings | **EXCLUDED** | Lacks certificate metadata in source CSV. |
| **Unreproduced** | Model-level rejection of impossible inputs | **EXCLUDED** | Tree regressors output floats regardless; rejection is an application-level software invariant. |

---

## 4. Remaining Scientific Limitations

1. **Absence of Optical Spectroscopic Telemetry:** CVD and HPHT synthetic diamonds share identical crystalline carbon structures ($sp^3$ cubic lattice, density $3.52\text{ g/cm}^3$, refractive index 2.417) with mined diamonds. Standard 4C tables cannot authenticate diamond genesis without FTIR or photoluminescence spectroscopy.
2. **Static Web-Scraped Dataset Nature:** Both datasets represent historical inventory snapshots. Cross-retailer validation and live secondary market depreciation remain uncaptured.
3. **Severe Missingness in Detailed Attributes:** `Cut` (66.4%), `Culet` (70.8%), and `Fluorescence` (81.2%) missingness limits extended attribute utility without synthetic imputation.
4. **Physical Volume Redundancy:** Geometric volume ($x \cdot y \cdot z$) is highly collinear ($r > 0.98$) with carat weight, yielding negligible predictive gain ($-\$0.70$ CV MAE).

