# DiamondIQ Claims Verification Ledger: Audit, Reproduction, and Limitations

**Project:** DiamondIQ — Diamond Price Prediction  
**Evaluation Standard:** Independent Verification Against Raw CSV Data, Saved Models, Notebooks, and Scripts  
**Date:** October 2026 (Final Evidence Audit)  
**Auditor:** Senior ML Research Engineer / DiamondIQ Mentorship Audit  

---

## 1. Verified Claims (Reproduced Directly from Repository Artifacts)

The following claims have been tested and reproduced directly from the physical files in the workspace:

### Claim 1.1: Production Model Artifact Performance on Documented Held-Out Split
* **Status:** **VERIFIED & REPRODUCED**
* **Artifacts:** `models/catboost_diamond_model.cbm`, `models/xgboost_diamond_model.pkl`, `models/diamond_preprocessor.pkl`, `models/ensemble_config.pkl`.
* **Notebook Origin:** `notebooks/Diamond_Price_Prediction_Final_Model.ipynb` (Cells 2, 11, 16, 23).
* **Exact Dataset & Split:** `data/diamonds (1).csv` with exact 10-feature duplicates removed ($N_{\text{total}} = 53,794$). Partitioned via `train_test_split(..., test_size=0.20, random_state=42)`:
  * $N_{\text{train}} = 43,035$ (used to fit CatBoost, XGBoost, and Preprocessor).
  * $N_{\text{test}} = 10,759$ (documented held-out test records with **0% exact 10-feature training overlap**).
  * *Profile Sharing Disclosure:* While exact 10-feature duplicates were dropped, **70 test rows (0.65%)** share identical 9-feature input profiles with training rows; thus, this evaluation is **not strictly profile-disjoint**.
* **Verified Held-Out Test Metrics:**
  * **CatBoost Regressor:** $\text{MAE} = \$266.81, \ \text{RMSE} = \$497.26, \ R^2 = 0.983777$
  * **XGBoost Regressor:** $\text{MAE} = \$258.77, \ \text{RMSE} = \$508.66, \ R^2 = 0.983025$
  * **Production Ensemble (0.55/0.45):** $\mathbf{\text{MAE} = \$257.31, \ \text{RMSE} = \$492.73, \ R^2 = 0.984071 \ (98.41\%)}$

### Claim 1.2: Standard Commercial Dataset Geometric Quality
* **Status:** **VERIFIED & REPRODUCED**
* **Evidence:** Running `research/data_quality/run_data_quality_audit.py` on `data/diamonds (1).csv`:
  * Total raw rows: **53,940**.
  * Exactly **20 records** have non-positive dimensions ($x=0$: 8, $y=0$: 7, $z=0$: 20). Row indices: `[2207, 2314, 4791, 5471, 10167, 11182, 11963, 13601, 15951, 24394, 24520, 26123, 26243, 27112, 27429, 27503, 27739, 49556, 49557, 51506]`.
  * Valid clean rows: **53,920**.
  * Exactly **685 records** share identical 9-feature input profiles across **340 distinct profiles**.
  * **198 profiles exhibit price variance** among duplicate entries, with a maximum spread of **\$1,447**.

### Claim 1.3: Dimension Cleaning Prevents Distortion on Clean Test Data
* **Status:** **VERIFIED & REPRODUCED**
* **Evidence:** Running `research/data_quality/run_data_quality_audit.py`:
  * Model trained with zero-dimension records: Clean Test MAE = \$282.73, RMSE = \$551.29, $R^2$ = 0.981043.
  * Model trained without zero-dimension records: Clean Test MAE = \$281.61, RMSE = \$548.45, $R^2$ = 0.981237.
  * Test error on clean held-out diamonds improves by **-\$1.12 MAE** and **-\$2.84 RMSE**.

### Claim 1.4: Paired Grouped Profile Validation
* **Status:** **VERIFIED & REPRODUCED**
* **Evidence:** Running `research/grouped_validation/run_grouped_validation.py` (`grouped_validation_results.json`):
  * Holdout: Random 80/20 (117 leaked test rows) achieved MAE \$273.44, RMSE \$517.60, $R^2$ 0.983289.
  * Holdout: Grouped 80/20 (0 leaked rows) achieved MAE \$272.05, RMSE \$528.19, $R^2$ 0.982094.
  * 5-Fold Cross-Validation: Random 5-Fold (555 leaked rows) achieved Mean MAE \$278.37 $\pm 4.85$, RMSE \$532.99 $\pm 9.71$, $R^2$ 0.982123.
  * 5-Fold Cross-Validation: Grouped 5-Fold (0 leaked rows) achieved Mean MAE \$278.87 $\pm 3.92$, RMSE \$534.74 $\pm 13.92$, $R^2$ 0.982000.

### Claim 1.5: Detailed Dataset Record Counts and Missingness
* **Status:** **VERIFIED & REPRODUCED**
* **Evidence:** Executed audit on `data/diamonds (cleaned).csv`:
  * Total rows: **6,485**.
  * Categorical counts of `Type`: `GIA` = 3,355; `IGI Lab-Grown` = 1,876; `GIA Lab-Grown` = 1,254.
  * Missing value counts: `Cut` = 4,303 (66.35%), `Culet` = 4,594 (70.84%), `Fluorescence` = 5,264 (81.17%).
  * Exactly 5 rows miss core attributes (`Shape`, `Color`, `Clarity`, `Carat Weight`, `Length/Width Ratio`): 3 in `GIA`, 2 in `IGI Lab-Grown`.
  * Carat-valid rows: **6,480** (`GIA` = 3,352; `IGI Lab-Grown` = 1,874; `GIA Lab-Grown` = 1,254).

### Claim 1.6: Feature Engineering Interaction Benefits Hold Under Grouped Validation
* **Status:** **VERIFIED & REPRODUCED**
* **Evidence:** Evaluated candidate feature sets under strict `GroupShuffleSplit` (0% profile leakage) via XGBoost (500 trees, lr=0.07):
  * Config 0 (Baseline 9 features): Grouped Test $\text{MAE} = \$283.04, \ \text{RMSE} = \$568.76, \ R^2 = 0.979237$.
  * Config 2 (Carat-Quality Interactions): Grouped Test $\text{MAE} = \$258.39, \ \text{RMSE} = \$518.18, \ R^2 = 0.982766$ ($\Delta = -\$24.65$ MAE, $-\$50.58$ RMSE).
  * Config 3 (Carat $\times$ Clarity Only): Grouped Test $\text{MAE} = \$263.37, \ \text{RMSE} = \$526.90, \ R^2 = 0.982181$ ($\Delta = -\$19.67$ MAE).
  * Config 5 (Full Engineered Set): Grouped Test $\text{MAE} = \$253.88, \ \text{RMSE} = \$513.42, \ R^2 = 0.983081$ ($\Delta = -\$29.16$ MAE, $-\$55.34$ RMSE).

### Claim 1.7: TreeSHAP Attribution Values
* **Status:** **VERIFIED & REPRODUCED**
* **Evidence:** Verified against `research/explainability/shap_explainability_results.json`:
  * Base expected value: **\$3,926.04**.
  * Global feature rankings: Dimension $y$ (\$2,791.90, 42.5%), Carat (\$1,185.46, 18.0%), Clarity (\$1,135.67, 17.3%), Color (\$769.56, 11.7%), Dimension $z$ (\$348.35, 5.3%), Dimension $x$ (\$186.47, 2.8%), Cut (\$89.59, 1.4%), Depth (\$40.99, 0.6%), Table (\$19.34, 0.3%).

---

## 2. Corrected Claims (Identified Inconsistencies & Fixes)

| # | Prior Text / Claim | Audited Finding | Corrective Action in Final Report |
| :---: | :--- | :--- | :--- |
| **2.1** | **Claim of $231.25 Production Independent Test MAE**<br>Report presented \$231.25 as independent test accuracy. | Auditing the test split showed that when the saved model was evaluated on the dimension-cleaned split ($N=53,920$), **8,407 test rows (77.96% of the test partition)** were identical to rows used to fit the model in `Final_Model.ipynb`. The lower MAE was an artifact of severe index-shift data leakage. | Removed \$231.25 as an independent test metric. Relabeled it in Section 5.2 as a contaminated index-shift artifact. Confirmed true independent test MAE is **\$257.31**. |
| **2.2** | **Discrepancy in Classifier Accuracy: 97.99% vs 98.15%**<br>Abstract stated 97.99% while the ablation table showed 98.15%. | 97.99% came from `run_origin_analysis.py` where 20 rows with missing dimensions were dropped ($N_{\text{test}} = 1,293$). 98.15% came from `run_origin_ablation.py` where missing dimensions were median-imputed ($N_{\text{test}} = 1,296$, stratified). | Synchronized the primary reported score to **98.15%** (Config 1, $N_{\text{test}} = 1,296$) across abstract and tables, while explicitly explaining the 97.99% complete-case result in Section 8.2. |
| **2.3** | **Claim of $215.96 "Test" MAE on Invalid Records**<br>Report stated: "When evaluated on the invalid records, Config A exhibited an artificial test MAE of \$215.96." | In `run_data_quality_audit.py`, all 20 zero-dimension records were included in Config A's training data. The \$215.96 was therefore a **training set memorization error**, not a test error. | Re-ran a controlled 50/50 held-out evaluation ($N_{\text{test, zero}} = 10$). Documented that Model A achieves training MAE \$424.07 (memorization) and held-out test MAE **\$1,542.45**, while Model B achieves held-out test MAE **\$2,527.22**. |
| **2.4** | **Claim of HTTP 400 for Invalid Dimensions**<br>Report claimed the Flask backend returns HTTP 400 when invalid dimensions are submitted. | Testing `app.test_client().post('/predict', ...)` with $x=0.0$ returns **HTTP 200 OK** because Flask re-renders `prediction.html` with an inline error banner for user guidance. | Section 3.3 updated to state that the application returns HTTP 200 with user-facing inline validation feedback, catching invalid inputs before model inference. |
| **2.5** | **Generalizing Grouped Validation Findings to All Literature**<br>Report claimed grouped validation results proved published studies contain optimistic bias. | In the clean random holdout split, only 1.08% of test records (117 of 10,784 rows) share identical nine-feature input profiles with training records, resulting in modest shifts (5-fold CV error increased by +\$0.50 MAE and +\$1.75 RMSE, while holdout MAE decreased by -\$1.39). | Section 4.3 now explicitly limits the finding to this specific dataset and holdout configuration, stating that these results cannot be generalized to all published diamond studies. |
| **2.6** | **Describing Detailed Dataset Type Differences as "Natural Discounts"**<br>Report referred to price differences between GIA and lab-grown listings as "natural-versus-lab-grown discounts." | GIA listings in this dataset lack laboratory report verification confirming mined physical origin. | Sections 3.4 and 8.1 now describe them strictly as "observed listing price differences across recorded categories (`GIA` vs. `GIA Lab-Grown` and `IGI Lab-Grown`)." |
| **2.7** | **Classifier Concluding Bias as Proven Fact Without Feature Ablation**<br>Report stated that classifier accuracy was proven to be entirely a missingness artifact. | Controlled feature ablation had not been isolated across configurations. | Executed `run_origin_ablation.py`. Proved that Carat Weight alone achieves 82.64% accuracy (selection bias), and removing missingness drops accuracy to 97.07%. Described the conclusion as a **potential dataset artifact**, not a proven universal fact. |
| **2.8** | **Reconciliation of Nine Benchmark Rows with Seven Algorithm Families**<br>Text stated seven algorithms were benchmarked while the table listed nine rows. | 7 algorithm families were evaluated across 9 concrete model configurations. | Section 5.1 updated to explicitly clarify the 7-family vs. 9-configuration distinction and explains why CatBoost lacked CV scores in the script. |

---

## 3. Unreproduced Claims (Claims Lacking Source Evidence / Excluded)

The following claims found in literature or earlier project drafts cannot be reproduced from repository files and are excluded from official reporting:
1. **$231.25 Independent Test MAE:** Invalid due to 77.96% training overlap caused by index-shifting across different preprocessing pipelines. Excluded from independent performance claims.
2. **$215.96 Test MAE on Invalid Records:** Invalid because all 20 records were in the training set. Corrected to training memorization error.
3. **Certified Natural Ground Truth for GIA Listings:** No independent laboratory certificates exist in the scraped dataset; cannot be claimed as certified natural stones.
4. **Physical Origin Classification from 4C Inputs:** CVD/HPHT diamonds share identical isometric carbon lattices ($sp^3$, refractive index 2.417, density $3.52\text{ g/cm}^3$) with mined diamonds. Claims that machine learning can determine physical origin from macroscopic 4Cs without optical spectroscopy are unscientific and refuted.

---

## 4. Remaining Limitations (Open Challenges for Future Research)

1. **Absence of Optical Spectroscopic Telemetry:** Mined and lab-grown diamonds cannot be physically authenticated from tabular 4C parameters without laboratory spectroscopy (FTIR, photoluminescence, DiamondView).
2. **Static Single-Retailer Aggregator Sampling:** Datasets represent historical web-scraped snapshots. Cross-retailer validation and live market dynamics remain uncollected.
3. **Severe Missingness in Detailed Laboratory Attributes:** High missingness in `Cut` (66.4%), `Culet` (70.8%), and `Fluorescence` (81.2%) limits practical utility for automated pricing without synthetic imputation.
4. **Dataset-Specific Scope of Grouped Validation:** Because only 1.08% of test records (117 of 10,784 rows) share identical nine-feature input profiles with training records in this specific clean random holdout split, grouped validation resulted in modest changes (+1.75 RMSE on 5-fold CV, while holdout MAE changed by -\$1.39). These results cannot be generalized as a universal claim across all diamond pricing literature.
5. **Redundancy of Geometric Volume:** Adding raw geometric volume ($x \cdot y \cdot z$) alongside carat weight yields negligible benefit ($-\$0.70$ CV MAE) due to extreme collinearity ($r > 0.98$) with gemstone mass.
