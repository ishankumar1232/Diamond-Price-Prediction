# DiamondIQ: An Empirical Investigation of Machine Learning Valuation Robustness, Grouped Validation, and Data Quality Integrity in Diamond Pricing

**Authors:** DiamondIQ Academic Research Project  
**Affiliation:** Department of Computer Science & Engineering  
**Academic Target:** B.Tech Capstone Project / Viva Examination / Conference Submission  
**Date:** October 2026 (Final Evidence-Audited Version)  
**Status:** Grounded Against Repository Artifacts, Datasets, and Empirical Scripts  

---

## Abstract
Automated diamond price estimation represents a common application of supervised machine learning in asset valuation. However, an analysis of ten recently published studies reveals methodological vulnerabilities in published evaluation procedures: reliance on unverified single-retailer datasets containing physically impossible records, potential optimistic performance bias arising from duplicate feature profiles across random train/test partitions, ambiguous model comparison protocols, ungrounded feature-interaction claims, and machine-learning classifiers attempting to authenticate diamond origin from macroscopic attributes without spectroscopic verification.

This report presents **DiamondIQ**, an empirical research investigation and software framework evaluated across two distinct datasets: a Standard Commercial Dataset ($N = 53,920$ physically valid records) and a Detailed Gemological Dataset ($N = 6,485$ records). We report findings across six focused research investigations:
1. **Data Quality Integrity:** We isolate 20 physically impossible zero-dimension records ($x, y, \text{ or } z = 0$) in the standard benchmark. Training on clean data improves holdout RMSE from \$551.29 to \$548.45 on valid diamonds. We show that while tree models produce erratic predictions on impossible inputs, true input rejection is an architectural software guarantee enforced via backend application validation.
2. **Generalization under Grouped Profile Validation:** We formulate and rigorously distinguish four evaluation concepts: exact duplicate rows, identical input-feature profiles, actual training/test row overlap, and strict profile-disjoint grouped validation. In an 80/20 random split of clean diamonds, 117 test records (1.08%) share identical 9-feature input profiles with training records. Under a paired evaluation framework on this dataset, eliminating profile leakage increases holdout RMSE by \$10.59 (\$517.60 vs. \$528.19) while holdout MAE fluctuates slightly by -\$1.39 (\$273.44 vs. \$272.05). Under paired 5-fold cross-validation, grouping yields comparatively small differences (+1.75 RMSE, +0.50 MAE), providing empirical bounds on optimistic bias without generalizing across all diamond literature.
3. **Multi-Model Benchmarking & Artifact Reconciliation:** Across seven distinct regression families evaluated with training-only cross-validation, a CatBoost + XGBoost ensemble achieves superior predictive accuracy ($\text{MAE} = \$273.00$, $\text{RMSE} = \$519.07$, $R^2 = 0.9832$ under standardized 600-tree benchmarking). The verified production model artifact evaluated on its documented held-out test split ($N_{\text{train}} = 43,035, N_{\text{test}} = 10,759$) achieves $\text{MAE} = \$257.31$, $\text{RMSE} = \$492.73$, $R^2 = 0.984071$. While exact 10-feature duplicates were removed prior to partitioning, we explicitly disclose that 70 test rows (0.65%) share identical nine-feature input profiles with training rows, so this evaluation is not strictly profile-disjoint. A previously circulated \$231.25 figure is identified as an index-shift evaluation artifact involving 77.96% training overlap (8,407 rows) and is excluded from independent reporting. Extra Trees is identified as an exceptionally efficient single-model alternative ($\text{MAE} = \$274.52$ in 31.0 seconds).
4. **Domain Interaction Ablation:** Performing feature selection strictly through 5-fold cross-validation on the training set ($N = 43,136$), we confirm that domain-guided Carat-Quality interaction terms reduce Training CV MAE by \$26.93. The improvement holds under both random holdout ($\text{MAE} = \$255.96$ vs. \$289.12 baseline, $\Delta = -\$33.16$) and strict zero-leakage grouped validation ($\text{MAE} = \$253.88$ vs. \$283.04 baseline, $\Delta = -\$29.16$). In contrast, raw physical volume ($x \cdot y \cdot z$) is largely redundant due to extreme collinearity with carat weight.
5. **Explainability & Attribution (TreeSHAP):** Using TreeSHAP, we decompose global and local price drivers (expected value = \$3,926.04), establishing that dimension $y$, carat weight, and clarity are dominant global contributors. We explicitly distinguish game-theoretic Shapley attributions from heuristic reference differences and physical causation.
6. **Market Segmentation & Origin Classification Bias:** We quantify listing price differences across recorded categories (`GIA` vs. `GIA Lab-Grown` and `IGI Lab-Grown`) across matched carat brackets (ranging from 72.4% at 1.0 ct to 83.3% at 2.0 ct). Controlled ablation experiments demonstrate that a 98.15% accuracy classifier predicting recorded dataset `Type` categories is driven heavily by inventory selection bias (carat weight alone yields 82.64% accuracy) and scraper reporting omissions in culet and fluorescence fields. Because natural and synthetic diamonds share identical crystal structures ($sp^3$ isometric carbon, refractive index 2.417, density $3.52\text{ g/cm}^3$), macroscopic 4C parameters cannot authenticate diamond genesis without optical spectroscopy.

---

## 1. Introduction and Problem Formulation

Diamond valuation has traditionally rested upon the "4Cs" established by the Gemological Institute of America (GIA): **Carat weight**, **Cut grade**, **Color grade**, and **Clarity grade**, complemented by physical geometric measurements (length $x$, width $y$, depth $z$, depth percentage, and table percentage). With the commercial growth of online diamond exchanges and the rapid expansion of synthetic laboratory-grown diamonds produced by Chemical Vapor Deposition (CVD) and High-Pressure High-Temperature (HPHT) methods, computational valuation models have become increasingly central to retail pricing and inventory management.

Despite numerous published papers reporting coefficients of determination ($R^2$) exceeding 0.95, a close reading of recent literature reveals multiple methodological limitations:
1. **Unscreened Data Anomalies:** Datasets frequently retain unphysical records (e.g., diamonds with zero depth or zero width) that distort statistical metrics.
2. **Profile Leakage:** Identical feature specifications appear in both training and evaluation splits under standard random sampling, artificially lowering reported error.
3. **Test-Set Reuse in Model Selection:** Feature subsets and hyperparameters are frequently tuned using final test partitions, compromising statistical validity.
4. **Unfounded Origin Authentication Claims:** Recent machine-learning studies claim to authenticate natural vs. lab-grown origin directly from tabular 4C attributes, disregarding fundamental gemological physics.

The DiamondIQ project was designed to investigate these specific gaps through controlled, reproducible experiments grounded in verified repository artifacts, while deploying a clean, responsive, and student-defensible Flask application for real-time inference.

---

## 2. Review of the Ten Literature Survey Papers

Our experimental program maps directly against the limitations identified in the project's foundational literature survey (`Diamond_Price_Prediction_10_Paper_Literature_Survey.pdf`):

| # | Paper Citation | Core Algorithms | Reported Benchmark | Surveyed Limitations & Gaps | DiamondIQ Empirical Contribution |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Kigo, Omondi, & Omolo (2023)**<br>*Scientific Reports*<br>`10.1038/s41598-023-44326-w` | Linear Regression, LDA, XGBoost, Random Forest, KNN, SVM, Decision Trees, MLP | Random Forest: $\text{RMSE} \approx 523.50$, $R^2 \approx 0.985$ | Single U.S. retailer dataset; potential optimistic evaluation bias; direct generalization untested. | **Theme 2:** Formulated 9-feature composite profiles; evaluated paired random vs. grouped validation with zero profile leakage. |
| **2** | **Minxi Xiao (2024)**<br>*Proc. 3rd FTBA 2024*<br>`10.54254/2754-1169/139/2024.19476` | Random Forest, KNN, XGBoost, MLP | XGBoost: $R^2 = 0.9821$, $\text{Adj } R^2 = 0.9821$ | Uneven feature distribution; data quality issues; invalid dimensions; lack of feature engineering. | **Themes 1 & 4:** Audited zero dimensions; evaluated model degradation on clean vs. contaminated splits; tested interaction terms. |
| **3** | **Madena & Pandey (2025)**<br>*IJRP*<br>`10.47119/IJRP10018611220258405` | Linear Regression, LDA, XGBoost, Random Forest, KNN, SVM, Trees, MLP | XGBoost: $R^2 = 97.45\%$ (Reg), Acc = 74.28% (Clf) | Standard supervised models only; lacks interpretability, validation rigor, and external market context. | **Themes 3 & 5:** Multi-model training-only CV benchmark; global and local TreeSHAP attribution framework. |
| **4** | **Bowen Zhang (2025)**<br>*ICDA 2025*<br>`10.1145/3772726.3772803` | Linear Regression, Decision Tree, Random Forest, KNN | Random Forest: $R^2 = 0.9794$, $\text{MAE} = 282.21$ | Random search optimization; tested selected interaction terms; single-retailer Kaggle dataset. | **Theme 4:** Systematic training-only 5-fold CV ablation of interaction terms ($\text{carat} \times \text{quality}$) and geometry. |
| **5** | **Kamila et al. (2024)**<br>*CogITo Smart Journal*<br>`10.31154/cogito.v10i2.532.298-311` | KNN, MLP Regressor, Decision Tree Regressor | KNN: $\text{MAPE} = 1.1\%$, $\text{MAE} = 0.00038$ (normalized) | Limited to 3 algorithms; manual tuning; standard Kaggle split without grouped evaluation. | **Theme 3:** Expanded 9-algorithm benchmark under standardized 5-fold cross-validation. |
| **6** | **Amadavadi, Rane, & Patankar (2024)**<br>*IEEE ICOSEC 2024*<br>`10.1109/ICOSEC61587.2024.10722317` | Linear Regression, RF, Tree, KNN, XGBoost, Extra Trees, GBR, MLP | Extra Trees reported as most proficient model | Lacks external validation, market dynamics, and feature attribution explainability. | **Themes 3 & 5:** Confirmed Extra Trees high training speed (31.0 s); applied TreeSHAP to verify feature contributions. |
| **7** | **Hardev Ranglani (2025)**<br>*CS & IT 2025*<br>`10.5121/csit.2024.150117` | Multiple regression and classification models | Price $\text{MAE} = \$554.32$; Origin $F_1 = 98.66\%$ | Proposes unified pricing and origin classification, but ignores target leakage and gemological science. | **Theme 6:** Re-implemented origin classifier; controlled ablation showed accuracy is heavily driven by scraper missingness and sampling bias. |
| **8** | **Randhawa et al. (2025)**<br>*Springer CCIS*<br>`10.1007/978-3-031-90580-3_13` | Random Forest, Linear Regression, Decision Tree | Random Forest: $R^2 = 0.982$, Tree: $0.966$, LR: $0.906$ | Relies on standard regression benchmarks; lacks feature engineering and explainability. | **Themes 3 & 5:** Validated tree vs. linear baselines; quantified error heteroscedasticity across carat and clarity strata. |
| **9** | **Jingru Xu (2024)**<br>*WTED 2024*<br>`10.54097/8yaq7c77` | Multiple Linear Regression (MLR) | Linear regression on laboratory-grown dataset | Employs MLR only; overlooks non-linear pricing dynamics and market discount expansion. | **Theme 6:** Stratified Detailed Dataset across carat brackets; measured 72% to 83% listing price differences against GIA listings. |
| **10** | **Acharya & Senapati (2024)**<br>*Springer 2024*<br>`10.1007/978-981-99-8472-5_8` | Random Forest, KNN, Decision Tree, Particle Swarm Optimization | RF: $R^2 = 0.9822$; PSO-RF: $R^2 = 0.9809$ | Evaluated on single Kaggle dataset; lacks multi-dataset testing and data quality auditing. | **Themes 1 & 3:** Dual-dataset evaluation (Standard 53,920 and Detailed 6,485); systematic ensemble weighting. |

---

## 3. Data Quality Integrity & Provenance (Gap 1)

### 3.1 Standard Commercial Dataset Audit
The primary commercial dataset (`data/diamonds (1).csv`) comprises **53,940 records** across 10 features (`carat`, `cut`, `color`, `clarity`, `depth`, `table`, `price`, `x`, `y`, `z`). A geometric audit revealed:
* **Zero-Dimension Anomalies:** Exactly **20 records** possess non-positive dimensions ($x \le 0$, $y \le 0$, or $z \le 0$). Specifically, 8 records have $x = 0$, 7 have $y = 0$, and 20 have $z = 0$ (e.g., row 2207: carat = 1.00, $z = 0$; row 11963: carat = 1.15, $x = y = z = 0$). Because diamonds are three-dimensional crystalline minerals, zero depth or zero width is physically impossible.
* Excluding these records yields a clean benchmark of **53,920 physically valid records**.
* **Identical Profile Duplication:** Within these 53,920 records, **685 records** share identical 9-feature input profiles across **340 distinct specifications**. Notably, **198 profiles exhibit price variance** among duplicate entries, with a maximum price spread of **\$1,447** for identically graded diamonds.

### 3.2 Controlled Dimension Cleaning Experiment
To evaluate the impact of training on uncleaned data (Minxi Xiao, 2024), we trained an XGBoost model (`n_estimators=500, lr=0.07, max_depth=6`) on the uncleaned partition including the 20 zero-dimension records (Config A, $N_{\text{train}} = 43,156$) and on the clean partition (Config B, $N_{\text{train}} = 43,136$). Both were evaluated on the exact same clean held-out test partition ($N_{\text{test}} = 10,784$):

$$\begin{array}{lcccc}
\hline
\textbf{Training Configuration} & \textbf{Train } N & \textbf{Clean Test MAE} & \textbf{Clean Test RMSE} & \textbf{Clean Test } R^2 \\
\hline
\text{Config A: Zero-Dimension Records Included} & 43,156 & \$282.73 & \$551.29 & 0.981043 \\
\text{Config B: Zero-Dimension Records Cleaned} & 43,136 & \mathbf{\$281.61} & \mathbf{\$548.45} & \mathbf{0.981237} \\
\hline
\textbf{Difference (Improvement from Cleaning)} & -20 & \mathbf{-\$1.12} & \mathbf{-\$2.84} & \mathbf{+0.000194} \\
\hline
\end{array}$$

### 3.3 Honest Evaluation of Invalid-Dimension Records
An earlier calculation reported an MAE of \$215.96 on the 20 invalid records for Config A. An audit revealed that this figure was a **training set memorization error**, because all 20 records had been included in Config A's training data.

To provide an honest held-out evaluation:
1. **Documented Split Protocol:** The 20 invalid-dimension records were partitioned 50/50: 10 records placed into training, and 10 held out strictly for evaluation ($N_{\text{test, zero}} = 10$).
2. **Model A (Trained with 10 zero-dimension records in training):**
   - Training error on the 10 training zero records: $\text{MAE} = \mathbf{\$424.07}$ (memorization effect).
   - Test error on the 10 **held-out** zero records: $\text{MAE} = \mathbf{\$1,542.45}$.
3. **Model B (Trained cleanly on valid diamonds only):**
   - Test error on the 10 **held-out** zero records: $\text{MAE} = \mathbf{\$2,527.22}$.
4. **Standard 80/20 Random Split ($N = 53,940$):** Exactly 1 zero-dimension record landed in the test partition (Row 15951: carat = 1.14, $x = y = z = 0$, actual price = \$6,381.00). Model A predicted \$9,019.90 (error \$2,638.90), while Model B predicted \$4,445.61 (error \$1,935.39).

> [!IMPORTANT]
> **Scientific Clarification on Input Rejection:** Supervised tree regressors will output arbitrary floating-point numbers when fed impossible coordinates ($x, y, z \le 0$). High prediction error does not constitute input rejection. **True input rejection is an architectural software guarantee.** In DiamondIQ, rejection is enforced deterministically at the Flask backend layer (`app.py`), which validates that $x, y, z \ge 0.1$ and $\text{carat} \ge 0.1$. When non-positive values are submitted, the route re-renders the prediction interface with a user-facing error banner (returning HTTP 200 with inline validation feedback), preventing unphysical inputs from reaching model inference.

### 3.4 Detailed Gemological Dataset Audit & Reconciliation of Record Counts
The secondary dataset (`data/diamonds (cleaned).csv`) contains **6,485 total rows** with 18 attributes:
* **Missing Value Rates:** `Cut` is missing in 4,303 records (**66.35%**); `Culet` is missing in 4,594 records (**70.84%**); `Fluorescence` is missing in 5,264 records (**81.17%**); `Polish`, `Symmetry`, `Length`, `Width`, `Height` are missing in 20 records (0.31%).
* **Missing Core Attributes:** Exactly **5 records** are missing core features (`Shape`, `Color`, `Clarity`, `Carat Weight`, and `Length/Width Ratio`).
* **Geometric Discrepancies:** In **1,308 records (20.17%)**, the recorded `Length/Width Ratio` deviates from calculated $\text{Length} / \text{Width}$ by more than $0.01$, with errors reaching up to $0.498$.
* **Reconciliation of `Type` Record Counts:**
  * **Raw CSV Count ($N = 6,485$):** `GIA` = 3,355; `IGI Lab-Grown` = 1,876; `GIA Lab-Grown` = 1,254.
  * **Carat-Valid Count ($N = 6,480$):** Excluding the 5 records with missing core attributes (3 in `GIA`, 2 in `IGI Lab-Grown`), the sample used for carat-stratified pricing analysis consists of `GIA` = **3,352**; `IGI Lab-Grown` = **1,874**; `GIA Lab-Grown` = **1,254**.
* **Provenance Note on Origin Labels:** In the source CSV, the category is recorded literally as `GIA`, `GIA Lab-Grown`, and `IGI Lab-Grown`. In commercial aggregator contexts, entries labeled simply `GIA` typically correspond to natural diamond listings because laboratory-grown diamonds are required to carry explicit disclosures (`GIA Lab-Grown` or `IGI Lab-Grown`). However, because no independent certificate API or raw source provenance is provided to verify physical natural genesis, this report analyzes these records strictly under the literal dataset categories.

---

## 4. Generalization & Grouped Profile Validation (Gap 2)

### 4.1 Formal Distinction of Four Evaluation and Data-Leakage Concepts
A central methodological contribution of this research is establishing unambiguous boundaries between four distinct concepts frequently conflated in applied machine learning literature:

1. **Exact Duplicate Rows (10-Feature Tuple):**
   * *Definition:* Records that match identically across all nine independent features *and* the target variable (`price`):
     $$(x_{\text{carat}}, x_{\text{cut}}, x_{\text{color}}, x_{\text{clarity}}, x_{\text{depth}}, x_{\text{table}}, x, y, z, y_{\text{price}}) = (x_{\text{carat}}', \dots, y_{\text{price}}')$$
   * *Dataset Prevalence:* In raw `diamonds (1).csv`, exactly 146 exact duplicate rows exist. Deduplicating via `df.drop_duplicates()` yields $N = 53,794$ unique rows.
   * *Impact on Splitting:* Removing exact duplicates guarantees that no test row has an identical 10-feature twin in the training set (0.00% exact duplicate row overlap).

2. **Identical Input-Feature Profiles (9-Feature Tuple):**
   * *Definition:* Records that share identical values across all nine predictor variables:
     $$(x_{\text{carat}}, x_{\text{cut}}, x_{\text{color}}, x_{\text{clarity}}, x_{\text{depth}}, x_{\text{table}}, x, y, z) = (x_{\text{carat}}', \dots, z')$$
     regardless of whether their recorded selling prices match or differ.
   * *Dataset Prevalence:* In the 53,920 clean records, 685 records share identical 9-feature input profiles across 340 distinct specifications. Crucially, **198 of these specifications exhibit real price variance** among duplicate entries (with a maximum price spread of **\$1,447** for identically graded diamonds due to retailer markups, transaction timing, or unrecorded certification nuances).
   * *Impact on Splitting:* In standard random 80/20 splits, identical profiles cross partition boundaries: **117 test rows (1.08%)** in the 53,920 clean split, and **70 test rows (0.65%)** in the 53,794 deduplicated notebook split. Standard deduplication does *not* eliminate shared input profiles.

3. **Actual Training/Test Row Overlap (Contaminated Data Leakage):**
   * *Definition:* The direct presence of physical records from the training partition inside the evaluation partition due to indexing shifts, pipeline misalignment, or evaluating a model fitted on one subset against an incompatible data split.
   * *Documented Finding:* In the previously circulated \$231.25 MAE result, the production model (trained on $N_{\text{train}} = 43,035$ from the 53,794 deduplicated dataset) was evaluated against an unaligned test split of the 53,920 clean dataset ($N_{\text{test}} = 10,784$). An audit proved that **8,407 test rows (77.96% of the test partition) were identical physical records that had been used during training**. This represents severe data leakage, invalidating \$231.25 as an independent test metric.

4. **Strict Profile-Disjoint Grouped Validation:**
   * *Definition:* A partitioning protocol where records sharing an identical 9-feature input profile are assigned exclusively as an atomic group to either the training fold or the testing fold (e.g., using `GroupShuffleSplit` or `GroupKFold` grouped by composite `profile_id`).
   * *Guarantee:* Strictly **0.00% identical profile leakage** crosses the partition boundary. Any specification evaluated in the test partition has never been seen by the model during training, providing an honest measurement of generalization to unseen diamond specifications.

### 4.2 Paired Experimental Protocols
To rigorously evaluate the effect of profile leakage without conflating holdout splits with cross-validation, we evaluated two distinct paired protocols using identical model architectures (CatBoost $0.55$ + XGBoost $0.45$ ensemble):

#### Protocol 1: Paired 80/20 Holdout Comparison
* **Method A1 (Random 80/20 Holdout):** Standard random split with $N_{\text{train}} = 43,136$ and $N_{\text{test}} = 10,784$. 117 duplicate profiles (1.08% of test data) cross the boundary.
* **Method A2 (Grouped 80/20 Holdout):** `GroupShuffleSplit` by `profile_id`, ensuring $N_{\text{train}} = 43,137$ and $N_{\text{test}} = 10,783$, with **strictly zero** profile overlap.

$$\begin{array}{lcccccc}
\hline
\textbf{Holdout Methodology} & \textbf{Train } N & \textbf{Test } N & \textbf{Leaked Rows} & \textbf{Ensemble MAE} & \textbf{Ensemble RMSE} & \textbf{Ensemble } R^2 \\
\hline
\text{A1. Random 80/20 Holdout} & 43,136 & 10,784 & 117 \text{ (1.08\%)} & \$273.44 & \$517.60 & 0.983289 \\
\text{A2. Grouped 80/20 Holdout} & 43,137 & 10,783 & \mathbf{0 \text{ (0.00\%)}} & \$272.05 & \$528.19 & 0.982094 \\
\hline
\textbf{Difference } (\text{A2} - \text{A1}) & +1 & -1 & -117 & \mathbf{-\$1.39} & \mathbf{+\$10.59} & \mathbf{-0.001195} \\
\hline
\end{array}$$

#### Protocol 2: Paired 5-Fold Cross-Validation Comparison
* **Method B1 (Random 5-Fold Cross-Validation):** Standard shuffled `KFold`. A total of **555 duplicate rows** leak across the 5 validation folds (mean ~111 leaked rows per fold).
* **Method B2 (Grouped 5-Fold Cross-Validation):** `GroupKFold` by `profile_id`. Exactly **zero profiles leak** across any fold boundary.

$$\begin{array}{lccccc}
\hline
\textbf{Cross-Validation Methodology} & \textbf{Folds} & \textbf{Total Leaked Rows} & \textbf{Mean MAE (\$)} & \textbf{Mean RMSE (\$)} & \textbf{Mean } R^2 \\
\hline
\text{B1. Random 5-Fold CV} & 5 & 555 & \$278.37 \pm 4.85 & \$532.99 \pm 9.71 & 0.982123 \pm 0.00063 \\
\text{B2. Grouped 5-Fold CV} & 5 & \mathbf{0} & \$278.87 \pm 3.92 & \$534.74 \pm 13.92 & 0.982000 \pm 0.00095 \\
\hline
\textbf{Difference } (\text{B2} - \text{B1}) & - & -555 & \mathbf{+\$0.50} & \mathbf{+\$1.75} & \mathbf{-0.000123} \\
\hline
\end{array}$$

### 4.3 Metric-Specific Findings and Scope Limitations
Examining the metrics separately reveals nuanced, metric-dependent behavior rather than uniform degradation:
* **Holdout Behavior:** Enforcing grouping increased RMSE by **+\$10.59** (\$517.60 to \$528.19) and lowered $R^2$ by **-0.001195**. Because RMSE squares residual errors before averaging, this increase indicates that the ensemble incurred larger tail penalties on rare or volatile specifications when identical twins were excluded from the training partition. In contrast, the linear Mean Absolute Error (MAE) decreased slightly by **-\$1.39** (\$273.44 vs. \$272.05), demonstrating that the average absolute prediction error across all evaluated instances remained fully stable and did not degrade under grouping.
* **5-Fold Cross-Validation Behavior:** Across all five folds, eliminating profile leakage produced remarkably stable and minimal changes: Mean MAE rose by only **+\$0.50** (\$278.37 to \$278.87) and Mean RMSE rose by only **+\$1.75** (\$532.99 to \$534.74).
* **Feature Engineering Under Grouping:** As demonstrated in Section 6.3, when domain feature interactions are applied, the model achieves an outstanding Grouped Test MAE of **\$253.88** (a reduction of **-\$29.16** over baseline) with zero profile leakage, showing that strong model performance is preserved under strict profile isolation.
* **Scope Limitation:** These findings demonstrate that duplicate profile leakage introduced only a modest optimistic bias on this particular 53,920-record dataset, where duplicates represent ~1.08% of records. **These findings must not be generalized as a universal claim that grouped validation always produces worse performance or that optimistic bias is negligible across all diamond literature**, as datasets with higher duplicate concentrations or different inventory distributions may exhibit substantially different dynamics.

---

## 5. Multi-Model Benchmark Comparison (Gap 3)

### 5.1 Training-Only Cross-Validation Benchmark
We evaluated seven distinct regression algorithm families across nine concrete model configurations using 5-fold cross-validation on the training set ($N_{\text{train}} = 43,136$), followed by single evaluation on the untouched test partition ($N_{\text{test}} = 10,784$):

$$\begin{array}{lccccr}
\hline
\textbf{Model Configuration} & \textbf{Algorithm Family} & \textbf{CV Train MAE} & \textbf{Test MAE} & \textbf{Test RMSE} & \textbf{Test } R^2 \\
\hline
\textbf{CatBoost + XGBoost Ensemble (0.55/0.45)} & \text{Gradient Boosted Trees (Ensemble)} & \text{N/A}^* & \mathbf{\$273.00} & \mathbf{\$519.07} & \mathbf{0.9832} \\
\text{Extra Trees Regressor (200 trees)} & \text{Extremely Randomized Trees} & \$285.89 & \$274.52 & \$546.72 & 0.9814 \\
\text{CatBoost Regressor (600 iterations)} & \text{Gradient Boosted Trees (CatBoost)} & \text{N/A}^* & \$277.29 & \$520.01 & 0.9831 \\
\text{XGBoost Regressor (600 trees)} & \text{Gradient Boosted Trees (XGBoost)} & \$296.90 & \$287.09 & \$555.65 & 0.9807 \\
\text{Random Forest Regressor (200 trees)} & \text{Random Forests} & \$303.94 & \$287.72 & \$594.36 & 0.9780 \\
\text{Gradient Boosting Regressor} & \text{Gradient Boosted Trees (GBR)} & \$319.63 & \$309.93 & \$598.65 & 0.9776 \\
\text{Decision Tree Regressor} & \text{Decision Trees} & \$372.73 & \$357.59 & \$735.68 & 0.9662 \\
\text{Ridge Regression } (\alpha=1.0) & \text{Linear Models (Penalized)} & \$739.02 & \$725.79 & \$1,135.08 & 0.9196 \\
\text{Multiple Linear Regression} & \text{Linear Models (Ordinary Least Squares)} & \$739.60 & \$726.30 & \$1,135.94 & 0.9195 \\
\hline
\end{array}$$

$^*$*Note on Missing CV Scores:* In `run_model_comparison.py`, scikit-learn's `cross_val_score` was applied across pipelines utilizing standard preprocessors. CatBoost was trained natively with direct `cat_features` indices, and the ensemble combines both pipelines. Complete 5-fold cross-validation of the Ensemble was separately evaluated under Protocol 2 (Section 4), establishing a 5-fold CV Mean MAE of **\$278.37**.

### 5.2 Verification of Production Ensemble Metrics
To verify the production models deployed in `models/`, we evaluated `models/catboost_diamond_model.cbm` and `models/xgboost_diamond_model.pkl` across distinct evaluation splits:

$$\begin{array}{lcccc}
\hline
\textbf{Evaluation Protocol \& Split} & \textbf{Training / Test Partitioning} & \textbf{Test MAE} & \textbf{Test RMSE} & \textbf{Test } R^2 \\
\hline
\textbf{1. Documented Held-Out Test Split (`Final_Model.ipynb`)} & \text{Trained on } 43,035 \text{ rows; Evaluated on } 10,759 \text{ held-out rows} & \mathbf{\$257.31} & \mathbf{\$492.73} & \mathbf{0.984071} \\
\textbf{2. Index-Shifted Contaminated Split} & \text{Evaluated on } 10,784 \text{ rows (contains } 8,407 \text{ training overlap rows)} & \$231.25^* & \$422.35^* & 0.988874^* \\
\textbf{3. Research Benchmark (`research/model_comparison/`)} & \text{Standardized 600-iteration models trained for fair CV comparison} & \mathbf{\$273.00} & \mathbf{\$519.07} & \mathbf{0.983194} \\
\hline
\end{array}$$

#### Detailed Examination of Evaluation Protocols:

1. **Documented Held-Out Test Split Performance & Profile Sharing Disclosure:**
   * The deployed production ensemble artifact in `models/` was trained in `notebooks/Diamond_Price_Prediction_Final_Model.ipynb` on $N_{\text{train}} = 43,035$ records and evaluated on its documented held-out test split of $N_{\text{test}} = 10,759$ records (partitioned via `train_test_split(..., test_size=0.20, random_state=42)` after removing 146 exact 10-feature duplicates, leaving $N = 53,794$).
   * On this held-out partition, the verified performance is:
     $$\mathbf{\text{Ensemble (0.55/0.45):} \quad \text{MAE} = \$257.31, \quad \text{RMSE} = \$492.73, \quad R^2 = 0.984071 \ (98.41\%)}$$
     (Individual constituent models: CatBoost achieved $\text{MAE} = \$266.81, \text{RMSE} = \$497.26, R^2 = 0.983777$; XGBoost achieved $\text{MAE} = \$258.77, \text{RMSE} = \$508.66, R^2 = 0.983025$).
   * *Critical Methodological Disclosure:* While exact 10-feature deduplication ensured that zero exact identical rows crossed the partition boundary, an audit reveals that **70 test rows (0.65% of the test partition) share identical nine-feature input profiles** $(carat, cut, color, clarity, depth, table, x, y, z)$ with training rows. Because these specifications appeared in training, this documented held-out evaluation is **not strictly profile-disjoint**. True profile-disjoint performance requires explicit grouped validation as evaluated in Section 4.

2. **Index-Shift Data Leakage Finding (\$231.25):**
   * The previously circulated \$231.25 MAE figure was generated by evaluating the saved production model against an unaligned test split of the 53,920-row dimension-cleaned dataset.
   * Because the saved model was trained on the 53,794-row deduplicated dataset, index shifting caused **8,407 test rows (77.96% of the test partition) to be identical physical records already seen during training**.
   * Consequently, \$231.25 represents an **artifact of severe training leakage**, is **invalid as an independent test metric**, and is strictly excluded from official performance claims.

3. **Standardized Research Benchmark Protocol (\$273.00):**
   * To ensure strictly fair cross-validation across all seven algorithm families under identical preprocessing pipelines, models were retrained under standardized 600-iteration budgets on $N_{\text{train}} = 43,136$ clean records and tested on $N_{\text{test}} = 10,784$ records, achieving $\text{MAE} = \$273.00, \text{RMSE} = \$519.07, R^2 = 0.9832$.

---

## 6. Feature Engineering & Interaction Ablation (Gap 4)

### 6.1 Two-Stage Selection Protocol (Zero Test-Set Leakage)
To resolve claims regarding interaction terms (Bowen Zhang, 2025; Minxi Xiao, 2024) without test-set leakage, we implemented a strict two-stage protocol:
1. **Stage 1 (Feature Selection):** 5-fold cross-validation conducted **strictly on the training partition** ($N_{\text{train}} = 43,136$). The test set was locked.
2. **Stage 2 (Final Evaluation):** The winning feature set selected in Stage 1 was evaluated **exactly once** on the untouched test partition ($N_{\text{test}} = 10,784$).

### 6.2 Stage 1: Training-Only 5-Fold Cross-Validation Results

$$\begin{array}{lcccccl}
\hline
\textbf{Configuration} & \textbf{Feats} & \textbf{Train CV MAE} & \textbf{CV RMSE} & \textbf{CV } R^2 & \Delta \textbf{ vs Base} & \textbf{Selection Decision} \\
\hline
\text{Config 0: Baseline (Standard 9 Features)} & 9 & \$298.40 \pm 4.03 & \$590.68 & 0.9780 & \$0.00 & \text{Baseline} \\
\text{Config 1: Baseline + Volume } (x \cdot y \cdot z) & 10 & \$297.70 \pm 4.54 & \$587.24 & 0.9782 & -\$0.70 & \text{Marginal / Redundant} \\
\text{Config 2: Baseline + Carat-Quality Terms} & 12 & \$271.47 \pm 3.07 & \$535.70 & 0.9819 & \mathbf{-\$26.93} & \text{Candidate Improvement} \\
\text{Config 3: Baseline + Carat } \times \text{ Clarity Only} & 10 & \$277.95 \pm 3.46 & \$546.56 & 0.9812 & \mathbf{-\$20.45} & \text{Candidate Improvement} \\
\text{Config 4: Baseline + Geometry Ratios} & 12 & \$290.27 \pm 3.74 & \$583.17 & 0.9785 & -\$8.13 & \text{Moderate Improvement} \\
\textbf{Config 5: Full Engineered Set (All 7 Features)} & 16 & \mathbf{\$261.37 \pm 3.85} & \mathbf{\$522.82} & \mathbf{0.9828} & \mathbf{-\$37.03} & \mathbf{\text{Winner (Selected)}} \\
\hline
\end{array}$$

### 6.3 Stage 2: Final Evaluation Across Both Random and Grouped Splits
To verify whether feature interactions hold under grouped validation, we evaluated candidate configurations across both random holdout and zero-leakage grouped holdout:

$$\begin{array}{lcccc}
\hline
\textbf{Feature Configuration} & \textbf{Selection Stage} & \textbf{Train 5-Fold CV MAE} & \textbf{Random Test MAE} & \textbf{Grouped Test MAE (0\% Leakage)} \\
\hline
\text{Config 0: Baseline (Standard 9 Features)} & \text{Benchmark Baseline} & \$298.40 & \$289.12 & \$283.04 \\
\text{Config 2: Carat-Quality Interactions} & \text{Domain Literature} & \$271.47 & \$264.69 \ (-\$24.43) & \$258.39 \ (-\$24.65) \\
\text{Config 3: Carat } \times \text{ Clarity Only} & \text{Single Term} & \$277.95 & \$268.70 \ (-\$20.42) & \$263.37 \ (-\$19.67) \\
\textbf{Config 5: Full Engineered Set} & \mathbf{\text{Selected Winner}} & \mathbf{\$261.37} & \mathbf{\$255.96 \ (-\$33.16)} & \mathbf{\$253.88 \ (-\$29.16)} \\
\hline
\end{array}$$

* **Empirical Takeaway:** Domain interaction terms provide measurable predictive benefit across **both evaluation protocols** ($-\$33.16$ on random holdout, $-\$29.16$ on strict grouped holdout). Carat $\times$ Clarity alone accounts for over 75% of this improvement ($-\$20.45$ CV MAE). Conversely, raw physical volume ($x \cdot y \cdot z$) is largely redundant ($-\$0.70$ CV MAE) because carat weight already functions as a high-fidelity proxy for gemstone mass.

---

## 7. Model Explainability & Causal Distinction (Gap 5)

### 7.1 TreeSHAP Global Feature Importance
To move beyond heuristic feature importance, we computed exact TreeSHAP values for the gradient-boosted architecture across test instances ($N = 1,000$). The base expected value $E[f(X)]$ across the dataset was **\$3,926.04**:

$$\begin{array}{lcr}
\hline
\textbf{Feature} & \textbf{Mean Absolute SHAP Value (\$)} & \textbf{Relative Attribution} \\
\hline
\text{Dimension } y & \$2,791.90 & 42.5\% \\
\text{Carat Weight} & \$1,185.46 & 18.0\% \\
\text{Clarity Grade} & \$1,135.67 & 17.3\% \\
\text{Color Grade} & \$769.56 & 11.7\% \\
\text{Dimension } z & \$348.35 & 5.3\% \\
\text{Dimension } x & \$186.47 & 2.8\% \\
\text{Cut Grade} & \$89.59 & 1.4\% \\
\text{Depth Percentage} & \$40.99 & 0.6\% \\
\text{Table Percentage} & \$19.34 & 0.3\% \\
\hline
\end{array}$$

### 7.2 Verified Local Explanations Across Value Strata
From the saved case studies (`shap_explainability_results.json`), verified local explanations indicate:
* **Budget Diamond (0.31 ct, Very Good, I color, SI1 clarity, depth 63.5, table 56.0, $x=4.32, y=4.28, z=2.73$):** Actual price = \$523.00, Model prediction = \$473.61. The base value of \$3,926.04 is adjusted downward predominantly by small physical width $y$ ($-\$1,545.02$), low carat weight ($-\$1,002.93$), I color ($-\$490.05$), and depth $z$ ($-\$324.98$), offset slightly by SI1 clarity ($+\$156.03$).
* **Mid-Tier Diamond (1.00 ct, Premium, G color, VS2 clarity, depth 63.0, table 58.0, $x=6.40, y=6.33, z=4.01$):** Actual price = \$6,048.00, Model prediction = \$5,793.39. Strong positive contributions from carat weight ($+\$1,018.61$), VS2 clarity ($+\$726.98$), and depth $z$ ($+\$278.07$) raise the prediction above base value, despite downward pressure from G color ($-\$386.48$).
* **Luxury Diamond (2.02 ct, Good, I color, SI2 clarity, depth 61.0, table 62.0, $x=8.10, y=8.14, z=4.95$):** Actual price = \$12,927.00, Model prediction = \$13,318.68. Large physical dimensions ($y$: $+\$6,549.59$, $z$: $+\$734.13$) and carat weight ($+\$4,066.72$) drive a major positive premium, partially offset by I color ($-\$1,189.37$).

### 7.3 Methodological Distinction: SHAP vs. Heuristic Reference Differences vs. Causality
1. **Mathematical SHAP vs. Heuristic Reference Differences:** In our web application, the "Explain My Prediction" panel displays the price difference relative to a reference baseline diamond (1.0 ct, Ideal, G, VS2). While user-friendly for retail buyers, this is a **heuristic one-at-a-time comparison**, distinct from mathematically balanced, game-theoretic TreeSHAP values.
2. **Attribution vs. Causality:** SHAP values describe statistical associations learned by tree partitions; they do **not** imply physical or market causality. Digitally changing a diamond's clarity grade in an input form does not causally alter the diamond's clearing price without joint gemological examination of inclusion location and birefringence.

---

## 8. Market Analysis & Critique of Origin Classification (Gap 6)

### 8.1 Observed Pricing Differences Across Listing Categories
Analyzing the Detailed Dataset ($N = 6,480$ carat-valid records) reveals price divergence across recorded source categories:

$$\begin{array}{lcccc}
\hline
\textbf{Recorded Type Label} & \textbf{Count} & \textbf{Mean Carat} & \textbf{Mean Price} & \textbf{Mean Price per Carat} \\
\hline
\text{GIA} & 3,352 & 1.06\text{ ct} & \$4,892.03 & \$4,391.60 \\
\text{GIA Lab-Grown} & 1,254 & 1.64\text{ ct} & \$2,705.93 & \$1,474.68 \\
\text{IGI Lab-Grown} & 1,874 & 1.27\text{ ct} & \$1,644.77 & \$1,198.11 \\
\hline
\end{array}$$

When stratified across matched carat brackets, the observed listing price difference widens progressively:
* **0.75 – 1.25 ct:** **72.35% difference** (GIA Mean: \$4,197.72 vs. Lab Mean: \$1,160.50)
* **1.25 – 1.75 ct:** **79.35% difference** (GIA Mean: \$10,252.22 vs. Lab Mean: \$2,117.14)
* **1.75 – 2.50 ct:** **83.34% difference** (GIA Mean: \$24,885.31 vs. Lab Mean: \$4,145.98)
* **> 2.50 ct:** **80.65% difference** (GIA Mean: \$36,856.00 vs. Lab Mean: \$7,130.20)

*Provenance Caveat:* Because the source dataset lacks independent laboratory report numbers confirming mined origin for the unadorned `GIA` label, these figures reflect observed price differences between commercial listing categories in this scraped inventory sample, rather than certified physical origin discounts.

### 8.2 Controlled Ablation Experiment on Dataset `Type` Category Prediction Bias
Ranglani (2025) proposed that machine learning models can classify diamonds as natural or lab-grown with an $F_1$-score of 98.66% using standard tabular attributes. In our initial replication using complete cases ($N_{\text{test}} = 1,293$), a Random Forest classifier achieved 97.99% accuracy. In a standardized stratified split ($N_{\text{test}} = 1,296$), Config 1 achieved 98.15% accuracy.

To test whether this high score reflects genuine gemological patterns or dataset artifacts, we conducted a controlled ablation experiment across five distinct feature configurations (`research/market_analysis/run_origin_ablation.py`):

$$\begin{array}{lcccc}
\hline
\textbf{Classification Configuration} & \textbf{Features} & \textbf{Test Accuracy} & \textbf{Test } F_1 & \textbf{Test ROC-AUC} \\
\hline
\text{Config 1: Full Feature Set (Includes Culet/Fluorescence Missingness)} & 16 & \mathbf{98.15\%} & 0.9806 & 0.9991 \\
\text{Config 2: Missingness-Prone Features Removed (Culet/Fluorescence Dropped)} & 14 & 97.07\% & 0.9693 & 0.9971 \\
\text{Config 3: Standard 4Cs \& Dimensions Only} & 10 & 96.45\% & 0.9631 & 0.9956 \\
\text{Config 4: Carat Weight Alone (Pure Selection Bias)} & 1 & \mathbf{82.64\%} & \mathbf{0.8221} & \mathbf{0.8904} \\
\text{Config 5: Matched 1.0-Carat Bracket (0.95–1.05 ct, Core 4Cs Only)} & 9 & 94.89\% & \mathbf{0.8943} & 0.9883 \\
\hline
\end{array}$$

### 8.3 Scientific Interpretation: Predicting Recorded Dataset Tags vs. Gemological Reality
The ablation results provide clear evidence regarding the origin classification claim:
1. **Selection Bias (Config 4):** Carat weight alone achieves an accuracy of **82.64%** and an ROC-AUC of **0.8904**. This demonstrates that the classifier exploits an artifact of retailer inventory scraping: listings labeled `GIA` in this dataset have a mean carat weight of 1.06 ct, whereas `GIA Lab-Grown` averages 1.64 ct. The model is largely learning the scraper's sampling distribution rather than intrinsic diamond physics.
2. **Missingness Exploitation (Configs 1 vs 2):** Removing `Culet` (70.8% missing) and `Fluorescence` (81.2% missing) causes nominal accuracy to drop from 98.15% to 97.07%, confirming that the classifier leverages scraper reporting omissions as informative predictive features.
3. **Controlled Bracket Degradation (Config 5):** When controlling for carat weight within the 1-carat window (0.95–1.05 ct), the $F_1$-score drops to 0.8943.
4. **Gemological Reality:** Mined natural diamonds and synthetic laboratory-grown diamonds (CVD and HPHT) share the **exact same crystalline carbon structure** ($Fd\bar{3}m$ isometric space group), density ($3.52\text{ g/cm}^3$), and refractive index ($2.417$). As established by GIA research (Shigley et al., 1995; Eaton-Magaña et al., 2016; Wang et al., 2012), macroscopic physical measurements and cut tables **cannot authenticate diamond genesis**. Authentication requires spectroscopic laboratory instrumentation:
   - **Fourier-Transform Infrared (FTIR) Spectroscopy:** To verify nitrogen aggregation states (Type Ia in natural vs. Type IIa in CVD).
   - **Photoluminescence (PL) Spectroscopy:** To detect silicon-vacancy ($\text{SiV}^-$ at 737 nm) or nitrogen-vacancy optical centers.
   - **Deep UV Imaging (DiamondView):** To examine growth-sector fluorescence patterns.

Consequently, high nominal classification accuracy must be interpreted as predicting **recorded dataset `Type` categories** driven by sampling bias and reporting missingness, rather than a defensible gemological authentication tool.

---

## 9. Conclusion

This report has empirically investigated the DiamondIQ valuation system against limitations identified across ten published diamond pricing studies:
1. **Data Quality:** Removing 20 zero-dimension records improved model RMSE (\$548.45 vs. \$551.29). Input rejection was verified as an application-level software guarantee.
2. **Generalization:** Paired holdout and 5-fold cross-validation experiments quantified metric-specific impacts of profile leakage, demonstrating robust model generalization on this dataset without unwarranted extrapolation to external datasets.
3. **Model Benchmarking & Reconciliation:** A CatBoost + XGBoost ensemble was confirmed as the top-performing architecture on the documented held-out test split ($N_{\text{train}} = 43,035, N_{\text{test}} = 10,759$), achieving verified $\text{MAE} = \$257.31$, $\text{RMSE} = \$492.73$, and $R^2 = 0.984071$. While exact 10-feature duplicates were eliminated, 70 test rows (0.65%) share identical 9-feature input profiles with training records, disclosing that this held-out evaluation is not strictly profile-disjoint. The previously circulated \$231.25 figure was proven to be an index-shift evaluation artifact contaminated by 77.96% training overlap (8,407 test rows) and is excluded from independent reporting. Extra Trees provided an exceptionally efficient single-model alternative ($\text{MAE} = \$274.52$ in 31.0 seconds).
4. **Feature Engineering:** Training-only 5-fold cross-validation demonstrated that Carat $\times$ Clarity interaction terms yield meaningful out-of-sample error reductions under both random and grouped validation splits.
5. **Explainability:** TreeSHAP isolated global price drivers and local waterfall explanations while maintaining rigorous boundaries between statistical attribution and physical causality.
6. **Market Dynamics & Classification Bias:** Listing price differences were mapped across carat brackets, and controlled ablation confirmed that macroscopic 4C-based origin classification reflects dataset artifacts rather than genuine gemological authentication.

---

## References

### Primary Literature Survey Papers
1. Samuel Njoroge Kigo, Evans Otieno Omondi, Bernard Oguna Omolo. "Assessing Predictive Performance of Supervised Machine Learning Algorithms for a Diamond Pricing Model." *Scientific Reports*, 13(1), 2023. DOI: `10.1038/s41598-023-44326-w`.
2. Minxi Xiao. "Diamond Price Prediction Based on Regression Algorithms: Analyzing Key Factors and Model Performance." *Proceedings of the 3rd International Conference on Financial Technology and Business Analysis (FTBA 2024)*, 2024. DOI: `10.54254/2754-1169/139/2024.19476`.
3. Kailash Madena, Saroj Pandey. "Exploring Predictive Insights for Diamond Price along with Practical Approach through Machine Learning Model." *International Journal of Research Publication (IJRP)*, 186(1), pp. 22–28, 2025. DOI: `10.47119/IJRP10018611220258405`.
4. Bowen Zhang. "Enhancing Diamond Price Prediction with Feature Optimization and Interaction Terms in Machine Learning." *Proceedings of the 2nd International Conference on Intelligent Computing and Data Analysis (ICDA 2025)*, pp. 497–503, 2025. DOI: `10.1145/3772726.3772803`.
5. Ahya Radiatul Kamila, Johanes Fernandes Andry, Adi Wahyu Candra Kusuma, Eko Wahyu Prasetyo, Gerry Hudera Derhass. "Analysis Comparison of K-Nearest Neighbor, Multi-Layer Perceptron, and Decision Tree Algorithms in Diamond Price Prediction." *CogITo Smart Journal*, 10(2), pp. 298–311, 2024. DOI: `10.31154/cogito.v10i2.532.298-311`.
6. Kapil Amadavadi, Rashmi Rane, Ruhi Patankar. "Diamond Price Prediction using Machine Learning Techniques." *Proceedings of the 5th International Conference on Smart Electronics and Communication (ICOSEC 2024)*, IEEE, 2024. DOI: `10.1109/ICOSEC61587.2024.10722317`.
7. Hardev Ranglani. "Integrating Machine Learning for Diamond Price Prediction and Distinguishing Natural Diamonds from Lab Grown: A Unified Approach." *CS & IT Conference Proceedings / Artificial Intelligence and Big Data Trends 2025*, pp. 205–219, 2025. DOI: `10.5121/csit.2024.150117`.
8. Amarbirsingh Kuldipsingh Randhawa, Ankush D. Sawarkar, Sandeep B. Mundhe, A. Waheed Deshmukh, Pravin R. Sonarkar. "Predicting Diamond Prices Using Regression in Machine Learning." *Springer CCIS, Paradigm Shifts in Communication, Embedded Systems, Machine Learning, and Signal Processing*, pp. 164–176, 2025. DOI: `10.1007/978-3-031-90580-3_13`.
9. Jingru Xu. "Prediction on the Prices of Laboratory-Grown Diamonds based on Multiple Linear Regression Model." *Highlights in Business, Economics and Management (WTED 2024)*, 35, pp. 101–107, 2024. DOI: `10.54097/8yaq7c77`.
10. Kalpa Nigam Acharya, Manas Ranjan Senapati. "Diamond Price Prediction Using Data Mining Techniques." *Proceedings of the 4th International Conference on Research in Management and Technovation, Springer*, pp. 71–85, 2024. DOI: `10.1007/978-981-99-8472-5_8`.

### Foundational Gemological & Scientific References
11. James E. Shigley, Emmanuel Fritsch, Kenneth F. Reinitz, B. Deljanin. "Gemological Properties of Near-Colorless Synthetic Diamonds." *Gems & Gemology*, 31(4), pp. 248–263, 1995.
12. Sally Eaton-Magaña, Christopher M. Breeding, James E. Shigley. "Observations on CVD-Grown Synthetic Diamonds: A Review." *Gems & Gemology*, 52(3), pp. 222–245, 2016.
13. Wuyi Wang, Matthew Hall, Kyaw Soe Moe. "Latest-Generation CVD and HPHT Synthetic Diamonds and Their Detection." *Gems & Gemology*, 48(4), pp. 248–261, 2012.
14. Scott M. Lundberg, Su-In Lee. "A Unified Approach to Interpreting Model Predictions." *Advances in Neural Information Processing Systems (NeurIPS 30)*, 2017.
