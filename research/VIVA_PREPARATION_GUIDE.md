# DiamondIQ — College Viva & Project Defense Guide

**Project Title:** DiamondIQ: An Explainable and Robust Machine Learning System for Diamond Valuation with Grouped Validation and Quality Integrity  
**Student Target:** B.Tech Computer Science Project Viva  
**Tone:** Clear, direct, professional, and easy to explain in simple English.  

---

## 1. Core Project Questions

### Q1. What is DiamondIQ in simple words?
**Answer:**  
"DiamondIQ is an intelligent diamond price estimation system built using machine learning. It takes the physical and quality attributes of a diamond—such as carat, cut, color, clarity, and dimensions—and estimates its market price within seconds. 

Unlike basic projects that just train a simple regression model on Kaggle, DiamondIQ addresses real scientific research gaps from 10 published research papers, including data quality integrity, profile leakage during validation, domain feature engineering, TreeSHAP explainability, and natural versus lab-grown price segmentation."

---

### Q2. Why does DiamondIQ have TWO separate datasets and model pipelines?
**Answer:**  
"In the real diamond industry, diamond data comes at different levels of detail depending on the certificate:
1. **Standard Pipeline (53,920 records):** Represents standard commercial grading containing the primary 4Cs and basic dimensions $(x, y, z, \text{depth}, \text{table})$. For this high-volume dataset, we deploy a **CatBoost + XGBoost weighted ensemble** achieving an $R^2$ of **98.41%** ($\text{RMSE} = \$492.73$, $\text{MAE} = \$257.31$) on its documented held-out test split ($N_{\text{train}} = 43,035, N_{\text{test}} = 10,759$). We strictly exclude an earlier circulated \$231.25 figure because our audit proved it suffered from 78% training leakage due to an index shift.
2. **Detailed Pipeline (6,485 records):** Represents advanced laboratory grading containing specialized gemological attributes such as Shape, Polish, Symmetry, Girdle, Culet, Fluorescence, and Lab-Grown/Natural certification type. For this dataset, we deploy a **Tuned Random Forest Pipeline** achieving an $R^2$ of **88.06%** and MAE of **\$388.72**."

---

### Q3. Why did you use CatBoost and XGBoost together in an ensemble?
**Answer:**  
"CatBoost and XGBoost use different tree-growing and categorical handling strategies:
* **CatBoost** uses symmetric, oblivious decision trees and ordered target statistics, which prevents target leakage on categorical grades (`cut`, `color`, `clarity`).
* **XGBoost** uses depth-wise tree growth with exact second-order gradient approximations (Hessian and gradient), which excels at modeling non-linear continuous physical dimensions ($x, y, z$, carat).

By combining them with optimal weights ($0.55 \text{ CatBoost} + 0.45 \text{ XGBoost}$), their residual errors cancel out, reducing test RMSE to **\$492.73** on our held-out test split (and **\$519.07** on the standardized 600-iteration benchmark), which is lower than either model can achieve individually."

---

### Q4. What do MAE, RMSE, and $R^2$ mean, and why did you report all three?
**Answer:**  
* **MAE (Mean Absolute Error):** The average dollar amount the prediction is off by. In our verified Standard model, $\text{MAE} = \$257.31$, meaning our estimate is on average within \$257 of the true price. It is robust to outliers and easy for a buyer to understand.
* **RMSE (Root Mean Squared Error):** Gives higher penalty to large mistakes because errors are squared before averaging. Our $\text{RMSE} = \$492.73$. If RMSE is much larger than MAE, it indicates prediction variance on rare, high-carat diamonds.
* **$R^2$ (Coefficient of Determination):** Measures the proportion of variance in diamond prices explained by our features. Our $R^2 = 0.9841$, meaning our model explains **98.41%** of price variations across the dataset."

---

## 2. The Six Research Contributions (The "Viva Highlights")

When the external examiner asks: *"What is your novel contribution beyond Kaggle tutorials?"*, present these six verified points:

### 1. Data Quality & Dimension Integrity
* **The Problem:** The raw Kaggle diamond dataset contains **20 records with $x=0, y=0$, or $z=0$**. In physical reality, a 3D gemstone cannot have 0 mm width or height while weighing 1 or 2 carats!
* **Our Experiment:** When we trained a model with these zero-dimension records included, the model learned to tolerate impossible diamonds. When we cleaned them, held-out clean test RMSE improved from \$551.29 to \$548.45. True input rejection is an application guarantee: our Flask backend strictly validates $x, y, z \ge 0.1$ and carat $\ge 0.1$, returning HTTP 200 with inline validation error guidance before inference.

### 2. Generalization & Grouped Profile Validation
* **The Problem:** In standard random 80/20 train/test splits, identical diamond specifications (685 records in 340 repeated profiles) leak across both partitions (117 test rows / 1.08%). Even in our documented held-out split with exact duplicates dropped, 70 test rows (0.65%) share 9-feature input profiles with training records.
* **Our Solution:** We implemented paired evaluations with **GroupShuffleSplit** and **GroupKFold** by composite 9-feature profile. In grouped validation, identical diamond profiles are strictly restricted to one side of the boundary (**0% profile leakage**).
* **Nuanced Finding:** Grouped validation does *not* uniformly worsen performance: under holdout, RMSE rose by +\$10.59 due to tail penalties on unseen profiles, while MAE actually improved by -\$1.39 (\$272.05 vs. \$273.44); across 5-fold CV, changes were modest (+\$0.50 MAE, +\$1.75 RMSE).

### 3. Fair Model Benchmarking & Training-Only Tuning
* **The Problem:** Many published papers compare models using different splits or tune hyperparameters on the test set (data snooping).
* **Our Solution:** We benchmarked 7 algorithms (Linear, Ridge, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost, CatBoost) under strictly identical training-only 5-fold cross-validation, keeping the final test set completely untouched until final scoring. We confirmed that the CatBoost + XGBoost ensemble achieves superior accuracy, while Extra Trees offers high training efficiency (31 seconds).

### 4. Domain Feature Engineering & Interaction Ablation
* **The Problem:** Bowen Zhang (2025) claimed that manual interaction terms ($\text{carat} \times \text{cut}$, $\text{carat} \times \text{clarity}$) dramatically boost prediction, while others add geometric volume ($x \cdot y \cdot z$).
* **Our Finding:** Using training-only 5-fold CV feature selection on 43,136 records, we found that geometric volume is redundant ($-\$0.70$ MAE) due to collinearity with carat weight ($r > 0.98$). However, Carat-Quality interaction terms (especially Carat $\times$ Clarity) reduced Training CV MAE by \$26.93. The full engineered feature set improved test MAE by **-\$33.16** on random holdout (\$255.96 vs. \$289.12) and **-\$29.16** on strict zero-leakage grouped holdout (\$253.88 vs. \$283.04).

### 5. Explainable AI (TreeSHAP) vs. Causality
* **The Problem:** Black-box models cannot explain why a price is high or low.
* **Our Solution:** We implemented TreeSHAP to calculate exact additive Shapley values for each prediction (base value \$3,926.04), identifying dimension $y$, carat weight, and clarity as the top global drivers.
* **Critical Viva Distinction:** *"Sir, we explicitly document that SHAP values reflect statistical associations within the training data, NOT physical or market causation. Changing a grade on paper does not causally alter real-world market value without joint gemological reality."*

### 6. Market Segmentation: Lab-Grown vs. Natural Listings
* **The Problem:** Hardev Ranglani (2025) claimed a machine learning classifier could detect whether a diamond is natural or lab-grown from 4C features with 98% accuracy.
* **Our Scientific Critique:** We analyzed 6,480 carat-valid diamonds and showed observed listing price differences widening from **72.4% at 1.0 ct to 83.3% at 2.0 ct**. However, CVD/HPHT lab diamonds have the exact same chemical structure (carbon lattice) as mined diamonds! Our controlled ablation proved that a classifier achieves 98.15% accuracy on dataset tags primarily because of **inventory selection bias** (carat alone yields 82.64% accuracy) and scraper missingness in culet and fluorescence fields. Physical origin certification requires infrared spectroscopy (FTIR), which is why we describe the model as predicting recorded dataset `Type` categories, not certified physical origin.

---

## 3. Likely Challenging Viva Questions & Answers

### Q: "Can your model predict the live price of a diamond today in the jewelry market?"
**Answer:**  
"No, sir. We explicitly disclaim this on our website and in our PDF report. Our model predicts the price distribution present in its training dataset. Live retail prices vary by inflation, geopolitical supply chains, retailer brand premiums (e.g. Tiffany vs wholesale), and updated Rapaport price sheets. Our project provides an intelligent valuation estimate, not a certified legal appraisal."

### Q: "Why is error much higher on 2-carat diamonds than on 0.5-carat diamonds?"
**Answer:**  
"This is due to **heteroscedasticity** and market rarity. In our error analysis:
* For diamonds under 0.5 carats, MAE is only **\$65.89** because there are thousands of examples in the market with narrow price margins.
* For diamonds over 2.0 carats, diamond prices scale exponentially (Pareto-like distribution) and represent less than 4% of the dataset, leading to wider price spreads based on subtle cut geometry and dealer negotiations."

### Q: "What if a user enters negative or zero dimensions in the form?"
**Answer:**  
"Our Flask backend has strict numeric bounds validation: $x, y, z$ must be between 0.1 mm and 30 mm, carat must be between 0.1 and 5.0, and depth/table percentages must be within physical limits. Any invalid or non-finite inputs immediately trigger clean user validation errors and are blocked from reaching the model."

