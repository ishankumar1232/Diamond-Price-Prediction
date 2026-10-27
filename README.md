# DiamondIQ — Precision Diamond Valuation & Price Prediction

DiamondIQ is an intelligent, machine-learning-based diamond price prediction and gemological valuation web application built with Python, Flask, and modern supervised tree ensembles. It provides real-time market value estimation in Indian Rupees (₹ INR) and US Dollars ($ USD), sensitivity analysis, and downloadable appraisal-style PDF reports.

---

## Key Features

1. **Dual Valuation Pipelines:**
   - **Standard Pipeline:** Evaluates the classic 4Cs (Carat, Cut, Color, Clarity) alongside physical millimeter dimensions ($x, y, z$) and table/depth percentages. Powered by a blended ensemble of CatBoost Regressor (55%) and XGBoost Regressor (45%).
   - **Detailed Pipeline:** Evaluates 17 gemological specifications (including Polish, Symmetry, Girdle thickness, Culet size, and Certificate origin agency) using a Tuned Random Forest pipeline.
2. **Carat Price Explorer (What-If Sensitivity Tool):**
   - An interactive slider and quick preset chips (0.50 ct to 3.00 ct) allowing users to observe how estimated price scales with carat weight while keeping all other physical and quality characteristics fixed.
3. **Dual Currency Presentation:**
   - Instant pricing displayed in both Indian Rupees (₹) and US Dollars ($).
   - Live exchange rates fetched automatically from the European Central Bank (ECB) via Open Exchange Rates API with a 1-hour cache and automatic fallback.
4. **Local Feature Attribution:**
   - Explains prediction deviations by comparing each entered diamond attribute against a standard 1.00 ct benchmark diamond (Ideal Cut, G Color, VS2 Clarity, 61.5% Depth, 55.0% Table).
5. **Downloadable PDF Valuation Reports:**
   - Generates client-ready, formatted PDF documentation using ReportLab with point estimates, empirical MAE error bands, specifications table, and currency exchange metadata.
6. **Research Prototype — Origin Dataset Screening:**
   - An experimental CatBoost classifier trained to categorize diamond specifications against historical dataset origin distributions. Includes model confidence score and clear academic disclaimers.

---

## Model Architecture & Performance

| Pipeline | Model Architecture | Training Records | Test Evaluation (Held-out) |
| :--- | :--- | :---: | :--- |
| **Standard Pipeline** | CatBoost (55%) + XGBoost (45%) | 43,035 rows | **MAE:** $257.31 · **RMSE:** $492.73 · **R²:** 0.9841 (Test $N=10,759$) |
| **Detailed Pipeline** | Tuned Random Forest | 5,188 rows | **MAE:** $648.20 · **R²:** 0.9620 (Test $N=1,297$) |
| **Origin Classifier** | CatBoost Classifier | 5,188 rows | **Accuracy:** 97.99% on recorded dataset categories |

### Dataset Disclosure & Limitations
- **Cleaned Data:** Trained on commercial diamond listings. Twenty non-physical outlier rows (where dimensions $x, y, z \le 0$) were systematically audited and filtered.
- **Overlap Transparency:** In the standard test hold-out split ($N=10,759$), 70 records (0.65%) share identical 9-feature input profiles with training rows. Grouped profile-disjoint cross-validation maintains high generalization ($R^2 > 0.98$).
- **Valuation Caveat:** Algorithmic estimates reflect historical market transactions and wholesale listings; they do not constitute legally binding appraisals or live retail quotes.

---

## Technology Stack

- **Backend:** Python 3.11+, Flask (WSGI), Gunicorn
- **Machine Learning:** Scikit-Learn, CatBoost, XGBoost, Joblib, NumPy, Pandas
- **Reporting:** ReportLab (vector PDF generation)
- **Frontend:** Vanilla JavaScript (ES6+), CSS3 with Luxury Dark Glassmorphism, Jinja2 Templating
- **Typography:** Cormorant Garamond (Editorial Serif) & Plus Jakarta Sans (UI Sans)

---

## Local Setup & Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd Diamond-Price-Prediction
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.

---

## Running Verification & Tests

The project includes an end-to-end verification and viva demonstration suite:

```bash
# 1. End-to-end smoke tests (all routes, prediction pipelines, PDF generation, validation)
python research/test_app_endpoints.py

# 2. Viva & currency verification suite (INR/USD conversion, simulator, error handling)
python research/test_viva_enhancements.py

# 3. Route integrity check
python research/verify_homepage_links.py
```

---

## Production Deployment (Render)

This application is production-ready for deployment on [Render](https://render.com) or any containerized/WSGI Python host.

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn app:app`
- **Health Check Path:** `/health`
- **Environment Variables:**
  - `FLASK_SECRET_KEY`: Set to a secure random string (Render generates this automatically via `render.yaml`).
  - `PORT`: Automatically assigned by Render (defaults to 5000).
