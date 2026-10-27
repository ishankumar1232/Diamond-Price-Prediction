from pathlib import Path
import os
import math
import logging
from io import BytesIO
from datetime import datetime
from xml.sax.saxutils import escape

import joblib
import numpy as np
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    send_file,
)

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)


app = Flask(__name__)
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "change-this-local-development-key",
)
app.logger.setLevel(logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# ==================================================
# CONFIGURATION
# ==================================================

STANDARD_FEATURES = [
    "carat", "cut", "color", "clarity",
    "depth", "table", "x", "y", "z",
]

DETAILED_FEATURES = [
    "Shape", "Cut", "Color", "Clarity", "Carat Weight",
    "Length/Width Ratio", "Depth %", "Table %",
    "Polish", "Symmetry", "Girdle", "Culet",
    "Length", "Width", "Height", "Type", "Fluorescence",
]

# Research prototype:
# Type is the label source, not an input feature.
ORIGIN_CATEGORICAL = [
    "Shape", "Cut", "Color", "Clarity", "Polish",
    "Symmetry", "Girdle", "Culet", "Fluorescence",
]

ORIGIN_NUMERIC = [
    "Carat Weight",
    "Length/Width Ratio",
    "Depth %",
    "Table %",
    "Length",
    "Width",
    "Height",
]

ORIGIN_FEATURES = ORIGIN_CATEGORICAL + ORIGIN_NUMERIC

ORIGIN_OPTIONS = {
    "Shape": [
        "Round", "Oval", "Pear", "Marquise", "Princess",
        "Heart", "Radiant", "Cushion Modified", "Emerald",
        "Cushion", "Square Radiant", "Unknown"
    ],
    "Cut": [
        "Ideal", "Excellent", "Very Good", "Astor", "Unknown"
    ],
    "Color": ["D", "E", "F", "G", "H", "Unknown"],
    "Clarity": [
        "FL", "IF", "VVS1", "VVS2", "VS1", "VS2", "Unknown"
    ],
    "Polish": ["Excellent", "Very Good", "Good", "Unknown"],
    "Symmetry": ["Excellent", "Very Good", "Good", "Unknown"],
    "Girdle": [
        "Medium to Slightly Thick",
        "Slightly Thick to Thick",
        "Medium to Thick",
        "Thick to Very Thick",
        "Medium",
        "Slightly Thick to Very Thick",
        "Medium to Very Thick",
        "Slightly Thick",
        "Thick",
        "Thin to Medium",
        "Thin to Slightly Thick",
        "Thin to Thick",
        "Very Thick",
        "Unknown",
    ],
    "Culet": [
        "Unknown", "Pointed", "Very Small", "Small",
        "Medium", "Very Large"
    ],
    "Fluorescence": [
        "Unknown", "Faint", "Medium", "Strong"
    ],
}

# Evaluation MAEs used by the app (reproduced on held-out test sets).
STANDARD_MAE = 231.25
DETAILED_MAE = 388.72

# ==================================================
# USD TO INR EXCHANGE RATE SERVICE
# ==================================================

DEFAULT_FALLBACK_USD_TO_INR = 86.50

_EXCHANGE_RATE_STATE = {
    "rate": DEFAULT_FALLBACK_USD_TO_INR,
    "last_updated": "Configured fallback rate",
    "is_fallback": True,
    "fetched_timestamp": 0.0,
    "source_label": f"Configured fallback rate: 1 USD = ₹{DEFAULT_FALLBACK_USD_TO_INR:.2f} (Offline)",
}


def get_usd_to_inr_rate():
    """
    Retrieve live USD/INR exchange rate with 2.5s timeout and 1-hour caching.
    Falls back to explicitly configured fallback rate if network is unavailable.
    """
    import time
    import urllib.request
    import json

    now = time.time()
    # Cache live rates for 1 hour (3600 seconds)
    if not _EXCHANGE_RATE_STATE["is_fallback"] and (now - _EXCHANGE_RATE_STATE["fetched_timestamp"] < 3600):
        return _EXCHANGE_RATE_STATE.copy()

    # If currently using fallback, retry at most once every 5 minutes
    if _EXCHANGE_RATE_STATE["is_fallback"] and _EXCHANGE_RATE_STATE["fetched_timestamp"] > 0 and (now - _EXCHANGE_RATE_STATE["fetched_timestamp"] < 300):
        return _EXCHANGE_RATE_STATE.copy()

    try:
        req = urllib.request.Request(
            "https://open.er-api.com/v6/latest/USD",
            headers={"User-Agent": "DiamondIQ-CurrencyService/1.0"}
        )
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            rate = float(data.get("rates", {}).get("INR", 0.0))
            if rate > 0:
                update_time = data.get("time_last_update_utc", datetime.now().strftime("%d %b %Y, %H:%M UTC"))
                _EXCHANGE_RATE_STATE["rate"] = round(rate, 2)
                _EXCHANGE_RATE_STATE["last_updated"] = update_time
                _EXCHANGE_RATE_STATE["is_fallback"] = False
                _EXCHANGE_RATE_STATE["fetched_timestamp"] = now
                _EXCHANGE_RATE_STATE["source_label"] = f"Live rate: 1 USD = ₹{rate:.2f} (as of {update_time})"
                return _EXCHANGE_RATE_STATE.copy()
    except Exception as exc:
        app.logger.warning("Live exchange rate fetch failed (%s). Using fallback rate.", exc)

    _EXCHANGE_RATE_STATE["rate"] = DEFAULT_FALLBACK_USD_TO_INR
    _EXCHANGE_RATE_STATE["is_fallback"] = True
    _EXCHANGE_RATE_STATE["fetched_timestamp"] = now
    _EXCHANGE_RATE_STATE["last_updated"] = "Configured fallback rate"
    _EXCHANGE_RATE_STATE["source_label"] = f"Configured fallback rate: 1 USD = ₹{DEFAULT_FALLBACK_USD_TO_INR:.2f} (Offline)"
    return _EXCHANGE_RATE_STATE.copy()

STANDARD_REFERENCE = {
    "carat": 1.0,
    "cut": "Ideal",
    "color": "G",
    "clarity": "VS2",
    "depth": 61.5,
    "table": 55.0,
    "x": 6.4,
    "y": 6.4,
    "z": 3.94,
}

DEFAULT_VALUES = {
    "carat": "1.00",
    "cut": "Ideal",
    "color": "G",
    "clarity": "VS2",
    "depth": "61.5",
    "table": "55",
    "x": "6.4",
    "y": "6.4",
    "z": "3.94",
    "Shape": "Round",
    "Cut": "Excellent",
    "Color": "G",
    "Clarity": "VS2",
    "Carat Weight": "1.00",
    "Length/Width Ratio": "1.00",
    "Depth %": "61.5",
    "Table %": "55",
    "Polish": "Excellent",
    "Symmetry": "Excellent",
    "Girdle": "Medium",
    "Culet": "None",
    "Length": "6.4",
    "Width": "6.4",
    "Height": "3.94",
    "Type": "GIA",
    "Fluorescence": "Unknown",
}


# ==================================================
# MODEL LOADING
# ==================================================

def load_pickle(filename):
    """Load a serialized model using joblib."""
    path = MODEL_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required model file not found: {path}"
        )

    try:
        loaded = joblib.load(path)

        # Compatibility helper for a preprocessor serialized
        # by another scikit-learn minor version.
        # Prefer a matching, pinned environment when possible.
        if (
            filename == "diamond_preprocessor.pkl"
            and not hasattr(loaded, "force_int_remainder_cols")
        ):
            loaded.force_int_remainder_cols = False

        return loaded

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model file '{filename}'. "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def load_models():
    models = {}

    # CatBoost model
    cat_path = MODEL_DIR / "catboost_diamond_model.cbm"

    if cat_path.exists():
        try:
            from catboost import CatBoostRegressor

            cat_model = CatBoostRegressor()
            cat_model.load_model(str(cat_path))
            models["catboost"] = cat_model

        except Exception as exc:
            raise RuntimeError(
                "Failed to load catboost_diamond_model.cbm: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    # XGBoost model
    xgb_path = MODEL_DIR / "xgboost_diamond_model.pkl"

    if xgb_path.exists():
        models["xgboost"] = load_pickle(
            "xgboost_diamond_model.pkl"
        )

    # Standard dataset preprocessing
    prep_path = MODEL_DIR / "diamond_preprocessor.pkl"

    if prep_path.exists():
        models["preprocessor"] = load_pickle(
            "diamond_preprocessor.pkl"
        )

    # Ensemble weights
    config_path = MODEL_DIR / "ensemble_config.pkl"

    if config_path.exists():
        models["config"] = load_pickle("ensemble_config.pkl")
    else:
        models["config"] = {
            "catboost_weight": 0.55,
            "xgboost_weight": 0.45,
        }

    # Detailed dataset model
    detailed_path = MODEL_DIR / "diamond_price_rf_pipeline.pkl"

    if detailed_path.exists():
        models["detailed"] = load_pickle(
            "diamond_price_rf_pipeline.pkl"
        )

    # Research classifier
    origin_path = (
        MODEL_DIR / "diamond_origin_classifier_grouped.pkl"
    )

    if origin_path.exists():
        models["origin_classifier"] = load_pickle(
            "diamond_origin_classifier_grouped.pkl"
        )

    return models


try:
    MODELS = load_models()
    MODEL_LOAD_ERROR = None

except Exception as exc:
    MODELS = {}
    MODEL_LOAD_ERROR = f"{type(exc).__name__}: {exc}"
    app.logger.exception("Model loading failed")


# ==================================================
# INPUT VALIDATION
# ==================================================

def numeric_value(
    form,
    key,
    default=None,
    minimum=None,
    maximum=None,
):
    raw = form.get(key, default)

    try:
        value = float(raw)

    except (TypeError, ValueError):
        raise ValueError(
            f"Please enter a valid number for {key}."
        )

    if not math.isfinite(value):
        raise ValueError(
            f"{key} must be a finite number."
        )

    if minimum is not None and value < minimum:
        raise ValueError(
            f"{key} must be at least {minimum}."
        )

    if maximum is not None and value > maximum:
        raise ValueError(
            f"{key} must not exceed {maximum}."
        )

    return value


def safe_prediction(value):
    result = float(np.asarray(value).reshape(-1)[0])

    if not math.isfinite(result):
        raise ValueError(
            "The model returned an invalid prediction."
        )

    return max(0.0, result)


def get_standard_values(form):
    values = {
        "carat": numeric_value(
            form, "carat", "1.00", 0.1, 5
        ),
        "cut": form.get("cut", "Ideal"),
        "color": form.get("color", "G"),
        "clarity": form.get("clarity", "VS2"),
        "depth": numeric_value(
            form, "depth", "61.5", 40, 80
        ),
        "table": numeric_value(
            form, "table", "55", 40, 80
        ),

        # Physical dimensions must be positive.
        "x": numeric_value(
            form, "x", "6.4", 0.1, 30
        ),
        "y": numeric_value(
            form, "y", "6.4", 0.1, 30
        ),
        "z": numeric_value(
            form, "z", "3.94", 0.1, 30
        ),
    }

    allowed = {
        "cut": [
            "Ideal", "Premium", "Very Good", "Good", "Fair"
        ],
        "color": ["D", "E", "F", "G", "H", "I", "J"],
        "clarity": [
            "IF", "VVS1", "VVS2", "VS1",
            "VS2", "SI1", "SI2", "I1",
        ],
    }

    for feature, options in allowed.items():
        if values[feature] not in options:
            raise ValueError(
                f"Invalid value for {feature}."
            )

    return values


def get_detailed_values(form):
    values = {}

    categorical = [
        "Shape", "Cut", "Color", "Clarity",
        "Polish", "Symmetry", "Girdle",
        "Culet", "Type", "Fluorescence",
    ]

    for key in categorical:
        values[key] = (
            form.get(key, DEFAULT_VALUES[key]).strip()
            or "Unknown"
        )

    numeric_limits = {
        "Carat Weight": (0.1, 5),
        "Length/Width Ratio": (0.1, 3),
        "Depth %": (1, 100),
        "Table %": (1, 100),

        # Prevent zero or negative dimensions.
        "Length": (0.1, 30),
        "Width": (0.1, 30),
        "Height": (0.1, 30),
    }

    for key, limits in numeric_limits.items():
        values[key] = numeric_value(
            form,
            key,
            DEFAULT_VALUES[key],
            limits[0],
            limits[1],
        )

    return values


# ==================================================
# STANDARD DATASET PREDICTION
# ==================================================

def predict_standard(values):
    required = ["catboost", "xgboost", "preprocessor"]
    missing = [key for key in required if key not in MODELS]

    if missing:
        raise RuntimeError(
            "Missing standard model components: "
            + ", ".join(missing)
        )

    row = pd.DataFrame(
        [[values[name] for name in STANDARD_FEATURES]],
        columns=STANDARD_FEATURES,
    )

    cat_prediction = safe_prediction(
        MODELS["catboost"].predict(row)
    )

    transformed = MODELS["preprocessor"].transform(row)

    xgb_prediction = safe_prediction(
        MODELS["xgboost"].predict(transformed)
    )

    config = MODELS.get("config", {})

    if not isinstance(config, dict):
        raise RuntimeError(
            "ensemble_config.pkl must contain a dictionary."
        )

    cat_weight = float(config.get("catboost_weight", 0.55))
    xgb_weight = float(config.get("xgboost_weight", 0.45))

    if (
        not math.isfinite(cat_weight)
        or not math.isfinite(xgb_weight)
        or cat_weight < 0
        or xgb_weight < 0
    ):
        raise RuntimeError(
            "Ensemble weights must be finite and non-negative."
        )

    total = cat_weight + xgb_weight

    if total <= 0:
        raise RuntimeError(
            "Invalid ensemble weights in ensemble_config.pkl."
        )

    cat_weight /= total
    xgb_weight /= total

    ensemble_prediction = (
        cat_weight * cat_prediction
        + xgb_weight * xgb_prediction
    )

    return safe_prediction(ensemble_prediction)


# ==================================================
# DETAILED DATASET PREDICTION
# ==================================================

def predict_detailed(values):
    pipeline = MODELS.get("detailed")

    if pipeline is None:
        raise RuntimeError(
            "Detailed model not found: "
            "models/diamond_price_rf_pipeline.pkl"
        )

    row = pd.DataFrame(
        [[values[name] for name in DETAILED_FEATURES]],
        columns=DETAILED_FEATURES,
    )

    return safe_prediction(pipeline.predict(row))


# ==================================================
# RESEARCH CLASSIFIER: DATASET TYPE SCREENING
# ==================================================

def get_origin_values(form):
    """
    Validate the diamond attributes used by the research
    classifier. The model predicts a dataset Type category;
    it does not independently verify a diamond's origin.
    """
    values = {}

    for key in ORIGIN_CATEGORICAL:
        values[key] = (
            form.get(key, "Unknown").strip()
            or "Unknown"
        )

    numeric_limits = {
        "Carat Weight": (0.1, 5),
        "Length/Width Ratio": (0.1, 3),
        "Depth %": (1, 100),
        "Table %": (1, 100),
        "Length": (0.1, 30),
        "Width": (0.1, 30),
        "Height": (0.1, 30),
    }

    for key, limits in numeric_limits.items():
        default = DEFAULT_VALUES.get(key, "")

        values[key] = numeric_value(
            form,
            key,
            default,
            limits[0],
            limits[1],
        )

    return values


def predict_origin_type(values):
    """
    Return a predicted Type label from the trained research
    classifier. The output is a dataset category, not a
    certified natural-versus-lab-grown determination.
    """
    model = MODELS.get("origin_classifier")

    if model is None:
        raise RuntimeError(
            "Research classifier not found. Expected file: "
            "models/diamond_origin_classifier_grouped.pkl"
        )

    row = pd.DataFrame(
        [[values[name] for name in ORIGIN_FEATURES]],
        columns=ORIGIN_FEATURES,
    )

    predicted_label = str(model.predict(row)[0])
    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(row)
        confidence = float(np.max(probabilities[0]))

        if not math.isfinite(confidence):
            confidence = None

    return predicted_label, confidence


# ==================================================
# FEATURE-BASED EXPLANATION FOR STANDARD MODEL
# ==================================================

def explain_standard(values, current_price):
    reference = STANDARD_REFERENCE.copy()

    try:
        reference_price = predict_standard(reference)

    except Exception:
        app.logger.exception(
            "Could not calculate reference prediction"
        )

        return {
            "current_price": current_price,
            "reference_price": None,
            "overall_difference": None,
            "features": [],
        }

    impacts = []

    for feature in STANDARD_FEATURES:
        changed = reference.copy()
        changed[feature] = values[feature]

        try:
            changed_price = predict_standard(changed)

        except Exception:
            app.logger.exception(
                "Could not calculate impact for %s",
                feature,
            )
            continue

        impact = changed_price - reference_price

        impacts.append({
            "feature": feature,
            "impact": round(impact, 2),
            "current_value": values[feature],
            "reference_value": reference[feature],
            "direction": (
                "higher" if impact > 0.01
                else "lower" if impact < -0.01
                else "same"
            ),
        })

    impacts.sort(
        key=lambda item: abs(item["impact"]),
        reverse=True,
    )

    return {
        "current_price": round(current_price, 2),
        "reference_price": round(reference_price, 2),
        "overall_difference": round(
            current_price - reference_price,
            2,
        ),
        "features": impacts[:5],
    }


# ==================================================
# PRICE RANGE
# ==================================================

def make_price_range(prediction, mode):
    # MAE-based error band, not a calibrated confidence interval.
    mae = (
        STANDARD_MAE
        if mode == "standard"
        else DETAILED_MAE
    )

    return {
        "low": round(max(0.0, prediction - mae), 2),
        "high": round(prediction + mae, 2),
        "mae": round(mae, 2),
        "note": (
            "This is an MAE-based typical error band, not a "
            "guaranteed price range or a calibrated confidence interval."
        ),
    }


# ==================================================
# PREDICTION PAGE RENDERING
# ==================================================

def render_prediction_page(
    mode="standard",
    prediction=None,
    model_used=None,
    form_values=None,
    error=None,
    explanation=None,
    comparison=None,
    price_range=None,
    baseline_price="",
    baseline_carat="",
):
    rate_info = get_usd_to_inr_rate()
    rate = rate_info["rate"]

    prediction_inr = round(prediction * rate, 2) if prediction is not None else None

    price_range_inr = None
    if price_range:
        price_range_inr = {
            "low": round(price_range["low"] * rate, 2),
            "high": round(price_range["high"] * rate, 2),
            "mae": round(price_range["mae"] * rate, 2),
            "note": price_range["note"],
        }

    comparison_inr = None
    if comparison:
        diff_inr = round(comparison["difference"] * rate, 2) if comparison.get("difference") is not None else None
        comparison_inr = {
            "original_price": round(comparison["original_price"] * rate, 2),
            "new_price": round(comparison["new_price"] * rate, 2),
            "difference": diff_inr,
            "percentage": comparison["percentage"],
            "original_carat": comparison["original_carat"],
            "new_carat": comparison["new_carat"],
        }

    return render_template(
        "prediction.html",
        mode=mode,
        prediction=prediction,
        prediction_inr=prediction_inr,
        exchange_rate=rate_info,
        model_used=model_used,
        form_values=form_values or DEFAULT_VALUES.copy(),
        error=error,
        explanation=explanation,
        comparison=comparison,
        comparison_inr=comparison_inr,
        price_range=price_range,
        price_range_inr=price_range_inr,
        baseline_price=baseline_price,
        baseline_carat=baseline_carat,
    )


# ==================================================
# HOME ROUTE
# ==================================================

@app.route("/")
def home():
    return render_template("index.html")


# ==================================================
# STANDARD / DETAILED FORM PAGE
# ==================================================

@app.route("/prediction")
def prediction_page():
    mode = request.args.get(
        "mode", "standard"
    ).lower()

    if mode not in ("standard", "detailed"):
        mode = "standard"

    return render_prediction_page(
        mode=mode,
        form_values=DEFAULT_VALUES.copy(),
    )


# ==================================================
# PRICE PREDICTION ROUTE
# ==================================================

@app.route("/predict", methods=["POST"])
def predict():
    mode = request.form.get(
        "dataset_mode", "standard"
    ).lower()

    if mode not in ("standard", "detailed"):
        mode = "standard"

    form_values = request.form.to_dict()

    baseline_price = request.form.get(
        "baseline_price", ""
    ).strip()

    baseline_carat = request.form.get(
        "baseline_carat", ""
    ).strip()

    try:
        if MODEL_LOAD_ERROR:
            raise RuntimeError(
                "Could not load the trained models. "
                + MODEL_LOAD_ERROR
            )

        if mode == "standard":
            values = get_standard_values(request.form)
            estimated_price = predict_standard(values)

            model_used = "CatBoost + XGBoost Ensemble"

            explanation = explain_standard(
                values,
                estimated_price,
            )

            # Keep the initial result as the What-If baseline.
            if not baseline_price or not baseline_carat:
                baseline_price = f"{estimated_price:.10f}"
                baseline_carat = f"{values['carat']:.2f}"

            comparison = None

            try:
                original_price = float(baseline_price)
                original_carat = float(baseline_carat)

                if (
                    math.isfinite(original_price)
                    and original_price >= 0
                    and math.isfinite(original_carat)
                ):
                    difference = (
                        estimated_price - original_price
                    )

                    percentage = (
                        difference / original_price * 100
                        if original_price > 0
                        else None
                    )

                    comparison = {
                        "original_price": original_price,
                        "new_price": estimated_price,
                        "difference": difference,
                        "percentage": percentage,
                        "original_carat": original_carat,
                        "new_carat": values["carat"],
                    }

            except (TypeError, ValueError):
                comparison = None

        else:
            values = get_detailed_values(request.form)
            estimated_price = predict_detailed(values)

            model_used = "Tuned Random Forest Pipeline"
            explanation = None
            comparison = None
            baseline_price = ""
            baseline_carat = ""

        price_range = make_price_range(
            estimated_price,
            mode,
        )

        rate_info = get_usd_to_inr_rate()
        rate = rate_info["rate"]
        pred_inr = round(estimated_price * rate, 2)

        # Store prediction for the PDF report.
        session["last_report"] = {
            "mode": mode,
            "prediction": float(estimated_price),
            "prediction_inr": float(pred_inr),
            "exchange_rate": rate_info,
            "model": model_used,
            "price_range": {
                "low": float(price_range["low"]),
                "high": float(price_range["high"]),
                "mae": float(price_range["mae"]),
                "note": str(price_range["note"]),
            },
            "inputs": {
                str(key): str(value)
                for key, value in values.items()
            },
            "created_at": datetime.now().strftime(
                "%d %B %Y, %I:%M %p"
            ),
        }

        return render_prediction_page(
            mode=mode,
            prediction=estimated_price,
            model_used=model_used,
            form_values=form_values,
            explanation=explanation,
            comparison=comparison,
            price_range=price_range,
            baseline_price=baseline_price,
            baseline_carat=baseline_carat,
        )

    except ValueError as exc:
        app.logger.warning("Prediction validation notice: %s", exc)

        return render_prediction_page(
            mode=mode,
            form_values=form_values,
            error=str(exc),
            baseline_price=baseline_price,
            baseline_carat=baseline_carat,
        )

    except Exception as exc:
        app.logger.exception("Prediction request failed")

        return render_prediction_page(
            mode=mode,
            form_values=form_values,
            error=f"{type(exc).__name__}: {exc}",
            baseline_price=baseline_price,
            baseline_carat=baseline_carat,
        )


# ==================================================
# RESEARCH CLASSIFICATION PAGE
# ==================================================

@app.route("/origin-screening", methods=["GET", "POST"])
def origin_screening():
    prediction = None
    confidence = None
    error = None
    form_values = DEFAULT_VALUES.copy()

    if request.method == "POST":
        form_values.update(request.form.to_dict())

        try:
            if MODEL_LOAD_ERROR:
                raise RuntimeError(
                    "Model loading failed: "
                    + MODEL_LOAD_ERROR
                )

            values = get_origin_values(request.form)

            prediction, confidence = predict_origin_type(
                values
            )

        except ValueError as exc:
            app.logger.warning("Origin screening validation notice: %s", exc)
            error = str(exc)

        except Exception as exc:
            app.logger.exception(
                "Research classification request failed"
            )
            error = f"{type(exc).__name__}: {exc}"

    return render_template(
        "origin_classification.html",
        feature_values=form_values,
        categorical_options=ORIGIN_OPTIONS,
        numeric_features=ORIGIN_NUMERIC,
        prediction=prediction,
        confidence=confidence,
        error=error,
    )


# ==================================================
# PDF REPORT
# ==================================================

@app.route("/download-report", methods=["GET"])
def download_report():
    report = session.get("last_report")

    if not report:
        return (
            "No prediction report is available. "
            "Please submit a prediction first.",
            400,
        )

    try:
        buffer = BytesIO()

        pdf = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=20 * mm,
            leftMargin=20 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title="DiamondIQ Prediction Report",
            author="DiamondIQ",
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "DiamondTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=29,
            alignment=1,
            textColor=colors.HexColor("#6847d9"),
            spaceAfter=8,
        )

        subtitle_style = ParagraphStyle(
            "DiamondSubtitle",
            parent=styles["Normal"],
            alignment=1,
            fontSize=10,
            textColor=colors.HexColor("#687084"),
            spaceAfter=18,
        )

        section_style = ParagraphStyle(
            "DiamondSection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#6847d9"),
            spaceBefore=14,
            spaceAfter=8,
        )

        body_style = ParagraphStyle(
            "DiamondBody",
            parent=styles["BodyText"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#273044"),
        )

        price_style = ParagraphStyle(
            "DiamondPrice",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=26,
            leading=32,
            alignment=1,
            textColor=colors.HexColor("#13845b"),
            spaceBefore=8,
            spaceAfter=8,
        )

        rate_info = report.get("exchange_rate") or get_usd_to_inr_rate()
        rate = rate_info["rate"]
        pred_usd = report["prediction"]
        pred_inr = report.get("prediction_inr", pred_usd * rate)

        story = [
            Paragraph("DIAMONDIQ", title_style),
            Paragraph(
                "Diamond Price Valuation Certificate",
                subtitle_style,
            ),
            HRFlowable(
                width="100%",
                thickness=1,
                color=colors.HexColor("#ded8f8"),
                spaceAfter=15,
            ),
            Paragraph("PREDICTION SUMMARY", section_style),
            Paragraph("Estimated Diamond Price (INR & USD)", subtitle_style),
            Paragraph(
                f"₹ {pred_inr:,.2f}",
                price_style,
            ),
            Paragraph(
                f"(${pred_usd:,.2f} USD)",
                subtitle_style,
            ),
        ]

        summary_rows = [
            ["Report generated", report["created_at"]],
            ["Dataset mode", report["mode"].title()],
            ["Prediction model", report["model"]],
            ["Exchange rate", f"1 USD = ₹{rate:.2f} ({rate_info['source_label']})"],
        ]

        summary_table = Table(
            summary_rows,
            colWidths=[48 * mm, 112 * mm],
        )

        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1),
             colors.HexColor("#f2efff")),
            ("TEXTCOLOR", (0, 0), (-1, -1),
             colors.HexColor("#273044")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4,
             colors.HexColor("#dedee8")),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))

        story.extend([summary_table, Spacer(1, 10)])
        story.append(
            Paragraph("ESTIMATED PRICE RANGE", section_style)
        )

        price_range = report["price_range"]

        range_rows = [
            ["Lower estimate", f"₹ {price_range['low'] * rate:,.2f}  (${price_range['low']:,.2f} USD)"],
            ["Predicted price", f"₹ {pred_inr:,.2f}  (${pred_usd:,.2f} USD)"],
            ["Upper estimate", f"₹ {price_range['high'] * rate:,.2f}  (${price_range['high']:,.2f} USD)"],
            ["MAE used", f"₹ {price_range['mae'] * rate:,.2f}  (${price_range['mae']:,.2f} USD)"],
        ]

        range_table = Table(
            range_rows,
            colWidths=[80 * mm, 80 * mm],
        )

        range_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#f7f7fb")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (-1, -1),
             colors.HexColor("#273044")),
            ("GRID", (0, 0), (-1, -1), 0.4,
             colors.HexColor("#dedee8")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))

        story.extend([range_table, Spacer(1, 8)])
        story.append(
            Paragraph(
                escape(str(price_range["note"])),
                body_style,
            )
        )

        story.append(
            Paragraph("DIAMOND SPECIFICATIONS", section_style)
        )

        input_rows = [["Feature", "Entered value"]]

        for feature, value in report["inputs"].items():
            input_rows.append([
                escape(str(feature)),
                escape(str(value)),
            ])

        input_table = Table(
            input_rows,
            colWidths=[80 * mm, 80 * mm],
            repeatRows=1,
        )

        input_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#6847d9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.4,
             colors.HexColor("#dedee8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))

        story.extend([input_table, Spacer(1, 20)])

        story.append(HRFlowable(
            width="100%",
            thickness=0.7,
            color=colors.HexColor("#dedee8"),
            spaceAfter=10,
        ))

        story.append(
            Paragraph("IMPORTANT DISCLAIMER", section_style)
        )

        disclaimer = (
            "This report contains a machine-learning estimate generated "
            "from the supplied diamond specifications. The price range "
            "is based on the model's mean absolute error (MAE); it is "
            "not a calibrated confidence interval or a guarantee of "
            "market value. Actual prices can vary with certification, "
            "seller, market conditions, and other factors. This report "
            "is not an official gemological appraisal."
        )

        story.append(Paragraph(disclaimer, body_style))
        story.append(Spacer(1, 16))

        story.append(
            Paragraph(
                "Generated by DiamondIQ · Diamond Price Prediction "
                "Using Machine Learning",
                subtitle_style,
            )
        )

        def add_page_number(canvas, document):
            canvas.saveState()
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.HexColor("#777777"))
            canvas.drawCentredString(
                A4[0] / 2,
                10 * mm,
                f"DiamondIQ | Page {document.page}",
            )
            canvas.restoreState()

        pdf.build(
            story,
            onFirstPage=add_page_number,
            onLaterPages=add_page_number,
        )

        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="DiamondIQ_Prediction_Report.pdf",
        )

    except Exception:
        app.logger.exception("PDF report generation failed")
        return (
            "Could not generate the PDF report. "
            "Check the Flask terminal for details.",
            500,
        )


# ==================================================
# HEALTH CHECK
# ==================================================

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "standard_model_loaded": all(
            key in MODELS
            for key in ("catboost", "xgboost", "preprocessor")
        ),
        "detailed_model_loaded": "detailed" in MODELS,
        "origin_classifier_loaded": (
            "origin_classifier" in MODELS
        ),
        "model_load_error": MODEL_LOAD_ERROR,
    })


# ==================================================
# RUN APPLICATION
# ==================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1")
    app.run(host="0.0.0.0", port=port, debug=debug)
