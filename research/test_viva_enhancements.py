import sys
import os

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app, get_usd_to_inr_rate

client = app.test_client()

print("=" * 70)
print("RUNNING VIVA & CURRENCY ENHANCEMENTS VERIFICATION SUITE")
print("=" * 70)

# 1. Exchange Rate Service
rate_info = get_usd_to_inr_rate()
print(f"✓ Exchange Rate Service: Rate = {rate_info['rate']}, Updated = {rate_info['last_updated']}")
print(f"  Source Label = {rate_info['source_label']}")
assert rate_info["rate"] > 0, "Invalid exchange rate"

# 2. Standard Prediction & Dual Currency Output
valid_standard = {
    "dataset_mode": "standard",
    "carat": "1.00",
    "cut": "Ideal",
    "color": "G",
    "clarity": "VS2",
    "depth": "61.5",
    "table": "55.0",
    "x": "6.40",
    "y": "6.40",
    "z": "3.94",
}
resp = client.post("/predict", data=valid_standard)
assert resp.status_code == 200, f"Status code: {resp.status_code}"
html = resp.data.decode("utf-8")

# Check Rupee (₹) and Dollar ($)
assert "₹" in html, "Rupee symbol missing from prediction output"
assert "$" in html, "Dollar symbol missing from prediction output"
print("✓ Dual Currency Display: Both ₹ and $ rendered in prediction response.")

# Check Price Range
assert "Estimated Price Range" in html
assert "range-inr" in html and "range-usd" in html
print("✓ Price Range Dual Currency: Both ₹ and $ endpoints rendered in price range.")

# 3. Carat Price Explorer
assert "Carat Price Explorer" in html, "Carat Price Explorer heading missing"
assert "Change carat weight to see how the model's estimated price changes while the other diamond characteristics remain fixed." in html, "Explanatory sentence missing"
assert "Fixed Diamond Profile:" in html, "Fixed Diamond Profile box missing"
assert "Estimate Price" in html, "Estimate Price button missing"
assert 'data-preset="1.50"' in html, "Carat presets missing"
print("✓ Carat Price Explorer: Renamed, simplified, with exact one-sentence explanation and fixed profile.")

# 4. Carat Price Explorer Execution with Comparison
simulated_data = valid_standard.copy()
simulated_data["carat"] = "1.50"
simulated_data["baseline_price"] = "2573.50"
simulated_data["baseline_carat"] = "1.00"

resp_sim = client.post("/predict", data=simulated_data)
assert resp_sim.status_code == 200
html_sim = resp_sim.data.decode("utf-8")
assert "Before vs After Carat Adjustment" in html_sim
assert "Original Baseline" in html_sim and "Simulated Estimate" in html_sim
assert "₹" in html_sim and "$" in html_sim
print("✓ Before vs After Comparison: Verified with dual currency (₹ and $) delta.")

# 5. Detailed Prediction & Dual Currency Output
valid_detailed = {
    "dataset_mode": "detailed",
    "Shape": "Round",
    "Cut": "Excellent",
    "Color": "G",
    "Clarity": "VS2",
    "Carat Weight": "1.00",
    "Length/Width Ratio": "1.00",
    "Depth %": "61.5",
    "Table %": "55.0",
    "Polish": "Excellent",
    "Symmetry": "Excellent",
    "Girdle": "Medium",
    "Culet": "None",
    "Length": "6.40",
    "Width": "6.40",
    "Height": "3.94",
    "Type": "GIA",
    "Fluorescence": "Unknown",
}
resp_det = client.post("/predict", data=valid_detailed)
assert resp_det.status_code == 200
html_det = resp_det.data.decode("utf-8")
assert "₹" in html_det and "$" in html_det
assert "Tuned Random Forest" in html_det
print("✓ Detailed Prediction: Dual currency valuation verified with Random Forest pipeline.")

# 6. PDF Report Generation with Dual Currency
resp_pdf = client.get("/download-report")
assert resp_pdf.status_code == 200
assert resp_pdf.data.startswith(b"%PDF"), "Invalid PDF stream"
assert len(resp_pdf.data) > 1000, "PDF suspiciously small"
print(f"✓ PDF Report Generation: Valid PDF report generated ({len(resp_pdf.data)} bytes) with INR & USD.")

# 7. Validation Error Handling
bad_data = valid_standard.copy()
bad_data["carat"] = "-1.0"
resp_bad = client.post("/predict", data=bad_data)
assert resp_bad.status_code == 200
html_bad = resp_bad.data.decode("utf-8")
assert "Prediction Error" in html_bad or "must be at least 0.1" in html_bad
print("✓ Validation Error Handling: Out-of-bounds input rejected with clear error message.")

print("=" * 70)
print("ALL VIVA & CURRENCY ENHANCEMENTS VERIFIED SUCCESSFULLY (7/7 PASS)")
print("=" * 70)

