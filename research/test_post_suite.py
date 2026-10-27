import sys
import os

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app

client = app.test_client()

print("Testing POST endpoints...")

# 1. Standard prediction with baseline
post_data = {
    "dataset_mode": "standard",
    "carat": "1.25",
    "cut": "Ideal",
    "color": "G",
    "clarity": "VS2",
    "depth": "61.5",
    "table": "55.0",
    "x": "6.80",
    "y": "6.80",
    "z": "4.18",
    "baseline_price": "5000.00",
    "baseline_carat": "1.00",
}
r1 = client.post("/predict", data=post_data)
assert r1.status_code == 200, f"Status: {r1.status_code}"
assert b"Before vs After" in r1.data or b"Sensitivity" in r1.data
print("Test 1: Standard with baseline comparison -> PASSED")

# 2. Detailed prediction
post_det = {
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
r2 = client.post("/predict", data=post_det)
assert r2.status_code == 200, f"Status: {r2.status_code}"
assert b"Tuned Random Forest" in r2.data
print("Test 2: Detailed prediction -> PASSED")

# 3. PDF download after prediction
r3 = client.get("/download-report")
assert r3.status_code == 200, f"Status: {r3.status_code}"
assert r3.data.startswith(b"%PDF"), "Not a valid PDF"
print("Test 3: PDF download -> PASSED")

# 4. Input validation errors
bad_carat = post_data.copy()
bad_carat["carat"] = "0.05"
r4 = client.post("/predict", data=bad_carat)
assert r4.status_code == 200
assert b"Prediction Error" in r4.data or b"must be at least 0.1" in r4.data
print("Test 4: Out-of-bounds rejection -> PASSED")

print("All backend checks passed!")

