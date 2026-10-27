"""
End-to-End Application Smoke Test & Verification Suite
=====================================================
Tests:
1. Flask app routes and status codes (/, /prediction, /health).
2. Standard Model prediction execution and response validation.
3. Detailed Model prediction execution and response validation.
4. Input validation (rejection of zero, negative, and out-of-range dimensions).
5. PDF Report generation and stream validity.
"""

import os
import sys

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

# Add repo root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app

def run_tests():
    print("=" * 70)
    print("RUNNING DIAMONDIQ APPLICATION SMOKE TESTS")
    print("=" * 70)

    client = app.test_client()
    passed = 0
    total = 0

    # Test 1: Health check
    total += 1
    resp = client.get("/health")
    data = resp.get_json()
    if resp.status_code == 200 and data.get("status") == "ok":
        print("✓ Test 1 Passed: /health endpoint returned 200 OK.")
        passed += 1
    else:
        print(f"✗ Test 1 Failed: /health status={resp.status_code}, data={data}")

    # Test 2: Homepage
    total += 1
    resp = client.get("/")
    if resp.status_code == 200 and b"Diamond" in resp.data:
        print("✓ Test 2 Passed: / homepage rendered successfully.")
        passed += 1
    else:
        print(f"✗ Test 2 Failed: / status={resp.status_code}")

    # Test 3: Prediction page (Standard mode)
    total += 1
    resp = client.get("/prediction?mode=standard")
    if resp.status_code == 200 and b"Standard Diamond Details" in resp.data:
        print("✓ Test 3 Passed: /prediction?mode=standard loaded.")
        passed += 1
    else:
        print(f"✗ Test 3 Failed: /prediction?mode=standard status={resp.status_code}")

    # Test 4: Prediction page (Detailed mode)
    total += 1
    resp = client.get("/prediction?mode=detailed")
    if resp.status_code == 200 and b"Detailed Diamond Details" in resp.data:
        print("✓ Test 4 Passed: /prediction?mode=detailed loaded.")
        passed += 1
    else:
        print(f"✗ Test 4 Failed: /prediction?mode=detailed status={resp.status_code}")

    # Test 5: Standard Prediction POST valid diamond
    total += 1
    valid_standard_data = {
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
    resp = client.post("/predict", data=valid_standard_data)
    if resp.status_code == 200 and (b"Estimated Price" in resp.data or b"ESTIMATED PRICE" in resp.data) and b"$" in resp.data:
        print("✓ Test 5 Passed: Standard prediction generated price successfully.")
        passed += 1
    else:
        print(f"✗ Test 5 Failed: Standard prediction status={resp.status_code}")

    # Test 6: Detailed Prediction POST valid diamond
    total += 1
    valid_detailed_data = {
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
    resp = client.post("/predict", data=valid_detailed_data)
    if resp.status_code == 200 and (b"Estimated Price" in resp.data or b"ESTIMATED PRICE" in resp.data):
        print("✓ Test 6 Passed: Detailed prediction generated price successfully.")
        passed += 1
    else:
        print(f"✗ Test 6 Failed: Detailed prediction status={resp.status_code}")

    # Test 7: Backend input validation - reject zero dimension
    total += 1
    invalid_zero_dim = valid_standard_data.copy()
    invalid_zero_dim["x"] = "0.00"
    resp = client.post("/predict", data=invalid_zero_dim)
    if b"Prediction Error" in resp.data or b"must be at least" in resp.data:
        print("✓ Test 7 Passed: Invalid zero dimension (x=0.00) properly rejected by backend validation.")
        passed += 1
    else:
        print("✗ Test 7 Failed: Backend permitted zero dimension x=0.00.")

    # Test 8: Backend input validation - reject negative dimension
    total += 1
    invalid_neg_dim = valid_standard_data.copy()
    invalid_neg_dim["z"] = "-1.50"
    resp = client.post("/predict", data=invalid_neg_dim)
    if b"Prediction Error" in resp.data or b"must be at least" in resp.data:
        print("✓ Test 8 Passed: Negative dimension (z=-1.50) properly rejected.")
        passed += 1
    else:
        print("✗ Test 8 Failed: Backend permitted negative dimension.")

    # Test 9: Backend input validation - reject negative carat
    total += 1
    invalid_carat = valid_standard_data.copy()
    invalid_carat["carat"] = "-0.50"
    resp = client.post("/predict", data=invalid_carat)
    if b"Prediction Error" in resp.data or b"must be at least" in resp.data:
        print("✓ Test 9 Passed: Negative carat properly rejected.")
        passed += 1
    else:
        print("✗ Test 9 Failed: Backend permitted negative carat.")

    # Test 10: PDF Report Download
    total += 1
    # First submit a valid prediction to populate session['last_report']
    client.post("/predict", data=valid_standard_data)
    pdf_resp = client.get("/download-report")
    if (
        pdf_resp.status_code == 200
        and pdf_resp.content_type == "application/pdf"
        and len(pdf_resp.data) > 1000
        and pdf_resp.data.startswith(b"%PDF")
    ):
        print(f"✓ Test 10 Passed: PDF report generated successfully ({len(pdf_resp.data)} bytes, valid PDF header).")
        passed += 1
    else:
        print(f"✗ Test 10 Failed: PDF report download status={pdf_resp.status_code}, type={pdf_resp.content_type}")

    # Test 11: Origin screening endpoint (GET & POST)
    total += 1
    origin_resp = client.get("/origin-screening")
    origin_post = client.post("/origin-screening", data={
        "Shape": "Round", "Cut": "Excellent", "Color": "G", "Clarity": "VS2",
        "Polish": "Excellent", "Symmetry": "Excellent", "Girdle": "Medium",
        "Culet": "None", "Fluorescence": "Unknown", "Carat Weight": "1.00",
        "Length/Width Ratio": "1.00", "Depth %": "61.5", "Table %": "55.0",
        "Length": "6.40", "Width": "6.40", "Height": "3.94"
    })
    if origin_resp.status_code == 200 and origin_post.status_code == 200 and b"Classification Result" in origin_post.data and b"Screening Error" not in origin_post.data:
        print("✓ Test 11 Passed: /origin-screening endpoint runs cleanly without errors.")
        passed += 1
    else:
        print(f"✗ Test 11 Failed: /origin-screening returned error or non-200 status.")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed} of {total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 70)

    if passed == total:
        print("ALL TESTS PASSED: Deployment verification clean.")
        return 0
    else:
        print("SOME TESTS FAILED: Please inspect errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())
