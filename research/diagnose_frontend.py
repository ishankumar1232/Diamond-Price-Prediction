import re
import sys
import os

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import STANDARD_FEATURES, DETAILED_FEATURES, DEFAULT_VALUES

with open("templates/prediction.html", encoding="utf-8") as f:
    html = f.read()

names = set(re.findall(r'name=["\']([^"\']+)["\']', html))
print("Form names found in prediction.html:", sorted(names))

missing_std = [f for f in STANDARD_FEATURES if f not in names]
missing_det = [f for f in DETAILED_FEATURES if f not in names]

print("Missing standard features in HTML:", missing_std)
print("Missing detailed features in HTML:", missing_det)

# Check IDs expected by script.js in prediction.html
with open("static/script.js", encoding="utf-8") as f:
    js = f.read()

ids_in_js = re.findall(r'getElementById\(["\']([^"\']+)["\']', js)
print("\nIDs queried in script.js:", set(ids_in_js))
for elem_id in set(ids_in_js):
    found = f'id="{elem_id}"' in html or f"id='{elem_id}'" in html
    print(f"  ID '{elem_id}': {'FOUND' if found else 'MISSING'}")

classes_in_js = re.findall(r'querySelector(?:All)?\(["\']\.([^"\'\s\[\],]+)', js)
print("\nClasses queried in script.js:", set(classes_in_js))
for cls in set(classes_in_js):
    found = f'class="' in html and cls in html
    print(f"  Class '.{cls}': {'FOUND' if found else 'MISSING'}")


