import sys
import os
import re

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app

client = app.test_client()

with open("templates/index.html", encoding="utf-8") as f:
    html = f.read()

links = set(re.findall(r'href=["\'](/[^"\'#]*)', html))
print("Checking internal routes referenced in index.html:", sorted(links))
for link in sorted(links):
    resp = client.get(link)
    print(f"  {link} -> {resp.status_code}")
    assert resp.status_code == 200, f"Dead link: {link}"

print("All navigation links verified clean!")

