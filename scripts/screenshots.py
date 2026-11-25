"""Capture the six app screens for the README (needs a running app: `make demo`)."""

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = Path(__file__).resolve().parents[1] / "reports" / "screenshots"
PAGES = [("1_feed", ""), ("2_compare", "Compare"), ("3_offline_results", "Offline_results"),
         ("4_tradeoffs", "Tradeoffs"), ("5_exposure_cold_start", "Exposure_and_cold_start"),
         ("6_engineering", "Engineering")]


def wait_idle(page):
    page.wait_for_selector("[data-testid='stApp']", timeout=90000)
    for _ in range(160):  # until Streamlit stops running scripts
        if page.locator("[data-testid='stStatusWidget']").count() == 0:
            break
        time.sleep(0.5)
    time.sleep(3)
    if page.locator("[data-testid='stMetricValue']").count():  # metrics render last on the feed
        page.wait_for_function("document.querySelectorAll('[data-testid=stMetricValue]').length >= 4", timeout=60000)
        time.sleep(2)


OUT.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1440, "height": 1000})
    for name, path in PAGES:
        page.goto(f"{BASE}/{path}")
        wait_idle(page)
        page.screenshot(path=str(OUT / f"{name}.png"), full_page=False)
        print("saved", name)
    b.close()
