"""Record a short demo GIF (feed -> diversity slider -> compare -> tradeoffs) from a running app (`make demo`)."""

import io
import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = Path(__file__).resolve().parents[1] / "reports" / "demo.gif"
frames: list[tuple[Image.Image, int]] = []


def settle(page, extra=2.0):
    page.wait_for_selector("[data-testid='stApp']", timeout=90000)
    for _ in range(160):
        if page.locator("[data-testid='stStatusWidget']").count() == 0:
            break
        time.sleep(0.4)
    time.sleep(extra)
    if page.locator("[data-testid='stMetricValue']").count():
        page.wait_for_function("document.querySelectorAll('[data-testid=stMetricValue]').length >= 4", timeout=60000)
        time.sleep(1.5)


def snap(page, ms):
    img = Image.open(io.BytesIO(page.screenshot())).convert("RGB").resize((1152, 800), Image.LANCZOS)
    frames.append((img, ms))


def scroll(page, y):
    page.evaluate(f"document.querySelector('[data-testid=\"stMain\"]')?.scrollTo(0, {y}) || window.scrollTo(0, {y})")
    time.sleep(1.2)


with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1440, "height": 1000})
    page.goto(BASE)
    settle(page, 3)
    snap(page, 3000)
    # switch to "two-tower only", then back to the full funnel with a lower MMR lambda
    page.get_by_text("Two-tower only").click()
    settle(page)
    snap(page, 2500)
    page.get_by_text("Full funnel (retrieve → rank → rerank)").click()
    settle(page)
    slider = page.get_by_role("slider").first
    slider.focus()
    for _ in range(4):
        slider.press("ArrowLeft")
    settle(page)
    snap(page, 3000)
    page.goto(f"{BASE}/Compare")
    settle(page, 3)
    snap(page, 3500)
    page.goto(f"{BASE}/Tradeoffs")
    settle(page, 3)
    snap(page, 3500)
    page.goto(f"{BASE}/Offline_results")
    settle(page, 3)
    snap(page, 3500)
    b.close()

imgs = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f, _ in frames]
imgs[0].save(OUT, save_all=True, append_images=imgs[1:], duration=[d for _, d in frames], loop=0, optimize=True)
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(imgs)} frames)")
