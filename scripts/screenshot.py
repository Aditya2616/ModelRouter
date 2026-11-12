"""Load the UI in headless Chromium at desktop and phone sizes; save screenshots, report console errors,
horizontal overflow, small tap targets and the stream's frame rate.

    python scripts/screenshot.py http://127.0.0.1:8109/ docs/assets/live
"""

from __future__ import annotations

import json
import sys

from playwright.sync_api import sync_playwright

FPS_JS = """() => new Promise(res => { let n = 0, t0 = performance.now(), worst = 0, last = t0;
  function f(t) { worst = Math.max(worst, t - last); last = t; n++; if (t - t0 < 3000) requestAnimationFrame(f);
    else res({ fps: n / ((t - t0) / 1000), worst_frame_ms: worst }); } requestAnimationFrame(f); })"""

SMALL_TARGETS_JS = """() => [...document.querySelectorAll('button, input[type=range], a.btn, textarea')]
  .filter(e => e.offsetParent !== null).map(e => ({ t: (e.textContent || e.id || e.tagName).trim().slice(0, 30), r: e.getBoundingClientRect() }))
  .filter(x => x.r.height < 43.5 || x.r.width < 43.5).map(x => `${x.t} ${Math.round(x.r.width)}x${Math.round(x.r.height)}`)"""


def run(url: str, prefix: str) -> dict:
    report: dict = {}
    with sync_playwright() as p:
        b = p.chromium.launch()
        for name, opts in {
            "desktop": dict(viewport={"width": 1440, "height": 900}),
            "phone": dict(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True),
        }.items():
            ctx = b.new_context(**opts)
            page = ctx.new_page()
            errors: list[str] = []
            page.on("console", lambda m, e=errors: e.append(m.text) if m.type in ("error", "warning") else None)
            page.on("pageerror", lambda x, e=errors: e.append(str(x)))
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(2500)
            page.screenshot(path=f"{prefix}-{name}-top.png")
            page.screenshot(path=f"{prefix}-{name}-full.png", full_page=True)
            report[name] = {
                "console_errors": errors,
                "horizontal_overflow": page.evaluate("document.documentElement.scrollWidth > window.innerWidth"),
                "small_targets": page.evaluate(SMALL_TARGETS_JS),
                "stream": page.evaluate(FPS_JS),
                "mode": page.inner_text("#mode-badge"),
            }
            ctx.close()
        b.close()
    return report


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1], sys.argv[2]), indent=1))
