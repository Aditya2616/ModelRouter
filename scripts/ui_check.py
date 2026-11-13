"""Browser check of the interface in headless Chromium, at 1440x900 and 390x844 (touch).

    python scripts/ui_check.py http://127.0.0.1:8109/ docs/assets/served      # served by `modelrouter serve`
    python scripts/ui_check.py http://127.0.0.1:8110/ docs/assets/static      # docs/ through a plain static file server

For each viewport it reports: console errors and warnings, failed or 404 requests, horizontal overflow, tap targets
under 44 px, the frame rate of the stream at 1x / 4x / 16x, whether moving the threshold slider changes which colours
the dots on the canvas are (re-routing in flight), and the state of the preset chips. It writes screenshots of the
top of the page (the stream mid-flow), the full page, the preset chips and the tables.
"""

from __future__ import annotations

import json
import sys

from playwright.sync_api import sync_playwright

FPS_JS = """() => new Promise(res => { let n = 0, t0 = performance.now(), worst = 0, last = t0, frames = [];
  function f(t) { const d = t - last; worst = Math.max(worst, d); frames.push(d); last = t; n++;
    if (t - t0 < 2500) requestAnimationFrame(f);
    else { frames.shift(); frames.sort((a, b) => a - b);
      res({ fps: Math.round(10 * n / ((t - t0) / 1000)) / 10, worst_frame_ms: Math.round(worst), p95_frame_ms: Math.round(frames[Math.floor(frames.length * .95)]) }); } }
  requestAnimationFrame(f); })"""

SMALL_TARGETS_JS = """() => [...document.querySelectorAll('button, input[type=range], a.btn, textarea, select')]
  .filter(e => e.offsetParent !== null).map(e => ({ t: (e.textContent || e.id || e.tagName).trim().slice(0, 30), r: e.getBoundingClientRect() }))
  .filter(x => x.r.height < 43.5 || x.r.width < 43.5).map(x => `${x.t} ${Math.round(x.r.width)}x${Math.round(x.r.height)}`)"""

# share of canvas pixels close to each of the three dot colours (read from the CSS variables)
COLOURS_JS = """() => {
  const cv = document.querySelector('#stream'), g = cv.getContext('2d');
  const css = getComputedStyle(document.documentElement);
  const hex = (n) => { const s = css.getPropertyValue(n).trim(); const v = parseInt(s.slice(1), 16); return [v >> 16, (v >> 8) & 255, v & 255]; };
  const cols = { lime: hex('--lime'), bone: hex('--bone'), ember: hex('--ember') };
  const d = g.getImageData(0, 0, cv.width, cv.height).data, out = { lime: 0, bone: 0, ember: 0 };
  for (let i = 0; i < d.length; i += 4 * 3) for (const k in cols) {
    const c = cols[k]; if (Math.abs(d[i] - c[0]) < 14 && Math.abs(d[i + 1] - c[1]) < 14 && Math.abs(d[i + 2] - c[2]) < 14) out[k]++; }
  return out; }"""


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
            bad: list[str] = []
            # the getImageData warning comes from this script's own pixel probe, not from the page
            page.on("console", lambda m, e=errors: e.append(m.text) if m.type in ("error", "warning") and "getImageData" not in m.text else None)
            page.on("pageerror", lambda x, e=errors: e.append(str(x)))
            page.on("response", lambda r, e=bad: e.append(f"{r.status} {r.url}") if r.status >= 400 else None)
            page.on("requestfailed", lambda r, e=bad: e.append(f"failed {r.url}"))
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(4000)  # let the stream fill: the first screenshot shows it mid-flow
            page.screenshot(path=f"{prefix}-{name}-top.png")
            r = {"console_errors": errors, "bad_requests": bad, "mode": page.inner_text("#mode-badge"),
                 "horizontal_overflow": page.evaluate("document.body.scrollWidth > document.documentElement.clientWidth || document.documentElement.scrollWidth > document.documentElement.clientWidth"),
                 "small_targets": page.evaluate(SMALL_TARGETS_JS), "fps": {}}
            for sp in ("1", "4", "16"):
                page.click(f'[data-speed="{sp}"]')
                page.wait_for_timeout(600)
                r["fps"][sp + "x"] = page.evaluate(FPS_JS)
            page.click('[data-speed="4"]')
            # re-routing in flight: all small (no dot goes big) then all big, colours on the canvas must change
            seen = {}
            for preset in ("0", "1", "cpt95"):
                page.locator(f'[data-preset="{preset}"]').scroll_into_view_if_needed()
                page.click(f'[data-preset="{preset}"]')
                page.wait_for_timeout(2500)
                seen[preset] = {"colours": page.evaluate(COLOURS_JS), "threshold": page.inner_text("#thr-out"),
                                "pressed": page.get_attribute(f'[data-preset="{preset}"]', "aria-pressed")}
            r["preset_states"] = seen
            r["rerouting_changes_colours"] = seen["0"]["colours"] != seen["1"]["colours"]
            page.click('[data-speed="1"]')
            page.evaluate("document.querySelector('.slider-box').scrollIntoView({block: 'center'})")
            page.wait_for_timeout(800)
            page.screenshot(path=f"{prefix}-{name}-presets.png")  # viewport: stage, slider and preset chips
            for tid in ("#metrics-table", "#lofo-table", "#live-table"):
                page.evaluate(f"document.querySelector('{tid}').scrollIntoView({{block: 'center'}})")
                page.wait_for_timeout(800)
                page.screenshot(path=f"{prefix}-{name}-{tid[1:]}.png")
            page.evaluate("document.querySelector('#stream').scrollIntoView({block: 'center'})")
            page.wait_for_timeout(2500)
            page.screenshot(path=f"{prefix}-{name}-stream.png")  # the stream mid-flow
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(500)
            page.screenshot(path=f"{prefix}-{name}-full.png", full_page=True)
            r["table_overflow"] = page.evaluate(
                "[...document.querySelectorAll('.tablewrap')].filter(e => e.scrollWidth > e.clientWidth + 1).map(e => e.querySelector('table').id)")
            report[name] = r
            ctx.close()
        b.close()
    return report


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1], sys.argv[2]), indent=1))
