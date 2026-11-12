"""Copy the web UI and its recorded data into docs/ so it works as a static site (GitHub Pages)."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    src, dst = ROOT / "web", ROOT / "docs"
    for f in ("style.css", "app.js"):
        shutil.copy2(src / f, dst / f)
    html = (src / "index.html").read_text(encoding="utf-8")
    html = html.replace('name="mr-mode" content="auto"', 'name="mr-mode" content="static"')  # skip the backend probe
    (dst / "index.html").write_text(html, encoding="utf-8")
    shutil.copytree(src / "data", dst / "data", dirs_exist_ok=True)
    (dst / ".nojekyll").write_text("")
    print("docs/ updated")


if __name__ == "__main__":
    main()
