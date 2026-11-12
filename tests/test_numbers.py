"""Every number in the README, the web data and the pricing config must match results/ and config/."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_web_data_and_prices_match_results():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_numbers.py"), "--check"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_no_vendor_model_ids_hard_coded_in_source():
    banned = ("gpt-4", "gpt-3.5", "gpt-5", "gpt-6", "sonnet", "opus", "haiku")
    for f in (ROOT / "src").rglob("*.py"):
        if f.name == "data.py":  # RouterBench's own column names for the recorded models, not routing defaults
            continue
        text = f.read_text(encoding="utf-8").lower()
        assert not any(b in text for b in banned), f"{f} hard-codes a hosted model id"
