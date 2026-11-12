"""Reproduce every number in the README, in order. Needs about 20 minutes on a laptop CPU plus downloads
(RouterBench 0-shot about 100 MB, Qwen2.5-0.5B/1.5B-Instruct about 4 GB).

    python scripts/reproduce_results.py            # everything
    python scripts/reproduce_results.py --skip-live
"""

from __future__ import annotations

import subprocess
import sys

STEPS = [
    ["modelrouter", "data"],
    ["modelrouter", "train", "--pair", "gpt35_vs_gpt4"],
    ["modelrouter", "train", "--pair", "mixtral_vs_gpt4"],
    ["modelrouter", "live"],
    [sys.executable, "scripts/build_web_data.py"],
    [sys.executable, "scripts/build_docs.py"],
]


def main() -> None:
    skip_live = "--skip-live" in sys.argv
    for cmd in STEPS:
        if skip_live and cmd[-1] == "live":
            continue
        print("\n$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
