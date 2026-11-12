"""One command that regenerates every published number: download, safe conversion, embeddings, duplicate check,
training and evaluation for both pairs, the live experiment, web data export, the static docs build and the
README numbers. Writes results/reproduce.json with the wall-clock time of each step and the hardware.

    python scripts/reproduce_results.py              # everything
    python scripts/reproduce_results.py --skip-live  # without the local-model experiment

The first run downloads RouterBench (about 100 MB) and, for the live experiment, the two Qwen models
(about 4 GB). Models run on CUDA when available, otherwise on the CPU (MODELROUTER_DEVICE=cpu forces that).
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
STEPS = [
    ("download and convert RouterBench", [PY, "-m", "modelrouter.cli", "data"]),
    ("embed prompts", [PY, "scripts/embed_routerbench.py"]),
    ("duplicate check", [PY, "scripts/check_duplicates.py"]),
    ("train and evaluate, pair A", [PY, "-m", "modelrouter.cli", "train", "--pair", "gpt35_vs_gpt4"]),
    ("train and evaluate, pair B", [PY, "-m", "modelrouter.cli", "train", "--pair", "mixtral_vs_gpt4"]),
    ("live local experiment", [PY, "-m", "modelrouter.cli", "live"]),
    ("export web data", [PY, "scripts/build_web_data.py"]),
    ("build docs", [PY, "scripts/build_docs.py"]),
    ("regenerate README numbers", [PY, "scripts/check_numbers.py", "--write"]),
]


def hardware() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def main() -> None:
    skip_live = "--skip-live" in sys.argv
    timings = {}
    t_all = time.time()
    for name, cmd in STEPS:
        if skip_live and name.startswith("live"):
            continue
        print(f"\n== {name}: {' '.join(cmd[1:])}", flush=True)
        t0 = time.time()
        subprocess.run(cmd, check=True, cwd=ROOT)
        timings[name] = round(time.time() - t0, 1)
    out = {"total_seconds": round(time.time() - t_all, 1), "steps_seconds": timings, "hardware": hardware(),
           "skipped_live": skip_live}
    (ROOT / "results" / "reproduce.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
