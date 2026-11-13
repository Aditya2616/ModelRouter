"""One command that regenerates every published number: download, safe conversion, embeddings, duplicate check,
training and evaluation for both pairs, the live experiment, web data export, the static docs build and the
README numbers. Writes results/reproduce.json with the wall-clock time of each step and the hardware.

    python scripts/reproduce_results.py              # everything
    python scripts/reproduce_results.py --skip-live  # without the local-model experiment

Run it in a clean clone: at the end it compares every regenerated results file with the committed one (timings
aside) and records the outcome in results/reproduce.json.

The first run downloads RouterBench (about 100 MB) and, for the live experiment, the two Qwen models
(about 4 GB). Models run on CUDA when available, otherwise on the CPU (MODELROUTER_DEVICE=cpu forces that).
"""

from __future__ import annotations

import json
import os
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


# measurements of time or machine load: these differ from run to run and are not part of the comparison
TIMING_KEYS = {"seconds", "wall_s", "items_per_s", "mean_s", "median_s", "p95_s", "prompt_tokens_per_s", "state_before"}


def _walk(a, b, path: str, out: list[str]) -> int:
    """Count the leaves of two JSON documents that differ (numbers: relative tolerance 1e-9)."""
    if isinstance(a, dict) and isinstance(b, dict):
        n = 0 if a.keys() == b.keys() else 1
        for k in a.keys() & b.keys():
            if k not in TIMING_KEYS:
                n += _walk(a[k], b[k], f"{path}/{k}", out)
        return n
    if isinstance(a, list) and isinstance(b, list):
        return (0 if len(a) == len(b) else 1) + sum(_walk(x, y, f"{path}[{i}]", out) for i, (x, y) in enumerate(zip(a, b, strict=False)))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        same = abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b)) or (a != a and b != b)
    else:
        same = a == b
    if not same:
        out.append(path)
    return 0 if same else 1


def compare_with_committed() -> dict:
    """Compare the regenerated results/*.json and *.parquet with the committed versions (git HEAD), ignoring timings."""
    import hashlib

    res: dict = {"json_files": {}, "parquet_files": {}}
    for f in sorted((ROOT / "results").glob("*.json")):
        if f.name == "reproduce.json":
            continue
        rel = f"results/{f.name}"
        old = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True)
        if old.returncode:
            res["json_files"][rel] = "not tracked"
            continue
        diffs: list[str] = []
        n = _walk(json.loads(old.stdout), json.loads(f.read_text(encoding="utf-8")), "", diffs)
        res["json_files"][rel] = {"differing_values": n, "paths": diffs[:10]}
    for f in sorted((ROOT / "results").glob("*.parquet")):
        old = subprocess.run(["git", "show", f"HEAD:results/{f.name}"], cwd=ROOT, capture_output=True)
        res["parquet_files"][f"results/{f.name}"] = (
            "identical" if old.returncode == 0 and hashlib.sha256(old.stdout).digest() == hashlib.sha256(f.read_bytes()).digest()
            else "differs")
    res["all_identical_except_timings"] = (all(isinstance(v, dict) and v["differing_values"] == 0 for v in res["json_files"].values())
                                           and all(v == "identical" for v in res["parquet_files"].values()))
    return res


def hardware() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(), "cpu_threads": os.cpu_count()}
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
    if "--compare-only" in sys.argv:   # re-do the comparison against the committed results after a finished run
        path = ROOT / "results" / "reproduce.json"
        out = json.loads(path.read_text(encoding="utf-8"))
        out["compared_with_committed"] = compare_with_committed()
        path.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(json.dumps(out["compared_with_committed"], indent=1))
        return
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
           "skipped_live": skip_live, "compared_with_committed": compare_with_committed()}
    (ROOT / "results" / "reproduce.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
