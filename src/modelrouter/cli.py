"""modelrouter command line."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="modelrouter", description="Difficulty-based LLM routing")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("data", help="download RouterBench (checksum-verified) and build the slim Parquet file")

    t = sub.add_parser("train", help="train and compare difficulty classifiers; write results/offline_*.json")
    t.add_argument("--pair", default="gpt35_vs_gpt4", choices=["gpt35_vs_gpt4", "mixtral_vs_gpt4"])
    t.add_argument("--fast", action="store_true", help="logistic regression + gradient boosting only")

    sub.add_parser("live", help="run the local Qwen small/big experiment (240 questions, about 25 min on CPU)")

    r = sub.add_parser("route", help="print the predicted difficulty of a prompt")
    r.add_argument("prompt")
    r.add_argument("--pair", default="gpt35_vs_gpt4")

    s = sub.add_parser("serve", help="start the gateway and web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8109)
    s.add_argument("--config", default=None)

    args = ap.parse_args(argv)
    if args.cmd == "data":
        from . import data

        data.build_slim(data.fetch())
        print("ok:", data.SLIM)
    elif args.cmd == "train":
        from . import train

        kinds = ["lr_lex", "lr", "hgb"] if args.fast else None
        res = train.run(args.pair, kinds=kinds)
        print(json.dumps({"selected": res["selected"], "test": {k: res["test"][k] for k in ("auc", "apgr", "cpt95", "aiq")}},
                         indent=1))
    elif args.cmd == "live":
        from . import live

        live.run()
    elif args.cmd == "route":
        from .router import DifficultyRouter

        rt = DifficultyRouter.load(f"models/router_{args.pair}.joblib")
        print(f"difficulty = {rt.score(args.prompt):.3f}")
    elif args.cmd == "serve":
        import os

        import uvicorn

        if args.config:
            os.environ["MODELROUTER_CONFIG"] = args.config
        uvicorn.run("modelrouter.gateway.app:create_app", factory=True, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
