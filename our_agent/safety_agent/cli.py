from __future__ import annotations
import argparse
import asyncio
import importlib
import json
from pathlib import Path
from .contracts import Config, GoalSpec
from .evidence import verify_chain


def main():
    parser = argparse.ArgumentParser(description="Q01-Q24 Runtime Assurance Agent")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("serve", help="Wrap an existing GR00T/Pi0.5 policy endpoint")
    p.add_argument("--goal", type=Path, required=True)
    p.add_argument("--upstream", required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18031)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--perception-provider", help="trusted local module:function returning PerceptionEvidence or None")
    p = sub.add_parser("verify-evidence")
    p.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "verify-evidence":
        print(json.dumps(verify_chain(args.path), indent=2))
    else:
        from .server import run_server
        provider = None
        if args.perception_provider:
            module, name = args.perception_provider.split(":", 1)
            provider = getattr(importlib.import_module(module), name)
        asyncio.run(run_server(GoalSpec.load(args.goal), Config.load(args.config), args.upstream,
                               args.host, args.port, args.output, provider))


if __name__ == "__main__":
    main()
