"""Unified entrypoint.

Examples
--------

.. code-block:: bash

    python run.py ui           # open the interactive UI
    python run.py train        # run the headless training loop
    python run.py interact     # CLI chat
    python run.py evolve       # run a self-evolution demo
    python run.py tools        # list built-in tools

Every subcommand supports ``--help`` for its own options.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


REPO_ROOT = Path(__file__).resolve().parent


def cmd_ui(args: argparse.Namespace) -> int:
    from eva.ui.server import EVARuntime, RuntimeConfig, run_server

    runtime = EVARuntime(
        RuntimeConfig(
            workspace=Path(args.workspace).resolve(),
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            vocab_size=args.vocab_size,
            max_ram_gb=args.max_ram_gb,
            device=args.device,
        )
    )
    print(
        f"Starting EVA UI on http://{args.host}:{args.port} "
        f"(workspace={runtime.workspace})",
        flush=True,
    )
    run_server(runtime, host=args.host, port=args.port)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    import subprocess

    return subprocess.call(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train.py"),
            "--config",
            args.config,
            "--steps",
            str(args.steps),
        ]
    )


def cmd_interact(args: argparse.Namespace) -> int:
    import subprocess

    return subprocess.call(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "interact.py"),
            "--checkpoint",
            args.checkpoint,
        ]
    )


def cmd_evolve(args: argparse.Namespace) -> int:
    from eva.core.baby_brain import BabyBrain
    from eva.evolution.evolver import ArchitectureGenome, Evolver

    brain = BabyBrain(
        vocab_size=128, d_model=64, n_layers=2, n_heads=4, dtype_str="float32"
    )
    genome = ArchitectureGenome(d_model=64, n_layers=2, n_heads=4)
    evolver = Evolver(brain=brain, genome=genome, max_ram_gb=args.max_ram_gb)

    import random

    rng = random.Random(0)
    reward_bias = 0.1
    for step in range(args.steps):
        reward = rng.gauss(reward_bias, 0.05)
        event = evolver.step(step, reward)
        if event is not None:
            print(
                f"[step {step}] {event.kind}: {event.detail} "
                f"-> {brain.parameter_count:,} params"
            )
        # after growth, pretend the plateau breaks
        if event is not None:
            reward_bias *= 1.1
    print(f"Final: {brain.parameter_count:,} params, genome={genome.to_dict()}")
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    from eva.tools.builtins import register_default_tools
    from eva.tools.registry import ToolRegistry

    registry = register_default_tools(
        ToolRegistry(),
        workspace_root=Path(args.workspace).resolve() / "tool_workspace",
    )
    for name in registry.list_all():
        info = registry.get_documentation(name)
        if info is None:
            continue
        available = name in registry.discover()
        print(
            f"{name} [{'available' if available else 'unavailable'}] — "
            f"{info.description}"
        )
        for ex in info.examples:
            print(f"  example: {ex}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="EVA — unified entrypoint.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    ui = sub.add_parser("ui", help="Open the interactive web UI.")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument(
        "--workspace", default=str(REPO_ROOT / "workspace" / "default")
    )
    ui.add_argument("--d-model", type=int, default=192)
    ui.add_argument("--n-layers", type=int, default=4)
    ui.add_argument("--n-heads", type=int, default=6)
    ui.add_argument("--vocab-size", type=int, default=256)
    ui.add_argument("--max-ram-gb", type=float, default=8.0)
    ui.add_argument("--device", default="auto")
    ui.set_defaults(func=cmd_ui)

    tr = sub.add_parser("train", help="Run headless training.")
    tr.add_argument("--config", default="configs/default.yaml")
    tr.add_argument("--steps", type=int, default=1000)
    tr.set_defaults(func=cmd_train)

    it = sub.add_parser("interact", help="CLI chat with a checkpoint.")
    it.add_argument("--checkpoint", required=True)
    it.set_defaults(func=cmd_interact)

    ev = sub.add_parser("evolve", help="Run a self-evolution demo.")
    ev.add_argument("--steps", type=int, default=2000)
    ev.add_argument("--max-ram-gb", type=float, default=8.0)
    ev.set_defaults(func=cmd_evolve)

    tl = sub.add_parser("tools", help="List built-in tools.")
    tl.add_argument(
        "--workspace", default=str(REPO_ROOT / "workspace" / "default")
    )
    tl.set_defaults(func=cmd_tools)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
