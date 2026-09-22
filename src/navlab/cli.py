"""Public command line for reproducible navigation demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _demo(arguments: argparse.Namespace) -> int:
    from navlab.evaluation.runner import run_navigation

    result = run_navigation(
        output=arguments.output, scenario_name=arguments.scenario, planner_name=arguments.planner,
        controller_name=arguments.controller, seed=arguments.seed, budget=arguments.budget,
        backward_pass=not arguments.no_backward_pass, lookahead_braking=not arguments.no_lookahead_braking,
        max_steps=arguments.max_steps,
    )
    print(json.dumps(result["metrics"], indent=2))
    return 0


def _benchmark(arguments: argparse.Namespace) -> int:
    from navlab.evaluation.benchmark import run_benchmark

    result = run_benchmark(arguments.output, quick=arguments.quick, seed=arguments.seed)
    print(json.dumps(result["groups"], indent=2))
    return 0


def _train(arguments: argparse.Namespace) -> int:
    try:
        from navlab.learning.ppo import PPOConfig, train_three_seeds
        result = train_three_seeds(
            arguments.output, arguments.seeds, arguments.steps,
            PPOConfig(
                rollout_steps=arguments.rollout_steps, update_epochs=arguments.update_epochs,
                torch_threads=arguments.torch_threads, easy_fraction=arguments.easy_fraction,
            ),
            validation_episodes=arguments.validation_episodes,
        )
    except RuntimeError as exc:
        if "PyTorch" in str(exc):
            print(str(exc))
            return 2
        raise
    print(json.dumps({seed: record["summary"] for seed, record in result["ppo_held_out_test"].items()}, indent=2))
    return 0


def _evaluate(arguments: argparse.Namespace) -> int:
    try:
        from navlab.evaluation.metrics import aggregate_runs
        from navlab.learning.ppo import evaluate_policy, load_policy
        model, _ = load_policy(arguments.checkpoint)
        rows = evaluate_policy(
            model, arguments.split, list(range(arguments.seed_start, arguments.seed_start + arguments.episodes)),
            output=arguments.output / "episodes", checkpoint=arguments.checkpoint,
        )
    except RuntimeError as exc:
        if "PyTorch" in str(exc):
            print(str(exc))
            return 2
        raise
    result = {"checkpoint": str(arguments.checkpoint), "split": arguments.split, "episodes": rows, "summary": aggregate_runs(rows)}
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    return 0


def _showcase(arguments: argparse.Namespace) -> int:
    from navlab.showcase.builder import build_showcase

    result = build_showcase(arguments.results, arguments.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="navlab", description="Reproducible robot navigation experiments")
    parser.add_argument("--version", action="version", version="navlab 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run plan → trajectory → control and save evidence")
    demo.add_argument("--scenario", choices=("open", "detour", "narrow"), default="detour")
    demo.add_argument("--planner", choices=("astar", "rrtstar"), default="astar")
    demo.add_argument("--controller", choices=("pure_pursuit", "stanley", "pid", "lqr", "lqr_rate"), default="pure_pursuit")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--budget", type=int, default=2000)
    demo.add_argument("--max-steps", type=int, default=2000)
    demo.add_argument("--no-backward-pass", action="store_true")
    demo.add_argument("--no-lookahead-braking", action="store_true")
    demo.add_argument("--output", type=Path, required=True)
    demo.set_defaults(handler=_demo)
    benchmark = subparsers.add_parser("benchmark", help="run the three declared benchmark groups")
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--quick", action="store_true", help="smaller matrix; recorded in benchmark.json")
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.set_defaults(handler=_benchmark)
    train = subparsers.add_parser("train", help="train bounded PPO steering runs on CPU")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--seeds", nargs=3, type=int, default=[0, 1, 2])
    train.add_argument("--steps", type=int, default=12288, help="environment steps per seed; must divide rollout steps")
    train.add_argument("--rollout-steps", type=int, default=512)
    train.add_argument("--update-epochs", type=int, default=6)
    train.add_argument("--torch-threads", choices=(1, 2), type=int, default=1)
    train.add_argument("--easy-fraction", type=float, default=0.35, help="fixed initial curriculum share; validation/test always use standard")
    train.add_argument("--validation-episodes", type=int, default=20)
    train.set_defaults(handler=_train)
    evaluate = subparsers.add_parser("evaluate", help="evaluate a versioned PPO checkpoint on a fixed split")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    evaluate.add_argument("--episodes", type=int, default=20)
    evaluate.add_argument("--seed-start", type=int, default=0)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(handler=_evaluate)
    showcase = subparsers.add_parser("showcase", help="build an offline interactive showcase from frozen results")
    showcase.add_argument("--results", type=Path, default=Path("docs/results"), help="frozen results directory (default: docs/results)")
    showcase.add_argument("--output", type=Path, required=True, help="standalone showcase output directory")
    showcase.set_defaults(handler=_showcase)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return int(arguments.handler(arguments))
