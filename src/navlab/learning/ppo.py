"""A compact, real PPO implementation for one-dimensional bounded steering.

This module deliberately owns its action transformation: the executed action
is tanh-squashed before it enters ``TrackingEnv``, so PPO's log probability
matches the actuator input rather than the legacy HW3 clamp-after-sampling
behavior.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

OBSERVATION_SCHEMA_VERSION = "tracking-env-40-v1"
CHECKPOINT_SCHEMA_VERSION = 1


def _torch():
    try:
        # This project deliberately trains PPO on CPU; hide an incompatible
        # driver/device before PyTorch initialises CUDA discovery.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - tested by installation path
        raise RuntimeError("PPO needs PyTorch. Install the optional dependency with `pip install -e '.[rl]'`.") from exc
    return torch, nn


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 512
    update_epochs: int = 6
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.002
    max_grad_norm: float = 0.5
    hidden_size: int = 64
    torch_threads: int = 1
    easy_fraction: float = 0.35
    validation_interval_steps: int = 8192


def make_actor_critic(obs_dim: int, action_dim: int = 1, hidden_size: int = 64):
    torch, nn = _torch()

    def layer(in_features: int, out_features: int, gain: float) -> Any:
        module = nn.Linear(in_features, out_features)
        nn.init.orthogonal_(module.weight, gain)
        nn.init.constant_(module.bias, 0.0)
        return module

    class ActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.actor = nn.Sequential(layer(obs_dim, hidden_size, math.sqrt(2)), nn.Tanh(), layer(hidden_size, hidden_size, math.sqrt(2)), nn.Tanh(), layer(hidden_size, action_dim, 0.01))
            self.critic = nn.Sequential(layer(obs_dim, hidden_size, math.sqrt(2)), nn.Tanh(), layer(hidden_size, hidden_size, math.sqrt(2)), nn.Tanh(), layer(hidden_size, 1, 1.0))
            self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        def distribution(self, observations):
            return torch.distributions.Normal(self.actor(observations), self.log_std.exp().expand_as(self.actor(observations)))

        def value(self, observations):
            return self.critic(observations).squeeze(-1)

        def action_value(self, observations, deterministic: bool = False):
            distribution = self.distribution(observations)
            raw_action = distribution.mean if deterministic else distribution.rsample()
            action = torch.tanh(raw_action)
            log_prob = distribution.log_prob(raw_action) - torch.log(1.0 - action.square() + 1e-6)
            return action, log_prob.sum(-1), self.value(observations)

        def evaluate_actions(self, observations, actions):
            bounded = actions.clamp(-0.999999, 0.999999)
            raw_action = 0.5 * (torch.log1p(bounded) - torch.log1p(-bounded))
            distribution = self.distribution(observations)
            log_prob = distribution.log_prob(raw_action) - torch.log(1.0 - bounded.square() + 1e-6)
            return log_prob.sum(-1), distribution.entropy().sum(-1), self.value(observations)

    return ActorCritic()


@dataclass
class Rollout:
    observations: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    next_values: np.ndarray
    terminated: np.ndarray
    episode_end: np.ndarray
    advantages: np.ndarray | None = None
    returns: np.ndarray | None = None

    def finish(self, gamma: float, gae_lambda: float) -> None:
        """GAE: timeout bootstraps its value but never leaks into a new episode."""
        advantages = np.zeros_like(self.rewards, dtype=np.float32)
        gae = 0.0
        for index in range(len(self.rewards) - 1, -1, -1):
            bootstrap = 0.0 if self.terminated[index] else 1.0
            continue_episode = 0.0 if self.episode_end[index] else 1.0
            delta = self.rewards[index] + gamma * self.next_values[index] * bootstrap - self.values[index]
            gae = delta + gamma * gae_lambda * continue_episode * gae
            advantages[index] = gae
        self.advantages = advantages
        self.returns = advantages + self.values


def _tracking_env(seed: int, split: str, curriculum_stage: str = "standard"):
    from navlab.core import VehicleConfig
    from navlab.maps import make_tracking_scenario
    from navlab.sim import TrackingEnv
    from navlab.trajectory import build_trajectory

    vehicle = VehicleConfig()
    scenario = make_tracking_scenario(seed, split, curriculum_stage=curriculum_stage)
    result = build_trajectory(scenario.metadata["reference_points"], scenario, vehicle, backward_pass=True)
    if result.trajectory is None:
        raise RuntimeError(f"tracking trajectory is infeasible: {result.reason}")
    return TrackingEnv(scenario, result.trajectory, vehicle=vehicle, max_steps=900)


def _set_seed(seed: int, threads: int) -> None:
    torch, _ = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def _collect(
    model: Any, env: Any, obs: np.ndarray, route_cursor: int, reset_seed: int,
    running_episode_return: float, config: PPOConfig, device: str, route_seeds: tuple[int, ...], curriculum_stage: str,
):
    torch, _ = _torch()
    obs_dim = int(env.observation_space.shape[0])
    actions = np.empty((config.rollout_steps, 1), dtype=np.float32)
    observations = np.empty((config.rollout_steps, obs_dim), dtype=np.float32)
    log_probs = np.empty(config.rollout_steps, dtype=np.float32)
    rewards = np.empty(config.rollout_steps, dtype=np.float32)
    values = np.empty(config.rollout_steps, dtype=np.float32)
    next_values = np.empty(config.rollout_steps, dtype=np.float32)
    terminated = np.empty(config.rollout_steps, dtype=bool)
    episode_end = np.empty(config.rollout_steps, dtype=bool)
    episode_outcomes: list[dict[str, Any]] = []
    for step in range(config.rollout_steps):
        observation_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_t, log_prob_t, value_t = model.action_value(observation_t)
        action = action_t.cpu().numpy()[0].astype(np.float32)
        next_obs, reward, did_terminate, did_truncate, info = env.step(action)
        with torch.no_grad():
            next_value = float(model.value(torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)).item())
        observations[step], actions[step] = obs, action
        log_probs[step], rewards[step], values[step], next_values[step] = float(log_prob_t.item()), reward, float(value_t.item()), next_value
        terminated[step] = bool(did_terminate)
        episode_end[step] = bool(did_terminate or did_truncate)
        running_episode_return += float(reward)
        if did_terminate or did_truncate:
            episode_outcomes.append({
                "return": running_episode_return, "success": bool(did_terminate and info.get("success", False)),
                "collision": bool(info.get("collision", False)), "termination_reason": info.get("termination_reason"),
                "final_progress_m": float(info.get("s", 0.0)),
            })
            running_episode_return = 0.0
            route_cursor = (route_cursor + 1) % len(route_seeds)
            reset_seed += 1
            # A new deterministic geometry seed supplies the declared train
            # distribution instead of repeating one hand-picked route.
            env.close()
            env = _tracking_env(route_seeds[route_cursor], "train", curriculum_stage)
            obs, _ = env.reset(seed=reset_seed)
        else:
            obs = next_obs
    rollout = Rollout(observations, actions, log_probs, rewards, values, next_values, terminated, episode_end)
    rollout.finish(config.gamma, config.gae_lambda)
    return rollout, env, obs, route_cursor, reset_seed, running_episode_return, episode_outcomes


def _update(model: Any, optimizer: Any, rollout: Rollout, config: PPOConfig, device: str) -> dict[str, float]:
    torch, _ = _torch()
    assert rollout.advantages is not None and rollout.returns is not None
    observations = torch.as_tensor(rollout.observations, dtype=torch.float32, device=device)
    actions = torch.as_tensor(rollout.actions, dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor(rollout.log_probs, dtype=torch.float32, device=device)
    advantages = torch.as_tensor(rollout.advantages, dtype=torch.float32, device=device)
    returns = torch.as_tensor(rollout.returns, dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    indices = np.arange(len(observations))
    losses: list[tuple[float, float, float]] = []
    for _ in range(config.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), config.minibatch_size):
            batch = indices[start : start + config.minibatch_size]
            log_prob, entropy, value = model.evaluate_actions(observations[batch], actions[batch])
            ratio = (log_prob - old_log_probs[batch]).exp()
            surrogate = torch.minimum(ratio * advantages[batch], ratio.clamp(1 - config.clip_ratio, 1 + config.clip_ratio) * advantages[batch])
            policy_loss = -surrogate.mean()
            value_loss = 0.5 * (returns[batch] - value).square().mean()
            entropy_bonus = entropy.mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_bonus
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            losses.append((float(policy_loss.item()), float(value_loss.item()), float(entropy_bonus.item())))
    return {"policy_loss": float(np.mean([x[0] for x in losses])), "value_loss": float(np.mean([x[1] for x in losses])), "entropy": float(np.mean([x[2] for x in losses]))}


def checkpoint_payload(
    model: Any, optimizer: Any, config: PPOConfig, seed: int, steps: int, obs_dim: int,
    route_seeds: tuple[int, ...] = tuple(range(20)),
) -> dict[str, Any]:
    torch, _ = _torch()
    from navlab.evaluation.runner import source_digest
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "observation_schema": OBSERVATION_SCHEMA_VERSION,
        "observation_dim": obs_dim,
        "action_dim": 1,
        "seed": seed,
        "steps": steps,
        "ppo_config": asdict(config),
        "environment_contract": {
            "route_generator": "make_tracking_scenario:c2",
            "training_route_seeds": list(route_seeds),
            "preview_distances_m": [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0],
            "normalization_schema": OBSERVATION_SCHEMA_VERSION,
            "source_digest_sha256": source_digest(),
        },
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()},
    }


def load_policy(path: Path | str, device: str = "cpu"):
    torch, _ = _torch()
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION or payload.get("observation_schema") != OBSERVATION_SCHEMA_VERSION:
        raise ValueError("checkpoint does not match the frozen TrackingEnv observation contract")
    if payload.get("action_dim") != 1 or payload.get("observation_dim") != 40:
        raise ValueError("checkpoint action/observation dimensions are incompatible")
    config = PPOConfig(**payload["ppo_config"])
    model = make_actor_critic(40, 1, config.hidden_size).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def _write_evaluation_episode(output: Path, env: Any, trace: list[dict[str, Any]], metrics: dict[str, Any], checkpoint: Path | None) -> None:
    """Save PPO evaluation evidence in the same raw-first shape as demo runs."""
    import hashlib
    from navlab.evaluation.render import overview_png, replay_gif
    from navlab.evaluation.runner import _scenario_manifest, _trajectory_rows, _write_csv, environment_manifest

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "trace.csv", trace)
    _write_csv(output / "trajectory.csv", _trajectory_rows(env.trajectory))
    checkpoint_info = None
    if checkpoint is not None and checkpoint.exists():
        checkpoint_info = {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    manifest = {
        "schema_version": 1, "kind": "ppo_evaluation", "checkpoint": checkpoint_info,
        "environment": environment_manifest(), "scenario": _scenario_manifest(env.scenario),
        "observation_schema": OBSERVATION_SCHEMA_VERSION,
        "metrics": metrics,
        "environment_contract": {
            "action": {"shape": [1], "range": [-1.0, 1.0], "meaning": "normalized steering"},
            "observation_shape": list(env.observation_space.shape), "preview_distances_m": env._preview_distances.tolist(),
            "goal": {"position_tolerance_m": env._goal_tolerance, "speed_tolerance_mps": env._goal_speed},
        },
    }
    (output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    overview_png(output / "overview.png", env.scenario, None, env.trajectory, trace, f"PPO / {env.scenario.name} / {str(metrics['termination_reason']).upper()}", env.vehicle)
    replay_gif(output / "replay.gif", env.scenario, None, env.trajectory, trace)


def evaluate_policy(
    model: Any, split: str, seeds: list[int], device: str = "cpu", output: Path | str | None = None,
    checkpoint: Path | str | None = None,
) -> list[dict[str, Any]]:
    torch, _ = _torch()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        env = _tracking_env(seed, split)
        obs, info = env.reset(seed=seed)
        trace: list[dict[str, Any]] = []
        for step in range(env.max_steps):
            old_delta = info["policy_input"].state.delta
            with torch.no_grad():
                action, _, _ = model.action_value(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
            trace.append({"step": step, "dt": env.vehicle.dt, "previous_delta": old_delta, "x": info["x"], "y": info["y"], "yaw": info["yaw"], "delta": info["delta"], "cte": info["cte"], "heading_error": info["heading_error"], "v": info["v"], "v_ref": info["v_ref"], "s": info["s"], "collision": info["collision"], "success": info["success"], "a_lat_kinematic_mps2": abs(info["v"] ** 2 * np.tan(info["delta"]) / env.vehicle.wheelbase)})
            if terminated or truncated:
                break
        from navlab.evaluation.metrics import trace_metrics
        metrics = trace_metrics(trace, info.get("termination_reason"))
        rows.append({"seed": seed, "metrics": metrics})
        if output is not None:
            _write_evaluation_episode(Path(output) / f"episode-{seed:03d}", env, trace, metrics, Path(checkpoint) if checkpoint is not None else None)
        env.close()
    return rows


def evaluate_classical(name: str, split: str, seeds: list[int]) -> list[dict[str, Any]]:
    """Evaluate a classical policy through the identical TrackingEnv/trace path."""
    from navlab.control import make_controller
    from navlab.evaluation.metrics import trace_metrics

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        env = _tracking_env(seed, split)
        policy = make_controller(name, env.vehicle)
        obs, info = env.reset(seed=seed)
        policy.reset()
        trace: list[dict[str, Any]] = []
        for step in range(env.max_steps):
            old_delta = info["policy_input"].state.delta
            action = policy.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            trace.append({"step": step, "dt": env.vehicle.dt, "previous_delta": old_delta, "x": info["x"], "y": info["y"], "yaw": info["yaw"], "delta": info["delta"], "cte": info["cte"], "heading_error": info["heading_error"], "v": info["v"], "v_ref": info["v_ref"], "s": info["s"], "collision": info["collision"], "success": info["success"], "a_lat_kinematic_mps2": abs(info["v"] ** 2 * np.tan(info["delta"]) / env.vehicle.wheelbase)})
            if terminated or truncated:
                break
        rows.append({"seed": seed, "metrics": trace_metrics(trace, info.get("termination_reason"))})
        env.close()
    return rows


def train_ppo(output: Path | str, seed: int, total_steps: int, config: PPOConfig | None = None, validation_episodes: int = 4) -> dict[str, Any]:
    """Run a bounded CPU PPO job and write checkpoint, config, and validation data."""
    torch, _ = _torch()
    config = config or PPOConfig()
    if total_steps < config.rollout_steps:
        raise ValueError("total_steps must be at least one rollout")
    if total_steps % config.rollout_steps:
        raise ValueError("total_steps must divide exactly by rollout_steps for a declared budget")
    if not 0.0 <= config.easy_fraction < 1.0:
        raise ValueError("easy_fraction must be in [0, 1)")
    if config.validation_interval_steps <= 0 or config.validation_interval_steps % config.rollout_steps:
        raise ValueError("validation_interval_steps must be a positive multiple of rollout_steps")
    _set_seed(seed, config.torch_threads)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Route geometry and neural-network randomness intentionally use separate
    # seeds. Every independent training run sees this same declared route set.
    route_seeds = tuple(range(20))
    route_cursor = 0
    easy_steps = int(total_steps * config.easy_fraction) // config.rollout_steps * config.rollout_steps
    stage = "easy" if easy_steps else "standard"
    env = _tracking_env(route_seeds[route_cursor], "train", stage)
    reset_seed = seed
    obs, _ = env.reset(seed=reset_seed)
    obs_dim = int(env.observation_space.shape[0])
    if obs_dim != 40:
        raise RuntimeError(f"expected frozen 40-D observation schema, received {obs_dim}")
    model = make_actor_critic(obs_dim, 1, config.hidden_size).to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    history: list[dict[str, Any]] = []
    running_episode_return = 0.0
    from navlab.evaluation.metrics import aggregate_runs

    def rank(summary: dict[str, Any]) -> tuple[float, float, float]:
        quality = summary["quality_all_rollouts"]
        progress, cte = quality["final_progress_m"], quality["mean_abs_cte_m"]
        return (
            float(summary["success_rate"]), float(progress) if progress is not None else -float("inf"),
            -(float(cte) if cte is not None else float("inf")),
        )

    validation_history: list[dict[str, Any]] = []
    best_validation: list[dict[str, Any]] | None = None
    best_summary: dict[str, Any] | None = None
    best_step: int | None = None
    for steps in range(0, total_steps, config.rollout_steps):
        desired_stage = "easy" if steps < easy_steps else "standard"
        if desired_stage != stage:
            env.close()
            stage = desired_stage
            reset_seed += 1
            env = _tracking_env(route_seeds[route_cursor], "train", stage)
            obs, _ = env.reset(seed=reset_seed)
            running_episode_return = 0.0
        rollout, env, obs, route_cursor, reset_seed, running_episode_return, outcomes = _collect(
            model, env, obs, route_cursor, reset_seed, running_episode_return, config, "cpu", route_seeds, stage,
        )
        losses = _update(model, optimizer, rollout, config, "cpu")
        progresses = [outcome["final_progress_m"] for outcome in outcomes]
        entry = {
            "steps": steps + config.rollout_steps, "stage": stage, "completed_episodes": len(outcomes),
            "mean_completed_episode_return": float(np.mean([outcome["return"] for outcome in outcomes])) if outcomes else None,
            "success_rate_completed": float(np.mean([outcome["success"] for outcome in outcomes])) if outcomes else None,
            "median_final_progress_completed_m": float(np.median(progresses)) if progresses else None,
            "termination_counts": {reason: sum(outcome["termination_reason"] == reason for outcome in outcomes) for reason in ("success", "collision", "timeout")},
            **losses,
        }
        current_steps = steps + config.rollout_steps
        if current_steps % config.validation_interval_steps == 0 or current_steps == total_steps:
            validation = evaluate_policy(model, "validation", list(range(validation_episodes)))
            summary = aggregate_runs(validation)
            entry["validation_summary"] = summary
            validation_history.append({"steps": current_steps, "summary": summary})
            if best_summary is None or rank(summary) > rank(best_summary):
                best_validation, best_summary, best_step = validation, summary, current_steps
                torch.save(checkpoint_payload(model, optimizer, config, seed, current_steps, obs_dim, route_seeds), output / "best_checkpoint.pt")
        history.append(entry)
    assert best_validation is not None and best_summary is not None and best_step is not None
    checkpoint = output / "checkpoint.pt"
    torch.save(checkpoint_payload(model, optimizer, config, seed, total_steps, obs_dim, route_seeds), checkpoint)
    result = {
        "seed": seed, "steps": total_steps, "checkpoint": str(checkpoint),
        "observation_schema": OBSERVATION_SCHEMA_VERSION, "training_route_seeds": list(route_seeds),
        "curriculum": {"easy_stage": "easy", "standard_stage": "standard", "easy_steps": easy_steps, "total_steps": total_steps},
        "validation_selection_rule": "success_rate, then mean final progress over all rollouts, then lower mean absolute CTE",
        "selected_step": best_step, "selected_checkpoint": str(output / "best_checkpoint.pt"),
        "history": history, "validation": best_validation, "validation_summary": best_summary,
        "validation_history": validation_history,
    }
    (output / "train.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    env.close()
    return result


def train_three_seeds(
    output: Path | str, seeds: list[int], total_steps: int, config: PPOConfig | None = None,
    validation_episodes: int = 20,
) -> dict[str, Any]:
    """Train three independent models, select by held-out validation only, then test once."""
    if len(seeds) != 3:
        raise ValueError("the portfolio training command requires exactly three independent seeds")
    from navlab.evaluation.metrics import aggregate_runs

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs = [train_ppo(output / f"seed-{seed}", seed, total_steps, config, validation_episodes=validation_episodes) for seed in seeds]
    def selection_key(run: dict[str, Any]) -> tuple[float, float, float]:
        summary = run["validation_summary"]
        quality = summary["quality_all_rollouts"]
        progress = quality["final_progress_m"]
        cte = quality["mean_abs_cte_m"]
        # Failures remain in these aggregates. A missing measurement is always
        # ranked below a real number; zero is deliberately preserved as zero.
        return (
            float(summary["success_rate"]),
            float(progress) if progress is not None else -float("inf"),
            -(float(cte) if cte is not None else float("inf")),
        )

    # Validation selection only: success, then progress, then tracking quality.
    chosen = max(
        runs,
        key=selection_key,
    )
    test_seeds = list(range(20))
    ppo_test = {}
    for run in runs:
        model, _ = load_policy(run["selected_checkpoint"])
        episode_output = output / "selected-checkpoint-test" if run["seed"] == chosen["seed"] else None
        rows = evaluate_policy(model, "test", test_seeds, output=episode_output, checkpoint=run["selected_checkpoint"])
        ppo_test[str(run["seed"])] = {"episodes": rows, "summary": aggregate_runs(rows)}
    classical_test = {}
    for name in ("pure_pursuit", "stanley", "pid", "lqr", "lqr_rate"):
        rows = evaluate_classical(name, "test", test_seeds)
        classical_test[name] = {"episodes": rows, "summary": aggregate_runs(rows)}
    result = {
        "schema_version": 1,
        "training_seeds": seeds,
        "total_steps_per_seed": total_steps,
        "selection_split": "validation",
        "selected_seed": chosen["seed"],
        "runs": runs,
        "held_out_test_seeds": test_seeds,
        "ppo_held_out_test": ppo_test,
        "classical_held_out_test": classical_test,
    }
    (output / "training-comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
