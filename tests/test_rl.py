"""RL local planner: env fidelity / determinism, information barrier, action limits, policy smoke, split + freeze invariants."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

import navlab.rl as rl_pkg
from navlab.benchmark.conditions import NOMINAL, Condition
from navlab.benchmark.config import code_digest
from navlab.benchmark.splits import TEST_BASE, TUNING_RANGE
from navlab.core import VehicleConfig
from navlab.local import Command, LocalObservation, LocalPlanner
from navlab.navigation import make_planner, run_episode
from navlab.perception import Lidar, LidarConfig
from navlab.rl.env import NavEnv, sample_condition
from navlab.rl.features import ObsEncoder, ObsSpec, decode_action
from navlab.rl.policy import RLPlanner, load_actor, save_actor
from navlab.rl.splits import TRAIN_RANGE, assert_train_seed
from navlab.world.generator import generate_scenario

VEH = VehicleConfig()
ROOT = Path(__file__).resolve().parents[1]


def _to_action(c: Command) -> np.ndarray:
    return np.array([c.accel / (VEH.max_accel if c.accel > 0 else VEH.max_decel), c.steer_rate / VEH.max_steer_rate])


def _random_layers(seed=0, hidden=32, scale=3.0):
    r = np.random.default_rng(seed)
    d = ObsSpec().dim
    return [(r.normal(0, scale / np.sqrt(d), (d, hidden)), np.zeros(hidden)), (r.normal(0, 1, (hidden, hidden)) / np.sqrt(hidden), np.zeros(hidden)),
            (r.normal(0, 5.0, (hidden, 2)), np.zeros(2))]


# ------------------------------------------------------------------ splits
def test_training_range_is_disjoint_from_tuning_and_test():
    assert TUNING_RANGE[1] <= TRAIN_RANGE[0] and TRAIN_RANGE[1] <= TEST_BASE
    assert_train_seed(1000), assert_train_seed(TEST_BASE - 1)
    for bad in (0, 999, TEST_BASE, TEST_BASE + 5):
        with pytest.raises(ValueError):
            assert_train_seed(bad)


# ------------------------------------------------------------------ environment
def test_env_is_deterministic():
    runs = []
    for _ in range(2):
        env = NavEnv()
        obs = [env.reset(1234, curriculum=0.8, pose="noisy")]
        rewards = []
        rng = np.random.default_rng(0)
        for _ in range(40):
            o, r, te, tr, _ = env.step(rng.uniform(-1, 1, 2))
            obs.append(o), rewards.append(r)
            if te or tr:
                break
        runs.append((np.array(obs), np.array(rewards), env.family, env.cond))
    assert np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1]) and runs[0][2:] == runs[1][2:]


@pytest.mark.parametrize("pose,family,seed", [("gt", "field", 100002), ("mcl", "rooms", 100004)])
def test_env_reproduces_run_episode(pose, family, seed):
    """Stepping NavEnv with pp_stop's commands gives exactly run_episode's outcome and final position."""
    env = NavEnv()
    env.reset(seed, family=family, condition=NOMINAL, pose=pose)
    planner = make_planner("pp_stop", VEH, 4.0, seed)
    planner.reset()
    info = {}
    while not env.done:
        _, _, _, _, info = env.step(_to_action(planner.act(env._last_obs)))
    ref = run_episode(generate_scenario(family, seed, NOMINAL.stress), "pp_stop", pose, seed)
    assert info["outcome"] == ref.summary["outcome"] and env.step_i == len(ref.trace)
    assert (info["x"], info["y"]) == pytest.approx((ref.trace[-1]["x"], ref.trace[-1]["y"]), abs=1e-6)


def test_env_actions_are_clipped_to_actuator_limits():
    env = NavEnv()
    env.reset(1500, family="field", condition=NOMINAL, pose="gt")
    for a in ([50.0, -50.0], [-9.0, 9.0], [1e9, 1e9]):
        o, r, te, tr, _ = env.step(np.array(a))
        assert np.all(np.isfinite(o)) and np.isfinite(r)
        assert abs(env.true.delta) <= VEH.max_steer + 1e-9 and -1e-9 <= env.true.v <= VEH.max_speed + 1e-9
        if env.done:
            break
    acc, rate = decode_action(np.array([5.0, -5.0]), VEH)
    assert acc == VEH.max_accel and rate == -VEH.max_steer_rate
    acc, rate = decode_action(np.array([-5.0, 5.0]), VEH)
    assert acc == -VEH.max_decel and rate == VEH.max_steer_rate


def test_curriculum_never_trains_on_the_worst_level():
    rng = np.random.default_rng(0)
    from navlab.benchmark.conditions import AXES
    conds = [sample_condition(rng, 1.0) for _ in range(300)]
    for fname, levels in AXES.values():
        vals = [getattr(c, fname) for c in conds]
        assert min(vals) >= levels[0] - 1e-12 and max(vals) <= levels[3] + 1e-12 < levels[4]


# ------------------------------------------------------------------ information barrier
def test_rl_modules_do_not_import_the_world():
    root = Path(rl_pkg.__file__).parent
    for name in ("features.py", "policy.py", "register.py"):
        for node in ast.walk(ast.parse((root / name).read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(n.startswith(("navlab.world", "navlab.sim", "navlab.maps", "navlab.benchmark", "navlab.perception")) for n in names), (name, names)
    assert list(__import__("inspect").signature(RLPlanner.act).parameters) == ["self", "obs"]
    assert isinstance(RLPlanner(VEH, _random_layers()), LocalPlanner)


def _scan_obs(hidden_mask, dyn, seed=5):
    world = dyn.__class__(**{**dyn.__dict__, "hidden": hidden_mask}).build_world(VEH)
    world.reset()
    lidar = Lidar(world.true_grid, LidarConfig(n_beams=120))
    true = dyn.scenario.start
    scan = lidar.scan(true, np.random.default_rng(seed), world.discs())
    path = np.array([[true.x, true.y], [true.x + 5, true.y], [true.x + 10, true.y]])
    return LocalObservation(0.0, 0.1, true, np.zeros((3, 3)), lidar.angles, scan, 20.0, path, dyn.scenario.goal, None)


def test_observation_ignores_hidden_obstacles_out_of_sensor_range_but_sees_those_in_range():
    dyn = generate_scenario("field", 100002, Condition("d", hidden_density=10.0, agent_density=0.0).stress)
    start = dyn.scenario.start
    hidden = dyn.hidden.copy()
    res = dyn.scenario.grid.resolution
    far = hidden.copy()
    rows, cols = np.nonzero(hidden)
    d = np.hypot((cols + 0.5) * res - start.x, (rows + 0.5) * res - start.y)
    assert (d > 24.0).any(), "scenario must have hidden cells beyond the 20 m sensor range"
    far[rows[d > 24.0], cols[d > 24.0]] = False
    enc_a, enc_b = ObsEncoder(VEH), ObsEncoder(VEH)
    prev = np.zeros(2)
    fa = enc_a.encode(_scan_obs(hidden, dyn), prev)
    fb = enc_b.encode(_scan_obs(far, dyn), prev)
    assert np.array_equal(fa, fb)
    # sensitivity: a box 5 m ahead of the robot must change the features
    near = hidden.copy()
    r0, c0 = int(start.y / res), int((start.x + 5.0) / res)
    near[r0 - 2:r0 + 2, c0:c0 + 2] = True
    assert not np.array_equal(ObsEncoder(VEH).encode(_scan_obs(near, dyn), prev), fa)


def test_features_have_fixed_dimension_and_range():
    dyn = generate_scenario("rooms", 100001)
    f = ObsEncoder(VEH).encode(_scan_obs(dyn.hidden, dyn), np.zeros(2))
    assert f.shape == (ObsSpec().dim,) and f.dtype == np.float32 and np.all(np.abs(f) <= 1.0 + 1e-6)


# ------------------------------------------------------------------ policy
def test_policy_roundtrip_and_command_limits(tmp_path):
    layers, spec = _random_layers(), ObsSpec()
    save_actor(tmp_path / "w.npz", layers, spec, {"note": "test"})
    loaded, spec2, meta = load_actor(tmp_path / "w.npz")
    assert spec2 == spec and meta["note"] == "test"
    dyn = generate_scenario("corridors", 100000)
    obs = _scan_obs(dyn.hidden, dyn)
    p1, p2 = RLPlanner(VEH, layers, spec), RLPlanner(VEH, loaded, spec2)
    for _ in range(5):
        c1, c2 = p1.act(obs), p2.act(obs)
        assert c1.accel == pytest.approx(c2.accel, abs=1e-4)
        assert -VEH.max_decel <= c1.accel <= VEH.max_accel and abs(c1.steer_rate) <= VEH.max_steer_rate
    p1.reset()
    assert np.all(p1._prev == 0)


def test_frozen_weights_load_and_act_through_the_registry():
    import navlab.rl.register as reg
    for s in reg.PPO_SEEDS:
        assert reg.weights_path(s).exists(), "trained weights are part of the repository"
        planner = make_planner(f"ppo_s{s}", VEH, 4.0, 0)
        assert isinstance(planner, LocalPlanner) and planner.name == f"ppo_s{s}"
        dyn = generate_scenario("field", 100002)
        c = planner.act(_scan_obs(dyn.hidden, dyn))
        assert np.isfinite([c.accel, c.steer_rate]).all()
        assert -VEH.max_decel <= c.accel <= VEH.max_accel and abs(c.steer_rate) <= VEH.max_steer_rate


# ------------------------------------------------------------------ freeze
def test_baseline_digest_is_unchanged_by_the_learned_planner():
    """The P3 freeze hashes the baseline stack's sources; navlab/rl lives outside them, so the digest still matches."""
    bench = ROOT / "docs/results/benchmark"
    cfg = json.loads((bench / "frozen_config.json").read_text())
    assert cfg["code_digest"] == code_digest()
    ppo = bench / "ppo_frozen.json"
    if ppo.exists():
        from navlab.benchmark.ppo_freeze import load_ppo_frozen
        art, base = load_ppo_frozen(ppo, bench / "frozen_config.json")
        assert art["baseline"]["code_digest"] == cfg["code_digest"] == base["code_digest"]
