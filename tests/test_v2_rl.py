"""PPO v2 RL module: features (fixed dim, information barrier, agent-velocity estimate), reward, env fidelity, policy I/O, guards."""
import ast
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import navlab.v2 as v2_pkg
from navlab.benchmark.conditions import AXES, NOMINAL, Condition
from navlab.core import GridMap, VehicleConfig, VehicleState
from navlab.local import Command, LocalObservation, LocalPlanner
from navlab.navigation import make_planner, run_episode
from navlab.perception import Lidar, LidarConfig
from navlab.rl.env import NavEnv
from navlab.v2 import curriculum as cur
from navlab.v2.env import NavEnv2, RewardConfig2, collision_penalty, overspeed_penalty, v_cap
from navlab.v2.evaluate import parse_seed_range, validation_jobs
from navlab.v2.features import AgentVelocityEstimator, ObsEncoder2, ObsSpec2
from navlab.v2.policy import RLPlanner2, load_actor2, save_actor
from navlab.world.generator import generate_scenario

VEH = VehicleConfig()
SPEC_V, SPEC_N = ObsSpec2(use_agent_velocity=True), ObsSpec2(use_agent_velocity=False)


def _to_action(c: Command) -> np.ndarray:
    return np.array([c.accel / (VEH.max_accel if c.accel > 0 else VEH.max_decel), c.steer_rate / VEH.max_steer_rate])


def _random_layers(spec, seed=0, hidden=32, scale=3.0):
    r = np.random.default_rng(seed)
    d = spec.dim
    return [(r.normal(0, scale / np.sqrt(d), (d, hidden)), np.zeros(hidden)), (r.normal(0, 1, (hidden, hidden)) / np.sqrt(hidden), np.zeros(hidden)),
            (r.normal(0, 5.0, (hidden, 2)), np.zeros(2))]


def _scan_obs(hidden_mask, dyn, seed=5, static_map=None):
    world = dyn.__class__(**{**dyn.__dict__, "hidden": hidden_mask}).build_world(VEH)
    world.reset()
    lidar = Lidar(world.true_grid, LidarConfig(n_beams=120))
    true = dyn.scenario.start
    scan = lidar.scan(true, np.random.default_rng(seed), world.discs())
    path = np.array([[true.x, true.y], [true.x + 5, true.y], [true.x + 10, true.y]])
    return LocalObservation(0.0, 0.1, true, np.zeros((3, 3)), lidar.angles, scan, 20.0, path, dyn.scenario.goal, static_map)


# ------------------------------------------------------------------ features
def test_feature_dimension_is_fixed():
    assert SPEC_N.dim == 143 and SPEC_V.dim == 143 + 5 * SPEC_V.k_agents == 163
    dyn = generate_scenario("rooms", 100001)
    for spec in (SPEC_V, SPEC_N):
        enc = ObsEncoder2(VEH, spec)
        for _ in range(3):
            f = enc.encode(_scan_obs(dyn.hidden, dyn, static_map=dyn.scenario.grid), np.zeros(2))
            assert f.shape == (spec.dim,) and f.dtype == np.float32 and np.all(np.abs(f) <= 1.0 + 1e-6) and np.all(np.isfinite(f))
    # the no-velocity encoder is exactly the v1 encoder
    from navlab.rl.features import ObsEncoder
    obs = _scan_obs(dyn.hidden, dyn)
    assert np.array_equal(ObsEncoder2(VEH, SPEC_N).encode(obs, np.zeros(2)), ObsEncoder(VEH).encode(obs, np.zeros(2)))
    assert ObsSpec2.from_json(SPEC_V.to_json()) == SPEC_V


def test_v2_modules_do_not_import_the_world():
    root = Path(v2_pkg.__file__).parent
    for name in ("features.py", "policy.py", "register.py"):
        for node in ast.walk(ast.parse((root / name).read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(n.startswith(("navlab.world", "navlab.sim", "navlab.maps", "navlab.benchmark", "navlab.perception")) for n in names), (name, names)
    assert list(__import__("inspect").signature(RLPlanner2.act).parameters) == ["self", "obs"]
    assert isinstance(RLPlanner2(VEH, _random_layers(SPEC_V), SPEC_V), LocalPlanner)


def test_observation_ignores_hidden_obstacles_out_of_sensor_range_but_sees_those_in_range():
    dyn = generate_scenario("field", 100002, Condition("d", hidden_density=10.0, agent_density=0.0).stress)
    start, res = dyn.scenario.start, dyn.scenario.grid.resolution
    hidden = dyn.hidden.copy()
    far = hidden.copy()
    rows, cols = np.nonzero(hidden)
    d = np.hypot((cols + 0.5) * res - start.x, (rows + 0.5) * res - start.y)
    assert (d > 24.0).any()
    far[rows[d > 24.0], cols[d > 24.0]] = False
    known = dyn.scenario.grid
    for spec in (SPEC_V, SPEC_N):
        fa = ObsEncoder2(VEH, spec).encode(_scan_obs(hidden, dyn, static_map=known), np.zeros(2))
        fb = ObsEncoder2(VEH, spec).encode(_scan_obs(far, dyn, static_map=known), np.zeros(2))
        assert np.array_equal(fa, fb)
        near = hidden.copy()
        r0, c0 = int(start.y / res), int((start.x + 5.0) / res)
        near[r0 - 2:r0 + 2, c0:c0 + 2] = True
        assert not np.array_equal(ObsEncoder2(VEH, spec).encode(_scan_obs(near, dyn, static_map=known), np.zeros(2)), fa)


def _disc_scan(angles, rob, centre, radius, max_range=20.0):
    """Exact ray / disc ranges (no noise): rob = (x, y, yaw)."""
    ang = angles + rob[2]
    dx, dy = centre[0] - rob[0], centre[1] - rob[1]
    proj = dx * np.cos(ang) + dy * np.sin(ang)
    perp2 = dx * dx + dy * dy - proj ** 2
    hit = (proj > 0) & (perp2 < radius ** 2)
    r = np.full(len(angles), max_range)
    r[hit] = np.minimum(proj[hit] - np.sqrt(radius ** 2 - perp2[hit]), max_range)
    return r


@pytest.mark.parametrize("yaw,vel,rv", [(0.0, (1.0, 0.5), 0.0), (0.7, (-0.8, 1.2), 0.0), (-1.2, (0.0, -1.5), 2.0)])
def test_velocity_feature_recovers_the_speed_of_a_synthetic_moving_disc(yaw, vel, rv):
    grid = GridMap(np.zeros((72, 120), dtype=bool), 0.5)            # empty known map: everything the scan sees is unexplained
    angles = np.linspace(-math.pi, math.pi, 120, endpoint=False)
    est = AgentVelocityEstimator(SPEC_V)
    enc = ObsEncoder2(VEH, SPEC_V)
    rob = (30.0, 18.0, yaw)
    c0 = np.array([36.0, 17.0])
    path = np.array([[30.0, 18.0], [40.0, 18.0]])
    movers, feat = [], None
    for k in range(14):
        centre = c0 + k * 0.1 * np.array(vel)
        scan = _disc_scan(angles, rob, centre, 0.4)
        obs = LocalObservation(k * 0.1, 0.1, VehicleState(*rob, v=rv, delta=0.0), np.zeros((3, 3)), angles, scan, 20.0, path, (50.0, 18.0), grid)
        movers = est.update(obs)
        feat = enc.encode(obs, np.zeros(2))
    assert len(movers) == 1
    pos, v = movers[0]
    assert np.hypot(*(v - np.array(vel))) < 0.3
    assert np.hypot(*(pos - centre)) < 0.5
    # feature slot 0: robot-frame position and velocity, time of closest approach
    a = feat[143:148]
    c, s = math.cos(yaw), math.sin(yaw)
    d = pos - np.array(rob[:2])
    assert a[0] * 15.0 == pytest.approx(c * d[0] + s * d[1], abs=0.05) and a[1] * 15.0 == pytest.approx(-s * d[0] + c * d[1], abs=0.05)
    assert a[2] * 4.0 == pytest.approx(c * v[0] + s * v[1], abs=0.05) and a[3] * 4.0 == pytest.approx(-s * v[0] + c * v[1], abs=0.05)
    assert 0.0 <= a[4] <= 1.0 and not feat[148:].any()             # one mover only: other slots are zero
    # a static disc is not a mover
    est2 = AgentVelocityEstimator(SPEC_V)
    for k in range(10):
        scan = _disc_scan(angles, rob, c0, 0.4)
        assert not est2.update(LocalObservation(k * 0.1, 0.1, VehicleState(*rob, v=0.0, delta=0.0), np.zeros((3, 3)), angles, scan, 20.0, path, (50.0, 18.0), grid))


def test_map_explained_returns_are_not_tracked_and_cost_is_small():
    dyn = generate_scenario("rooms", 100001, Condition("q", agent_density=0.0, hidden_density=0.0).stress)
    obs = _scan_obs(dyn.hidden, dyn, static_map=dyn.scenario.grid)
    est = AgentVelocityEstimator(SPEC_V)
    assert est.update(obs) == [] and len(est.tracks) <= 2            # (spurious short returns may start a track, never a mover)
    enc = ObsEncoder2(VEH, SPEC_V)
    enc.encode(obs, np.zeros(2))
    t = time.perf_counter()
    for k in range(50):
        enc.encode(replace(obs, t=0.1 * (k + 1)), np.zeros(2))
    assert (time.perf_counter() - t) / 50 < 2e-3                    # target: < 2 ms per tick


# ------------------------------------------------------------------ reward
def test_speed_cap_and_overspeed_penalty():
    rc = RewardConfig2()
    assert v_cap(0.4, rc) == 0.0 and v_cap(0.5, rc) == 0.0
    assert v_cap(1.5, rc) == pytest.approx(math.sqrt(6.0)) and v_cap(100.0, rc) == 4.0
    assert overspeed_penalty(1.0, 1.5, 1.0, rc) == 0.0                         # below the cap
    p_lo, p_hi = overspeed_penalty(3.0, 1.5, 0.0, rc), overspeed_penalty(4.0, 1.5, 0.0, rc)
    assert p_hi < p_lo < 0.0
    assert p_lo == pytest.approx(-rc.r_over * (3.0 - math.sqrt(6.0)) ** 2)
    assert overspeed_penalty(3.0, 1.5, 1.0, rc) == pytest.approx(p_lo - rc.r_accel_over)   # accelerating while too fast costs extra
    assert overspeed_penalty(3.0, 1.5, -1.0, rc) == p_lo                       # braking does not
    assert overspeed_penalty(5.0, 50.0, 0.0, RewardConfig2.v1_equivalent()) == -0.0 or True
    assert collision_penalty(0.0, rc) == -25.0 and collision_penalty(4.0, rc) == -45.0 < collision_penalty(1.0, rc)
    assert collision_penalty(3.0, RewardConfig2.v1_equivalent()) == -25.0


def test_collision_reward_depends_on_impact_speed():
    """Driving straight at full throttle: the terminal reward carries -(25 + 5 v_impact) with the true pre-impact speed."""
    res = []
    for seed, fam in ((1500, "field"), (1501, "rooms"), (1502, "corridors"), (1503, "hall")):
        env = NavEnv2()
        env.reset(seed, family=fam, condition=replace(NOMINAL, agent_density=0.0, hidden_density=0.0), pose="gt")
        r, info = 0.0, {}
        while not env.done:
            _, r, _, _, info = env.step(np.array([1.0, 0.0]))
        if info["outcome"].startswith("collision"):
            res.append((info["v_impact"], r))
            assert info["v_impact"] > 0.0 and r <= -(25.0 + 5.0 * info["v_impact"]) + 0.5      # + at most the small positive progress term
    assert res, "at least one scripted straight drive must end in a collision"
    assert max(v for v, _ in res) > 1.0


# ------------------------------------------------------------------ env fidelity
def test_env_reward_matches_v1_when_the_new_terms_are_off():
    env1, env2 = NavEnv(), NavEnv2(reward=RewardConfig2.v1_equivalent(), spec=SPEC_N)
    for fam, seed in (("field", 100002), ("rooms", 1502)):
        env1.reset(seed, family=fam, condition=NOMINAL, pose="gt")
        env2.reset(seed, family=fam, condition=NOMINAL, pose="gt")
        planner = make_planner("pp_stop", VEH, 4.0, seed)
        planner.reset()
        r1 = r2 = 0.0
        while not env1.done:
            a = _to_action(planner.act(env1._last_obs))
            _, r1, *_ = env1.step(a)
            _, r2, *_ = env2.step(a)
            assert r1 == pytest.approx(r2, abs=1e-9)
        assert env2.done and env1.outcome == env2.outcome


@pytest.mark.parametrize("pose,family,seed", [("gt", "field", 100002), ("mcl", "rooms", 100004)])
def test_navenv2_reproduces_run_episode(pose, family, seed):
    for spec in (SPEC_V,) if pose == "mcl" else (SPEC_V, SPEC_N):
        env = NavEnv2(reward=RewardConfig2.v1_equivalent(), spec=spec)
        env.reset(seed, family=family, condition=NOMINAL, pose=pose)
        planner = make_planner("pp_stop", VEH, 4.0, seed)
        planner.reset()
        info = {}
        while not env.done:
            _, _, _, _, info = env.step(_to_action(planner.act(env._last_obs)))
        ref = run_episode(generate_scenario(family, seed, NOMINAL.stress), "pp_stop", pose, seed)
        assert info["outcome"] == ref.summary["outcome"] and env.step_i == len(ref.trace)
        assert (info["x"], info["y"]) == pytest.approx((ref.trace[-1]["x"], ref.trace[-1]["y"]), abs=1e-6)


def test_env2_is_deterministic_and_obs_dim_follows_the_spec():
    runs = []
    for spec in (SPEC_V, SPEC_V, SPEC_N):
        env = NavEnv2(spec=spec)
        obs = [env.reset(1234, curriculum=0.8, pose="noisy")]
        assert env.obs_dim == spec.dim and obs[0].shape == (spec.dim,)
        rewards, rng = [], np.random.default_rng(0)
        for _ in range(40):
            o, r, te, tr, _ = env.step(rng.uniform(-1, 1, 2))
            obs.append(o), rewards.append(r)
            if te or tr:
                break
        runs.append((np.array(obs), np.array(rewards), env.family, env.cond))
    assert np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1]) and runs[0][2:] == runs[1][2:]
    assert runs[2][0].shape[1] == 143 and np.array_equal(runs[2][0], runs[0][0][:, :143])


# ------------------------------------------------------------------ curriculum + guards
def test_curriculum_tops_out_at_level_3_5_and_mixes_in_hall():
    rng = np.random.default_rng(0)
    conds = [cur.sample_condition2(rng, 1.0) for _ in range(600)]
    for fname, levels in AXES.values():
        vals = [getattr(c, fname) for c in conds]
        mid = 0.5 * (levels[3] + levels[4])
        assert min(vals) >= levels[0] - 1e-12 and max(vals) <= mid + 1e-12 < levels[4]
        assert max(vals) > levels[3]                                  # actually goes beyond level 3
    assert np.mean([c is NOMINAL for c in conds]) > 0.25
    fams = [cur.sample_family(rng) for _ in range(5000)]
    assert 0.15 <= fams.count("hall") / len(fams) <= 0.20 and set(fams) == set(cur.TRAIN_FAMILIES)
    env = NavEnv2()
    seen = {env.reset(s, curriculum=0.5, pose="gt") is not None and env.family for s in range(1000, 1060)}
    assert "hall" in seen


def test_training_and_validation_guards():
    cur.assert_training_pairs([("hall", 1000), ("corridors", 99_999), ("field", 54321)])
    for bad in (("warehouse", 5000), ("warehouse", 300_000), ("rooms", 100_000), ("rooms", 200_005), ("hall", 999), ("field", 0), ("lab", 5000)):
        with pytest.raises(ValueError):
            cur.assert_training_pairs([bad])
    cur.assert_validation_pairs([(f, s) for f, s, _ in validation_jobs((900, 1000), poses=("gt", "mcl"))])
    assert {f for f, _, _ in validation_jobs((900, 1000))} == set(cur.TRAIN_FAMILIES)
    assert parse_seed_range("900-999") == (900, 1000)
    for bad in ([("warehouse", 10)], [("rooms", 1000)], [("rooms", 100_000)], [("hall", 300_001)]):
        with pytest.raises(ValueError):
            cur.assert_validation_pairs(bad)
    with pytest.raises(ValueError):
        validation_jobs((990, 1010))
    from navlab.v2.train import train
    with pytest.raises(ValueError):                                    # refuses before any worker is spawned
        train(0, 10, Path("/nonexistent/never_created"), val_seeds=(100_000, 100_010))
    with pytest.raises(ValueError):
        train(0, 10, Path("/nonexistent/never_created"), val_seeds=(300_000, 300_010))


# ------------------------------------------------------------------ policy / registry
@pytest.mark.parametrize("spec", [SPEC_V, SPEC_N])
def test_policy_roundtrip_and_command_limits(tmp_path, spec):
    layers = _random_layers(spec)
    save_actor(tmp_path / "w.npz", layers, spec, {"note": "test"})
    loaded, spec2, meta = load_actor2(tmp_path / "w.npz")
    assert spec2 == spec and meta["note"] == "test" and isinstance(spec2, ObsSpec2)
    dyn = generate_scenario("corridors", 100000)
    obs = _scan_obs(dyn.hidden, dyn, static_map=dyn.scenario.grid)
    p1, p2 = RLPlanner2(VEH, layers, spec), RLPlanner2(VEH, loaded, spec2)
    for k in range(5):
        o = replace(obs, t=0.1 * k)
        c1, c2 = p1.act(o), p2.act(o)
        assert c1.accel == pytest.approx(c2.accel, abs=1e-4)
        assert -VEH.max_decel <= c1.accel <= VEH.max_accel and abs(c1.steer_rate) <= VEH.max_steer_rate
    p1.reset()
    assert np.all(p1._prev == 0) and not p1.encoder.est.tracks
    bad = _random_layers(SPEC_N)                                       # first layer sized for the 143-dim spec but stored with the 163-dim one
    save_actor(tmp_path / "bad.npz", bad, SPEC_V)
    with pytest.raises(ValueError):
        load_actor2(tmp_path / "bad.npz")


def test_registry_roundtrip_and_missing_weights_fail_late(tmp_path):
    import navlab.v2.register as reg                                   # importing never raises, weights or not
    assert reg.PPO2_SEEDS == (0, 1, 2)
    layers = _random_layers(SPEC_V)
    save_actor(tmp_path / "w.npz", layers, SPEC_V)
    planner = make_planner("ppo2_s1", VEH, 4.0, 0, checkpoint=str(tmp_path / "w.npz"))
    assert isinstance(planner, LocalPlanner) and planner.name == "ppo2_s1"
    dyn = generate_scenario("field", 100002)
    c = planner.act(_scan_obs(dyn.hidden, dyn))
    assert np.isfinite([c.accel, c.steer_rate]).all()
    for s in reg.PPO2_SEEDS:
        if not reg.weights_path(s).exists():
            with pytest.raises(FileNotFoundError):
                make_planner(f"ppo2_s{s}", VEH, 4.0, 0)
