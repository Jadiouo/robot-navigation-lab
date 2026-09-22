import numpy as np
import pytest

from navlab.learning.ppo import PPOConfig, Rollout
from navlab.evaluation.metrics import aggregate_runs


def test_timeout_bootstraps_but_true_terminal_does_not():
    rollout = Rollout(
        observations=np.zeros((2, 40), dtype=np.float32), actions=np.zeros((2, 1), dtype=np.float32),
        log_probs=np.zeros(2, dtype=np.float32), rewards=np.ones(2, dtype=np.float32),
        values=np.zeros(2, dtype=np.float32), next_values=np.array([10.0, 10.0], dtype=np.float32),
        terminated=np.array([True, False]), episode_end=np.array([True, True]),
    )
    rollout.finish(gamma=0.9, gae_lambda=1.0)
    # First is a physical terminal transition, so it must not see value=10.
    assert rollout.advantages[0] == pytest.approx(1.0)
    # Second represents a time limit: bootstrapping from the final observation is required.
    assert rollout.advantages[1] == pytest.approx(10.0)


def test_timeout_bootstrap_does_not_receive_advantage_from_next_episode():
    rollout = Rollout(
        observations=np.zeros((3, 40), dtype=np.float32), actions=np.zeros((3, 1), dtype=np.float32),
        log_probs=np.zeros(3, dtype=np.float32), rewards=np.array([1.0, 1.0, 100.0], dtype=np.float32),
        values=np.zeros(3, dtype=np.float32), next_values=np.array([0.0, 5.0, 0.0], dtype=np.float32),
        terminated=np.array([False, False, True]), episode_end=np.array([False, True, True]),
    )
    rollout.finish(gamma=1.0, gae_lambda=1.0)
    # The timeout's own final-state bootstrap is present (1 + 5), whereas the
    # following episode's reward 100 is excluded from its advantage.
    assert rollout.advantages[1] == pytest.approx(6.0)
    assert rollout.advantages[0] == pytest.approx(7.0)


def test_ppo_config_is_explicitly_bounded_for_cpu_runs():
    config = PPOConfig()
    assert config.rollout_steps > 0
    assert config.minibatch_size <= config.rollout_steps
    assert config.torch_threads in {1, 2}
    assert 0.0 <= config.easy_fraction < 1.0


def test_checkpoint_refuses_an_observation_contract_mismatch(tmp_path):
    torch = pytest.importorskip("torch")
    from navlab.learning.ppo import checkpoint_payload, load_policy, make_actor_critic

    model = make_actor_critic(40)
    optimizer = torch.optim.Adam(model.parameters())
    good = checkpoint_payload(model, optimizer, PPOConfig(), seed=3, steps=512, obs_dim=40)
    checkpoint = tmp_path / "good.pt"
    torch.save(good, checkpoint)
    _, restored = load_policy(checkpoint)
    assert restored["observation_dim"] == 40
    bad = dict(good)
    bad["observation_schema"] = "legacy-hw3-14d"
    bad_checkpoint = tmp_path / "bad.pt"
    torch.save(bad, bad_checkpoint)
    with pytest.raises(ValueError, match="observation contract"):
        load_policy(bad_checkpoint)


def test_aggregate_keeps_lowercase_serialized_timeout_in_the_denominator():
    summary = aggregate_runs([
        {"metrics": {"success": False, "collision": False, "termination_reason": "timeout", "final_progress_m": 2.0}},
        {"metrics": {"success": True, "collision": False, "termination_reason": "success", "final_progress_m": 4.0, "time_to_goal_s": 1.0}},
    ])
    assert summary["timeout_rate"] == pytest.approx(0.5)
    assert summary["success_rate"] == pytest.approx(0.5)
