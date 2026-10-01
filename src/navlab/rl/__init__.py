"""RL local planner (PPO) for the navigation-under-uncertainty benchmark.

Modules: ``features`` (observation encoder, action decoder), ``policy`` (NumPy actor behind the ``LocalPlanner`` contract),
``env`` (training environment around the benchmark episode machinery), ``train`` (PPO), ``select`` (checkpoint selection on the
tuning split), ``register`` (planner registry entries ``ppo_s0..2`` / ``ppo``), ``splits`` (training seed range).
Importing this package is light; ``import navlab.rl.register`` registers the planners.
"""
