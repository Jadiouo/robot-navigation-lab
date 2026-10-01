"""Experimental / legacy code kept for reference; not part of the headline navigation-under-uncertainty results.

``navlab.experimental.learning`` is the earlier hand-written PPO for *path tracking* (one-dimensional steering on the
40-D ``TrackingEnv`` observation, perfect localisation, no obstacles).  The headline learned method is the RL *local planner*
in :mod:`navlab.rl`, evaluated on the same frozen benchmark as DWA / MPPI.
"""
