# Planning and trajectory design

This document describes the static, known-map navigation pipeline implemented
in `navlab`.  It is a simulation study in SI units with a y-up world frame and
an occupancy array indexed as `[y, x]`; it does not claim arbitrary-map,
perception, localisation, or real-vehicle performance.

## Global planning

`astar` searches an 8-connected lattice at the map resolution.  Its reported
path cost is optimal **on that fixed, footprint-clear lattice** for the
Euclidean edge costs used here.  This is not a claim of continuous-space
optimality and is not a comparison claim against RRT*.

`rrtstar` samples with a seeded local random generator, steers by a bounded
step, selects the least-cost visible parent within its neighbourhood, and
rewires cheaper neighbours.  Rewiring updates every descendant's
cost-to-come.  Each node that can connect to the goal is retained; after the
iteration budget, the final goal parent is selected using the rewired costs.
The result records both the first solution cost and final returned-path cost.
A finite sampling budget is never evidence that no geometric route exists.

Both planners reject an invalid start or goal.  A* distinguishes a completed
lattice search with no route from an expansion budget limit.  RRT* reports
`budget_exhausted` when it has no goal connection by its limit.  A returned
geometric path may still yield `trajectory infeasible` after vehicle-aware
conversion.

## Collision model

Planning uses a conservative point-path clearance: the circumscribed radius of
the rear-axle-referenced vehicle rectangle plus the configured safety margin.
This lets the grid planners reject passages that cannot contain the vehicle
without globally inflating the map.

The shared `CollisionChecker` performs the stronger check used by trajectory
validation and simulation.  It tests the padded oriented vehicle rectangle
against occupied cells, expanding it by a cell half-diagonal so cell-edge and
diagonal grazes are conservative.  A segment is sampled no farther apart than
0.4 grid cells; an SE(2) sweep is sampled no farther apart than 0.35 grid
cells or 0.05 rad.  The simulator calls this same swept checker on every
actual model step.  Thus the footprint is represented once in each check, not
by both a permanently inflated map and a second vehicle expansion.

## From a path to a reference trajectory

The converter first visibility-shortcuts a grid or tree path under the same
planning clearance.  If safe, it preserves a 4 m lead-in aligned with the
declared initial yaw rather than replacing the initial pose with a path
tangent.

It then tries a bounded candidate set.  The first candidate is a C2 cubic
spline with explicit start and final tangents.  It is resampled at at most
`min(0.18 m, 0.7 * grid resolution)`.  Because a spline can overshoot,
acceptance always requires the common validation: initial heading error at
most 0.30 rad, curvature within `tan(max_steer) / wheelbase`, and collision-
free swept rectangular body segments.  If that candidate fails, the bounded
fallbacks are two circular-fillet radii, the polyline, and three Chaikin
rounding levels.  A fallback is identified in trajectory metadata; no failed
candidate is silently repaired.

Arc length `s` is strictly increasing.  The reference speed is limited by
configured maximum speed, a 2.5 m/s² lateral-acceleration envelope, steering
rate through the spatial gradient of `atan(wheelbase * kappa)`, and a forward
acceleration pass.  The endpoint reference speed is zero.  When enabled, the
backward pass also enforces the configured deceleration limit; without it,
metadata explicitly marks terminal deceleration as unvalidated.

## Experiments and outcomes

The benchmark retains raw runs and includes failures in its denominator.

- The end-to-end group runs `open`, `detour`, and `narrow` through the real
  funnel.  `narrow` is intentionally a vehicle-footprint counterexample.
- The controller group gives each controller one literal immutable A*
  detour trajectory, so it does not compare controllers on different plans.
- The declared 2×2 ablation keeps the detour, A*, and pure pursuit fixed while
  varying backward speed pass and look-ahead braking independently.

The `--quick` benchmark intentionally changes the planner/controller matrix,
budget, and maximum steps, so it is a smoke test rather than a substitute for
the full matrix.

## Procedural tracking routes

`make_tracking_scenario(seed, split, curriculum_stage)` generates fixed-split
routes.  Geometry uses distinct seed offsets for `train`, `validation`, and
`test`; each scenario exposes JSON-safe `reference_points` that must still
pass the same trajectory validator.  The standard stage is a broad
alternating-wall S-route.  It provides repeatable tracking episodes, not a
claim of general road coverage.

The optional `easy` stage is train-only.  It has a distant single wall and a
wide turn, giving a policy useful forward progress and preview while still
requiring non-zero steering.  Requests for `easy` on validation or test are
rejected so curriculum geometry cannot leak into evaluation.
