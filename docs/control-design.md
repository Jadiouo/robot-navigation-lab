# Control and simulation design

This document specifies the implemented control contract for
`robot-navigation-lab`. It is a deterministic **kinematic simulation** of a
single car-like robot. Results apply to this stated model, its static
occupancy maps, and its collision checker. They are not claims of real-vehicle
safety, sensing, or robustness.

## Frame, bicycle model, and shared actuator

The frame uses metres, seconds, and radians; (x) points right and (y)
points up. The state reference point is the rear-axle centre. Positive steering
is a left turn. Given (q=(x,y,\psi,v,\delta)), every policy passes through
the same steering-angle and steering-rate actuator:

\[
\begin{aligned}
\delta_c &= \operatorname{clip}(\delta_{cmd},-\delta_{max},\delta_{max}),\\
\delta^+ &= \operatorname{clip}(\delta_c,\delta-\dot\delta_{max}\Delta t,
                                      \delta+\dot\delta_{max}\Delta t),\\
a^+ &= \operatorname{clip}(a,-a_{dec,max},a_{acc,max}),\\
v^+ &= \operatorname{clip}(v+a^+\Delta t,0,v_{max}),\\
\dot\psi &= v^+\tan(\delta^+)/L.
\end{aligned}
\]

Position is updated with semi-implicit Euler using (v^+). Thus an LQR-rate
policy has no separate rate actuator, and PPO has no separate speed actuator.

| Parameter | Default | Meaning |
|---|---:|---|
| (L) | 2.5 m | wheelbase |
| width | 1.6 m | rectangular body width |
| front / rear overhang | 0.8 / 0.8 m | measured from the axles |
| (Delta t) | 0.05 s | simulation step |
| (delta_{max}) | 0.6 rad | steering-angle limit |
| (dot\delta_{max}) | 0.8 rad/s | steering-rate limit |
| (a_{acc,max}) / (a_{dec,max}) | 2 / 3 m/s² | acceleration and braking limits |
| (v_{max}) | 8 m/s | speed limit |
| safety margin | 0.15 m | added once to the collision footprint |

Non-finite states and actions are rejected by the shared data/environment
validation. Yaw is wrapped to ([-pi,pi)).

## Projection, preview, and fair policy information

The environment projects the rear axle only within a local, forward path
window. It does not globally re-search the reference at every step, so a
crossing route cannot jump directly to its endpoint. For projected reference
((x_r,y_r,\psi_r)), it uses

\[
e=-(x-x_r)\sin\psi_r+(y-y_r)\cos\psi_r,
\qquad \theta_e=\operatorname{wrap}(\psi-\psi_r).
\]

Therefore (e>0) is left of the reference direction and
(	heta_e>0) is pointing left of it.

Each policy receives the same `PolicyInput`: state, (e), (	heta_e),
progress (s), current (v_{ref}), and a finite route-order preview at
([1,2,4,6,8,12,16]) m (clamped at the endpoint). Preview positions are in
the vehicle body frame; relative yaw, curvature, and reference speed are also
available. Classical policies never receive the full trajectory.

PPO encodes exactly that contract as 40 clipped float32 values:

\[
[e/4,\theta_e/\pi,v/v_{max},\delta/\delta_{max},(v-v_{ref})/v_{max}]
\]

followed by seven copies of

\[
[x_b/16,y_b/16,\psi_{rel}/\pi,L\kappa,v_{ref}/v_{max}].
\]

The sole action is shape `(1,)`, normalized desired steering in ([-1,1]).
The environment maps it to a requested angle and applies the actuator above.
PPO and every classical policy therefore share preview, vehicle, actuator,
collision, longitudinal, success, and evaluation contracts.

## Lateral control policies

All policies output a normalized desired angle; the model applies the physical
clipping afterwards.

### Pure Pursuit

The look-ahead is (L_d=2.0+0.45v) metres. The policy scans preview points in
route order and selects the first whose actual body-frame range meets (L_d);
it deliberately does not assume Euclidean ranges are sorted on a bend. For
selected target ((x_t,y_t)), (r=sqrt{x_t^2+y_t^2}),

\[
\delta=\arctan2(2Ly_t,r^2).
\]

### Front-axle Stanley

Stanley is evaluated at its front axle. It transforms the common rear-axle
projection rather than mislabeling rear error:

\[
e_f=e+L\sin\theta_e,
\qquad
\delta=-\theta_e-\arctan2(k e_f,\max(v,0.5)),
\]

with (k=1.2). The signs make a vehicle left of the path command a right
turn.

### PID baseline

The PID baseline uses signed rear-axle error and heading:

\[
\delta=-\left(0.65e+0.02\int e\,dt+0.16\frac{de}{dt}+1.0\theta_e\right).
\]

Its integral resets per episode and is clipped to ([-2,2]). It is a
baseline, not an optimality claim.

### Discrete LQR: angle and rate variants

Both LQR variants solve the DARE with SciPy and verify the residual. At actual
near-zero speed, their linearization uses a conservative 1 m/s floor to avoid
an uncontrollable numerical pair; the physical bicycle still uses its true
speed and limits.

The steering-angle form uses

\[
x=[e,\theta_e]^T,
A=\begin{bmatrix}1&v\Delta t\\0&1\end{bmatrix},
B=\begin{bmatrix}0\\v\Delta t/L\end{bmatrix},
\qquad \delta=\delta_{ff}-Kx,
\]

where (delta_{ff}=\arctan(L\kappa)) comes from local preview curvature.
It uses (Q=\operatorname{diag}(3,3)) and
(R=\max(1.5,(v/3)^2)).

The rate form uses

\[
x=[e,\theta_e,\delta-\delta_{ff}]^T,
A=\begin{bmatrix}1&v\Delta t&0\\0&1&v\Delta t/L\\0&0&1\end{bmatrix},
B=\begin{bmatrix}0\\0\\\Delta t\end{bmatrix},
\]

with (Q=\operatorname{diag}(3,3,0.8)) and the same (R). It computes
(dot\delta=-Kx), then (delta_{cmd}=delta+\dot\delta\Delta t). This is
a true rate command, not an unscaled angle correction, and it still passes
through the common actuator.

## Shared longitudinal loop and terminal braking

Longitudinal control is outside the lateral-policy comparison. The target
uses the current and first-forward speed reference so a trajectory beginning
with (v_{ref}(0)=0) can launch. With look-ahead braking, every preview limit
at distance (D_i) supplies

\[
v_{safe,i}=\sqrt{v_{ref,i}^2+2a_{eff}D_i},
\qquad a_{eff}=0.8a_{dec,max}.
\]

The controller follows the minimum safe envelope. Positive speed error uses
PID with (k_p=1.4,k_i=0.25,k_d=0.05) and a bounded non-negative integral.
On braking it resets the integral and uses (k_p=4.0).

The terminal-braking correction is deliberate. A feasible nominal profile can
still be followed late by P feedback, pass a zero-speed endpoint, and make the
actual body collide at a boundary. For each preview it also computes

\[
a_{needed,i}=\frac{v_{ref,i}^2-v^2}{2D_i},
\]

uses the most negative finite request clipped by physical braking, and chooses
the more conservative of it and the P-brake command. This fix is shared by all
lateral policies. The backward-smoothing × look-ahead-braking ablation changes
only the declared profile/longitudinal switch, never the vehicle or lateral
actuator.

## Collision, termination, reward, and limits

Trajectory construction validates a nominal vehicle-footprint sweep, but
nominal feasibility is not evidence that a tracking policy remains clear.
`TrackingEnv` invokes the same shared `CollisionChecker.swept_free` on every
**actual** state transition. The rectangular footprint and safety margin are
applied once; simulation does not inflate it again.

Success requires an actual collision-free state that is within physical goal
tolerance (default 1.25 m for rear-axle reference), near terminal progress,
and travelling at or below 0.45 m/s. Collision is `terminated`; exhausting
the step budget is `truncated` with reason `timeout`. PPO can therefore
bootstrap time limits but not physical terminals.

The dense reward contains only progress, tracking, heading, and action-change
terms:

\[
r=1.5\max(\Delta s,0)-0.18e^2-0.12\theta_e^2
  -0.025(u_t-u_{t-1})^2,
\]

plus terminal success/collision/timeout terms. Waiting cannot collect a
positive survival reward.

## Tests and boundaries of the claim

The control/simulation tests cover actuator clipping/sign, non-finite inputs,
seeded full-trace determinism, Gymnasium API compliance, launch and terminal
braking, actual swept collision, physical low-speed success, an open A* to
trajectory to stop regression, and a curved detour regression for all five
policy variants. The verified command was:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_sim.py tests/test_control.py tests/test_env.py
```

It completed with 23 passing tests when this document was written. That is a
behavioral regression result, not a claim that a controller is optimal or that
an error/lap-time value generalizes.

The model omits tyre forces, slip, unmodelled delays, state-estimation error,
moving objects, perception failures, and map uncertainty. Grid collision is
resolution-dependent. Reports must say “collision-free under this configured
kinematic simulation and checker,” never “safe for a real robot.”
