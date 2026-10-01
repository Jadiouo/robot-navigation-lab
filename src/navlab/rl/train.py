"""PPO training of the RL local planner (``python -m navlab.rl.train``).

One training run = one seed: ``workers`` env processes (``envs_per_worker`` :class:`NavEnv` each), a learner on the GPU if
there is one.  Scenario seeds are drawn from the training range only (:mod:`navlab.rl.splits`); every ``val_every`` env steps the
current actor is evaluated through the real episode runner on tuning-split scenarios (GT pose, nominal condition) and a checkpoint
is written; the learning curve goes to ``curve.csv``.  Checkpoint *selection* is a separate step (:mod:`navlab.rl.select`).

PPO: clipped surrogate, GAE(gamma, lambda), advantage normalisation per minibatch, return-std reward scaling, linear learning-rate
decay, separate actor / critic MLPs (tanh), state-independent log-std, tanh-squashed Gaussian action (the executed action is
tanh(raw); the importance ratio uses the Gaussian density of the raw sample, the tanh Jacobian cancels in the ratio).
Truncation (time-out) bootstraps from the value of the last observation; collision / success / stuck do not.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from navlab.rl.features import ObsSpec
from navlab.rl.splits import TRAIN_RANGE


@dataclass(frozen=True)
class PPOConfig:
    n_steps: int = 256              # per env per update
    epochs: int = 8
    minibatch: int = 1024
    lr: float = 3e-4
    lr_final_frac: float = 0.15
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.003
    max_grad_norm: float = 0.5
    hidden: int = 256
    log_std_init: float = -0.7
    target_kl: float = 0.03
    curriculum_frac: float = 0.5    # fraction of the budget over which the stress level ramps 0.15 -> 1
    pose_gt_prob: float = 0.4       # training episodes with exact pose; the rest use the "noisy" surrogate


# ------------------------------------------------------------------ env worker
def _worker(conn, worker_id: int, train_seed: int, n_envs: int, pose_gt_prob: float) -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    from navlab.rl.env import NavEnv
    from navlab.rl.evaluate import run_actor_episode

    rng = np.random.default_rng(np.random.SeedSequence([train_seed, worker_id, 1234]))
    envs = [NavEnv() for _ in range(n_envs)]
    u = 0.0

    def start(env) -> np.ndarray:
        while True:
            seed = int(rng.integers(*TRAIN_RANGE))
            pose = "gt" if rng.random() < pose_gt_prob else "noisy"
            try:
                return env.reset(seed, curriculum=u, pose=pose)
            except RuntimeError:     # no feasible layout for this seed (very rare): draw another
                continue

    while True:
        cmd, payload = conn.recv()
        if cmd == "reset":
            u = payload
            conn.send([start(e) for e in envs])
        elif cmd == "curriculum":
            u = payload
        elif cmd == "step":
            out = []
            for env, a in zip(envs, payload):
                obs, r, term, trunc, info = env.step(a)
                final = None
                if term or trunc:
                    if trunc:
                        final = env.terminal_observation()
                    info = dict(info)
                    obs = start(env)
                out.append((obs, r, term, trunc, final, info))
            conn.send(out)
        elif cmd == "eval":
            layers, spec, jobs = payload
            conn.send([run_actor_episode(layers, spec, f, s, p) for f, s, p in jobs])
        elif cmd == "close":
            conn.close()
            return


class EnvPool:
    def __init__(self, train_seed: int, workers: int, envs_per_worker: int, pose_gt_prob: float) -> None:
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for w in range(workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, w, train_seed, envs_per_worker, pose_gt_prob), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.epw, self.n = envs_per_worker, workers * envs_per_worker

    def reset(self, u: float) -> np.ndarray:
        for c in self.conns:
            c.send(("reset", u))
        return np.concatenate([np.stack(c.recv()) for c in self.conns])

    def set_curriculum(self, u: float) -> None:
        for c in self.conns:
            c.send(("curriculum", u))

    def step(self, actions: np.ndarray):
        for i, c in enumerate(self.conns):
            c.send(("step", actions[i * self.epw:(i + 1) * self.epw]))
        res = []
        for c in self.conns:
            res.extend(c.recv())
        return res

    def evaluate(self, layers, spec, jobs: list[tuple[str, int, str]]) -> list[dict]:
        shards = [jobs[i::len(self.conns)] for i in range(len(self.conns))]
        for c, sh in zip(self.conns, shards):
            c.send(("eval", (layers, spec, sh)))
        out = []
        for c in self.conns:
            out.extend(c.recv())
        return out

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)


# ------------------------------------------------------------------ learner
def _build(torch, obs_dim: int, hidden: int, log_std_init: float):
    nn = torch.nn

    def lin(i, o, g):
        m = nn.Linear(i, o)
        nn.init.orthogonal_(m.weight, g)
        nn.init.zeros_(m.bias)
        return m

    actor = nn.Sequential(lin(obs_dim, hidden, math.sqrt(2)), nn.Tanh(), lin(hidden, hidden, math.sqrt(2)), nn.Tanh(), lin(hidden, 2, 0.01))
    critic = nn.Sequential(lin(obs_dim, hidden, math.sqrt(2)), nn.Tanh(), lin(hidden, hidden, math.sqrt(2)), nn.Tanh(), lin(hidden, 1, 1.0))
    log_std = nn.Parameter(torch.full((2,), log_std_init))
    return actor, critic, log_std


def _actor_layers(actor) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(m.weight.detach().cpu().numpy().T.copy(), m.bias.detach().cpu().numpy().copy()) for m in actor if hasattr(m, "weight")]


def train(seed: int, total_steps: int, out_dir: Path, workers: int = 4, envs_per_worker: int = 2, val_every: int = 150_000,
          n_val: int = 60, max_hours: float = 2.5, cfg: PPOConfig | None = None, log=print) -> dict:
    import torch
    from navlab.benchmark.splits import scenario_list
    from navlab.rl.policy import save_actor

    cfg = cfg or PPOConfig()
    out_dir = Path(out_dir)
    (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    spec = ObsSpec()
    actor, critic, log_std = _build(torch, spec.dim, cfg.hidden, cfg.log_std_init)
    actor.to(device), critic.to(device), log_std.data.to(device)
    log_std = torch.nn.Parameter(log_std.data.to(device))
    params = list(actor.parameters()) + list(critic.parameters()) + [log_std]
    opt = torch.optim.Adam(params, lr=cfg.lr, eps=1e-5)
    pool = EnvPool(seed, workers, envs_per_worker, cfg.pose_gt_prob)
    N, T = pool.n, cfg.n_steps
    val_jobs = [(f, s, "gt") for f, s in scenario_list("tuning", n_val, start=200)]
    curve_path = out_dir / "curve.csv"
    fields = ["step", "wall_s", "curriculum", "train_success", "train_return", "val_success", "val_ttg", "val_coll", "val_timeout", "val_stuck", "std_accel", "std_steer", "kl", "ev"]
    t_start = time.time()
    u_of = lambda step: min(1.0, 0.15 + 0.85 * step / (cfg.curriculum_frac * total_steps))
    obs = pool.reset(u_of(0))
    ret_acc = np.zeros(N)
    ret_var, ret_cnt, ret_mean = 1.0, 1e-4, 0.0
    recent: list[tuple[bool, float]] = []
    ep_ret = np.zeros(N)
    step, next_val, best_updates = 0, 0, 0
    rows = []
    n_updates = total_steps // (N * T)
    fh = curve_path.open("w", newline="")
    wr = csv.DictWriter(fh, fieldnames=fields)
    wr.writeheader()
    last_kl = last_ev = 0.0

    def validate(step_now: int) -> dict:
        layers = _actor_layers(actor)
        res = pool.evaluate(layers, spec, val_jobs)
        k = sum(r["success"] for r in res)
        ttg = [r["ttg"] for r in res if r["success"]]
        frac = lambda name: float(np.mean([r["outcome"].startswith(name) for r in res]))
        save_actor(out_dir / "ckpt" / f"step{step_now:09d}.npz", layers, spec,
                   {"train_seed": seed, "step": step_now, "val_success": k / len(res), "config": asdict(cfg)})
        return {"val_success": k / len(res), "val_ttg": float(np.mean(ttg)) if ttg else float("nan"), "val_coll": frac("collision"),
                "val_timeout": frac("timeout"), "val_stuck": frac("stuck")}

    for upd in range(n_updates):
        frac_done = upd / max(1, n_updates)
        for g in opt.param_groups:
            g["lr"] = cfg.lr * (1.0 - (1.0 - cfg.lr_final_frac) * frac_done)
        pool.set_curriculum(u_of(step))
        b_obs = torch.zeros((T, N, spec.dim), device=device)
        b_raw = torch.zeros((T, N, 2), device=device)
        b_logp, b_val, b_boot = (torch.zeros((T, N), device=device) for _ in range(3))
        b_rew, b_done = np.zeros((T, N), np.float32), np.zeros((T, N), np.float32)
        finals: list[tuple[int, int, np.ndarray]] = []
        for t in range(T):
            ob = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                mean = actor(ob)
                std = log_std.exp().expand_as(mean)
                raw = mean + std * torch.randn_like(mean)
                logp = (-0.5 * ((raw - mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
                val = critic(ob).squeeze(-1)
            res = pool.step(torch.tanh(raw).cpu().numpy())
            b_obs[t], b_raw[t], b_logp[t], b_val[t] = ob, raw, logp, val
            new_obs = np.zeros_like(obs)
            for i, (o, r, term, trunc, final, info) in enumerate(res):
                new_obs[i] = o
                b_rew[t, i] = r
                ep_ret[i] += r
                ret_acc[i] = ret_acc[i] * cfg.gamma + r
                if term or trunc:
                    b_done[t, i] = 1.0
                    recent.append((info["success"], ep_ret[i]))
                    ep_ret[i] = 0.0
                    ret_acc[i] = 0.0
                    if trunc:
                        finals.append((t, i, final))
                    else:
                        b_boot[t, i] = 0.0
            # running std of discounted returns (reward scaling)
            m = ret_acc.mean()
            ret_cnt_new = ret_cnt + N
            delta = m - ret_mean
            ret_mean += delta * N / ret_cnt_new
            ret_var = (ret_var * ret_cnt + ((ret_acc - m) ** 2).sum() + delta ** 2 * ret_cnt * N / ret_cnt_new) / ret_cnt_new
            ret_cnt = ret_cnt_new
            obs = new_obs
            step += N
        scale = 1.0 / math.sqrt(ret_var + 1e-8)
        with torch.no_grad():
            nxt = critic(torch.as_tensor(obs, dtype=torch.float32, device=device)).squeeze(-1)
            boot = torch.zeros((T, N), device=device)
            boot[:-1] = b_val[1:]
            boot[-1] = nxt
            for t, i, fo in finals:
                boot[t, i] = critic(torch.as_tensor(fo, dtype=torch.float32, device=device)[None]).squeeze()
            done_t = torch.as_tensor(b_done, device=device)
            trunc_mask = torch.zeros((T, N), device=device)
            for t, i, _ in finals:
                trunc_mask[t, i] = 1.0
            # terminal (non-truncated) episode ends bootstrap with 0; truncated ends with V(final obs); others with V(next)
            nonterm = 1.0 - done_t * (1.0 - trunc_mask)
            rew = torch.as_tensor(b_rew * scale, device=device)
            adv = torch.zeros((T, N), device=device)
            gae = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                delta_t = rew[t] + cfg.gamma * boot[t] * nonterm[t] - b_val[t]
                gae = delta_t + cfg.gamma * cfg.lam * (1.0 - done_t[t]) * gae
                adv[t] = gae
            returns = adv + b_val
        f_obs, f_raw, f_logp = b_obs.reshape(-1, spec.dim), b_raw.reshape(-1, 2), b_logp.reshape(-1)
        f_adv, f_ret = adv.reshape(-1), returns.reshape(-1)
        n_all = f_obs.shape[0]
        stop = False
        for ep in range(cfg.epochs):
            perm = torch.randperm(n_all, device=device)
            for s in range(0, n_all, cfg.minibatch):
                idx = perm[s:s + cfg.minibatch]
                mean = actor(f_obs[idx])
                logp = (-0.5 * ((f_raw[idx] - mean) / log_std.exp()) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
                ratio = (logp - f_logp[idx]).exp()
                a_n = f_adv[idx]
                a_n = (a_n - a_n.mean()) / (a_n.std() + 1e-8)
                pg = -torch.min(ratio * a_n, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a_n).mean()
                v_loss = 0.5 * ((critic(f_obs[idx]).squeeze(-1) - f_ret[idx]) ** 2).mean()
                ent = (log_std + 0.5 * math.log(2 * math.pi * math.e)).sum()
                loss = pg + cfg.vf_coef * v_loss - cfg.ent_coef * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    last_kl = float(((ratio - 1) - (ratio.log())).mean())
            if last_kl > cfg.target_kl:
                break
        with torch.no_grad():
            var_y = f_ret.var()
            last_ev = float(1 - ((f_ret - b_val.reshape(-1)) ** 2).mean() / (var_y + 1e-8))
        elapsed = time.time() - t_start
        if step >= next_val or upd == n_updates - 1 or elapsed > max_hours * 3600:
            next_val = step + val_every
            tail = recent[-300:]
            row = {"step": step, "wall_s": round(elapsed, 1), "curriculum": round(u_of(step), 3),
                   "train_success": round(float(np.mean([s for s, _ in tail])), 4) if tail else float("nan"),
                   "train_return": round(float(np.mean([r for _, r in tail])), 2) if tail else float("nan"),
                   **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in validate(step).items()},
                   "std_accel": round(float(log_std.detach().exp()[0]), 3), "std_steer": round(float(log_std.detach().exp()[1]), 3),
                   "kl": round(last_kl, 4), "ev": round(last_ev, 3)}
            wr.writerow(row)
            fh.flush()
            rows.append(row)
            log(f"[seed {seed}] step {step} wall {elapsed / 60:.1f}m  train_succ {row['train_success']}  val_succ {row['val_success']}  u {row['curriculum']}  ev {row['ev']}")
        if elapsed > max_hours * 3600:
            log(f"[seed {seed}] wall-clock cap reached at step {step}")
            break
    fh.close()
    pool.close()
    meta = {"seed": seed, "steps": step, "wall_s": time.time() - t_start, "workers": workers, "envs_per_worker": envs_per_worker,
            "device": str(device), "config": asdict(cfg), "total_steps_budget": total_steps}
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--envs-per-worker", type=int, default=2)
    ap.add_argument("--val-every", type=int, default=150_000)
    ap.add_argument("--max-hours", type=float, default=2.5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    train(a.seed, a.steps, a.out / f"seed{a.seed}", a.workers, a.envs_per_worker, a.val_every, max_hours=a.max_hours)


if __name__ == "__main__":
    main()
