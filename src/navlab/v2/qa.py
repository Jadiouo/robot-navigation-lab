"""Visual QA of warehouse maps: the known map, start / goal and the A* route only (no planner, no episode)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from navlab.v2.worlds import generate_scenario_v2, warehouse_route_stats  # noqa: E402
from navlab.world import generator as g  # noqa: E402


def plot_warehouse(seed: int, out: Path) -> Path:
    dyn = generate_scenario_v2("warehouse", seed, g.ScenarioStress(agent_density=0.0, hidden_density=0.0))
    sc = dyn.scenario
    path = g._plan(sc.grid.occupancy, (sc.start.x, sc.start.y), sc.goal)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.imshow(~sc.grid.occupancy, origin="lower", extent=(0, g.WIDTH_M, 0, g.HEIGHT_M), cmap="gray", vmin=0, vmax=1)
    ax.plot(path[:, 0], path[:, 1], "c-", lw=1.5, label="A* (known map)")
    ax.plot(sc.start.x, sc.start.y, "go", ms=9, label="start")
    ax.plot(sc.goal[0], sc.goal[1], "r*", ms=13, label="goal")
    ax.set_title(f"warehouse seed {seed}"), ax.legend(loc="upper right", fontsize=7), ax.set_aspect("equal")
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"warehouse_{seed}.png"
    fig.savefig(f, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return f


def plot_overview(seeds=range(12), out: Path = Path("notes/v2_qa"), ncols: int = 4) -> Path:
    """3 x 4 overview of the known map + start / goal + A* route for ``seeds``."""
    seeds = list(seeds)
    nrows = -(-len(seeds) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.2 * nrows))
    for ax, seed in zip(axes.ravel(), seeds):
        dyn = generate_scenario_v2("warehouse", seed, g.ScenarioStress(agent_density=0.0, hidden_density=0.0))
        sc = dyn.scenario
        path = g._plan(sc.grid.occupancy, (sc.start.x, sc.start.y), sc.goal)
        ax.imshow(~sc.grid.occupancy, origin="lower", extent=(0, g.WIDTH_M, 0, g.HEIGHT_M), cmap="gray", vmin=0, vmax=1)
        ax.plot(path[:, 0], path[:, 1], "c-", lw=1.3)
        ax.plot(sc.start.x, sc.start.y, "go", ms=6)
        ax.plot(sc.goal[0], sc.goal[1], "r*", ms=10)
        st = warehouse_route_stats(sc.grid.occupancy, (sc.start.x, sc.start.y), sc.goal, path)
        ax.set_title(f"seed {seed}: ratio {st['ratio']:.2f}, rows {st['rows_crossed']}", fontsize=9)
        ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
    for ax in axes.ravel()[len(seeds):]:
        ax.axis("off")
    out.mkdir(parents=True, exist_ok=True)
    f = out / "overview.png"
    fig.savefig(f, dpi=80, bbox_inches="tight")
    plt.close(fig)
    return f


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[3] / "notes" / "v2_qa"
    for sd in range(12):
        plot_warehouse(sd, out)
    print(plot_overview(range(12), out))
