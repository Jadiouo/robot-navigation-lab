"""Headless figures and GIFs generated solely from one saved rollout trace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from matplotlib.patches import Polygon


def _map_extent(grid: Any) -> tuple[float, float, float, float]:
    height, width = grid.occupancy.shape
    ox, oy = grid.origin
    return ox, ox + width * grid.resolution, oy, oy + height * grid.resolution


def _as_xy(points: Any) -> np.ndarray:
    if points is None:
        return np.empty((0, 2), dtype=float)
    array = np.asarray(points, dtype=float)
    return array.reshape((-1, 2)) if array.size else np.empty((0, 2), dtype=float)


def overview_png(
    path: Path,
    scenario: Any,
    plan_points: Any,
    trajectory: Any | None,
    trace: list[dict[str, Any]],
    title: str,
    vehicle: Any | None = None,
) -> None:
    """Render geometry plus the actual trace; works without an interactive GUI."""
    fig, (ax_map, ax_speed) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    x0, x1, y0, y1 = _map_extent(scenario.grid)
    ax_map.imshow(
        scenario.grid.occupancy,
        origin="lower",
        extent=(x0, x1, y0, y1),
        cmap="Greys",
        interpolation="nearest",
        alpha=0.5,
    )
    plan = _as_xy(plan_points)
    if len(plan):
        ax_map.plot(plan[:, 0], plan[:, 1], "--", color="#e68a00", label="geometric path")
    if trajectory is not None:
        points = _as_xy(trajectory.points)
        ax_map.plot(points[:, 0], points[:, 1], color="#0066cc", label="reference trajectory")
    if trace:
        xy = np.asarray([(row["x"], row["y"]) for row in trace], dtype=float)
        ax_map.plot(xy[:, 0], xy[:, 1], color="#d62728", label="vehicle trace")
        final = trace[-1]
        color = "#cc0000" if bool(final.get("collision", False)) else "#222222"
        ax_map.scatter([final["x"]], [final["y"]], color=color, marker="x" if color == "#cc0000" else "o", label="collision" if color == "#cc0000" else "final", zorder=5)
        if vehicle is not None:
            yaw = float(final["yaw"])
            forward = np.array([np.cos(yaw), np.sin(yaw)])
            left = np.array([-np.sin(yaw), np.cos(yaw)])
            rear = np.array([final["x"], final["y"]]) - vehicle.rear_overhang * forward
            front = np.array([final["x"], final["y"]]) + (vehicle.wheelbase + vehicle.front_overhang) * forward
            corners = np.array([rear + vehicle.width / 2 * left, front + vehicle.width / 2 * left, front - vehicle.width / 2 * left, rear - vehicle.width / 2 * left])
            ax_map.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=color, linewidth=1.2))
    ax_map.scatter([scenario.start.x], [scenario.start.y], color="#228833", marker="o", label="start", zorder=3)
    ax_map.scatter([scenario.goal[0]], [scenario.goal[1]], color="#aa3377", marker="*", s=90, label="goal", zorder=3)
    ax_map.set(title=title, xlabel="x [m]", ylabel="y [m]", aspect="equal")
    ax_map.legend(loc="best", fontsize=8)

    if trace:
        t = np.asarray([(row["step"] + 1) * row["dt"] for row in trace], dtype=float)
        ax_speed.plot(t, [row.get("v", np.nan) for row in trace], label="v")
        ax_speed.plot(t, [row.get("v_ref", np.nan) for row in trace], label="v_ref", linestyle="--")
    ax_speed.set(xlabel="time [s]", ylabel="speed [m/s]", title="Longitudinal behavior")
    if trace:
        ax_speed.legend(loc="best")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def replay_gif(path: Path, scenario: Any, plan_points: Any, trajectory: Any | None, trace: list[dict[str, Any]]) -> None:
    """Create a compact replay GIF; no invented frames or graphics are used."""
    x0, x1, y0, y1 = _map_extent(scenario.grid)
    width, height = 720, 720

    def px(x: float, y: float) -> tuple[int, int]:
        return (int((x - x0) / (x1 - x0) * (width - 1)), int((y1 - y) / (y1 - y0) * (height - 1)))

    base = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(base)
    occ = np.asarray(scenario.grid.occupancy, dtype=bool)
    rows, cols = occ.shape
    for cy, cx in zip(*np.nonzero(occ)):
        gx0, gy0 = x0 + cx * scenario.grid.resolution, y0 + cy * scenario.grid.resolution
        gx1, gy1 = gx0 + scenario.grid.resolution, gy0 + scenario.grid.resolution
        first, second = px(gx0, gy0), px(gx1, gy1)
        draw.rectangle((min(first[0], second[0]), min(first[1], second[1]), max(first[0], second[0]), max(first[1], second[1])), fill="#333333")

    def line(points: Any, color: str, width_px: int) -> None:
        xy = _as_xy(points)
        if len(xy) > 1:
            draw.line([px(x, y) for x, y in xy], fill=color, width=width_px)

    line(plan_points, "#e68a00", 2)
    if trajectory is not None:
        line(trajectory.points, "#0066cc", 2)
    draw.ellipse([*px(scenario.start.x, scenario.start.y), *px(scenario.start.x, scenario.start.y)], fill="#228833")
    gx, gy = px(*scenario.goal)
    draw.ellipse((gx - 6, gy - 6, gx + 6, gy + 6), fill="#aa3377")

    if not trace:
        base.save(path, format="GIF")
        return
    indices = np.unique(np.linspace(0, len(trace) - 1, min(160, len(trace)), dtype=int))
    frames: list[Image.Image] = []
    for index in indices:
        frame = base.copy()
        frame_draw = ImageDraw.Draw(frame)
        traveled = [(row["x"], row["y"]) for row in trace[: index + 1]]
        if len(traveled) > 1:
            frame_draw.line([px(x, y) for x, y in traveled], fill="#d62728", width=3)
        row = trace[index]
        cx, cy = px(row["x"], row["y"])
        radius = 5
        frame_draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill="#d62728")
        frames.append(frame)
    frames[0].save(path, format="GIF", save_all=True, append_images=frames[1:], duration=50, loop=0)
