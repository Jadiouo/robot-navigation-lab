"""Geometry of a polyline global path in the world frame, and small shared helpers."""
from __future__ import annotations

import math

import numpy as np


class PathFrame:
    """Arc-length parametrised polyline: projection, point/tangent lookup, remaining length."""

    def __init__(self, path: np.ndarray) -> None:
        self.points = np.asarray(path, dtype=float)
        seg = np.diff(self.points, axis=0)
        self._len = np.hypot(seg[:, 0], seg[:, 1])
        self._seg, self._seglen = seg, np.maximum(self._len, 1e-9)
        self.s = np.concatenate(([0.0], np.cumsum(self._len)))
        self.length = float(self.s[-1])

    def project(self, xy: tuple[float, float]) -> tuple[float, float]:
        """Return ``(s, distance)`` of the nearest point on the polyline."""
        a = self.points[:-1]
        p = np.asarray(xy, dtype=float) - a
        t = np.clip(np.einsum("ij,ij->i", p, self._seg) / self._seglen ** 2, 0.0, 1.0)
        nearest = a + t[:, None] * self._seg
        d2 = np.sum((nearest - np.asarray(xy)) ** 2, axis=1)
        i = int(np.argmin(d2))
        return float(self.s[i] + t[i] * self._len[i]), math.sqrt(float(d2[i]))

    def point_at(self, s: float) -> np.ndarray:
        s = min(max(s, 0.0), self.length)
        i = min(int(np.searchsorted(self.s, s, side="right")) - 1, len(self._len) - 1)
        return self.points[i] + (s - self.s[i]) / self._seglen[i] * self._seg[i]

    def tangent_at(self, s: float) -> float:
        s = min(max(s, 0.0), self.length)
        i = min(int(np.searchsorted(self.s, s, side="right")) - 1, len(self._len) - 1)
        return math.atan2(self._seg[i, 1], self._seg[i, 0])

    def tangents_at(self, s: np.ndarray) -> np.ndarray:
        """Vectorised path heading (world frame) at arc lengths ``s`` (clamped to the path)."""
        s = np.clip(np.asarray(s, dtype=float), 0.0, self.length)
        i = np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self._len) - 1)
        return np.arctan2(self._seg[i, 1], self._seg[i, 0])

    def sample(self, step: float, s_from: float, s_to: float) -> tuple[np.ndarray, np.ndarray]:
        """Points (and their arc lengths) every ``step`` metres on ``[s_from, s_to]``."""
        s_to = min(s_to, self.length)
        ss = np.arange(max(s_from, 0.0), s_to + 1e-9, step)
        if not len(ss):
            ss = np.array([min(max(s_from, 0.0), self.length)])
        idx = np.clip(np.searchsorted(self.s, ss, side="right") - 1, 0, len(self._len) - 1)
        pts = self.points[idx] + ((ss - self.s[idx]) / self._seglen[idx])[:, None] * self._seg[idx]
        return pts, ss


def to_robot_frame(points: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(points, dtype=float) - np.array([x, y])
    return np.stack((c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]), axis=-1)


def goal_speed_limit(remaining: float, decel: float = 1.5, stop_offset: float = 1.0) -> float:
    """Speed from which a comfortable constant deceleration stops ``stop_offset`` before the path end."""
    return math.sqrt(2.0 * decel * max(remaining - stop_offset, 0.0))


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi
