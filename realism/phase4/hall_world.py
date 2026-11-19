#!/usr/bin/env python3
"""Build the Gazebo equivalent of one hall_v3 scenario (read-only use of navlab).

Usage: hall_world.py SEED OUTDIR   ->  OUTDIR/{hall.sdf.xacro, hall.pgm, hall.yaml, scenario.json}

Coordinates: Gazebo world frame == 2D map frame (origin at the lower-left corner of the 60 x 36 m map), so start/goal
from navlab.v3.worlds.generate_scenario_v3('hall_v3', seed) are used verbatim.
Hidden obstacles and moving agents of the 2D scenario are NOT built (minimal hall; see DESIGN.md).
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("NAVLAB_SRC", os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, SRC)
from navlab.v3.worlds import generate_scenario_v3  # noqa: E402

BAND_M = 0.5   # occupied wall band thickness in the localisation map
MAP_RES = 0.05  # m / px for the AMCL / map_server map (2D grid is 0.5 m)


def hall_geometry(known: np.ndarray, res: float):
    """Corridor rectangle (x0,x1,y0,y1) recovered from the 2D occupancy grid (free cells)."""
    rr, cc = np.where(~known)
    return cc.min() * res, (cc.max() + 1) * res, rr.min() * res, (rr.max() + 1) * res


def box(name, x0, x1, y0, y1, h=1.0):
    cx, cy, sx, sy = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    return f"""
      <collision name="{name}_c"><pose>{cx} {cy} {h/2} 0 0 0</pose><geometry><box><size>{sx} {sy} {h}</size></box></geometry></collision>
      <visual name="{name}_v"><pose>{cx} {cy} {h/2} 0 0 0</pose><geometry><box><size>{sx} {sy} {h}</size></box></geometry>
        <material><ambient>0.6 0.6 0.65 1</ambient><diffuse>0.6 0.6 0.65 1</diffuse></material></visual>"""


def build(seed: int, outdir: str) -> dict:
    os.makedirs(outdir, exist_ok=True)
    d = generate_scenario_v3("hall_v3", seed)
    sc = d.scenario
    known = np.asarray(sc.grid.occupancy, dtype=bool)
    res = float(sc.grid.resolution)
    W, H = known.shape[1] * res, known.shape[0] * res
    x0, x1, y0, y1 = hall_geometry(known, res)
    # --- SDF: 4 static slabs fill everything outside the corridor rectangle (same as the 2D map: outside == occupied)
    walls = (box("south", 0, W, 0, y0) + box("north", 0, W, y1, H) + box("west", 0, x0, y0, y1) + box("east", x1, W, y0, y1))
    sdf = f"""<?xml version="1.0"?>
<sdf version="1.6" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:arg name="headless" default="true"/>
  <world name="hall">
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <light name="sun" type="directional"><cast_shadows>0</cast_shadows><pose>30 {H/2} 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse><specular>0.8 0.8 0.8 1</specular><direction>-0.5 0.1 -0.9</direction></light>
    <model name="ground_plane"><static>1</static><link name="link">
      <collision name="collision"><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry></collision>
      <visual name="visual"><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
        <material><ambient>0.8 0.8 0.8 1</ambient><diffuse>0.8 0.8 0.8 1</diffuse></material></visual></link></model>
    <scene><shadows>0</shadows></scene>
    <physics name="3ms" type="ode"><max_step_size>0.003</max_step_size><real_time_factor>1</real_time_factor></physics>
    <model name="hall_walls"><static>1</static><link name="link">{walls}
    </link></model>
  </world>
</sdf>
"""
    open(os.path.join(outdir, "hall.sdf.xacro"), "w").write(sdf)
    # --- occupancy map (pgm P5, 0 = occupied, 254 = free), image row 0 = top (y = H)
    nx, ny = int(round(W / MAP_RES)), int(round(H / MAP_RES))
    xs = (np.arange(nx) + 0.5) * MAP_RES
    ys = (np.arange(ny) + 0.5) * MAP_RES
    free = ((ys[:, None] > y0) & (ys[:, None] < y1)) & ((xs[None, :] > x0) & (xs[None, :] < x1))
    # Occupied only in a BAND_M thick band around the corridor, 128 (= unknown in trinary mode) further out.  A solid-black exterior
    # lets AMCL particles that wander into the wall mass score ~perfectly (observed: estimate jumped 6 m out of the hall at S=1).
    band = ((ys[:, None] > y0 - BAND_M) & (ys[:, None] < y1 + BAND_M)) & ((xs[None, :] > x0 - BAND_M) & (xs[None, :] < x1 + BAND_M))
    img = np.where(free, 254, np.where(band, 0, 128)).astype(np.uint8)[::-1]
    with open(os.path.join(outdir, "hall.pgm"), "wb") as f:
        f.write(f"P5\n{nx} {ny}\n255\n".encode()); f.write(img.tobytes())
    open(os.path.join(outdir, "hall.yaml"), "w").write(
        f"image: hall.pgm\nmode: trinary\nresolution: {MAP_RES}\norigin: [0.0, 0.0, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
    gx, gy = sc.goal
    s = sc.start
    meta = dict(seed=seed, hall=dict(x0=x0, x1=x1, y0=y0, y1=y1, width_m=y1 - y0, length_m=x1 - x0, map_w=W, map_h=H),
                start=[s.x, s.y, s.yaw], goal=[gx, gy], goal_yaw=math.atan2(gy - s.y, gx - s.x),
                route_len_m=sc.metadata["route_length_m"], twoD_max_time=d.max_time,
                twoD_hidden=sc.metadata["n_hidden"], twoD_agents=sc.metadata["n_agents"],
                end_wall_dist_start=min(s.x - 1, 59 - s.x), end_wall_dist_goal=min(gx - 1, 59 - gx))
    json.dump(meta, open(os.path.join(outdir, "scenario.json"), "w"), indent=1)
    return meta


if __name__ == "__main__":
    print(json.dumps(build(int(sys.argv[1]), sys.argv[2])))
