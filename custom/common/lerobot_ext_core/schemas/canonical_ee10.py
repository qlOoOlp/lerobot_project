"""EE-pose canonical representation (10D) — dim, axes.

ONE representation among possible others — NOT "the" schema.
Layout (10D): [x, y, z, rot6d(6), gripper]
  - pose    : xyz(3) + rot6d(6) = 9D
  - gripper : 1D
Shared by every EE-Cartesian embodiment we use (metaworld/Sawyer, franka, UMI):
swapping the *robot* needs no new module — only a *representation* change does.
A new representation (e.g. joint-space) -> add a flat sibling module
(`canonical_joint7.py`); do not edit this one.

Outward contract (every representation module exposes these):
    STATE_DIM, STATE_AXES, ACTION_DIM, ACTION_AXES
Representation-internal (this module only):
    POSE_DIM, POSE_AXES, GRIPPER_DIM, GRIPPER_AXES

Used for LeRobotDataset feature `shape`/`names.axes`  
"""
from __future__ import annotations

# Dim
POSE_DIM: int = 9
GRIPPER_DIM: int = 1
STATE_DIM: int = POSE_DIM + GRIPPER_DIM
ACTION_DIM: int = POSE_DIM + GRIPPER_DIM

# Axes
POSE_AXES: tuple[str, ...] = ("x", "y", "z", "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5")
GRIPPER_AXES: tuple[str, ...] = ("gripper",)
STATE_AXES: tuple[str, ...] = POSE_AXES + GRIPPER_AXES
ACTION_AXES: tuple[str, ...] = POSE_AXES + GRIPPER_AXES

__all__ = [
    "POSE_DIM", "GRIPPER_DIM", "STATE_DIM", "ACTION_DIM",
    "POSE_AXES", "GRIPPER_AXES", "STATE_AXES", "ACTION_AXES",
]
