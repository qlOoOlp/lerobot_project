"""EE-pose canonical representation (10D) — dim, axes.

ONE representation among possible others — NOT "the" schema.
Layout (10D): [x, y, z, rot6d(6), gripper]
  - pose    : xyz(3) + rot6d(6) = 9D
  - gripper : 1D

Channel semantics (the contract every source/sink converts TO):
  - x, y, z   : end-effector position [m]
  - rot6d(6)  : first two columns of the 3x3 rotation matrix, flattened.
                Recover R by Gram-Schmidt; 3rd column = b1 x b2.
                No rotation -> IDENTITY_ROT6D.
  - gripper   : openness in [0, 1] — 0 = closed, 1 = open.

This is a *definition*, not a description of any one embodiment: sources are free
to disagree with it (most do, and some disagree with themselves between their
observation and their action). Every boundary translates to/from this at its own
adapter, and each adapter documents its own translation — nothing embodiment-
specific belongs in this module.

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

# Constants of this representation
# rot6d encoding of the identity rotation (= "no rotation"). Same value for every
# embodiment — only *whether* a source uses it is embodiment-specific (metaworld's
# Sawyer EE never rotates, so it is filled in there; UMI/franka carry real rotation).
# Kept as a tuple so ext_core stays dependency-free; consumers do np.asarray(...).
IDENTITY_ROT6D: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

__all__ = [
    "POSE_DIM", "GRIPPER_DIM", "STATE_DIM", "ACTION_DIM",
    "POSE_AXES", "GRIPPER_AXES", "STATE_AXES", "ACTION_AXES",
    "IDENTITY_ROT6D",
]