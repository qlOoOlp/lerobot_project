"""Meta-World env adapter — everything that crosses the env <-> dataset boundary.

The *mapping core* of the metaworld env adapter (부록 D.1), covering both halves of
the observation plus the action going back out:
    render_frame()            env camera  -> dataset image  (un-flip + resize)
    state4_to_canonical10()   env obs[:4] -> canonical 10D  (gripper binarized)
    canonical10_to_env_action()  canonical -> env's 4D action

Plain functions on purpose, because each has two consumers:
  - collection  (Phase 1) : called directly, port_droid style (no processor)
  - rollout     (Phase 5) : wrapped in a thin ObservationProcessorStep
Both go through the SAME functions, which is what makes train == inference hold —
so anything with that requirement belongs here, not in a collection script.
No Robot / robot_processor here: metaworld is a gym Env path.

GRIPPER — everything WE own is binary {0, 1}, 0 = closed, 1 = open:
      observation.state[9] (policy input)  : {0, 1}
      action[9]            (policy output) : {0, 1}
  The policy only ever sees and emits those. [-1, 1] is NOT our action: Meta-World's
  env.step() enforces spaces.Box(low=-1, high=1) (lerobot envs/metaworld.py:137), so
  we translate to its closing effort ONLY on the last step out to the env. That value
  exists in neither the dataset nor the policy's I/O.

      [dataset / policy in / policy out]  ---- binary {0,1}, 0=closed, 1=open ----
                   |                                        ^
                   | canonical10_to_env_action()            | state4_to_canonical10()
                   v   (last boundary only)                 |   (binarizes obs[3])
      [metaworld env.step()]  ---- [-1,1] closing effort (API-enforced) ----

  Sourced from obs[3], the *measured* openness, binarized against a threshold:
      obs[3] >= threshold  ->  1.0 (open)
      obs[3] <  threshold  ->  0.0 (closed)
  obs[3] has the same polarity as canonical (1=open), so only the range collapses.
  Since obs[3] is a genuine state, it is directly observable at rollout — no
  command tracking or frame shift is needed anywhere.

  ⚠ THE THRESHOLD IS TASK-DEPENDENT — never hardcode it at a call site.
    obs[3] is continuous and pick-place measures [0.3955, 1.0]: the fingers stop on
    the block, so it never reaches 0. It is bimodal at ~1.0 (open) and ~0.40-0.46
    (gripping), so ~0.7 separates them. A THICKER object grips at a higher openness
    and a thinner one lower, so the same value silently mislabels another task.
    Measure per task, and pass the SAME value at collection and at rollout — a
    mismatch breaks train == inference. The value is baked into the dataset, so
    changing it means re-collecting (re-running the env).

Rotation: the Sawyer EE never rotates (mocap weld holds its orientation), and the
env obs does not even expose a rotation. The canonical rot6d slot is filled with
the constant IDENTITY_ROT6D — a true (constant) value, not fake padding.

Binarization is baked at conversion here (no runtime processor). See retargeting.md.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from lerobot_canonical.schemas import canonical_ee10 as sch

# Meta-World env facts (NOT representation facts -> they live here, not in lerobot_canonical)
STATE4_DIM: int = 4       # env obs prefix: [ee_x, ee_y, ee_z, gripper]
ENV_ACTION_DIM: int = 4   # env action: [dx, dy, dz, closing_effort]

# Measured on pick_place_v3: obs[3] is bimodal at ~1.0 (open) / ~0.40-0.46 (gripping
# the block), so 0.7 separates them with margin. TASK-DEPENDENT — re-measure for any
# other task (see the module docstring); do not reuse blindly.
PICK_PLACE_GRIPPER_THRESHOLD: float = 0.7

# The env's own action->displacement gain, NOT a statistic of any dataset:
#   sawyer_xyz_env.py:327   pos_delta = np.clip(action, -1, 1) * self.action_scale
#   sawyer_xyz_env.py:182   action_scale: float = 1.0 / 100
# Verified by driving the env: a constant action of 1.0/0.5/0.25 settles at
# 0.01003/0.00513/0.00261 m per step — exactly action * action_scale.
# Task-independent (it is the env's constant), unlike the gripper threshold.
#
# ⚠ Do NOT fit this to the observed |dxyz| distribution. The hand lags the mocap
# (weld + frame_skip=5), so it ramps up over ~10 steps and can transiently exceed
# action_scale while catching up (pick-place measures mean 0.008, max 0.016). Those
# are the *response*, not the gain: using them would make every command undershoot.
ENV_XYZ_SCALE: float = 0.01


def render_frame(env: Any, image_size: int) -> np.ndarray:
    """Render the corner2 camera as an (image_size, image_size, 3) uint8 RGB frame.

    The image half of the observation adapter. It must produce byte-identical
    framing at collection and at rollout, or the policy sees a different world
    than it trained on — same contract as state4_to_canonical10().

    `env` is the INNER env (`wrapper._env`), which does not correct anything: the
    lerobot wrapper un-flips corner2 in its own render()/_format_raw_obs(), but we
    bypass it because the expert needs the raw 39D obs. So both the un-flip and the
    resize are ours to do. The resize target must match what build_features()
    declares, or add_frame() rejects the frame.

    Args:
        env: inner mujoco env; render_mode/camera_name were fixed at construction.
        image_size: target square size (the dataset's declared H and W).

    Returns:
        (image_size, image_size, 3) uint8, C-contiguous.
    """
    img = env.render()
    img = np.flip(img, (0, 1))
    return np.asarray(Image.fromarray(img).resize((image_size, image_size)))


def state4_to_canonical10(state4: np.ndarray, gripper_threshold: float) -> np.ndarray: # ee xyz + measured gripper 4D (abs) -> ee xyz + rotation 6D + binary gripper 10D (abs)
    state4 = np.asarray(state4, dtype=np.float32)
    if state4.shape[-1] != STATE4_DIM:
        raise ValueError(f"Expected trailing dim {STATE4_DIM}, got {state4.shape}.")

    xyz = state4[..., :3]
    rot = np.broadcast_to(
        np.asarray(sch.IDENTITY_ROT6D, dtype=np.float32),
        (*state4.shape[:-1], len(sch.IDENTITY_ROT6D)),
    )
    gripper = (state4[..., 3:4] >= gripper_threshold).astype(np.float32)
    return np.concatenate((xyz, rot, gripper), axis=-1).astype(np.float32)


def canonical10_to_env_action( # canonical 10D (abs) -> env 4D (rel)
    target10: np.ndarray,
    current_ee_xyz: np.ndarray,
    xyz_scale: float,
) -> np.ndarray:
    target10 = np.asarray(target10, dtype=np.float32)
    if target10.shape != (sch.STATE_DIM,):
        raise ValueError(f"Expected shape ({sch.STATE_DIM},), got {target10.shape}.")

    delta_xyz = (target10[:3] - np.asarray(current_ee_xyz, dtype=np.float32)) / float(xyz_scale)
    openness = float(target10[sch.POSE_DIM])
    closing_effort = (0.5 - openness) * 2.0
    action = np.array([*delta_xyz, closing_effort], dtype=np.float32)
    return np.clip(action, -1.0, 1.0)


__all__ = [
    "STATE4_DIM",
    "ENV_ACTION_DIM",
    "ENV_XYZ_SCALE",
    "PICK_PLACE_GRIPPER_THRESHOLD",
    "render_frame",
    "state4_to_canonical10",
    "canonical10_to_env_action",
]
