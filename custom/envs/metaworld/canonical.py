"""Meta-World env adapter — everything that crosses the env <-> dataset boundary.

The *mapping core* of the metaworld env adapter (부록 D.1), covering both halves of
the observation plus the action going back out:
    render_frame()            env camera  -> dataset image  (un-flip if FLIP_CAMERAS, + resize)
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

  The threshold is task-dependent — never hardcode it at a call site.
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

WHY ALL OF THIS IS HERE AND NOT IN THE POLICY — the flip/binarize/resize are FIXED
(a camera's mount angle, a threshold, a size), so they are baked at collection and
replayed identically at rollout; the policy never sees them and cannot tell metaworld
pixels from UMI pixels. Contrast anchor-relative, which depends on the sampled window
and therefore CANNOT be baked -> it lives in the (shared) policy processor and runs at
training AND inference. The dividing question is always "can this be baked offline?":
    yes -> env adapter, applied at collection + rollout   (this file)
    no  -> policy processor, applied at training + inference
That is what lets metaworld and UMI share one policy: each embodiment's quirks are
translated away HERE, so the policy only ever sees canonical.
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

# Cameras whose raw frames come out 180deg rotated and must be corrected.
# A property of the CAMERA — not of Meta-World, not of the task. That is a third axis next
# to the two constants above (task-dependent / task-independent):
#   metaworld assets/objects/assets/xyz_base.xml (the package's only corner2 definition)
#   <camera name="corner2" fovy="60" mode="fixed" pos="1.3 -0.2 1.1" euler="3.9 2.3 0.6"/>
# that euler rolls the camera past 180deg, so mujoco faithfully renders an upside-down scene.
# Verified by rendering: the raw frame has the table hanging from the ceiling and the arm
# pointing down (tmp/real/corner2_raw.png vs corner2_flipped.png).
#
# np.flip(img, (0,1)) is a 180deg ROTATION, not a mirror -> it undoes the mount angle rather
# than transforming the data. Hence it applies at BOTH collection and rollout: the camera keeps
# emitting inverted frames, so every read needs the same correction. "The dataset already has
# the flip, so skip it at rollout" would feed upside-down frames to a right-side-up policy.
#
# lerobot guards identically (envs/metaworld.py:147). Sibling cameras are defined differently
# and must NOT be flipped blindly:
#   corner  xyaxes="-1 1 0 -0.2 -0.2 -1"   behindGripper  quat="0 1 0 0"   gripperPOV  quat="-1 -1.3 0 0"
FLIP_CAMERAS: frozenset[str] = frozenset({"corner2"})

# The env's own action->displacement gain, NOT a statistic of any dataset:
#   sawyer_xyz_env.py:327   pos_delta = np.clip(action, -1, 1) * self.action_scale
#   sawyer_xyz_env.py:182   action_scale: float = 1.0 / 100
# Verified by driving the env: a constant action of 1.0/0.5/0.25 settles at
# 0.01003/0.00513/0.00261 m per step — exactly action * action_scale.
# Task-independent (it is the env's constant), unlike the gripper threshold.
#
# Do NOT fit this to the observed |dxyz| distribution. The hand lags the mocap
# (weld + frame_skip=5), so it ramps up over ~10 steps and can transiently exceed
# action_scale while catching up (pick-place measures mean 0.008, max 0.016). Those
# are the *response*, not the gain: using them would make every command undershoot.
ENV_XYZ_SCALE: float = 0.01


def render_frame(env: Any, image_size: int) -> np.ndarray:
    """Render env's camera as an (image_size, image_size, 3) uint8 RGB frame.

    The image half of the observation adapter. It must produce byte-identical
    framing at collection and at rollout, or the policy sees a different world
    than it trained on — same contract as state4_to_canonical10().

    `env` must be the inner env (`wrapper._env`), which corrects nothing. The lerobot
    wrapper un-flips FLIP_CAMERAS in its own render() (metaworld.py:149) and
    _format_raw_obs() (:172); we bypass it because the expert needs the raw 39D obs. So
    both the un-flip and the resize are ours to do. Handing the WRAPPER here instead
    would flip an already-corrected frame -> upside-down input (double flip). The resize
    target must match what build_features() declares, or add_frame() rejects the frame.

    The camera comes from `env.camera_name` rather than an argument on purpose: it is a
    fact the env already owns, so it cannot desync from the camera actually being
    rendered. (Contrast gripper_threshold / xyz_scale, which are OUR decisions the env
    knows nothing about — those are passed in.)

    Args:
        env: inner mujoco env; render_mode/camera_name were fixed at construction.
        image_size: target square size (the dataset's declared H and W).

    Returns:
        (image_size, image_size, 3) uint8, C-contiguous.

    Raises:
        ValueError: if `env` exposes no `camera_name` — we refuse to guess whether the
            frame needs correcting, since guessing wrong is silent and breaks
            train == inference.
    """
    camera_name = getattr(env, "camera_name", None)
    if camera_name is None:
        raise ValueError(
            f"Expected `env` to expose `camera_name`, got {type(env).__name__} without it. "
            "It decides whether the frame needs un-flipping (see FLIP_CAMERAS); pass the "
            "inner env (`wrapper._env`), not a bare renderer."
        )

    img = env.render()
    if camera_name in FLIP_CAMERAS:
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
    servo_gain: float = 1.0,
    direction_preserving: bool = False,
) -> np.ndarray:
    """canonical 절대 타겟 -> Meta-World 4D 액션.

    env 액션은 위치가 아니라 mocap setpoint 의 속도 명령이다 (mocap += action * xyz_scale).
    그래서 이 함수는 pose -> velocity 변환기이고, 정체는 resolved-rate P 제어기다.

    servo_gain 은 그 P 게인이다. 물리 단위로 환산하면 (dt = 1/80s, xyz_scale = 0.01):
        servo_gain=1.0 -> Kp = 1/dt = 80/s   "오차를 한 스텝에 0으로"
        servo_gain=0.1 -> Kp = 8/s           metaworld scripted expert 와 같은 게인
    기본 1.0 은 검증된 값이다 (30k ckpt, seeds 0~19, 18/20 = 90%).

    게인을 낮출 때의 함정: 우리 기준(타겟)은 6.75mm/스텝으로 움직이므로 P 제어의 정상 상태
    추종 오차가 v/Kp 로 남는다. gain=1.0 이면 6.75mm(= 정확히 한 스텝 뒤)지만 gain=0.1 이면
    67.5mm 다. expert 는 고정 목표를 겨냥해 이 항이 없다 — 게인만 베껴오면 안 되는 이유다.
    """
    target10 = np.asarray(target10, dtype=np.float32)
    if target10.shape != (sch.STATE_DIM,):
        raise ValueError(f"Expected shape ({sch.STATE_DIM},), got {target10.shape}.")

    delta_xyz = float(servo_gain) * (
        target10[:3] - np.asarray(current_ee_xyz, dtype=np.float32)
    ) / float(xyz_scale)

    # 축별 클리핑은 어느 한 축이 1 을 넘는 순간 그 축만 눌러 이동 방향을 틀어놓는다. peak 로 나누면
    # 축 사이 비율이 보존되어 방향이 유지된다. 액션 공간이 [-1,1]^3 인 것은 env 의 고정 사실이라
    # 여기(어댑터)에 둔다.
    #
    # 기본값이 False 인 이유 (2026-07-17, 30k ckpt, pick-place 실측):
    #   축별 클리핑은 스텝의 41% 에서 포화하며 그때 방향을 평균 7.4도(최대 23.5도) 틀어놓는다.
    #   이걸 켜면 청크 중반(k=4)의 방향 변화가 30.3도 -> 18.4도로 줄어 원인은 확인된다. 그러나
    #   정작 스텝 수를 결정하는 접근 굴곡도는 1.36 -> 1.30 밖에 안 움직이고(expert 는 1.04),
    #   seed 1 이 성공 -> 200스텝 타임아웃으로 회귀했다. 포화가 클 때 부축 속도가 peak 배수만큼
    #   느려지는 대가로 보인다 (실측 peak 최대 3.93 -> 부축 4배 감속).
    #   즉 진단으로는 유효하지만 개선으로는 미검증이다. 20~30 에피소드 A/B 로 성공률이 확인되기
    #   전까지 기본값은 검증된 축별 클리핑(90%, seeds 0~19)을 유지한다.
    if direction_preserving:
        peak = float(np.abs(delta_xyz).max())
        if peak > 1.0:
            delta_xyz = delta_xyz / peak

    openness = float(target10[sch.POSE_DIM])
    closing_effort = (0.5 - openness) * 2.0
    action = np.array([*delta_xyz, closing_effort], dtype=np.float32)
    return np.clip(action, -1.0, 1.0)


__all__ = [
    "STATE4_DIM",
    "ENV_ACTION_DIM",
    "ENV_XYZ_SCALE",
    "FLIP_CAMERAS",
    "PICK_PLACE_GRIPPER_THRESHOLD",
    "render_frame",
    "state4_to_canonical10",
    "canonical10_to_env_action",
]
