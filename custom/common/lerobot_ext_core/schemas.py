"""ext_core canonical schema — dim, axes

canonical layout (10D): [x, y, z, rot6d(6), gripper]
  - pose    : xyz(3) + rot6d(6) = 9D
  - gripper : 1D
Used for LeRobotDataset feature `shape`/`names.axes`  
"""
from __future__ import annotations

# ── 차원 ──────────────────────────────────────────────────────
POSE_DIM: int = 9       # 구현 ①  primitive: xyz(3) + rot6d(6)
GRIPPER_DIM: int = 1    # 구현 ②  primitive: gripper 스칼라
STATE_DIM: int = POSE_DIM + GRIPPER_DIM      # 구현 ③  derived: pose + gripper
ACTION_DIM: int = POSE_DIM + GRIPPER_DIM     # 구현 ④  derived: action 도 같은 레이아웃

# ── 축 이름 (feature 의 names.axes 재료 · 채널 순서와 일치) ─────
POSE_AXES: tuple[str, ...] = ("x", "y", "z", "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5")     # 구현 ⑤  primitive: 9개 (x,y,z,rot6d_0..5)
GRIPPER_AXES: tuple[str, ...] = ("gripper",)  # 구현 ⑥  primitive: 1개 (1-튜플 = 끝에 콤마)
STATE_AXES: tuple[str, ...] = POSE_AXES + GRIPPER_AXES    # 구현 ⑦  derived: pose + gripper
ACTION_AXES: tuple[str, ...] = POSE_AXES + GRIPPER_AXES   # 구현 ⑧  derived

__all__ = [
    "POSE_DIM", "GRIPPER_DIM", "STATE_DIM", "ACTION_DIM",
    "POSE_AXES", "GRIPPER_AXES", "STATE_AXES", "ACTION_AXES",
]
