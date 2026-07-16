"""런타임 pose 표현 step — 정책이 "지금 내 자세 기준"으로 보게 만든다.

■ 이 층의 책임 (dev_plan §7.3/§9.5)
    codec(schemas/canonical_ee10_se3.py) = 순수 변환 수학       <- 여기서 import
    이 파일                              = **앵커 의미** + Step 래퍼
    processor_mypolicy.py                = pipeline 조립만

■ ★ 왜 런타임인가 (dev_plan §3.2)
    "relative/delta 는 frame 의 고정 속성이 아니라 **sample window 의 anchor 에 종속된 표현**"
    앵커 = 샘플된 관측 윈도우의 마지막 프레임. 학습 때 윈도우를 어디서 자르냐에 따라 달라지므로
    **오프라인 고정 저장이 원천적으로 불가능**하다.
    (대비: 그리퍼 이진화는 threshold 가 고정 상수라 오프라인 bake 가능 — retargeting.md 4절)

■ ★ 왜 custom policy 가 필요한가 (dev_plan §12.1)
    policy.type=diffusion 을 그대로 쓰면 factory 가 만드는 기본 processor 에 이 step 들이 없다.
    => custom policy 의 존재 이유가 곧 이 파일이다.

■ 공통 규약
    - **pose 9D 만 변환**, gripper 1D 는 손대지 않고 그대로 이어붙인다.
    - **차원 유지** 10D -> 10D (dev_plan §9.4)
    - 앵커는 항상 `state[:, -1]` (관측 히스토리의 마지막 = 지금 내 자세)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStepRegistry
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.pipeline import ProcessorStep
from lerobot.utils.constants import ACTION, OBS_STATE

from custom.common.lerobot_ext_core.schemas import canonical_ee10 as sch
from custom.common.lerobot_ext_core.schemas.canonical_ee10_se3 import (
    pose9d_to_transform,
    relative_transform,
    transform_to_pose9d,
)


def _split_pose_and_gripper(value: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(..., 10) -> (pose (..., 9), gripper (..., 1)).

    ■ 유의
      - gripper 를 `[..., sch.POSE_DIM]` 로 뽑으면 **축이 사라진다**. 슬라이스
        `[..., sch.POSE_DIM:]` 를 써서 (..., 1) 을 유지해야 나중에 cat 이 된다.
        (GRIPPER_AXES=("gripper",) 콤마, state4[..., 3:4] 와 같은 함정)
      - trailing dim 이 sch.STATE_DIM 인지 검증하고 아니면 ValueError.
    """
    ...  # 구현 ①


@ProcessorStepRegistry.register("canonical_pose_to_relative_observation")
@dataclass
class CanonicalPoseToRelativeObservationStep(ProcessorStep):
    """관측 히스토리를 **현재(마지막) 관측 기준**으로 재표현.

    입력  observation[state_key] : (B, T, 10) 절대 canonical
    출력  같은 키                : (B, T, 10) 앵커 기준 relative

    ■ 동작
        state_transform  = pose9d_to_transform(pose_state)        # (B, T, 4, 4)
        anchor_transform = state_transform[:, -1:, :, :].expand_as(state_transform)
        relative_pose    = transform_to_pose9d(relative_transform(anchor, state))
        out              = cat(relative_pose, gripper_state)      # 그리퍼는 원본 그대로

    ■ 유의
      - `[:, -1:]`(슬라이스)로 축을 **유지**한 뒤 expand. `[:, -1]` 이면 축이 사라져 broadcast 가 깨짐.
      - **결과의 마지막 프레임 pose 는 항상 (0,0,0, 1,0,0,0,1,0)** (anchor⁻¹@anchor = I).
        정보량 0이지만 정상 — 정보는 이전 히스토리 프레임과 이미지에 있다.
      - `state.ndim != 3` 이면 ValueError. AddBatchDimension 뒤에 놓여야 (B,T,10) 이 보장된다
        => pipeline 순서 의존성 (processor_mypolicy.py 참고).
      - state_key 가 observation 에 없으면 **그냥 통과**시킨다(방어적).
      - obs_pose_repr 은 "relative" 만 지원 — config __post_init__ 에서 이미 막는다.
    """

    state_key: str = OBS_STATE

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        ...  # 구현 ②

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """차원이 안 바뀌므로(10->10) features 를 그대로 반환. (dev_plan §9.4)"""
        return features

    def get_config(self) -> dict[str, Any]:
        return {"state_key": self.state_key}


@ProcessorStepRegistry.register("canonical_pose_to_action_pose_repr")
@dataclass
class CanonicalPoseToActionPoseReprStep(ProcessorStep):
    """액션(학습 정답)을 **관측과 같은 앵커** 기준으로 재표현.

    입력  transition[ACTION] : (B, H, 10) 절대 canonical
    출력  같은 키            : (B, H, 10) relative 또는 delta

    ■ 앵커는 **관측**에서 가져온다 (액션이 아니라!)
        anchor = pose9d_to_transform(state_pose[:, -1, :]).unsqueeze(1)
      => 이 step 은 observation 이 **반드시** 있어야 한다. 없으면 ValueError.

    ■ relative vs delta (원본 UMI pose_repr_util.py)
        relative : a_i' = inv(anchor) @ a_i          전부 같은 앵커 -> 오차 전파 없음
        delta    : d_0 = a_0 - anchor, d_i = a_i - a_{i-1}   연쇄 -> 오차 누적
      기본값은 relative.

    ■ ★★ 학습/추론 대칭 (원본 UMI 의 버그를 피하는 지점)
      원본 umi_dataset.py:349 는 액션 변환에 `pose_rep=self.obs_pose_repr` 을 넘긴다
      (action_pose_repr 이어야 함). 기본값이 둘 다 relative 라 안 드러나지만
      action_pose_repr="delta" 로 두면 **학습=relative / 추론=delta** 로 조용히 갈라진다.
      => 이 step 과 decode_policy_action() 이 **반드시 같은 action_pose_repr** 을 받아야 한다.

    ■ ★ action 이 없으면 그냥 통과 (dev_plan §9.3)
      학습 배치엔 action 이 있지만 **eval/추론 관측엔 없다**. `if action is None: return`.
      이걸 빠뜨리면 추론 때 터진다.

    ■ 유의
      - action.ndim != 3 이면 ValueError. state 와 batch 크기 일치도 검증.
      - 그리퍼(action[..., 9:])는 변환하지 않고 그대로 이어붙인다.
    """

    action_pose_repr: str = "relative"
    state_key: str = OBS_STATE

    def __post_init__(self) -> None:
        """action_pose_repr 이 {"relative","delta"} 인지 검증 → 아니면 ValueError."""
        ...  # 구현 ③

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        ...  # 구현 ④

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {"action_pose_repr": self.action_pose_repr, "state_key": self.state_key}


def decode_policy_action(
    action: torch.Tensor,
    anchor_state: torch.Tensor,
    *,
    action_pose_repr: str = "relative",
) -> torch.Tensor:
    """★ 역변환 — 정책의 relative 출력을 **절대 canonical** 로 되돌린다. 추론의 필수 절반.

    CanonicalPoseToActionPoseReprStep 의 **정확한 역함수**. 그래서 여기(같은 파일)에 둔다 —
    떨어뜨려 놓으면 둘이 갈라져도 아무도 모른다(원본 UMI 가 정확히 그렇게 깨졌다).

    ■ 왜 pipeline 밖의 평범한 함수인가
      policy_post 는 `PolicyAction` 만 받아 **앵커(현재 관측)에 접근할 수 없다**.
      그래서 원본 UMI(get_real_umi_action)·lerobot_hong(decode_policy_action) 모두
      **추론 루프가 직접 호출**한다. Phase 5 rollout 도 같은 형태.

    ■ 없으면 어떻게 되나
      정책의 "지금 기준 +2cm" 를 "월드 좌표 2cm 지점"으로 오해 -> **완전히 틀린 명령**.

    Args:
        action: (10,) | (H, 10) | (B, H, 10) — 정책 출력(역정규화 후)
        anchor_state: (10,) | (B, 10) — **현재(마지막) 관측**의 canonical state
        action_pose_repr: forward step 과 **반드시 같은 값**

    Returns:
        절대 canonical, 입력과 같은 shape

    ■ 동작
        relative : decoded = anchor_transform.unsqueeze(1) @ pose9d_to_transform(action_pose)
        delta    : 앵커에서 시작해 스텝마다 누적
                     t_i = t_{i-1} + delta_t_i
                     R_i = delta_R_i @ R_{i-1}
                   (원본 UMI 의 backward delta 와 동일: cumsum(pos) + base, 회전은 순차 곱)

    ■ 유의
      - 그리퍼는 변환하지 않고 그대로 이어붙인다.
      - 입력 ndim 에 따라 unsqueeze 했다가 **끝에 원래 shape 로 되돌릴 것**.
      - 회귀 테스트 필수: decode(forward(a, anchor), anchor) == a  (relative/delta 둘 다)
    """
    ...  # 구현 ⑤


__all__ = [
    "CanonicalPoseToRelativeObservationStep",
    "CanonicalPoseToActionPoseReprStep",
    "decode_policy_action",
]
