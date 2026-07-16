"""런타임 pose 표현 step — 정책이 "지금 내 자세 기준"으로 보게 만든다.

■ 이 층의 책임 (dev_plan §7.3/§9.5)
    codec(schemas/canonical_ee10_se3.py) = 순수 변환 수학       <- 여기서 import
    이 파일                              = **앵커 의미** + Step 래퍼
    processor_umidiffusion.py                = pipeline 조립만

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

from lerobot_canonical.schemas import canonical_ee10 as sch
from lerobot_canonical.schemas.canonical_ee10_se3 import (
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
    if value.shape[-1] != sch.STATE_DIM:
        raise ValueError(
            f"Expected `{name}` trailing dim {sch.STATE_DIM}, got {tuple(value.shape)}."
        )
    pose = value[..., : sch.POSE_DIM]
    # ★ `[..., sch.POSE_DIM:]` (슬라이스) — `[..., sch.POSE_DIM]` (정수 인덱스) 이면 축이 사라져
    #   (..., ) 가 되고, 나중에 cat 할 때 shape 이 안 맞는다.
    gripper = value[..., sch.POSE_DIM :]
    return pose, gripper


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
        => pipeline 순서 의존성 (processor_umidiffusion.py 참고).
      - state_key 가 observation 에 없으면 **그냥 통과**시킨다(방어적).
      - obs_pose_repr 은 "relative" 만 지원 — config __post_init__ 에서 이미 막는다.

    ■ ★ (B, T, 10) 은 누가 만드나 — 학습/추론이 다르다
        학습  : DataLoader 가 delta_timestamps 로 윈도우를 잘라 (B, T, 10) 로 준다.
        추론  : 정책의 _queues stack 은 predict_action_chunk **안**에서 일어나므로
                프로세서보다 **뒤**다. select_action() 을 쓰면 이 step 은 (B, 10) 을 받아
                앵커를 만들 수 없다.
                => rollout 이 **자체 히스토리 버퍼**로 (B, T, 10) 을 만들어 preprocessor 에
                   넘기고, select_action() 대신 policy.diffusion.generate_actions() 를
                   직접 불러야 한다. lerobot_hong rollout_metaworld_mypolicy.py:83-93 이
                   정확히 이 형태다. 아래 ndim 검사가 그 계약을 강제한다. (Phase 5)
    """

    state_key: str = OBS_STATE

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict) or self.state_key not in observation:
            # 방어적: state 가 없는 transition 은 우리 관심사가 아니다.
            return transition

        state = observation[self.state_key]
        if state.ndim != 3:
            raise ValueError(
                f"Expected `{self.state_key}` to be (B, T, {sch.STATE_DIM}) after batching, "
                f"got shape {tuple(state.shape)}. This step must run after "
                f"AddBatchDimensionProcessorStep, and at rollout the caller must supply an "
                f"observation window (see the class docstring)."
            )
        if state.shape[-1] != sch.STATE_DIM:
            raise ValueError(
                f"Expected `{self.state_key}` trailing dim {sch.STATE_DIM}, got {tuple(state.shape)}."
            )

        pose, gripper = _split_pose_and_gripper(state, self.state_key)

        state_transform = pose9d_to_transform(pose)                     # (B, T, 4, 4)
        # ★ `[:, -1:]` (슬라이스) 로 T 축을 **유지**한 뒤 expand. `[:, -1]` 이면 축이 사라져
        #   (B, 4, 4) 가 되고 expand_as 가 깨진다.
        anchor_transform = state_transform[:, -1:, :, :].expand_as(state_transform)
        relative = relative_transform(anchor_transform, state_transform)

        # 그리퍼는 좌표계와 무관하므로 원본 그대로 이어붙인다 ("0.7만큼 열려라"는 내가
        # 어디 있든 같은 뜻). 이래야 10D -> 10D 로 차원이 유지된다.
        new_state = torch.cat((transform_to_pose9d(relative), gripper), dim=-1)

        # transition/observation 을 in-place 로 고치지 않는다 — 호출자가 원본을 들고 있을 수 있고,
        # lerobot 의 다른 step 들도 copy 패턴을 쓴다(normalize_processor.py:446).
        new_observation = dict(observation)
        new_observation[self.state_key] = new_state
        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

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

    ■ 학습/추론 대칭 (원본 UMI 의 버그를 피하는 지점)
      원본 umi_dataset.py:349 는 액션 변환에 `pose_rep=self.obs_pose_repr` 을 넘긴다
      (action_pose_repr 이어야 함). 기본값이 둘 다 relative 라 안 드러나지만
      action_pose_repr="delta" 로 두면 **학습=relative / 추론=delta** 로 조용히 갈라진다.
      => 이 step 과 decode_policy_action() 이 **반드시 같은 action_pose_repr** 을 받아야 한다.

    ■ action 이 없으면 그냥 통과 (dev_plan §9.3)
      학습 배치엔 action 이 있지만 **eval/추론 관측엔 없다**. `if action is None: return`.
      이걸 빠뜨리면 추론 때 터진다.

    ■ 유의
      - action.ndim != 3 이면 ValueError. state 와 batch 크기 일치도 검증.
      - 그리퍼(action[..., 9:])는 변환하지 않고 그대로 이어붙인다.
    """

    action_pose_repr: str = "relative"
    state_key: str = OBS_STATE

    def __post_init__(self) -> None:
        """action_pose_repr 이 {"relative","delta"} 인지 검증 → 아니면 ValueError.

        config.__post_init__ 에도 같은 검증이 있다. 중복이 아니라 **각자 필요**하다 —
        이 step 은 config 없이 직접 만들어질 수 있기 때문(ProcessorStepRegistry 역직렬화:
        체크포인트가 get_config() 의 dict 를 그대로 생성자에 넣는다).
        """
        if self.action_pose_repr not in {"relative", "delta"}:
            raise ValueError(
                f'`action_pose_repr` must be one of {{"relative", "delta"}}. Got {self.action_pose_repr}.'
            )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition

        observation = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict) or self.state_key not in observation:
            raise ValueError(
                f"`{type(self).__name__}` needs `{self.state_key}` in the observation to build the "
                f"anchor, but got {type(observation).__name__}. It must run inside the policy "
                f"preprocessor, not standalone."
            )

        if action.ndim != 3:
            raise ValueError(
                f"Expected `action` to be (B, H, {sch.ACTION_DIM}) after batching, "
                f"got shape {tuple(action.shape)}."
            )
        if action.shape[-1] != sch.ACTION_DIM:
            raise ValueError(
                f"Expected `action` trailing dim {sch.ACTION_DIM}, got {tuple(action.shape)}."
            )

        state = observation[self.state_key]
        if state.shape[0] != action.shape[0]:
            raise ValueError(
                f"Batch size mismatch: `{self.state_key}` has {state.shape[0]}, "
                f"`action` has {action.shape[0]}."
            )

        action_pose, action_gripper = _split_pose_and_gripper(action, "action")
        state_pose, _ = _split_pose_and_gripper(state, self.state_key)

        # ★ 앵커는 **관측**의 마지막 프레임. 액션의 첫 프레임이 아니다 —
        #   그러면 관측과 기준이 갈라져 정책이 배울 게 없어진다.
        anchor_transform = pose9d_to_transform(state_pose[:, -1, :])    # (B, 4, 4)
        action_transform = pose9d_to_transform(action_pose)             # (B, H, 4, 4)

        if self.action_pose_repr == "relative":
            # a_i' = inv(anchor) @ a_i — 전부 같은 앵커라 오차가 전파되지 않는다.
            # 원본 UMI pose_repr_util.py:63-64 `'relative'` 와 동일.
            out_transform = relative_transform(
                anchor_transform.unsqueeze(1).expand_as(action_transform), action_transform
            )
        else:  # "delta"
            # ★ delta 는 SE(3) 합성이 **아니다** — 원본 UMI(pose_repr_util.py:65-77)가
            #   위치/회전을 분리해 정의했으므로 그대로 따른다:
            #       위치: np.diff([base_t, t_0..t_{H-1}])  =>  t_i - t_{i-1}   (뺄셈)
            #       회전: R_i @ inv(R_{i-1})                                    (곱셈)
            #   위치는 뺄셈인데 회전은 곱셈인 비대칭이 이상해 보여도, decode_policy_action 과
            #   짝이 맞아야 하므로 원본을 따른다. 둘이 갈라지면 조용히 깨진다.
            prev = torch.cat((anchor_transform.unsqueeze(1), action_transform[:, :-1]), dim=1)
            delta_t = action_transform[..., :3, 3] - prev[..., :3, 3]
            delta_r = torch.matmul(
                action_transform[..., :3, :3], prev[..., :3, :3].transpose(-1, -2)
            )
            out_transform = action_transform.new_zeros(action_transform.shape)
            out_transform[..., :3, :3] = delta_r
            out_transform[..., :3, 3] = delta_t
            out_transform[..., 3, 3] = 1.0

        # 그리퍼는 변환하지 않고 그대로 이어붙인다 (좌표계와 무관).
        new_action = torch.cat((transform_to_pose9d(out_transform), action_gripper), dim=-1)

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = new_action
        return new_transition

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
    if action_pose_repr not in {"relative", "delta"}:
        raise ValueError(
            f'`action_pose_repr` must be one of {{"relative", "delta"}}. Got {action_pose_repr}.'
        )

    # 입력 shape 을 (B, H, 10) 으로 통일한 뒤, 마지막에 원래 모양으로 되돌린다.
    # rollout 은 (H,10) 을, 단일 스텝 경로는 (10,) 를 넘길 수 있어서 셋 다 받아야 한다.
    original_ndim = action.ndim
    if original_ndim == 1:
        action = action[None, None]        # (10,)     -> (1, 1, 10)
    elif original_ndim == 2:
        action = action[None]              # (H, 10)   -> (1, H, 10)
    elif original_ndim != 3:
        raise ValueError(f"Expected action ndim in {{1, 2, 3}}, got {tuple(action.shape)}.")

    if anchor_state.ndim == 1:
        anchor_state = anchor_state[None]  # (10,) -> (1, 10)
    elif anchor_state.ndim != 2:
        raise ValueError(f"Expected anchor_state ndim in {{1, 2}}, got {tuple(anchor_state.shape)}.")

    if anchor_state.shape[0] != action.shape[0]:
        raise ValueError(
            f"Batch size mismatch: anchor_state has {anchor_state.shape[0]}, "
            f"action has {action.shape[0]}."
        )

    action_pose, action_gripper = _split_pose_and_gripper(action, "action")
    anchor_pose, _ = _split_pose_and_gripper(anchor_state, "anchor_state")

    anchor_transform = pose9d_to_transform(anchor_pose)      # (B, 4, 4)
    action_transform = pose9d_to_transform(action_pose)      # (B, H, 4, 4)

    if action_pose_repr == "relative":
        # forward 가 inv(anchor) @ a 였으므로 역은 anchor @ a'.
        # 원본 UMI pose_repr_util.py:94-95 backward `'relative'` 와 동일.
        decoded = torch.matmul(anchor_transform.unsqueeze(1), action_transform)
    else:  # "delta"
        # 원본 UMI pose_repr_util.py:97-106 backward `'delta'`:
        #   위치: cumsum(delta_t) + anchor_t          (forward 의 diff 를 되돌림)
        #   회전: R_i = delta_R_i @ R_{i-1},  R_{-1} = anchor_R   (순차 곱)
        position = torch.cumsum(action_transform[..., :3, 3], dim=1) + anchor_transform[
            ..., :3, 3
        ].unsqueeze(1)

        # ★ 회전은 벡터화가 안 된다 — R_i 가 R_{i-1} 에 의존하는 **연쇄**라서.
        #   이게 delta 가 relative 보다 오차에 취약한 이유이기도 하다(오차가 누적된다).
        rotations = []
        current = anchor_transform[..., :3, :3]              # (B, 3, 3)
        for i in range(action_transform.shape[1]):
            current = torch.matmul(action_transform[:, i, :3, :3], current)
            rotations.append(current)
        rotation = torch.stack(rotations, dim=1)             # (B, H, 3, 3)

        decoded = action_transform.new_zeros(action_transform.shape)
        decoded[..., :3, :3] = rotation
        decoded[..., :3, 3] = position
        decoded[..., 3, 3] = 1.0

    out = torch.cat((transform_to_pose9d(decoded), action_gripper), dim=-1)

    # 들어온 모양 그대로 돌려준다.
    if original_ndim == 1:
        return out[0, 0]
    if original_ndim == 2:
        return out[0]
    return out


__all__ = [
    "CanonicalPoseToRelativeObservationStep",
    "CanonicalPoseToActionPoseReprStep",
    "decode_policy_action",
]
