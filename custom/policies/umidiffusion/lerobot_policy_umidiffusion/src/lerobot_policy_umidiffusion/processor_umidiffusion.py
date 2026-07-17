"""UmiDiffusion pre/post processor — step 들을 pipeline 에 끼우는 역할만 한다.

변환 수학은 steps.py 에 있다. 여기 직접 쓰지 말 것.

함수 이름은 고정이다. lerobot 의 _make_processors_from_policy_config 가
make_<policy_type>_pre_post_processors 라는 컨벤션으로 찾기 때문에, 바꾸면 조용히 못 찾는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.pipeline import ProcessorStep
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .steps import (
    CanonicalPoseToActionPoseReprStep,
    CanonicalPoseToRelativeObservationStep,
)

from lerobot_canonical import keys
from .configuration_umidiffusion import UmiDiffusionConfig


@dataclass
class DropObservationKeysProcessorStep(ProcessorStep):
    """관측 dict 에서 특정 키를 제거한다. depth 게이트의 런타임 절반.

    config.apply_depth_gate() 는 모델이 depth 인코더를 안 만들게 하고(정확성), 이 step 은 관측
    dict 에서 실제로 빼서 GPU 로 안 올라가게 한다(효율). 그래서 Device step 앞에 놓아야 한다.

    observation 이 dict 가 아니거나 제거할 키가 없으면 transition 을 그대로 통과시킨다.
    """

    keys: tuple[str, ...] = (keys.DEPTH_KEY,)

    def __post_init__(self) -> None:
        self.keys = tuple(self.keys)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict):
            return transition

        if not any(key in observation for key in self.keys):
            # 제거할 게 없으면 그대로 반환 — 불필요한 dict 복사를 만들지 않는다.
            return transition

        new_observation = {k: v for k, v in observation.items() if k not in self.keys}
        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {"keys": list(self.keys)}


def make_umidiffusion_pre_post_processors(
    config: UmiDiffusionConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """pre/post pipeline 을 만든다.

    input_steps 는 순서가 의미를 가진다:

        [0] RenameObservations       키 정규화. rename_map={} 이라 no-op 이지만 규약상 둔다.
        [1] AddBatchDimension        이 뒤부터 (B,T,10)/(B,H,10) 보장. relative step 들이
                                     ndim==3 을 요구하므로 반드시 앞에 와야 한다.
        [2] CanonicalPoseToActionPoseRepr   action 을 먼저 변환한다 (아래 참조).
        [3] CanonicalPoseToRelativeObservation   그다음 관측.
        [4] Device                   GPU 로.
        [5] Normalizer               STATE/ACTION 은 IDENTITY 라 사실상 VISUAL 만 정규화된다.

    use_depth=False 면 index 1 에 DropObservationKeys 를 넣는다 — Rename 뒤, AddBatch 앞이라
    depth 가 배치·GPU 로 올라가기 전에 빠진다.

    output_steps 는 Unnormalizer, Device(cpu) 뿐이고 역변환이 없다. 정책은 relative 를 뱉는데
    post pipeline 은 PolicyAction 만 받아 앵커에 접근할 수 없기 때문이다. 추론 루프가
    decode_policy_action(action, anchor_state=<현재 관측>, action_pose_repr=...) 을 직접 불러야
    하고, 빠뜨리면 완전히 틀린 명령이 조용히 나간다 (retargeting.md 6절).

    Normalizer 의 features 는 input 과 output 을 합쳐 넘긴다. dataset_stats 가 None 이면
    STATE/ACTION 은 IDENTITY 라 무방하지만 VISUAL 은 통계가 필요하다 — make_policy 가
    ds_meta.stats 를 넘겨준다.
    """

    # depth 게이트 1겹: input_features 에서 depth 를 빼면 Normalizer 의 features 도 따라서 빠진다.
    # 정책 __init__ 도 같은 호출을 한다 — idempotent 이고, 누가 먼저 불릴지 보장이 없어 양쪽에서 부른다.
    config.apply_depth_gate()

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        # 순서 고정: action step 이 obs step 보다 먼저다. 액션 step 이 앵커로 쓰는 state 가 아직
        # 절대여야 하기 때문. 뒤집으면 obs step 이 state 를 relative 로 바꿔 마지막 프레임이
        # 항등(0,0,0, 1,0,0, 0,1,0)이 되고, 액션 step 이 그 항등을 앵커로 삼아 액션이 전혀 변환되지
        # 않는다 — 에러 없이 조용히 전부 틀린다.
        CanonicalPoseToActionPoseReprStep(action_pose_repr=config.action_pose_repr),
        CanonicalPoseToRelativeObservationStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    # depth 게이트 2겹: 관측 dict 에서 실제로 제거한다. 1겹만으로도 정확성은 확보되지만
    # (모델이 인코더를 안 만듦) 배치엔 여전히 depth 텐서가 실려 GPU 로 전송된다.
    if not config.use_depth:
        input_steps.insert(1, DropObservationKeysProcessorStep())

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
