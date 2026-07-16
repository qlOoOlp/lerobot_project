"""UmiDiffusion pre/post processor — step 들을 pipeline 에 **끼우는 역할만** (dev_plan §9.5).

■ 이 파일의 책임 경계
    steps.py               = 순수 변환 로직 + 앵커 의미
    이 파일                 = pipeline 조립                <- 여기
  변환 수학을 여기 직접 쓰지 말 것. 재사용성·역할 분리(§9.5).

■ 플러그인 자동 탐지 (lerobot 무수정)
    lerobot 의 _make_processors_from_policy_config 폴백이 컨벤션으로
    `make_<name>_pre_post_processors` 를 찾는다 => factory 패치 불필요.
    이름을 바꾸면 못 찾으니 고정.
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
    """관측 dict 에서 특정 키를 제거 — **depth 게이트의 런타임 절반**.

    ■ 왜 config 게이트만으로 부족한가
      apply_depth_gate() 는 **모델이 depth 인코더를 안 만들게** 한다(정확성).
      이 step 은 **관측 dict 에서 실제로 빼서** GPU 로 안 올라가게 한다(효율 + 방어).
      lerobot_hong 도 두 겹을 다 쓴다.

    ■ 유의
      - observation 이 dict 가 아니면 그냥 통과(방어적).
      - 제거할 키가 없으면 transition 을 **그대로 반환**(불필요한 copy 방지).
      - Device 앞에 놓아야 depth 가 GPU 로 안 올라간다 -> pipeline 순서 의존.
    """

    keys: tuple[str, ...] = (keys.DEPTH_KEY,)

    def __post_init__(self) -> None:
        self.keys = tuple(self.keys)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict):
            return transition

        if not any(key in observation for key in self.keys):
            # 제거할 게 없으면 transition 을 **그대로** 반환 — 불필요한 dict 복사를 만들지 않는다.
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
    """pre/post pipeline 생성. **이름 고정** — 플러그인 폴백이 컨벤션으로 찾는다.

    ■ ★ input_steps 순서가 의미를 가진다
        [0] RenameObservations        키 정규화 (rename_map={} 이면 no-op이지만 규약상 유지)
        [1] AddBatchDimension         ← 이 뒤부터 (B,T,10)/(B,H,10) 보장.
                                        relative step 들이 ndim==3 을 요구하므로 **반드시 앞에**
        [2] CanonicalPoseToActionPoseRepr(config.action_pose_repr)
                                      ★ action 을 **먼저** 변환. 이유: 앵커로 쓰는 state 가
                                        아직 **절대**여야 한다. 순서를 뒤집으면 이미 relative 로
                                        바뀐 state(마지막 프레임=항등)를 앵커로 삼아 **전부 망가진다**.
        [3] CanonicalPoseToRelativeObservation()
                                      그다음 관측을 relative 로.
        [4] Device                    GPU 로
        [5] Normalizer                STATE/ACTION 은 IDENTITY(dev_plan §11) -> 사실상 VISUAL 만

      use_depth=False 면 index 1 에 DropObservationKeys 삽입
      (Rename 뒤, AddBatch 앞 — depth 를 배치·GPU 로 올리기 전에 제거)

    ■ output_steps
        [0] Unnormalizer  [1] Device(cpu)
      ★ 여기에 **역변환이 없다**. 정책은 relative 를 뱉으므로 추론 루프가
        `decode_policy_action(action, anchor_state=<현재 관측>, action_pose_repr=config.action_pose_repr)`
        을 직접 호출해야 한다. post 는 PolicyAction 만 받아 **앵커에 접근할 수 없기 때문**.
        빠뜨리면 완전히 틀린 명령이 나간다 (retargeting.md 6절).

    ■ 유의
      - Normalizer 의 features 는 input+output 을 합쳐 넘긴다(config.input_features | output_features).
      - dataset_stats 가 None 이면 정규화 통계가 없다 -> IDENTITY 라 STATE/ACTION 은 무방하나
        VISUAL 은 필요. make_policy 가 ds_meta.stats 를 넘겨준다.
      - post pipeline 은 to_transition/to_output 을 지정해야 PolicyAction <-> transition 변환이 된다.
    """

    # depth 게이트 **1겹**: input_features 에서 depth 를 뺀다 -> Normalizer 의 features 가
    # 자동으로 depth 를 제외하고, 모델도 인코더를 안 만든다(정책 __init__ 이 또 부른다 — idempotent).
    # 정책과 프로세서 중 누가 먼저 불릴지 보장이 없어 양쪽에서 부른다.
    config.apply_depth_gate()

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        # ★★ 순서 고정: action step 이 obs step 보다 **먼저**.
        #    액션 step 이 앵커로 쓰는 state 가 아직 '절대'여야 하기 때문. 뒤집으면 obs step 이
        #    state 를 relative 로 바꿔 마지막 프레임이 항등(0,0,0,1,0,0,0,1,0)이 되고, 액션 step 이
        #    그 항등을 앵커로 삼아 **액션이 전혀 변환되지 않는다** — 에러 없이 전부 망가진다.
        #    지금은 둘 다 항등(2-3)이라 순서가 결과를 안 바꾸지만, 2-5 에서 수학이 들어오는
        #    순간 이 순서가 정답과 재앙을 가른다.
        CanonicalPoseToActionPoseReprStep(action_pose_repr=config.action_pose_repr),
        CanonicalPoseToRelativeObservationStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    # depth 게이트 **2겹째**: 관측 dict 에서 실제로 제거해 배치·GPU 로 올라가지 않게 한다.
    # 1겹(config)만으로도 '정확성'은 확보되지만(모델이 인코더를 안 만듦), 배치엔 여전히
    # depth 텐서가 실려 GPU 로 전송된다 -> 낭비. 그래서 index 1 = **Rename 뒤 / AddBatch 앞**.
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
