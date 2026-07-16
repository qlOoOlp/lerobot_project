"""MyPolicy config — DiffusionConfig + pose 표현 옵션 + depth 게이트.

■ 플러그인 등록 (lerobot 무수정)
    @PreTrainedConfig.register_subclass("mypolicy") 만 붙이면
    lerobot 의 _get_policy_cls_from_policy_name 폴백이 `lerobot_policy_mypolicy` 를
    컨벤션으로 찾아 import 한다 => factory 패치 불필요.
    단, **패키지가 import 되어야 등록이 일어난다** -> __init__.py 에서 export 할 것.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

from custom.common.lerobot_ext_core import keys

DEPTH_KEY = keys.DEPTH_KEY


@PreTrainedConfig.register_subclass("mypolicy")
@dataclass
class MyPolicyConfig(DiffusionConfig):
    """
    ■ 필드 의미
      obs_pose_repr    : 관측 표현. **"relative" 만** 지원(정책이 관측을 만들지 않으니 backward 가 없음).
      action_pose_repr : 액션 표현. {"relative", "delta"}. 기본 relative.
                         ★ 이 값이 **학습(step)·추론(decode_policy_action) 양쪽**에 흘러야 한다.
                            원본 UMI 는 학습에서 obs_pose_repr 을 잘못 써서 delta 설정 시 조용히 깨짐.
      use_depth        : depth ablation 스위치. metaworld=False(depth 없음), UMI=True.

    ■ ★ normalization_mapping 이 STATE/ACTION = IDENTITY 인 이유 (dev_plan §11)
        dataset stats 는 **canonical(절대)** 기준으로 계산되는데, 런타임 step 이 이를
        **relative** 로 바꾼다 -> canonical stats 로 relative 를 정규화하면 표현 공간이 안 맞는다.
        1차 전략: 정규화하지 않음(relative 값은 이미 0 근처 작은 범위).
        부수효과: metaworld 의 rot6d 6채널은 **std=0**(Sawyer 무회전)이라 MEAN_STD 였다면
                  0-나눗셈이 났을 것 — IDENTITY 가 이것도 함께 회피한다.
        확장: 필요해지면 relative 기준 stats 를 따로 계산.
      VISUAL 은 default diffusion 전략(MEAN_STD) 유지.
    """

    obs_pose_repr: str = "relative"
    action_pose_repr: str = "relative"
    use_depth: bool = True
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self) -> None:
        """super().__post_init__() 후 obs_pose_repr=="relative", action_pose_repr in {relative,delta} 검증."""
        ...  # 구현 ①

    def apply_depth_gate(self) -> None:
        """use_depth=False 면 input_features 에서 depth 키를 제거한다.

        ■ 이게 lerobot 패치 60줄을 대체한다
          lerobot_hong 은 datasets/factory.py(+15) + policies/factory.py(+45) 를 패치해
          depth 를 걸러냈다. 그 로직을 **config 로 옮겨** 무수정을 달성한다.

        ■ 원리
          depth 를 input_features 에서 빼면 DiffusionPolicy 가 **depth 인코더를 안 만들고**
          배치의 depth 를 무시한다 -> 별도 필터 불필요.

        ■ hook 위치가 중요
          make_policy 는 input_features 를 채운 뒤(factory.py:517) validate_features 를
          **안 부르고** 바로 정책을 생성한다 => 필터는 **MyPolicyPolicy.__init__ 의 super() 직전**이
          정답(이때 input_features 는 세팅됐고 모델은 아직 안 만들어짐).

        ■ 유의
          - **idempotent** 해야 한다(정책/프로세서 어느 쪽이 먼저 불러도 동일). 이미 없으면 no-op.
          - input_features 가 비어있을 수 있으니 방어.
          - 체크포인트 config 에 필터된 input_features 가 저장되므로 로드 시에도 일관.
          - ⚠ depth 게이트는 **두 겹**: 이 config 게이트(모델이 인코더를 안 만들게) +
            DropObservationKeysProcessorStep(관측 dict 에서 실제 제거). 둘 다 필요.
        """
        ...  # 구현 ②
