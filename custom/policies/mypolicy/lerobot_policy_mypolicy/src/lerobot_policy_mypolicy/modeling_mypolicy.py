"""MyPolicy 모델 — default DiffusionPolicy 를 최대한 재사용 (dev_plan §12.3).

■ 이 파일이 짧아야 정상이다
  모델 구조를 바꾸는 게 목적이 아니다. custom policy 의 존재 이유는 **런타임 pose step 을
  processor 에 끼우는 것**(dev_plan §12.1)이지 네트워크가 아니다.
  lerobot_hong 의 modeling_mypolicy.py 도 251B(4줄)뿐이었다.

■ 최소 연결값 (dev_plan §12.3)
    config_class = MyPolicyConfig
    name = "mypolicy"          <- 플러그인 폴백이 이 이름으로 찾는다
"""
from __future__ import annotations

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

from .configuration_mypolicy import MyPolicyConfig


class MyPolicyPolicy(DiffusionPolicy):
    config_class = MyPolicyConfig
    name = "mypolicy"

    def __init__(self, config: MyPolicyConfig, *args, **kwargs) -> None:
        """★ super() **직전**에 config.apply_depth_gate() 를 호출한다.

        ■ 왜 하필 여기인가
          make_policy 가 input_features 를 채운 뒤(factory.py:517) validate_features 를
          안 부르고 바로 정책을 생성한다. 즉 이 시점이:
            - input_features 는 이미 세팅됨 ✓
            - 모델은 아직 안 만들어짐 ✓
          => depth 를 빼면 DiffusionPolicy 가 depth 인코더를 **안 만든다**. lerobot 패치 불필요.

        ■ 유의
          - super().__init__() **뒤에** 부르면 이미 인코더가 만들어져서 소용없다.
          - apply_depth_gate 는 idempotent 라 processor 쪽에서 또 불러도 안전.
          - config 를 in-place 로 수정하므로, 체크포인트 저장 시 필터된 input_features 가
            함께 저장된다 -> 로드 시 자동 일관.
        """
        ...  # 구현 ①


# lerobot_hong 호환 별칭 (dev_plan §12.3 은 `MyPolicy` 를 export 하라고 함)
MyPolicy = MyPolicyPolicy

__all__ = ["MyPolicyPolicy", "MyPolicy"]
