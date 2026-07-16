"""lerobot_policy_umidiffusion — BYO Policy 플러그인.

■ ★ 이 파일은 선택이 아니라 **실질적 wiring 파일** (dev_plan §12.3)
  lerobot 의 플러그인 폴백은 `lerobot_policy_<name>` 패키지를 import 한다.
  그 import 시점에 **config 등록(@register_subclass)이 일어나야** 하므로,
  여기서 UmiDiffusionConfig 를 import 하는 것 자체가 등록 트리거다.
  export 를 빼먹으면 `make_policy(type="umidiffusion")` 가 조용히 실패한다.

■ export 해야 할 3종 (dev_plan §12.3)
    UmiDiffusionConfig
    UmiDiffusion
    make_umidiffusion_pre_post_processors
"""

from .configuration_umidiffusion import UmiDiffusionConfig
from .modeling_umidiffusion import UmiDiffusion, UmiDiffusionPolicy
from .processor_umidiffusion import DropObservationKeysProcessorStep, make_umidiffusion_pre_post_processors

__all__ = [
    "UmiDiffusionConfig",
    "UmiDiffusion",
    "UmiDiffusionPolicy",
    "make_umidiffusion_pre_post_processors",
    "DropObservationKeysProcessorStep",
]
