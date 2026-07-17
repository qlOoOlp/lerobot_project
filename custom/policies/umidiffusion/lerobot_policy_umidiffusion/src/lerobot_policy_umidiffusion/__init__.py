"""lerobot_policy_umidiffusion — BYO Policy 플러그인.

이 파일은 형식적인 것이 아니라 실질적인 wiring 이다. lerobot 은 lerobot_policy_<name> 패키지를
import 해서 정책을 찾는데, 등록(@register_subclass)이 그 import 시점에 일어나야 한다. 즉 여기서
UmiDiffusionConfig 를 import 하는 것 자체가 등록 트리거다. export 를 빼면
make_policy(type="umidiffusion") 가 조용히 실패한다.
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
