"""lerobot_processor_robot_maps — 재사용 가능한 런타임 pose 표현 step (dev_plan §7.3).

여기엔 **앵커 의미 + Step 래퍼**만 둔다. 순수 변환 수학은
`custom.common.lerobot_ext_core.schemas.canonical_ee10_se3` (표현 codec) 에 있고
이 패키지는 그걸 import 해서 쓴다 — 중복 구현 금지(refactoring.md 부록 D.5).

`decode_policy_action` 은 `CanonicalPoseToActionPoseReprStep` 의 **정확한 역함수**라
같은 모듈에 둔다: 떨어뜨려 놓으면 둘이 갈라져도 아무도 모른다(원본 UMI 가 그렇게 깨졌다).
"""

from .steps import (
    CanonicalPoseToActionPoseReprStep,
    CanonicalPoseToRelativeObservationStep,
    decode_policy_action,
)

__all__ = [
    "CanonicalPoseToRelativeObservationStep",
    "CanonicalPoseToActionPoseReprStep",
    "decode_policy_action",
]
