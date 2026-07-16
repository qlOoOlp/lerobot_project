"""ext_core 데이터셋 키.
lerobot 규약(observation.images.<cam>, observation.state, action) 위에
canonical 데이터셋이 쓰는 키를 얇게 얹는다. 값은 전부 lerobot 상수에서
파생한다 — 문자열 하드코딩 금지(규약이 바뀌어도 자동 추종).
"""
from __future__ import annotations

from lerobot.utils.constants import OBS_IMAGES, OBS_STATE, ACTION


def image_key(cam_name: str) -> str:
    """카메라 이름 → 이미지 feature 전체 키 생성
    예) "rgb" -> "observation.images.rgb"
    """
    return f"{OBS_IMAGES}.{cam_name}"


# Dataset keys
RGB_KEY: str = image_key("rgb")  
DEPTH_KEY: str = image_key("depth")
STATE_KEY: str = OBS_STATE 
ACTION_KEY: str = ACTION 

__all__ = ["image_key", "RGB_KEY", "DEPTH_KEY", "STATE_KEY", "ACTION_KEY"]
