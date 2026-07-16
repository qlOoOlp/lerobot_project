from __future__ import annotations

from lerobot.utils.constants import OBS_IMAGES, OBS_STATE, ACTION


def image_key(cam_name: str) -> str:
    """camera name -> image feature key
    예) "rgb" -> "observation.images.rgb"
    """
    return f"{OBS_IMAGES}.{cam_name}"


# Dataset keys
RGB_KEY: str = image_key("rgb")  
DEPTH_KEY: str = image_key("depth")
STATE_KEY: str = OBS_STATE 
ACTION_KEY: str = ACTION 

__all__ = ["image_key", "RGB_KEY", "DEPTH_KEY", "STATE_KEY", "ACTION_KEY"]
