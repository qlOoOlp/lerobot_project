"""추론용 관측 히스토리 버퍼 — 정책이 요구하는 (B, T, ...) 윈도우를 만든다.

학습은 DataLoader 가 delta_timestamps 로 윈도우를 잘라 (B, T, 10) 을 준다. 추론에는 그게 없다.
정책에도 큐(_queues)가 있지만 stack 이 predict_action_chunk 안, 즉 프로세서보다 뒤에서 일어난다:

    select_action(obs)              # obs = (B, 10). 프로세서는 이미 지나갔다.
      -> populate_queues(...)
      -> predict_action_chunk()
           -> torch.stack(queues)   # 여기서 (B, T, 10) 이 된다. 너무 늦다.

CanonicalPoseToRelativeObservationStep 은 앵커로 state[:, -1] 이 필요해 (B, T, 10) 을 프로세서
단계에서 받아야 한다. select_action 을 쓰면 (B, 10) 이 와서 ndim != 3 ValueError 로 죽는다.
그래서 rollout 이 자체 히스토리로 윈도우를 만들어 preprocessor 에 넘기고, select_action 대신
policy.diffusion.generate_actions() 를 직접 부른다.

이 파일이 정책 패키지 안에 있는 이유는 embodiment 를 모르기 때문이다. metaworld 든 UMI 든
franka 든 canonical 10D 관측을 T개 모아 정책 입력을 만드는 일은 같다. 아는 것은 정책의 계약
(n_obs_steps, canonical 키, STATE_DIM)뿐이라 정책과 운명을 같이한다.

지금은 동기 전용이다. sim 은 우리가 env.step() 을 부를 때만 시간이 흐르므로 정책이 얼마나 오래
생각하든 세상은 멈춰 있고 지연이라는 개념이 없다. 실기는 다르다 — 원본 UMI 는
robot_action_latency=0.1s 를 명시적으로 모델링한다(eval_robots_config.yaml). 그때 이 버퍼 위에
타임스탬프 기반 정렬과 멀티스레딩을 얹는다. TIMESTAMP_KEY 를 지금부터 들고 다니는 게 그 대비다.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch

from lerobot.utils.constants import OBS_STATE

from lerobot_canonical import keys
from lerobot_canonical.schemas import canonical_ee10 as sch

TIMESTAMP_KEY = "timestamp"


class ObservationHistoryBuffer:
    """canonical 관측을 n_obs_steps 개 들고 있다가 (T, ...) 윈도우로 내준다.

    에피소드 시작 직후엔 히스토리가 1개뿐이다. 그때 가장 오래된 프레임을 복제해 T개를 채운다.
    lerobot 의 정책 큐도 같은 규약이고("for the first steps, the observation is copied
    n_obs_steps times"), 학습 데이터의 첫 프레임 패딩과도 맞는다. 복제되는 건 과거뿐이라
    앵커(마지막 프레임)는 항상 진짜 지금이다.

    append 에서 값을 copy() 하는 이유는 env 가 관측 배열을 in-place 로 재사용하는 경우가 있기
    때문이다. 복사하지 않으면 히스토리 전체가 최신 값으로 덮여 앵커-relative 가 전부 0이 된다.
    """

    def __init__(self, n_obs_steps: int, include_depth: bool = False) -> None:
        if n_obs_steps <= 0:
            raise ValueError(f"`n_obs_steps` must be positive, got {n_obs_steps}.")
        self.n_obs_steps = int(n_obs_steps)
        self.include_depth = bool(include_depth)
        self._rgb: deque = deque(maxlen=self.n_obs_steps)
        self._depth: deque | None = deque(maxlen=self.n_obs_steps) if include_depth else None
        self._state: deque = deque(maxlen=self.n_obs_steps)
        self._timestamp: deque = deque(maxlen=self.n_obs_steps)

    def __len__(self) -> int:
        return len(self._state)

    @property
    def is_full(self) -> bool:
        return len(self) == self.n_obs_steps

    def clear(self) -> None:
        self._rgb.clear()
        self._state.clear()
        self._timestamp.clear()
        if self._depth is not None:
            self._depth.clear()

    def append(self, observation: dict[str, Any]) -> None:
        """canonical 관측 하나를 넣는다.

        Args:
            observation: {RGB_KEY: (H,W,3) uint8, OBS_STATE: (10,) float32,
                          [DEPTH_KEY], [timestamp]}
        """
        if keys.RGB_KEY not in observation:
            raise KeyError(f"Observation is missing `{keys.RGB_KEY}`.")
        if OBS_STATE not in observation:
            raise KeyError(f"Observation is missing `{OBS_STATE}`.")

        rgb = np.asarray(observation[keys.RGB_KEY])
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"`{keys.RGB_KEY}` must be (H, W, 3), got {rgb.shape}.")

        state = np.asarray(observation[OBS_STATE], dtype=np.float32)
        if state.shape != (sch.STATE_DIM,):
            raise ValueError(f"`{OBS_STATE}` must be ({sch.STATE_DIM},), got {state.shape}.")

        self._rgb.append(rgb.copy())      # copy: env 가 배열을 재사용할 수 있다
        self._state.append(state.copy())
        self._timestamp.append(float(observation.get(TIMESTAMP_KEY, 0.0)))

        if self.include_depth:
            if keys.DEPTH_KEY not in observation:
                raise KeyError(f"Observation is missing `{keys.DEPTH_KEY}` (include_depth=True).")
            depth = np.asarray(observation[keys.DEPTH_KEY])
            if depth.ndim != 3 or depth.shape[-1] != 3:
                raise ValueError(f"`{keys.DEPTH_KEY}` must be (H, W, 3), got {depth.shape}.")
            self._depth.append(depth.copy())

    def as_window(self) -> dict[str, np.ndarray]:
        """(T, ...) 윈도우. 부족하면 가장 오래된 것을 앞에 복제해 채운다."""
        if len(self) == 0:
            raise ValueError("Observation history is empty — append() at least once.")

        def pad(items: list) -> list:
            # 앞쪽(과거)에 복제한다. 뒤에 붙이면 앵커가 가짜가 되어 전부 망가진다.
            return [items[0]] * (self.n_obs_steps - len(items)) + items

        window = {
            keys.RGB_KEY: np.stack(pad(list(self._rgb)), axis=0),
            OBS_STATE: np.stack(pad(list(self._state)), axis=0),
            TIMESTAMP_KEY: np.asarray(pad(list(self._timestamp)), dtype=np.float64),
        }
        if self.include_depth:
            window[keys.DEPTH_KEY] = np.stack(pad(list(self._depth)), axis=0)
        return window

    def anchor_state(self) -> np.ndarray:
        """앵커 = 히스토리의 마지막 = 지금 내 자세. (10,) 절대 canonical.

        preprocessor 를 통과시키기 전에 뽑아야 한다. 통과 후엔 relative 로 바뀌어 마지막
        프레임이 항등(0,0,0, 1,0,0, 0,1,0)이 되므로 앵커로 쓸 수 없다. decode_policy_action 이
        이 값을 받아 정책의 relative 출력을 절대로 되돌린다.
        """
        if len(self) == 0:
            raise ValueError("Observation history is empty.")
        return np.asarray(self._state[-1], dtype=np.float32).copy()


def build_model_input(
    window: dict[str, np.ndarray], device: torch.device | str = "cpu"
) -> dict[str, torch.Tensor]:
    """윈도우(numpy) -> preprocessor 가 받는 텐서 dict.

    학습 배치와 같은 형태여야 한다. LeRobotDataset 은 이미지를 (3,H,W) float32 [0,1] 로, state 를
    (10,) float32 로 준다. 여기서는 (1,T,3,H,W) 와 (1,T,10) 을 만든다. 즉 uint8 [0,255] ->
    float [0,1] 과 (H,W,C) -> (C,H,W) 를 여기서 한다. 데이터셋이 해주던 일을 추론에선 우리가 한다.

    배치축도 여기서 붙인다. AddBatchDimensionProcessorStep 은 1D state / 3D 이미지일 때만 축을
    붙이는데(batch_processor.py:104-114) 우리 윈도우는 이미 (T,10)/(T,H,W,3) 이라 그 조건에
    안 걸린다. 여기서 (1,T,...) 로 만들어야 우리 step 이 ndim==3 을 본다.
    """
    state = torch.from_numpy(np.asarray(window[OBS_STATE], dtype=np.float32)).unsqueeze(0)
    out: dict[str, torch.Tensor] = {OBS_STATE: state.to(device)}

    for key in (keys.RGB_KEY, keys.DEPTH_KEY):
        if key not in window:
            continue
        image = np.asarray(window[key])
        tensor = torch.from_numpy(np.ascontiguousarray(image))          # (T,H,W,3) uint8
        tensor = tensor.permute(0, 3, 1, 2).float() / 255.0             # (T,3,H,W) [0,1]
        out[key] = tensor.unsqueeze(0).to(device)                       # (1,T,3,H,W)
    return out


__all__ = ["ObservationHistoryBuffer", "build_model_input", "TIMESTAMP_KEY"]
