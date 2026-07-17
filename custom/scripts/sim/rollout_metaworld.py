#!/usr/bin/env python
"""umidiffusion 정책을 Meta-World 에서 rollout 시켜 성공률을 잰다.

    MUJOCO_GL=egl python custom/scripts/sim/rollout_metaworld.py \
        --checkpoint outputs/train/.../checkpoints/030000/pretrained_model --n-episodes 20

동기 구현이다. Meta-World 는 env.step() 을 부를 때만 시간이 흐르므로 추론 시간이 세계에
영향을 주지 않는다. 실기는 다르고(원본 UMI 는 robot_action_latency=0.1s 를 명시적으로
모델링한다) 그때 이 루프 위에 멀티스레딩을 얹게 되는데, ObservationHistoryBuffer 가 이미
분리돼 있어 generate/decode/env 만 갈리면 된다.

train == inference 는 수집과 같은 함수·같은 상수를 쓰는 것이 전부다:

    render_frame(wrapper._env, 240)         flip + resize
    state4_to_canonical10(obs[:4], 0.7)     PICK_PLACE_GRIPPER_THRESHOLD (데이터셋에 구워진 값)
    canonical10_to_env_action(..., 0.01)    ENV_XYZ_SCALE (env 상수)

어기면 안 되는 것이 넷 있다. 앞의 둘은 어겨도 에러가 안 나므로 특히 위험하다.

1. render_frame 에 wrapper 를 넘기지 말 것. wrapper 는 render() 와 _format_raw_obs() 양쪽에서
   이미 np.flip 하므로 이중 flip 이 되어 거꾸로 된 그림으로 추론하게 된다. 수집과 똑같이
   wrapper._env 를 넘긴다. wrapper.pixels 를 쓰는 것도 안 된다 — flip 은 맞지만 리사이즈
   구현이 달라진다.
2. decode_policy_action 을 빠뜨리지 말 것. 정책은 relative 를 뱉으므로 그대로 env 에 주면
   "지금 기준 +2cm" 가 "월드 좌표 2cm" 로 해석된다.
3. select_action() 을 쓰지 말 것. 정책의 큐 stack 은 predict_action_chunk 안, 즉 프로세서보다
   뒤에서 일어난다. 쓰면 relative step 이 (B,10) 을 받아 앵커를 만들 수 없다(ndim!=3 ValueError).
   자체 버퍼로 (B,T,10) 을 만들어 preprocessor 에 넘기고 diffusion.generate_actions() 를 직접 부른다.
4. register_third_party_plugins() 를 빠뜨리지 말 것. 등록이 import 시점에 일어나므로 없으면
   체크포인트 로드가 KeyError 로 죽는다. lerobot CLI 는 스스로 부르지만 이 스크립트는 아니다.

eval seed 0~9 가 홀드아웃이려면 env.seeded_rand_vec = True 가 필요하다. 안 켜면 Meta-World 가
물체·목표를 전역 np.random 으로 뽑아 seed 가 아무 일도 하지 않는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.envs.metaworld import MetaworldEnv
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins

# 등록은 import 시점에 일어난다. 없으면 체크포인트 로드가 KeyError 로 죽는다.
register_third_party_plugins()

from lerobot_canonical import keys  # noqa: E402
from lerobot_policy_umidiffusion import UmiDiffusionPolicy  # noqa: E402
from lerobot_policy_umidiffusion.runtime_buffer import (  # noqa: E402
    ObservationHistoryBuffer,
    build_model_input,
)
from lerobot_policy_umidiffusion.steps import decode_policy_action  # noqa: E402

from custom.envs.metaworld.canonical import (  # noqa: E402
    ENV_XYZ_SCALE,
    PICK_PLACE_GRIPPER_THRESHOLD,
    canonical10_to_env_action,
    render_frame,
    state4_to_canonical10,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="…/checkpoints/<step>/pretrained_model")
    p.add_argument("--env-task", type=str, default="pick-place-v3")
    p.add_argument("--n-episodes", type=int, default=20,
                   help="비교엔 20~30 권장 (10개는 ±15%%p 노이즈).")
    p.add_argument("--max-steps", type=int, default=200, help="수집의 max_steps 와 맞출 것.")
    p.add_argument("--consume-steps", type=int, default=0,
                   help="한 번 추론으로 실행할 스텝 수. 0 -> policy.config.n_action_steps. "
                        "낮추면 더 자주 재계획한다. 스윕 대상.")
    p.add_argument("--lookahead", type=int, default=0,
                   help="청크의 L 칸 앞 지점을 겨냥한다. 0 -> 기존 동작(검증됨, 90%%). "
                        "근거는 execution_target() 참조. 스윕 대상.")
    p.add_argument("--seed", type=int, default=0,
                   help="eval 시작 시드. 수집이 seed_base=100 이므로 0~9 는 홀드아웃.")
    p.add_argument("--image-size", type=int, default=240, help="학습 데이터셋 해상도와 반드시 일치.")
    p.add_argument("--gripper-threshold", type=float, default=PICK_PLACE_GRIPPER_THRESHOLD,
                   help="수집 때 데이터셋에 구워진 값과 같아야 한다.")
    p.add_argument("--xyz-scale", type=float, default=ENV_XYZ_SCALE,
                   help="env 상수. 데이터 통계로 잡지 말 것.")
    p.add_argument("--servo-gain", type=float, default=1.0,
                   help="resolved-rate P 게인. 1.0 -> Kp=1/dt=80/s (검증됨, 90%%). "
                        "0.1 -> expert 와 같은 8/s. canonical10_to_env_action 참조. 스윕 대상.")
    p.add_argument("--expert", action="store_true",
                   help="정책 대신 scripted expert 로 돌린다 (대조군). env 무죄 확인용.")
    p.add_argument("--gif-dir", type=Path, default=Path("tmp/real"))
    p.add_argument("--gif-episodes", type=int, default=1, help="앞 N개를 gif 로 저장.")
    p.add_argument("--torch-seed", type=int, default=42, help="diffusion 샘플링 재현용.")
    return p.parse_args()


def execution_target(chunk: np.ndarray, k: int, lookahead: int) -> np.ndarray:
    """청크의 k 번째를 실행할 때 실제로 겨냥할 canonical 타겟을 만든다.

    chunk[k+lookahead] 를 통째로 겨냥한다. lookahead=0 이면 chunk[k] 그대로.

    왜 멀리 겨냥하나 (2026-07-17 실측, 30k ckpt, pick-place):
      서보는 100 x (타겟 - 손) 이고 타겟이 6.75mm 앞이라, 모델 예측 오차가 그대로 방향 오차가 된다.
      실측 오차는 평균 5.23mm 라 arctan(5.23/6.75) = 37.8도까지 틀어진다. expert 는 같은 오차를
      갖고도 190mm 앞 최종 목표를 겨냥해 arctan(5.23/190) = 1.58도로 눌러버린다. 겨냥 거리가
      멀수록 같은 오차가 덜 증폭된다. 청크 뒤로 갈수록 예측 오차 자체는 커지지만(1.66 -> 8.19mm)
      누적 거리가 더 빨리 자라서 순이득이다.
      기준선(look=0)이 굴곡도 1.36, expert 가 1.04 다. look=4 는 3 seed 에서 1.11 까지 내려갔고
      성공 에피소드 스텝도 52/49 로 expert(53/47)와 같아졌다.

    그리퍼도 같이 앞당기는 이유 (실측으로 갈림):
      청크는 (위치, 그리퍼) 쌍의 궤적이므로 둘을 같은 인덱스에서 뽑아야 일관된다. xyz 만 앞당기고
      그리퍼를 k 에 두면 손은 포화 명령으로 전속력으로 가는데 그리퍼는 원래 시각표를 따라가
      동기가 깨진다. 3 seed 실측 (30k ckpt, pick-place, 굴곡도 / 성공):
        xyz 만    look=1: 1.52 3/3   look=2: 1.39 2/3   look=4: 1.28 1/3
        xyz+그리퍼 look=1: 1.40 3/3   look=2: 1.21 2/3   look=4: 1.11 2/3
    """
    if lookahead <= 0:
        return chunk[k]
    return chunk[min(k + lookahead, len(chunk) - 1)]


def canonical_obs(raw_obs: np.ndarray, env, image_size: int, gripper_threshold: float) -> dict:
    """raw 39D obs + 내부 env -> canonical 관측 하나. 수집과 같은 경로.

    env 는 반드시 wrapper._env 여야 한다. wrapper 를 넘기면 이중 flip 이 되어, 에러 없이
    거꾸로 된 그림으로 추론하게 된다.
    """
    return {
        keys.RGB_KEY: render_frame(env, image_size),                       # flip(+resize) — 수집과 동일
        OBS_STATE: state4_to_canonical10(raw_obs[:4], gripper_threshold),  # 수집과 같은 threshold
    }


@torch.no_grad()
def predict_chunk(policy, preprocessor, postprocessor, buffer: ObservationHistoryBuffer) -> np.ndarray:
    """버퍼 -> 절대 canonical 액션 청크 (n_action_steps, 10).

    select_action() 이 아니라 diffusion.generate_actions() 를 직접 부르고,
    decode_policy_action 으로 relative 를 절대로 되돌린다.
    """
    window = buffer.as_window()
    anchor_state = buffer.anchor_state()      # preprocessor 통과 전에 뽑는다 (통과 후엔 항등이 된다)

    device = get_device_from_parameters(policy)
    processed = preprocessor(build_model_input(window, device=device))

    # 카메라 축(OBS_IMAGES)을 쌓는다. 정책은 학습(modeling:201)과 추론(modeling:183) 두 경로에서
    # 각자 이걸 만드는데, select_action 을 건너뛰면 :183 의 스택도 함께 건너뛰므로 여기서 대신 한다.
    # config.image_features 를 순회하므로 카메라 수에 자동 적응한다 (UMI 의 rgb+depth 면 n=2).
    # 단 validate_features 가 모든 이미지의 shape 일치를 요구한다 — torch.stack 의 제약이다.
    if policy.config.image_features:
        processed = dict(processed)
        processed[OBS_IMAGES] = torch.stack(
            [processed[k] for k in policy.config.image_features], dim=-4
        )

    # generate_actions 는 horizon(16) 을 만든 뒤 이미 actions[:, 1:9] 로 잘라서 준다. 액션 윈도우가
    # 관측 윈도우 시작(t=-1)에 정렬돼 있어 첫 1개가 과거이기 때문이다. 반환은 (1, 8, 10) 이다.
    chunk = policy.diffusion.generate_actions(processed)        # (1, 8, 10) relative
    chunk = postprocessor(chunk)                                # ACTION=IDENTITY 라 사실상 no-op
    chunk = torch.as_tensor(chunk).detach().cpu().numpy().astype(np.float32)
    if chunk.ndim == 3:
        chunk = chunk[0]                                        # (8, 10)

    return decode_policy_action(
        torch.from_numpy(chunk),
        torch.from_numpy(anchor_state),
        action_pose_repr=policy.config.action_pose_repr,
    ).numpy()                                                   # (8, 10) 절대 canonical


def main() -> None:
    args = parse_args()
    args.gif_dir.mkdir(parents=True, exist_ok=True)
    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)

    wrapper = MetaworldEnv(task=args.env_task, obs_type="pixels_agent_pos", camera_name="corner2")
    env = wrapper._env                  # 내부 env. 수집과 동일 (wrapper 를 쓰면 이중 flip)
    env.seeded_rand_vec = True          # 없으면 seed 가 장면을 결정하지 않는다
    expert = wrapper.expert_policy

    policy = preprocessor = postprocessor = None
    if not args.expert:
        policy = UmiDiffusionPolicy.from_pretrained(args.checkpoint)
        policy.eval()
        # pretrained_path 를 주면 lerobot 이 저장된 파이프라인을 그대로 복원한다
        # (dataset_stats 는 그 경로에서 쓰이지 않는다 — factory.py 의 pretrained_path 분기).
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config, pretrained_path=str(args.checkpoint)
        )
        consume_steps = args.consume_steps or int(policy.config.n_action_steps)
        print(f"[info] ckpt={args.checkpoint}")
        print(f"[info] n_obs_steps={policy.config.n_obs_steps} horizon={policy.config.horizon} "
              f"consume_steps={consume_steps} action_pose_repr={policy.config.action_pose_repr}")
    else:
        consume_steps = 1
        print("[info] scripted expert (대조군)")
    print(f"[info] task={args.env_task} seeds={args.seed}~{args.seed + args.n_episodes - 1} "
          f"gripper_threshold={args.gripper_threshold} xyz_scale={args.xyz_scale}")

    successes: list[bool] = []
    for episode in range(args.n_episodes):
        env.seed(args.seed + episode)   # reset(seed=) 는 Meta-World 가 무시한다
        raw_obs, _ = env.reset()

        buffer = ObservationHistoryBuffer(n_obs_steps=int(policy.config.n_obs_steps) if policy else 1,
                                          include_depth=False)
        buffer.append(canonical_obs(raw_obs, env, args.image_size, args.gripper_threshold))
        frames = [render_frame(env, args.image_size)]
        success = False
        steps = 0

        while steps < args.max_steps and not success:
            if args.expert:
                chunk, n_exec = None, 1                          # expert 는 매 스텝 직접 계산
            else:
                # 자르지 않고 청크 전체(8개)를 들고 있는다 — lookahead 가 consume_steps 너머를
                # 겨냥할 수 있어야 한다. 실행 개수만 consume_steps 로 제한한다.
                chunk = predict_chunk(policy, preprocessor, postprocessor, buffer)
                n_exec = min(consume_steps, len(chunk))

            for k in range(n_exec):
                if args.expert:
                    env_action = np.clip(expert.get_action(raw_obs), -1, 1)
                else:
                    env_action = canonical10_to_env_action(
                        execution_target(chunk, k, args.lookahead),
                        raw_obs[:3], xyz_scale=args.xyz_scale, servo_gain=args.servo_gain,
                    )
                raw_obs, _, terminated, truncated, info = env.step(env_action)
                steps += 1
                buffer.append(
                    canonical_obs(raw_obs, env, args.image_size, args.gripper_threshold)
                )
                frames.append(render_frame(env, args.image_size))
                if int(info.get("success", 0)) == 1:
                    success = True
                if success or terminated or truncated or steps >= args.max_steps:
                    break

        successes.append(success)
        print(f"[episode {episode:2d}] seed={args.seed + episode:3d} success={success} steps={steps}")

        if episode < args.gif_episodes:
            tag = "expert" if args.expert else "policy"
            path = args.gif_dir / f"rollout_{args.env_task}_{tag}_ep{episode}.gif"
            imageio.mimsave(path, frames, fps=20)
            print(f"            gif -> {path}")

    n = len(successes)
    print(f"\n[ok] success: {sum(successes)}/{n} = {sum(successes) / n:.0%}")
    if n < 20:
        print("     [warn] 20개 미만은 노이즈가 크다 (구 보고서: 10개는 ±15%p). 비교엔 20~30 사용")


if __name__ == "__main__":
    main()
