#!/usr/bin/env python
"""umidiffusion 정책을 Meta-World 에서 closed-loop rollout 시켜 성공률을 잰다. (Phase 5)

═══════════════════════════════════════════════════════════════════════════════
■ 실행
    MUJOCO_GL=egl python custom/scripts/sim/rollout_metaworld.py \
        --checkpoint outputs/train/umidiffusion_pick_place_v4/checkpoints/020000/pretrained_model \
        --n-episodes 20

■ ★ 동기(synchronous) 구현이다 — sim 에선 그게 정답
    metaworld 는 우리가 env.step() 을 부를 때만 시간이 흐른다. 정책이 100ms 를 생각해도
    물체는 안 움직인다 => **지연 개념이 없다**.
    실기(Phase 9-11)는 다르다 — 원본 UMI 는 robot_action_latency=0.1s 를 명시적으로
    모델링한다(eval_robots_config.yaml:8). 그때 멀티스레딩으로 비동기를 얹는다.
    지금 구조는 그걸 막지 않는다: ObservationHistoryBuffer 가 이미 분리돼 있고
    (정책 패키지 안, Phase 7 의 runtime_buffer 와 같은 물건),
    generate → decode → env 만 비동기로 갈리면 된다.

■ ★★ 이 스크립트가 피해야 하는 지뢰 4개 (전부 refactoring.md 에 기록됨)

  1. select_action() 을 쓰면 안 된다  [시끄러운 실패]
     정책의 _queues stack 은 predict_action_chunk **안**에서 = 프로세서보다 **뒤**에 일어난다.
     쓰면 우리 relative step 이 (B,10) 을 받아 앵커를 못 만든다 -> ndim!=3 ValueError.
     => 자체 버퍼로 (B,T,10) 을 만들어 preprocessor 에 넘기고
        policy.diffusion.generate_actions() 를 직접 부른다.

  2. render_frame 에 wrapper 를 넘기면 안 된다  [★조용한 실패]
     wrapper 는 render()(:149)·_format_raw_obs()(:172) 양쪽에서 이미 np.flip 한다.
     wrapper 를 넘기면 **이중 flip = 거꾸로 된 그림으로 추론**. 에러가 안 난다.
     => 수집과 **똑같이** wrapper._env (내부 env) 를 넘긴다.
        실측: inner.render()=(480,480,3) 거꾸로 / wrapper.pixels=(480,480,3) 이미 flip 됨.
     ⚠ wrapper 의 pixels 를 쓰는 것도 안 된다 — flip 은 맞지만 **리사이즈 구현이 다르다**
        (우리 수집은 PIL, lerobot_hong 의 옛 rollout 은 torch bilinear+antialias).

  3. register_third_party_plugins() 를 빠뜨리면 안 된다  [시끄러운 실패]
     등록은 import 시점에 일어난다. 안 하면 PreTrainedConfig.from_pretrained 가
     KeyError: 'umidiffusion' 로 죽는다. lerobot-train/eval 은 스스로 부르지만 이 스크립트는 아니다.

  4. decode_policy_action 을 빠뜨리면 안 된다  [★조용한 실패]
     정책은 **relative 를 뱉는다**. 그대로 env 에 주면 "지금 기준 +2cm" 를
     "월드 좌표 2cm 지점" 으로 오해 -> 완전히 틀린 명령. 에러가 안 난다.

■ train == inference — 수집과 **같은 함수·같은 상수**
    render_frame(wrapper._env, 240)              수집과 동일 (flip+resize 를 같은 코드로)
    state4_to_canonical10(obs[:4], 0.7)          PICK_PLACE_GRIPPER_THRESHOLD — 데이터셋에 bake 된 값
    canonical10_to_env_action(..., 0.01)         ENV_XYZ_SCALE — env 상수
                                                 (구 보고서의 "실측 ~9mm/unit" 과 일치.
                                                  0.004 는 상시 포화, 0.014 는 과대였다)
    camera=corner2                               FLIP_CAMERAS 가드가 env.camera_name 으로 확인

■ ★ 시딩 — eval seed 0~9 홀드아웃이 성립하려면
    env.seeded_rand_vec = True 를 켜야 한다. 안 켜면 Meta-World 가 물체·목표를 전역
    np.random 으로 뽑아 seed 가 **아무 일도 안 한다**(sawyer_xyz_env.py:697).
    수집(collect_metaworld.py)이 seed_base=100 을 쓰므로 0~9 는 홀드아웃이다 —
    단 이 플래그가 전제. 상세: information.md §1.3 "시딩 함정"
═══════════════════════════════════════════════════════════════════════════════
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

# ★ 지뢰 3: 등록은 import 시점에 일어난다. 이게 없으면 체크포인트 로드가 KeyError 로 죽는다.
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
                   help="구 보고서: 10개는 ±15%%p 노이즈 -> 비교엔 20~30 권장.")
    p.add_argument("--max-steps", type=int, default=200, help="수집의 max_steps 와 맞출 것.")
    p.add_argument("--consume-steps", type=int, default=0,
                   help="한 번 추론으로 실행할 스텝 수. 0 -> policy.config.n_action_steps. "
                        "구 보고서에서 reach 가 4 로 낮춰서야 80%% 달성했다 -> 스윕 대상.")
    p.add_argument("--seed", type=int, default=0,
                   help="eval 시작 시드. 수집이 seed_base=100 이므로 0~9 는 홀드아웃.")
    p.add_argument("--image-size", type=int, default=240, help="학습 데이터셋 해상도와 반드시 일치.")
    p.add_argument("--gripper-threshold", type=float, default=PICK_PLACE_GRIPPER_THRESHOLD,
                   help="수집 때 데이터셋에 bake 된 값과 **같아야** 한다.")
    p.add_argument("--xyz-scale", type=float, default=ENV_XYZ_SCALE,
                   help="env 상수. 데이터 통계로 잡지 말 것 (retargeting.md 5절).")
    p.add_argument("--expert", action="store_true",
                   help="정책 대신 scripted expert 로 돌린다 (대조군). env 무죄 확인용.")
    p.add_argument("--gif-dir", type=Path, default=Path("tmp/real"))
    p.add_argument("--gif-episodes", type=int, default=1, help="앞 N개를 gif 로 저장.")
    p.add_argument("--torch-seed", type=int, default=42, help="diffusion 샘플링 재현용.")
    return p.parse_args()


def canonical_obs(raw_obs: np.ndarray, env, image_size: int, gripper_threshold: float) -> dict:
    """raw 39D obs + **내부** env -> canonical 관측 하나. 수집(collect_metaworld.py)과 같은 경로.

    ★ `env` 는 반드시 `wrapper._env` — 지뢰 2. wrapper 를 넘기면 이중 flip 이 되고
      **에러 없이** 거꾸로 된 그림으로 추론한다.
    """
    return {
        keys.RGB_KEY: render_frame(env, image_size),                       # flip(+resize) — 수집과 동일
        OBS_STATE: state4_to_canonical10(raw_obs[:4], gripper_threshold),  # 수집과 같은 threshold
    }


@torch.no_grad()
def predict_chunk(policy, preprocessor, postprocessor, buffer: ObservationHistoryBuffer) -> np.ndarray:
    """버퍼 -> 절대 canonical 액션 청크 (H, 10).

    ★ 지뢰 1: select_action() 이 아니라 diffusion.generate_actions() 를 직접 부른다.
    ★ 지뢰 4: decode_policy_action 으로 relative -> 절대 를 반드시 거친다.
    """
    window = buffer.as_window()
    anchor_state = buffer.anchor_state()          # ★ preprocessor 통과 '전'에 뽑는다 (통과 후엔 항등)

    device = get_device_from_parameters(policy)
    processed = preprocessor(build_model_input(window, device=device))

    # ★ 카메라 축(OBS_IMAGES)을 여기서 쌓는다.
    #   정책은 두 경로에서 **각자** 이걸 만든다:
    #       forward()       (우리 modeling_umidiffusion.py:201)  <- 학습 경로. 우리 학습이 그대로 쓴다 (건드리지 않음)
    #       select_action   (우리 modeling_umidiffusion.py:183)  <- 추론 경로. ★ 우리가 우회하는 건 '이쪽'
    #   select_action 을 건너뛰면 :183 의 스택도 함께 건너뛰므로 우리가 대신 한다.
    #   (forward 를 우회하는 게 아니다 — forward 는 애초에 학습 경로다.)
    #   config.image_features 를 순회하므로 카메라 수에 자동 적응한다: UMI 의 rgb+depth -> n=2.
    #   ⚠ 단 validate_features 가 **모든 이미지의 shape 일치**를 요구한다 (torch.stack 의 제약).
    if policy.config.image_features:
        processed = dict(processed)
        processed[OBS_IMAGES] = torch.stack(
            [processed[k] for k in policy.config.image_features], dim=-4
        )

    # ★ generate_actions 는 horizon(16) 을 만든 뒤 **이미 잘라서** 준다:
    #       start = n_obs_steps - 1 = 1,   end = start + n_action_steps = 9   ->  actions[:, 1:9]
    #   action_delta_indices 가 [-1, 0, 1, ... 14] 라 액션 윈도우가 관측 윈도우 시작(t=-1)에
    #   정렬돼 있고, 그 첫 1개는 과거라 버린다. 즉 반환은 (1, n_action_steps=8, 10) — 16 이 아니다.
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
    env = wrapper._env                       # ★ 지뢰 2: 내부 env. 수집과 동일
    env.seeded_rand_vec = True               # ★ 없으면 seed 가 아무 일도 안 한다 (information.md §1.3)
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
        env.seed(args.seed + episode)        # ★ reset(seed=) 는 Meta-World 가 무시한다
        raw_obs, _ = env.reset()

        buffer = ObservationHistoryBuffer(n_obs_steps=int(policy.config.n_obs_steps) if policy else 1,
                                          include_depth=False)
        buffer.append(canonical_obs(raw_obs, env, args.image_size, args.gripper_threshold))
        frames = [render_frame(env, args.image_size)]
        success = False
        steps = 0

        while steps < args.max_steps and not success:
            if args.expert:
                chunk = [None]                                   # expert 는 매 스텝 직접 계산
            else:
                chunk = predict_chunk(policy, preprocessor, postprocessor, buffer)[:consume_steps]

            for target10 in chunk:
                if args.expert:
                    env_action = np.clip(expert.get_action(raw_obs), -1, 1)
                else:
                    env_action = canonical10_to_env_action(
                        target10, raw_obs[:3], xyz_scale=args.xyz_scale
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
            print(f"            gif -> {path}   ★ flip 방향은 육안으로만 확인 가능")

    n = len(successes)
    print(f"\n[ok] success: {sum(successes)}/{n} = {sum(successes) / n:.0%}")
    if n < 20:
        print("     ⚠ 20개 미만은 노이즈가 크다 (구 보고서: 10개는 ±15%p) — 비교엔 20~30 사용")


if __name__ == "__main__":
    main()
