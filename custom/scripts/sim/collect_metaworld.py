"""Collect canonical(EE-pose 10D) episodes in Meta-World with the built-in expert.

port_droid pattern (부록 D.1): no Robot, no processor pipeline — just
`LeRobotDataset.create` -> `add_frame` -> `save_episode`. The gym Env IS the
observation source and metaworld's scripted expert IS the action source, so no
teleoperator either. This script and the Phase 5 rollout share
`envs/metaworld/canonical.py`, which is what makes train == inference hold.

Why collect in-env instead of converting `lerobot/metaworld_mt50`: the per-step
dynamics must match what the policy faces at rollout. mt50 has EE moves up to
~16mm/step while this env tops out near ~8.6mm/step, and that gap was traced to
grasp-phase failures.

Only SUCCESSFUL expert episodes are saved. A fraction of episodes gets Gaussian
noise injected into the expert's xyz action, so the dataset contains
"slightly-off -> corrected" recovery states that clean demos never show.

Gripper: obs[3] is binarized against --gripper-threshold inside
state4_to_canonical10(). The SAME value must be passed to the Phase 5 rollout or
train != inference (retargeting.md 4절).

Example:
    MUJOCO_GL=egl python custom/envs/metaworld/collect.py \
        --env-task pick-place-v3 \
        --output-root ~/datasets/metaworld_canonical/pick_place_v3_bin \
        --repo-id local/metaworld_canonical_pick_place \
        --n-episodes 300
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.metaworld import MetaworldEnv

from lerobot_canonical import keys
from lerobot_canonical.schemas import canonical_ee10 as sch
from custom.envs.metaworld.canonical import (
    PICK_PLACE_GRIPPER_THRESHOLD,
    render_frame,
    state4_to_canonical10,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-task", type=str, default="pick-place-v3")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--repo-id", type=str, default="local/metaworld_canonical_pick_place")
    p.add_argument("--n-episodes", type=int, default=300)
    p.add_argument("--image-size", type=int, default=240)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--fps", type=int, default=80, help="fps label; must match the training config.")
    p.add_argument(
        "--gripper-threshold", type=float, default=PICK_PLACE_GRIPPER_THRESHOLD,
        help="obs[3] >= thresh -> open(1). TASK-DEPENDENT; pass the same value at rollout.",
    )
    p.add_argument(
        "--seed-base", type=int, default=100,
        help="First reset seed. Keep >= 100 so eval seeds 0..9 stay held out.",
    )
    p.add_argument(
        "--noise-fraction", type=float, default=0.3,
        help="Fraction of episodes with xyz action noise (recovery data).",
    )
    p.add_argument("--noise-std", type=float, default=0.15, help="Gaussian std in action units.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def build_features(image_size: int) -> dict[str, dict[str, Any]]:
    """Feature dict for the canonical 10D dataset (no depth: metaworld has none).

    The converter — not ext_core — owns this: it declares what THIS dataset holds.
    Policies derive their own features from it later (`dataset_to_policy_features`).

    Derive every key/shape/axis name from ext_core (`keys`, `sch`); do not hardcode
    "observation.state", 10, or the axis list.

    Returns:
        {RGB_KEY: {dtype "image", shape (H,W,3), names [...]},
         STATE_KEY / ACTION_KEY: {dtype "float32", shape (STATE_DIM,), names {"axes": [...]}}}
    """
    return {keys.RGB_KEY: dict(dtype="image", shape=(image_size,image_size,3), names=["height","width","channels"]),
            keys.STATE_KEY: dict(dtype="float32", shape=(sch.STATE_DIM,), names={"axes": list(sch.STATE_AXES)}),
            keys.ACTION_KEY: dict(dtype="float32", shape=(sch.ACTION_DIM,), names={"axes": list(sch.ACTION_AXES)})}


def collect_episode(
    env: Any,
    expert: Any,
    seed: int,
    max_steps: int,
    image_size: int,
    noise_std: float | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[np.ndarray], bool]:
    """Roll the scripted expert once and record the frames.

    Args:
        seed: scene seed (object/goal placement). Applied via env.seed(), NOT via
            reset(seed=...) — Meta-World discards the latter by design; see the
            seeded_rand_vec note in main(). Requires env.seeded_rand_vec = True.
        noise_std: if not None, add N(0, noise_std) to the expert's xyz action
            (recovery data). The gripper channel is never noised.
        rng: seeded generator, so a run is reproducible.

    Returns:
        (states4, images, success)
          states4 : (N, 4) float32, each = env obs[:4] = [ee_x, ee_y, ee_z, gripper]
          images  : list of N rendered frames
          success : True if info["success"] was ever 1
        The reset frame is included, so N == steps + 1.
    """
    env.seed(seed)
    obs, _ = env.reset()
    states4 = [obs[:4]]
    images = [render_frame(env, image_size)]
    success = False

    for _ in range(max_steps):
        action = expert.get_action(obs)
        if noise_std is not None:
            action[:3] += rng.normal(0, noise_std, 3)
        action = np.clip(action, -1, 1)
        obs, _, term, trunc, info = env.step(action)
        states4.append(obs[:4])  
        images.append(render_frame(env, image_size))
        if int(info.get("success", 0)) == 1:
            success = True; break 
        if term or trunc:
            break
    return np.stack(states4).astype(np.float32), images, success

def to_canonical_and_actions(
    states4: np.ndarray, gripper_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """(N,4) env states -> (canonical (N,10), actions (N,10)).

    action[t] = state[t+1] (absolute target pose). The last frame has no successor,
    so it repeats itself.

    Returns:
        (canonical, actions), both (N, sch.STATE_DIM) float32.
    """
    canonical = state4_to_canonical10(states4, gripper_threshold)          # Step 2 함수 재사용
    actions   = np.concatenate((canonical[1:], canonical[-1:]), axis=0)    # 한 칸 당기기
    return canonical, actions

def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    # The wrapper applies the corner2 camera tweak and owns the expert; we drive the
    # inner env directly because the expert needs the raw 39D obs.
    wrapper = MetaworldEnv(task=args.env_task, obs_type="pixels_agent_pos", camera_name="corner2")
    env = wrapper._env
    expert = wrapper.expert_policy
    task_text = wrapper.task_description

    # Required for --seed-base to do anything at all. Meta-World picks the object/goal
    # placement in one of three ways (sawyer_xyz_env.py:697):
    #     _freeze_rand_vec=True  -> reuse last vec        (no randomization)
    #     seeded_rand_vec=True   -> self.np_random        (env.seed(n) controls it)  <- we want this
    #     else (DEFAULT)         -> global np.random      (nothing we pass controls it)
    # The lerobot wrapper sets _freeze_rand_vec=False but leaves seeded_rand_vec=False
    # (envs/metaworld.py:163), landing us in the global-np.random branch. There,
    # `env.reset(seed=n)` is silently discarded — Meta-World's own reset() docstring says
    # "seed: The seed to use. Ignored, use `seed()` instead." (sawyer_xyz_env.py:670) — and
    # even env.seed(n) is inert, because self.np_random is not the RNG being read.
    # Consequence when unset: every run draws different scenes, so a dataset can never be
    # reproduced and two policies can never be evaluated on the same 10 scenes.
    # Verified: with this flag + env.seed(n), repeated runs give identical scenes; without
    # it, they differ on the very first episode.
    env.seeded_rand_vec = True

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=build_features(args.image_size),
        root=output_root,
        use_videos=False,
    )

    rng = np.random.default_rng(args.seed_base)
    delta_norms: list[float] = []
    saved = attempts = failed = total_frames = 0
    max_attempts = args.n_episodes * 2

    while saved < args.n_episodes and attempts < max_attempts:
        seed = args.seed_base + attempts
        attempts += 1
        noisy = (saved % 100) < int(args.noise_fraction * 100)

        states4, images, success = collect_episode(
            env, expert, seed, args.max_steps, args.image_size,
            args.noise_std if noisy else None, rng,
        )
        if not success:
            failed += 1
            print(f"[warn] seed {seed}: expert failed ({len(states4)} frames) — skipped", flush=True)
            continue

        canonical, actions = to_canonical_and_actions(states4, args.gripper_threshold)
        delta_norms.extend(np.linalg.norm(canonical[1:, :3] - canonical[:-1, :3], axis=-1).tolist())

        for state10, action10, img in zip(canonical, actions, images):
            dataset.add_frame({
                keys.RGB_KEY: img,
                keys.STATE_KEY: state10,
                keys.ACTION_KEY: action10,
                "task": task_text,
            })
            total_frames += 1
        dataset.save_episode()

        saved += 1
        if saved % 10 == 0 or saved == args.n_episodes:
            print(f"[info] saved {saved}/{args.n_episodes} ({total_frames} frames, "
                  f"noisy={noisy}, expert fails={failed})", flush=True)

    dataset.finalize()
    print(f"[ok] wrote {saved} episodes / {total_frames} frames -> {output_root} "
          f"(expert fails: {failed})")

    if delta_norms:
        norms = np.asarray(delta_norms, dtype=np.float32)
        print(
            "[scale] per-step |dxyz| meters: "
            f"mean={norms.mean():.5f} p50={np.percentile(norms, 50):.5f} "
            f"p95={np.percentile(norms, 95):.5f} max={norms.max():.5f}"
        )

if __name__ == "__main__":
    main()
