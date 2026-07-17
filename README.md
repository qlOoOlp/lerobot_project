# lerobot_share

A custom robot-learning stack built on [HuggingFace `lerobot`](https://github.com/huggingface/lerobot).

Policies, environments, and robots are separated as plugins, and **`lerobot` itself is never patched**.
Everything is expressed in one canonical schema — a 10-D end-effector pose
`[x, y, z, rot6d(6), gripper]` — so the same policy trains on Meta-World and, later, on UMI data.

The policy (`umidiffusion`) is a Diffusion Policy that consumes and predicts poses **relative to the
current end-effector pose**, which makes it independent of any dataset's world frame.

---

## 1. Environment setup

**Requirements**: Linux, conda, an NVIDIA GPU. The commands below target an RTX 5090 (Blackwell);
for other GPUs only the torch index URL changes.

### 1.1 Clone this repository

```bash
git clone <this-repo-url> lerobot_share
cd lerobot_share
```

### 1.2 Clone lerobot and pin it to v0.4.4

`lerobot` is not vendored here (it is git-ignored). Fetch it yourself:

```bash
git clone https://github.com/huggingface/lerobot.git
git -C lerobot checkout v0.4.4          # commit 8fff0fde
```

The pin matters: this project is verified against v0.4.4 and never modifies it.
Custom code attaches through lerobot's plugin conventions, so no patch is required.
You can confirm this at any time with `git -C lerobot diff` — it should stay empty.

### 1.3 Create the conda environment

```bash
conda create -n lerobot_hong2 python=3.10 -y
conda activate lerobot_hong2
```

### 1.4 Install torch

lerobot requires `torch<2.11`, and Blackwell GPUs require `cu128+`. This satisfies both:

```bash
pip install "torch==2.10.*" "torchvision==0.25.*" --index-url https://download.pytorch.org/whl/cu128
```

Do not use a bare `pip install torch` — it will not respect the version cap or your CUDA version.
For a different GPU, pick an `--index-url` matching your CUDA within the `torch<2.11` range.

### 1.5 Install lerobot and the custom packages

Install in dependency order. `--no-deps` prevents pip from replacing the torch build you just installed.

```bash
pip install -e lerobot
pip install -e custom/utils/lerobot_canonical --no-deps
pip install -e custom/policies/umidiffusion/lerobot_policy_umidiffusion --no-deps
```

All installs are editable, so code changes take effect without reinstalling. Reinstall only when a
`pyproject.toml` changes or a package directory moves — editable installs record absolute paths.

### 1.6 Install the Meta-World simulator

```bash
pip install metaworld==3.0.0
```

### 1.7 Verify

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.10.x+cu128 True
python -c "import lerobot; print(lerobot.__version__)"                          # 0.4.4

# Plugin discovery — run this from outside the repo (e.g. cd /tmp) and the policy must still appear.
lerobot-train --help | grep -A2 "policy.type"                                   # lists "umidiffusion"
```

---

## 2. Project structure

`custom/` mirrors the top level of `lerobot/` itself (`policies/`, `envs/`, `scripts/`, `utils/`),
so there is no new layout to learn.

```
lerobot_share/
├── lerobot/                        HF lerobot v0.4.4 — cloned separately, git-ignored
├── custom/
│   ├── utils/lerobot_canonical/                  Installable library: the shared vocabulary
│   │   └── src/lerobot_canonical/
│   │       ├── keys.py                           Dataset keys, derived from lerobot constants
│   │       └── schemas/
│   │           ├── canonical_ee10.py             The 10-D EE-pose representation: dims and axes
│   │           └── canonical_ee10_se3.py         Its codec: rot6d <-> R, pose9d <-> transform
│   ├── policies/umidiffusion/lerobot_policy_umidiffusion/    Installable plugin: the policy
│   │   └── src/lerobot_policy_umidiffusion/
│   │       ├── configuration_umidiffusion.py     UmiDiffusionConfig(PreTrainedConfig)
│   │       ├── modeling_umidiffusion.py          UmiDiffusionPolicy(PreTrainedPolicy)
│   │       ├── steps.py                          Runtime anchor-relative transforms
│   │       ├── processor_umidiffusion.py         Pre/post pipeline assembly
│   │       └── runtime_buffer.py                 Observation history for inference
│   ├── envs/metaworld/canonical.py               Meta-World adapter, shared by collection and rollout
│   └── scripts/
│       ├── data_processing/raw_inspect.py        Raw-data inspector
│       └── sim/
│           ├── collect_metaworld.py              Data collection
│           └── rollout_metaworld.py              Inference / evaluation
├── outputs/                        Run artifacts — git-ignored
└── tmp/real/                       Verification artifacts (gifs, images) — git-ignored
```

The two installable packages play different roles:

- **`lerobot_canonical`** is a library. Policies, environments, and scripts agree on it without
  knowing about each other. Its name deliberately avoids lerobot's auto-discovery prefixes, because
  it is not a plugin.
- **`lerobot_policy_umidiffusion`** is a plugin. The `lerobot_policy_` prefix lets lerobot import it
  automatically, which is why `--policy.type=umidiffusion` works without patching lerobot.

`custom/envs/metaworld/canonical.py` stays a plain module: only scripts import it, and lerobot
already ships a Meta-World environment.

---

## 3. Collecting Meta-World data

Meta-World's scripted expert drives the environment; each frame is converted to the canonical schema
and written as a `LeRobotDataset`.

```bash
MUJOCO_GL=egl python custom/scripts/sim/collect_metaworld.py \
    --output-root ~/datasets/metaworld_canonical/pick_place_v4 \
    --n-episodes 300
```

`MUJOCO_GL=egl` selects headless rendering. Only successful expert episodes are stored, and runs are
reproducible — the same command produces the same dataset.

### 3.1 Options

Defaults are the verified configuration; the command above overrides only the output path and episode
count.

| Option | Default | Notes |
|---|---|---|
| `--output-root` | required | Where the dataset is written. |
| `--n-episodes N` | 300 | Successful episodes to collect. Failed expert attempts are retried, not stored. |
| `--overwrite` | off | Replace an existing output directory instead of aborting. |
| `--env-task` | `pick-place-v3` | Meta-World task id. |
| `--repo-id` | `local/metaworld_canonical_pick_place` | Dataset id recorded in the dataset. |
| `--max-steps N` | 200 | Episode cap. Match this at rollout. |

### 3.2 Options that end up baked into the dataset

Changing any of these means re-collecting, and rollout must be given the same values or the policy
acts on a world it was not trained on — silently, with no error.

| Option | Default | Notes |
|---|---|---|
| `--image-size N` | 240 | Image resolution. Training reads it from the dataset. |
| `--gripper-threshold F` | 0.7 | `obs[3] >= F` counts as open. Task-dependent — measure it per task. |
| `--fps N` | 80 | Label only; must match Meta-World's actual rate (`frame_skip 5 x 0.0025s`). |

### 3.3 Seeding and noise

| Option | Default | Notes |
|---|---|---|
| `--seed-base N` | 100 | First reset seed. Keep at 100 or above so evaluation seeds stay held out. |
| `--noise-fraction F` | 0.3 | Fraction of episodes with Gaussian noise added to the expert's action. |
| `--noise-std F` | 0.15 | Noise magnitude, in action units. |

The noise injection is deliberate. It produces "drifted, then recovered" states that clean
demonstrations never contain, which is what the policy needs in order to recover at rollout.

The seed base matters more than it looks. Meta-World ignores `reset(seed=)` by design and picks
object placement from the global RNG unless `seeded_rand_vec` is set, which the collection script
does. Without it, `--seed-base` would do nothing and the evaluation seeds would not be held out.

Inspect a dataset afterwards:

```bash
python custom/scripts/data_processing/raw_inspect.py \
    --raw-root ~/datasets/metaworld_canonical/pick_place_v4 \
    --format lerobot_dataset --target-fps 80
```

---

## 4. Training

```bash
lerobot-train \
    --policy.type=umidiffusion \
    --policy.push_to_hub=false \
    --policy.use_depth=false \
    --dataset.repo_id=local/x \
    --dataset.root=~/datasets/metaworld_canonical/pick_place_v4 \
    --steps=30000 --batch_size=64 --policy.device=cuda --num_workers=8 \
    --save_freq=5000 --wandb.enable=false \
    --output_dir=outputs/train/umidiffusion_pick_place_v4
```

`--policy.push_to_hub=false` is required; the default is `true`, which demands a `policy.repo_id`
and otherwise aborts. `--policy.use_depth=false` matches the Meta-World dataset, which has no depth.

Loss should fall quickly — roughly 0.96 to 0.03 within the first epoch.

To continue an interrupted or finished run, resume from a checkpoint. Passing a new `--steps`
rebuilds the learning-rate schedule for the new total:

```bash
python -u -m lerobot.scripts.lerobot_train \
    --config_path=outputs/train/umidiffusion_pick_place_v4/checkpoints/010000/pretrained_model/train_config.json \
    --resume=true --steps=30000
```

---

## 5. Inference

Roll the trained policy out in Meta-World and measure its success rate.

```bash
MUJOCO_GL=egl python custom/scripts/sim/rollout_metaworld.py \
    --checkpoint outputs/train/umidiffusion_pick_place_v4/checkpoints/030000/pretrained_model \
    --n-episodes 20
```

`MUJOCO_GL=egl` selects headless rendering; without it the script looks for a display and fails.
No window opens — results are printed and written to gifs.

### 5.1 What to run

| Option | Default | Notes |
|---|---|---|
| `--checkpoint` | required | Path to a `pretrained_model` directory. Use `/dev/null` with `--expert`. |
| `--n-episodes N` | 20 | Episodes to run, using seeds `--seed` through `--seed + N - 1`. |
| `--seed N` | 0 | First evaluation seed. Collection starts at 100, so 0–99 are held out. |
| `--expert` | off | Run the scripted expert instead of the policy. See 5.3. |

Use 20–30 episodes for any comparison. Ten episodes carry roughly ±15 percentage points of noise,
which is wider than most differences worth measuring.

### 5.2 Saving gifs

The command above already writes one gif, because `--gif-episodes` defaults to 1.

| Option | Default | Notes |
|---|---|---|
| `--gif-episodes N` | 1 | Save the first N episodes. `0` disables saving. |
| `--gif-dir PATH` | `tmp/real` | Where to write them. |

Filenames are `rollout_<task>_<policy\|expert>_ep<N>.gif` and do not encode any settings, so a second
run overwrites the first. When comparing configurations, give each run its own `--gif-dir`.

Gif encoding costs more than the rollout itself — roughly 30 s per episode against a few seconds of
simulation. Pass `--gif-episodes 0` when you only need the success rate.

### 5.3 The expert control group

The scripted expert drives the same environment through the same loop, so it isolates the policy from
everything else. It should succeed on every episode; if it does not, the problem is the environment
or the rollout loop rather than the policy.

```bash
MUJOCO_GL=egl python custom/scripts/sim/rollout_metaworld.py \
    --checkpoint /dev/null --expert --n-episodes 10
```

Note that the expert bypasses the canonical-to-action conversion and issues environment actions
directly. It therefore validates the environment, not the action mapping.

### 5.4 Options that must match collection

These are baked into the dataset. A mismatch breaks the train/inference contract silently — no error,
just a policy acting on a world it was not trained on.

| Option | Default | Notes |
|---|---|---|
| `--env-task` | `pick-place-v3` | Must match the collected task. |
| `--image-size N` | 240 | Must match the dataset resolution. |
| `--gripper-threshold F` | 0.7 | Task-dependent, and baked into the dataset at collection. |
| `--max-steps N` | 200 | Match collection so episode lengths are comparable. |

### 5.5 Controller options

These convert the policy's canonical pose targets into Meta-World's 4-D action. They affect inference
only; the policy is untouched. Defaults are the verified configuration (90% on seeds 0–19 at 30k).

| Option | Default | Notes |
|---|---|---|
| `--xyz-scale F` | 0.01 | The environment's `action_scale`. A constant, not something to fit to data. |
| `--servo-gain F` | 1.0 | Resolved-rate P gain. 1.0 means Kp = 1/dt: null the error in one step. |
| `--consume-steps N` | 0 | Actions executed per inference. 0 uses the policy's `n_action_steps` (8). |
| `--lookahead N` | 0 | Aim N chunk entries ahead instead of at the next one. |
| `--torch-seed N` | 42 | Diffusion sampling is stochastic; fix this for reproducible runs. |

Lowering `--consume-steps` re-plans more often, which sounds safer but is not: the gripper needs
several consecutive closing commands to actually grasp, and truncating the chunk drops them. Measured
at 30k: 8 gives 90%, 4 gives 5%.

`--lookahead` reduces path wander but removes the controller's ability to decelerate, because the aim
point stays a fixed distance ahead however close the target is. Measured at 30k over 20 episodes:
0 and 1 give 90%, 2 gives 60%, 4 gives 55%.

