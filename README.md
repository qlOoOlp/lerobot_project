# lerobot_project

A custom robot-learning stack built on [HuggingFace `lerobot`](https://github.com/huggingface/lerobot).

Policies, environments, and robots are separated as plugins, and **`lerobot` itself is never patched**.
Everything is expressed in one canonical schema — a 10-D end-effector pose
`[x, y, z, rot6d(6), gripper]` — so the same policy trains on Meta-World and, later, on UMI data.

The policy (`umidiffusion`) is a Diffusion Policy that consumes and predicts poses **relative to the
current end-effector pose**, which makes it independent of any dataset's world frame.

---

## 1. Environment setup

**Requirements**: Linux, conda, an NVIDIA GPU.

### 1.1 Clone this repository

```bash
git clone git@github.com:qlOoOlp/lerobot_project.git
cd lerobot_project
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
conda create -n lerobot_project python=3.10 -y
conda activate lerobot_project
```

### 1.4 Install torch

Install torch yourself, before lerobot, so that pip does not pick a build that mismatches your GPU.
lerobot v0.4.4 requires:

```
torch >= 2.2.1, < 2.11.0
torchvision >= 0.21.0, < 0.26.0
```

Pick the newest version in that range whose CUDA build your driver supports, and install it from the
matching index. See [pytorch.org](https://pytorch.org/get-started/locally/) for the URL:

```bash
pip install "torch==<version>" "torchvision==<version>" --index-url https://download.pytorch.org/whl/<cuXXX>
```

A bare `pip install torch` respects neither the version cap nor your CUDA version. If
`torch.cuda.is_available()` comes back `False` after installing, the CUDA build does not match your
driver — pick a different one.

### 1.5 Install lerobot and the custom packages

Install in this order — each package depends on the ones above it, and `lerobot_canonical` is local,
so pip must find it already installed rather than looking for it on PyPI.

```bash
pip install -e lerobot
pip install -e custom/utils/lerobot_canonical
pip install -e custom/policies/umidiffusion/lerobot_policy_umidiffusion
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
and written as a `LeRobotDataset`. Only successful episodes are stored, and runs are reproducible.

```bash
python custom/scripts/sim/collect_metaworld.py \
    --output-root $HOME/datasets/metaworld_canonical/pick_place_v4 \
    --n-episodes 300
```

| Option | Default | |
|---|---|---|
| `--output-root` | required | Where the dataset is written. |
| `--n-episodes` | 300 | Successful episodes to collect. |
| `--overwrite` | off | Replace an existing output directory. |
| `--seed-base` | 100 | First reset seed. Keep >= 100 so evaluation seeds 0-9 stay held out. |

Everything else defaults to the verified configuration: `pick-place-v3`, 240x240 images, 80 fps,
gripper threshold 0.7, and 30% of episodes with noise injected into the expert's action to produce
recovery states.

`--image-size` and `--gripper-threshold` are baked into the dataset. If you change either, pass the
same value at rollout — a mismatch is silent, not an error.

Inspect a dataset afterwards:

```bash
python custom/scripts/data_processing/raw_inspect.py \
    --raw-root $HOME/datasets/metaworld_canonical/pick_place_v4 \
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
    --dataset.root=$HOME/datasets/metaworld_canonical/pick_place_v4 \
    --steps=30000 --batch_size=64 --policy.device=cuda --num_workers=8 \
    --save_freq=5000 --wandb.enable=false \
    --output_dir=outputs/train/umidiffusion_pick_place_v4
```

`--policy.use_depth=false` matches the Meta-World dataset, which has no depth.

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
python custom/scripts/sim/rollout_metaworld.py \
    --checkpoint outputs/train/umidiffusion_pick_place_v4/checkpoints/030000/pretrained_model \
    --n-episodes 20
```

| Option | Default | |
|---|---|---|
| `--checkpoint` | required | Path to a `pretrained_model` directory. |
| `--n-episodes` | 20 | Episodes to run, using seeds `--seed` through `--seed + N - 1`. |
| `--seed` | 0 | First evaluation seed. Collection starts at 100, so 0-99 are held out. |
| `--gif-episodes` | 1 | Save the first N episodes as gifs. `0` disables. |
| `--gif-dir` | `tmp/real` | Where to write them. |

Gif filenames do not encode any settings, so a second run overwrites the first — give each run its
own `--gif-dir` when comparing. Encoding costs about 30 s per episode, so pass `--gif-episodes 0`
when you only need the success rate.

Run `--help` for the remaining options.
