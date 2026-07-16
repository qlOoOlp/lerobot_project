# LeRobot 기반 custom 스택 리팩터링 (`lerobot_share`)

> 기존 `lerobot_hong` 의 구현을 `lerobot_share` 로 새로 이식하며 리팩터링한다.
> **핵심 목적: 옮기는 행위 자체로 lerobot 구조를 한 층씩 이해하고, 제대로 모듈화한다.**
>
> **작업 원칙 (매 Phase 동일 루프)**: 옮긴다 → 왜 이렇게 생겼는지 lerobot 소스를 읽는다 → 그 조각만 단독 실행해 검증 → 다음 층과 연결. **한 층이 혼자 도는 걸 확인하기 전엔 다음으로 안 넘어간다.**
>
> 목표: policy / observer / env·robot 어댑터 모듈 분리 · lerobot **완전 무수정**(플러그인 규칙) · 각 층 **독립 swap** · **ROS2 제외**(별도 레이어).

## 우선순위: metaworld 먼저, UMI 후순위

metaworld 로 **코어 루프(데이터→정책→observer→학습→eval)** 를 먼저 완성하고, 같은 스키마로 UMI 를 확장한다. 이유:
- metaworld 는 **sim 이라 데이터가 깨끗**(pose jump·fps 혼재·non-monotonic 없음), **self-contained**(gym), **Robot 불필요**(env 경로)
- metaworld canonical 스키마 = **UMI 와 동일한 10D `[xyz, rot6d, gripper]`** (일부러 맞춰 설계됨) → 뼈대·스키마를 **그대로 UMI 로 이식**

## 전체 작업 순서 (한눈에)

| Phase | 층 | 할 일 | lerobot 수정 |
|---|---|---|---|
| **0** | 뼈대 | 환경 부트스트랩 (clone·env·torch·lerobot) ✅ | 무수정 |
| **── 코어 (metaworld) ──** |
| **1** | 데이터 | lerobot_canonical(keys/dims) + metaworld→LeRobotDataset ✅ | 무수정 |
| **2** ★ | 정책 + observer | **핵심.** top-down: baseline → 정책 껍데기 → 파이프라인 → step 껍데기 → codec → step 내부 → 역변환 → depth 게이트 | 무수정 |
| **4** | 학습 | lerobot-train (metaworld dataset) | 무수정 |
| **5** | eval | metaworld rollout (env_processor) | 무수정 |
| **── UMI 확장 (코어 검증 후) ──** |
| **6** | UMI 데이터 | raw 인스펙터(✅) + umi2lerobot 컨버터(**Phase 2 표현 codec 재사용**) | 무수정 |
| **7** | UMI 오프라인 추론 | runtime buffer/sync, 녹화 관측 vs GT | 무수정 |
| **8** | 정리 | 무수정 확인·setup.sh 최종화 | 무수정 |
| **── 이후 (deferred) ──** |
| 9–11 | robot(franka) + real-world env + ROS2 | — | — |

> **lerobot 전 Phase 무수정.** 등록=플러그인 폴백, depth=`UmiDiffusionConfig.apply_depth_gate()` → `datasets/policies factory.py` 패치 **불필요**(Phase 2). 패치 파일(`patches/`)은 fallback 참고용.
> 관측 어댑터(env_processor vs robot_processor vs 컨버터)와 canonical 계약은 **부록 D** 참조 (핵심 설계).

---

# Phase 0 — 뼈대 (환경 부트스트랩) ✅ 완료

lerobot 을 clone 해 `v0.4.4` 로 고정하고, 깨끗한 conda env 만 세운다. (lerobot 패치 없음 — 전 Phase 무수정)

### 디렉토리 배치
```
lerobot_share/
├── lerobot/            # HF lerobot v0.4.4 (clone, detached HEAD)
├── custom/             # lerobot 플러그인/변환 코드 (ROS2 제외)
├── custom/scripts/     # 실행 스크립트 전부 (raw_inspect, collect_metaworld)
├── patches/            # fallback 참고용 (기본은 무수정)
├── outputs/            # 실행 산출물 (인스펙터 로그 등)
├── information.md      # 구현 동작 상세
└── refactoring.md      # (이 문서) 계획·순서
```

### 실행 (완료)
```bash
git clone https://github.com/huggingface/lerobot.git
git -C lerobot checkout v0.4.4          # commit 8fff0fde
conda create -n lerobot_hong2 python=3.10 -y   # env 이름=lerobot_hong2
conda activate lerobot_hong2
# torch: bare 금지 — lerobot 캡(<2.11) + RTX 5090(Blackwell, cu128+) 둘 다 만족
pip install "torch==2.10.*" "torchvision==0.25.*" --index-url https://download.pytorch.org/whl/cu128
pip install -e lerobot
```
> `~/.local` 제거됨(부록 C.1) → `pip install --user` 금지, env 안에만 설치. 다른 GPU면 `torch<2.11` + 맞는 `--index-url`.

### 검증 (완료)
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.10.0+cu128 True
python -c "import lerobot; print(lerobot.__version__)"                          # 0.4.4
```

### 남은 것
- [x] `custom/{utils,policies,envs,scripts}` 골격 ✅ (Phase 1–2 에서 실물로 채워짐 — `lerobot/` 최상위의 부분집합)
- [ ] (선택) `setup.sh`

---

# ══════ 코어 (metaworld) ══════

# Phase 1 — 데이터 (metaworld → LeRobotDataset) ✅ 완료

**목표**: metaworld env 데이터를 **LeRobotDataset(canonical 10D)** 으로 만들며 LeRobotDataset(create/add_frame/features/stats)을 이해. sim 이라 깨끗 → 필터/리샘플/회전 불필요.

**읽을 것**: `lerobot-dataset-v3.mdx`, **`porting_datasets_v3.mdx` + `examples/port_datasets/port_droid.py`**(로봇 없이 `create`+`add_frame` 변환 정석 ⭐) / 소스 `datasets/lerobot_dataset.py`(`create`/`add_frame`/`save_episode`/`meta`), `configs/types.py`.

**스키마 (canonical 10D, `use_depth=False`)**:
```
observation.images.rgb : (240,240,3)                       ← depth 없음
observation.state      : (10,) [x,y,z, rot6d(6), gripper]  ← UMI 와 동일, **절대** 자세
action                 : (10,) 같은 10D, **절대 target** = state[t+1]
fps 80
```
> **state·action 둘 다 절대값**(delta 아님). `delta` 는 env 경계(`canonical10_to_env_action`)에서만 생김 — 폴더명 `umi2lerobot_delta_pose` 에 낚이지 말 것(옛 UMI 변환본도 실측상 절대값). rot6d 채택 근거·복원식·절대 vs delta 상세: **retargeting.md 1절**.

### ✅ 산출물
| | |
|---|---|
| **코드** ✅ | `custom/utils/lerobot_canonical/` · `custom/envs/metaworld/canonical.py` · `custom/scripts/sim/collect_metaworld.py` |
| **계약 문서** ✅ | `retargeting.md` (그리퍼 규약 · `xyz_scale` 근거 · 변경 이력) |
| **데이터셋** ⏳ | **재수집 필요** → `~/datasets/metaworld_canonical/pick_place_v3_bin` (이 경로로 만들면 아래 명령·문서가 전부 그대로 유효) |

> **왜 재수집인가 (2026-07-17)**: 옛 300ep(16,370프레임)은 **시딩 함정** 시절 산출물이라 재현이 안 됐다
> — `env.reset(seed=)` 을 Meta-World 가 설계상 버려서 `--seed-base` 가 아무 일도 안 했다. 매 수집이 다른
> 장면을 뽑았고, "eval seed 0~9 홀드아웃" 도 성립하지 않았다. 고친 지금은 같은 명령 = 같은 데이터셋
> (50ep 2회로 실측 확인). 옛 데이터셋 3.6GB 는 남길 가치가 없어 전부 삭제. 원인·해법: information.md §1.3 "시딩 함정".
>
> **수집 명령**:
> ```bash
> MUJOCO_GL=egl python custom/scripts/sim/collect_metaworld.py \
>     --output-root ~/datasets/metaworld_canonical/pick_place_v3_bin --n-episodes 300
> ```
> 나머지는 전부 기본값(`pick-place-v3`, 240px, 80fps, threshold 0.7, seed-base 100, noise 30%/std 0.15).

### ✅ 확정된 상수 (Phase 5 가 **같은 값**을 써야 함 → train==inference)
| 상수 | 값 | 성격 |
|---|---|---|
| `PICK_PLACE_GRIPPER_THRESHOLD` | **0.7** | **태스크 의존** — obs[3] 이중봉우리 ≈1.0/≈0.40~0.46 실측. 물체 두께 바뀌면 재실측 |
| `ENV_XYZ_SCALE` | **0.01** | **태스크 무관** — env 의 `action_scale` 상수. 실측 검증: action 1.0/0.5/0.25 → 0.01003/0.00513/0.00261 m/step (retargeting.md 5절) |

### ✅ 검증 결과 (Step 5)
```
raw_inspect      300ep 전부 80fps 균일 · 10D 일관 · no-warning
이진 그리퍼       state[9]/action[9] 고유값 {0,1} (옛 연속 88종) · 열림 55.5%
meta.stats       rot6d std=0 (상수 채널, 정상) · gripper mean 0.555/std 0.497 · xyz std ≈[0.047,0.063,0.050]
스키마 대조       옛 pick_place_v3_inenv 와 차이 딱 1개: 축이름 gripper_width -> gripper (의도)
flip 방향        gif 육안 확인 ✓ (테이블 아래, 팔 위, 블록 쥐고 목표로 이동) -> tmp/real/
per-step |dxyz|  mean 0.0072 / p95 0.0131 / max 0.0169  -> mean/ENV_XYZ_SCALE=0.72 (범위 적절 사용)
```

### 세부 작업
1. [x] **lerobot_canonical (최소)** ✅: `utils/lerobot_canonical/` = `keys.py` + `schemas/canonical_ee10.py`
   - `keys.py`: lerobot `utils/constants.py`(`OBS_IMAGES`,`OBS_STATE`,`ACTION`) 위에 `image_key()`/`RGB_KEY`/`DEPTH_KEY`/`STATE_KEY`/`ACTION_KEY` — 전부 상수 **파생**(문자열 하드코딩 0). 표현 무관 → 모든 표현이 공유
   - `schemas/` = **표현별 모듈 하나씩** 두는 자리 (`__init__.py` 에 규약 문서화, **재노출 없음** → 어떤 모듈도 "the schema" 행세 못 함)
     - `schemas/canonical_ee10.py`: dims/axes (`POSE_DIM=9`, `GRIPPER_DIM=1`, `STATE_DIM=10`, rot6d `STATE_AXES`)
     - schema 는 **로봇이 아니라 표현(EE-Cartesian pose+gripper)에 종속** → metaworld/Sawyer·franka·UMI 가 공유. **로봇 교체 = 재사용**, **표현 변경 = 새 모듈**
     - 새 표현(관절 등)은 `schemas/` 안에 **sibling** 추가(`canonical_joint7.py`), 기존 모듈 불변(open/closed)
     - 대외 계약(모든 표현 모듈 공통): `STATE_DIM`/`STATE_AXES`/`ACTION_DIM`/`ACTION_AXES` · 내부 사정: `POSE_DIM`/`POSE_AXES`/`GRIPPER_*`
     - 다운스트림은 **모듈 import** 로 표현만 갈아끼움: `from lerobot_canonical.schemas import canonical_ee10 as sch` → `sch.STATE_DIM`
   - ★ **feature 빌더·정책 feature 는 여기 두지 말 것** — 정책은 데이터셋에서 파생(`dataset_to_policy_features`), feature dict 는 변환기가 정의 (lerobot 정석, 부록 D)
2. [x] **metaworld env 어댑터** ✅: `custom/envs/metaworld/canonical.py`
   - ★ **경계를 넘는 것 전부**가 여기 — 셋 다 **수집·rollout 공유**라 train==inference 요구가 걸리기 때문:
     | 함수 | 방향 |
     |---|---|
     | `render_frame(env, size)` | env 카메라 → 데이터셋 이미지 (**corner2 양축 flip 보정** + resize) |
     | `state4_to_canonical10(state4, thresh)` | env obs[:4] (abs) → canonical 10D (abs, **그리퍼 이진화**) |
     | `canonical10_to_env_action(t10, cur_xyz, scale)` | canonical (abs) → env 4D (**rel**, 그리퍼 극성 반전) |
   - "공유 요구가 있는 건 여기(모듈), 수집 전용은 `scripts/sim/collect_metaworld.py`" 가 배치 기준 (`render_frame` 을 수집 스크립트에서 여기로 옮긴 이유). **재사용되는 것=모듈 / 실행되는 것=`scripts/`**
   - env 사실은 여기: `STATE4_DIM=4`, `ENV_ACTION_DIM=4`, `ENV_XYZ_SCALE=0.01`, `PICK_PLACE_GRIPPER_THRESHOLD=0.7` / 표현 사실은 `lerobot_canonical` 에서 import (`sch.STATE_DIM` 등, `10`/`9` 하드코딩 0)
   - ⚠ `env.render()` 는 **인자를 받지 않음** (gymnasium 0.26+; `render_mode`/`camera_name` 은 생성 시 고정). 우리는 expert 가 raw 39D obs 를 필요로 해 `wrapper._env`(내부, 보정 없음)를 직접 몰므로 **flip·resize 는 우리 몫**
3. [x] **수집 → LeRobotDataset** ✅: `custom/scripts/sim/collect_metaworld.py` (**port_droid 패턴, Robot 없음**)
   - `build_features` (컨버터 책임) + `LeRobotDataset.create` + `add_frame` 루프 + `save_episode` + `finalize`
   - ⚠ 이미지 feature 의 `names[2]` 는 **`"channels"`** 여야 함 — lerobot 이 이 값으로 `(H,W,C)→(C,H,W)` 변환 여부를 결정(`datasets/utils.py:724`). `"channel"` 은 옛 오타 호환 경로라 쓰지 말 것
   - **성공 에피소드만** 저장(`info["success"]`), **30%에 xyz 노이즈**(회복 데이터, **그리퍼엔 금지**), `seed_base=100`(eval seed 0~9 홀드아웃)
   - **시프트 없음**: `obs[3]` 이 진짜 상태라 `action[t] = state[t+1]` 규칙 그대로
   - `convert_`(mt50) 경로는 미구현 — in-env 수집이 per-step dynamics 를 rollout 과 맞추므로 우선
4. [x] `pip install metaworld==3.0.0` ✅ (metaworld 3.0.0 / mujoco 3.10.0). 실행 시 **`MUJOCO_GL=egl`**(헤드리스)
   - ⚠ `metaworld.__version__` 없음 → `importlib.metadata.version("metaworld")` 로 확인
5. [x] **검증** ✅ (결과는 위 "검증 결과" 표)
   - ★ **인스펙터 재사용**: `raw_inspect.py --format lerobot_dataset --raw-root <dataset> --target-fps 80`
   - **gif 는 코드로 대체 불가** — `render_frame` 의 flip 방향은 육안 확인만 가능 (`tmp/real/`)

> **핵심**: metaworld 는 gym Env 경로 → **Robot·robot_processor 불필요.** env↔canonical 매핑이 계약이고 수집·rollout 양쪽에서 같은 함수 → 자동 일치 (부록 D.1).
>
> **옛 데이터셋 상태**: `pick_place_v3`(mt50 변환)·`pick_place_v3_inenv`(옛 in-env) 는 **그리퍼가 연속**이라 새 규약과 다름 → 학습엔 `pick_place_v3_bin` 사용.

---

# Phase 2 — 정책 + 런타임 프로세서 (lerobot 무수정)

**목표**: umidiffusion 를 BYO Policy 규칙대로 만들고, **런타임 anchor-relative 변환까지** 붙여 `pre→forward→post` end-to-end. lerobot 무수정(등록=플러그인 폴백, depth=config 게이트).
**이 Phase 가 이 프로젝트의 핵심**이다 — 데이터(P1)는 재료였고, 여기서 "정책이 무엇을 보고 무엇을 뱉는가"가 결정된다.

> **왜 정책과 프로세서가 한 Phase 인가** (dev_plan §12.1):
> *"`policy.type=diffusion` 을 그대로 쓰면 factory 가 기본 processor 를 만든다. 그 기본 processor 에는 **runtime relative/delta pose step 이 없다**."*
> → **custom policy 의 존재 이유가 곧 이 step 들**이다. 떼어놓으면 심장 없는 정책이 됨.
> → 그래서 **2-0 에서 기본 diffusion 을 먼저 돌려본다**: 뭐가 없는지 봐야 왜 만드는지 안다.

**BYO Policy 규칙** (`lerobot_policy_umidiffusion`) — **lerobot 공식 문서(`docs/source/bring_your_own_policies.mdx`) 규약 그대로**:
- config `@PreTrainedConfig.register_subclass("umidiffusion")`, **`UmiDiffusionConfig(PreTrainedConfig)`**
- class **`UmiDiffusionPolicy(PreTrainedPolicy)`**, `name="umidiffusion"`
- processor `make_umidiffusion_pre_post_processors`
- **자동 탐지(무수정)**: `_get_policy_cls_from_policy_name` + `_make_processors_from_policy_config` 폴백이 컨벤션으로 찾음.
- `__init__.py` 에서 `UmiDiffusionConfig`/`UmiDiffusion`/`make_umidiffusion_pre_post_processors` **export**(dev_plan §12.3) — import 시 등록+노출이 함께 일어나게

> ★ **기존 정책의 config 를 상속하면 안 된다** (2026-07-17, 반나절 소모). lerobot_hong 처럼
> `UmiDiffusionConfig(DiffusionConfig)` 로 하면 `make_pre_post_processors` 의
> `elif isinstance(policy_cfg, DiffusionConfig)` (`factory.py:296`)에 걸려 **우리 프로세서 팩토리가
> 영원히 안 불린다** — 그것도 **조용히**(lerobot 의 diffusion 프로세서가 대신 일해 학습이 "성공"하고,
> 정책은 anchor-relative 없이 절대 canonical 로 학습된다). lerobot_hong 이 factory 를 패치한 진짜 이유가
> 이것이고, 우리는 **diffusion 정책을 통으로 복사**해 규약으로 돌아왔다. 근거·검증: **부록 D.7**

**층 분리는 lerobot_hong 을 따름** (§7.2/§9.5: `steps.py`=순수 로직 / `processor_umidiffusion.py`=조립):
```python
input_steps = [Rename, AddBatch,
               CanonicalPoseToActionPoseReprStep(action_pose_repr),   # 액션 → relative/delta
               CanonicalPoseToRelativeObservationStep(),              # 관측 → anchor-relative
               Device, Normalizer]
# use_depth=False 면 index 1 에 DropObservationKeysProcessorStep 삽입
output_steps = [Unnormalizer, Device(cpu)]
```

### 세부 작업 — **top-down 순서** (2-0 → 2-7)

> ★ **의존성 순서 ≠ 구현 순서.** 파일은 `codec ← step ← processor` 로 의존하지만(설치도 `lerobot_canonical → umidiffusion`), **구현은 위에서 내려간다**. 수학부터 채우면 *왜* 하는지 모른 채 손만 움직이게 된다. 아래는 **매 단계 돌려볼 수 있게** 짠 순서다.

0. [x] **2-0 데이터셋 무죄 확인** ✅ — `lerobot-train --policy.type=diffusion` 을 우리 데이터셋에 **그대로**
   - **목적 = 변수 분리** (성능 baseline 이 아님 — 표현이 달라 애초에 비교 대상이 아니다). lerobot 기본(검증된 것) × 우리 데이터셋(미검증) 을 먼저 붙여, **이후 실패는 전부 우리 코드**임을 확정
   ```bash
   lerobot-train --policy.type=diffusion --policy.push_to_hub=false \
     --dataset.repo_id=local/metaworld_canonical_pick_place_bin \
     --dataset.root=~/datasets/metaworld_canonical/pick_place_v3_bin \
     --steps=200 --batch_size=8 --output_dir=outputs/train/baseline_diffusion
   ```
   - ⚠ **`--policy.push_to_hub=false` 필수** — 기본값이 True 라 `policy.repo_id` 를 요구하며 죽는다(`configs/train.py:138`)
   - **결과**: 200스텝 36초, `loss:0.843 grdn:4.579`, **NaN 없음**, 체크포인트 저장 OK → **데이터셋 무죄**
   - **실측 발견**: `rot6d std=0` 이 **안 터진다**. lerobot 이 `denom = std + eps`(eps=1e-8)로 막음(`processor/normalize_processor.py:94, :335`) → **"std=0 나눗셈 회피"는 IDENTITY 의 근거가 아님**(2-7 참고). 상수 채널은 `0/1e-8=0` 이 되어 **죽은 채로 들어갈 뿐**
1. [x] **2-1 정책 껍데기** ✅ (2026-07-17) — `configuration_umidiffusion.py` + `modeling_umidiffusion.py`
   - 배우는 것: **BYO Policy 플러그인 규칙**, 정책 발견이 **이름 컨벤션**으로 되는 이유(`_get_policy_cls_from_policy_name`)
   - `__init__.py` export 가 **등록 트리거** — 빼면 조용히 실패 (dev_plan §12.3)
   - ✅ **결과**: `UmiDiffusionPolicy` **263,196,458 params** / input `{rgb (3,240,240), state (10,)}` / output `{action (10,)}`.
     `lerobot-train --policy.type=umidiffusion` **완주**.
   - ★ **여기서 두 가지가 터졌다** (둘 다 **조용한** 실패라 찾는 데 오래 걸렸다):
     1. **패키지화** — 설치된 배포판이 `custom/...` 경로를 import → `ModuleNotFoundError` →
        `register_third_party_plugins()` 가 **삼켜서** `invalid choice` 로만 보였다. 해결 = 배포판 2개로 분리 (**부록 D.6**).
        이때 개명도 함께: `mypolicy`→`umidiffusion` · `lerobot_ext_core`→`lerobot_canonical` · `common/`→`utils/` ·
        `tools/`+수집 스크립트→`custom/scripts/` · `robot_maps` 패키지 소멸(→ 정책 안 `steps.py`).
     2. **프로세서 팩토리가 안 불림** — `UmiDiffusionConfig(DiffusionConfig)` 라서
        `isinstance(cfg, DiffusionConfig)`(`factory.py:296`)에 걸렸다. 해결 = **diffusion 정책 통 복사** (**부록 D.7**).
   - ⚠ **여기서 "0-patch 첫 실증" 이라고 단언했던 것은 과장이었다.** `lerobot-train` 이 완주한 것은
     **정책 발견 경로**만 증명한다. 프로세서 발견 경로는 확인하지 않았고, 실제로는 깨져 있었다
     (우리 함수 본문이 `...`=None 인데도 정상 반환 = 한 번도 안 불림). **검증한 범위만 주장할 것.**
2. [ ] **2-2 파이프라인 조립** — `make_umidiffusion_pre_post_processors`, **lerobot 기본 step 만** (Rename/AddBatch/Device/Normalizer)
   - 배우는 것: `PolicyProcessorPipeline` 구조, step 순서, feature 계약, post 의 `to_transition`/`to_output`
   - 이름 고정 — `_make_processors_from_policy_config` 폴백이 컨벤션으로 찾음
   - 검증: `pre(sample) → select_action → post(action)` end-to-end
3. [x] **2-3 우리 step 껍데기** ✅ (2026-07-17) — 2개 step 을 **pass-through(항등)** 로 만들어 파이프라인에 끼움
   - 배우는 것: `ProcessorStep` 규약(**추상은 `__call__` + `transform_features` 둘뿐** — 나머지 `get_config`/`reset`/`state_dict`/`load_state_dict`/`transition` 은 기본 구현 있음), `@ProcessorStepRegistry.register`, **파이프라인 순서 의존성**
   - ★ **action step 이 obs step 보다 먼저** — 앵커로 쓸 `state` 가 아직 **절대**여야 한다. 뒤집으면 이미 relative 가 된 state(마지막=항등)를 앵커로 삼아 **전부 망가짐**
   - ★ **`if action is None: return`** — 추론엔 action 이 없다(정답을 모르니 정책을 돌린다). 빠뜨리면 Phase 5 에서 터짐
   - ✅ **검증**: 순서(action idx2 < obs idx3) · **항등**(obs/act 통과 전후 완전 동일) · action 없이 통과 ·
     `ndim!=3` 가드 발화 · `action_pose_repr` 검증 발화 · **레지스트리 왕복**(`get_config()` → 생성자 복원)
   - ★ **여기서 Phase 5 계약이 드러났다** — 아래 참조

> ### ★ (B, T, 10) 은 누가 만드나 — 학습과 추론이 다르다 (2-3 에서 발견)
> ```
> 학습 : DataLoader 가 delta_timestamps 로 윈도우를 잘라 (B, T, 10) 을 준다     → 우리 step OK
> 추론 : 정책의 _queues stack 은 predict_action_chunk **안**에서 일어난다
>        = 프로세서보다 **뒤** → select_action() 을 쓰면 우리 step 은 (B, 10) 을 받고
>        → 앵커 state[:, -1] 을 만들 수 없다
> ```
> **Phase 5 rollout 은 `select_action()` 을 쓰면 안 된다.** 대신:
> ```python
> window       = buffer.as_window()                       # 자체 히스토리 버퍼로 (T, 10)
> anchor_state = window[OBS_STATE][-1]                    # 앵커를 여기서 확보
> processed    = preprocessor(build_model_obs_dict(window))   # pre 에 (B, T, 10) 을 넘김
> chunk        = policy.diffusion.generate_actions(processed) # ★ select_action 우회
> action       = decode_policy_action(chunk, anchor_state, action_pose_repr=policy.config.action_pose_repr)
> ```
> lerobot_hong `custom/scripts/sim/rollout_metaworld_mypolicy.py:83-98` 이 정확히 이 형태다.
> 우리 step 의 `ndim != 3` ValueError 가 이 계약을 **강제**한다 — 어기면 조용히 틀리는 대신 터진다.
> (관측 히스토리 버퍼는 Phase 7 의 `runtime_buffer` 와 같은 물건 — 그때 공유 검토)
4. [x] **2-4 표현 codec** ✅ (2026-07-17) — `schemas/canonical_ee10_se3.py`: `rot6d↔R`, `pose9d↔transform`, `invert_transform`, `relative_transform` (torch, 배치 `...`)
   - 배우는 것: rot6d 를 쓰는 이유(연속성), **SE3 역변환은 `R^T`**(일반 `inv` 금지 — lerobot_hong 이 그 실수), 좌표계 상쇄
   - **표현 옆에 두는 이유·lerobot_hong 중복 증거**: 부록 D.5
   - 검증(**의존성 0 단독**): `invert(T)@T == I` · `R→rot6d→R == R`(**비대칭 회전으로!** 항등행렬은 행/열이 같아 안 걸림) · **좌표계 불변성** `(T@a)⁻¹@(T@b) == a⁻¹@b`
5. [x] **2-5 step 내부** ✅ (2026-07-17) — 2-3 의 껍데기를 codec 으로 채움
   - 배우는 것: **앵커 의미** — 왜 `state[:,-1]` 인지, 왜 마지막 프레임이 항등이 되는지(정보량 0이지만 정상)
   - **pose 9D 만 변환, gripper 1D 는 그대로** / **차원 유지** 10D→10D (§9.4)
   - ⚠ **`action` 없으면 skip** — eval/추론 raw 관측엔 action 이 없음 (§9.3)
   - 검증: 2-3 항등 대비 **값이 바뀌는지**, gripper 는 **불변**인지
6. [x] **2-6 역변환** ✅ (2026-07-17) — `decode_policy_action(action, anchor_state, action_pose_repr)`
   - **정책 relative 출력 → 절대 canonical**. `relative`: `base @ action` / `delta`: 누적 적분
   - **파이프라인 밖**: `policy_post` 는 `PolicyAction` 만 받아 **앵커 접근 불가** → 원본 UMI(`get_real_umi_action`)·lerobot_hong(`decode_policy_action`, 호출부 5곳) 모두 추론 루프가 직접 호출
   - ⚠ **없으면 Phase 5 가 통째로 틀린 명령을 냄**
   - forward step 과 **같은 파일**에 둔다 — 떨어뜨리면 갈라져도 모름(원본 UMI 가 그렇게 깨짐)
   - 검증: **왕복** `decode(forward(a, anchor), anchor) == a` (relative/delta **둘 다**)
7. [x] **2-7 depth 게이트** ✅ (2026-07-17) — `apply_depth_gate()`(config) + `DropObservationKeysProcessorStep`(런타임 절반)
   - hook 위치: `UmiDiffusionPolicy.__init__` 의 **`super()` 직전** / **idempotent** 필수
   - `normalization_mapping`: `VISUAL=MEAN_STD`, **`STATE=IDENTITY`, `ACTION=IDENTITY`**
     - **유일한 근거 = dev_plan §11**: canonical stats 로 relative 를 정규화하면 **표현공간 불일치**
     - ⚠ ~~"std=0 나눗셈 회피"~~ 는 **근거 아님** — 2-0 실측으로 반증(lerobot 이 eps 로 막음)
   - 검증: `use_depth` on/off 둘 다

**설치**: `pip install -e` (`custom/utils/lerobot_canonical` → `custom/policies/umidiffusion/lerobot_policy_umidiffusion` 순, 둘 다 `--no-deps`).
**코드 수정은 재설치 불필요**(editable). 단 `pyproject.toml` 변경 **또는 디렉토리 이동** 시에는 재설치 — editable 은 절대경로를 박아둔다.

### 왜 lerobot 무수정 (lerobot_hong 은 60줄 패치)
- lerobot_hong 실측: `datasets/factory.py`(+15) · `policies/factory.py`(+45) = **전부 depth 필터**
- → **`apply_depth_gate()`(config)로 이전**: `use_depth=False` 면 `input_features` 에서 depth 제거 → 모델이 depth 인코더 안 만듦 → `policies/factory.py` 패치 불필요
- `datasets/factory.py` 패치는 "depth 미로드 효율"뿐 → 생략(로드되나 `DropObservationKeys` 가 제거)
- → **패치 0파일.** 상세: `information.md` §3.1
- ⚠ depth 게이트는 **두 겹**: config(모델이 인코더를 안 만들게) + `DropObservationKeys` step(관측에서 실제 제거). 둘 다 필요

---

# Phase 4 — 학습 (metaworld dataset)

**목표**: umidiffusion 를 `lerobot-train` 에 연결, metaworld 데이터로 소량 overfit 검증.

### 세부 작업
1. [ ] train config: `n_obs_steps`, `horizon`, `use_depth=False`, dataset root = **`~/datasets/metaworld_canonical/pick_place_v3_bin`** (Phase 1 산출물, 300ep/16,370프레임) (delta_timestamps=`index/fps` 자동)
2. [ ] `lerobot-train --policy.type=umidiffusion --dataset...`
   - ⚠ **`--policy.push_to_hub=false` 필수** — 기본 True 라 `policy.repo_id` 요구하며 죽음 (`configs/train.py:138`). 2-0 에서 확인
   - `rot6d` std=0 은 **문제없음** — lerobot 이 `denom = std + 1e-8` 로 막음(2-0 실측). 우리는 어차피 `STATE=IDENTITY`
   - 2-0 참고: 우리 데이터셋은 lerobot 학습 경로를 **이미 통과**함(200스텝 loss 0.843) → 여기서 실패하면 **umidiffusion 쪽 문제**
3. [ ] 소량 overfit + `pip install matplotlib`
4. [ ] **검증**: loss 하강, 체크포인트 저장/로드

---

# Phase 5 — eval (metaworld rollout) — 코어 루프 완성

**목표**: 학습 정책을 metaworld 에서 rollout. **env_processor(=canonical 매핑)** 로 온라인 관측을 학습 데이터와 일치시킴.

### ★ 먼저 알아야 할 것 — lerobot 에 metaworld 가 **이미 있다** (2026-07-17 확인)

`@EnvConfig.register_subclass("metaworld")` (`envs/configs.py:349`) + `create_metaworld_envs`(`envs/metaworld.py:272`)
가 본체에 내장. **env 패키지를 만들 필요 없다**(부록 D.6.1). `fps=80` 도 우리 데이터셋과 일치.

**하지만 내장 feature 는 env-native 라 우리 canonical 과 다르다** — 이 갭이 곧 env_processor 의 존재 이유:
| | lerobot 내장 `MetaworldEnv` | 우리 canonical (데이터셋·정책) |
|---|---|---|
| state | `agent_pos` **(4,)** | `observation.state` **(10,)** |
| action | **(4,)** (env 4D) | `action` **(10,)** |
| 이미지 | `pixels/top` → `observation.image` **480×480** | `observation.images.rgb` **240×240** |

> **flip 은 학습과 무관하다.** 대칭 쌍은 학습↔추론이 아니라 **수집↔추론**이다:
> ```
> 수집:  env raw ──flip──resize──> 데이터셋    ← 여기서 구워짐 (파일에 박힘)
> 학습:  데이터셋 ─────────────-─> 정책        ← flip 할 게 없음. env 를 만난 적도 없음
> 추론:  env raw ──flip──resize──> 정책        ← 수집과 '같은 함수' render_frame()
> ```
> 즉 `train==inference` 의 실제 의미는 **`데이터셋 == 추론 입력`**. flip/resize/그리퍼 이진화는
> **고정 상수라 오프라인에 구울 수 있어** 런타임 프로세서에 **없다**(수집 때 1번, 추론 때 1번).
> 반대로 **anchor-relative 는 학습·추론 둘 다** 런타임 프로세서에서 한다 — 윈도우의 anchor 에
> 종속이라 못 굽는다(dev_plan §3.2). **판정: 구울 수 있나? → 수집에 bake / 없으면 → 런타임 양쪽.**
>
> ⚠ **이중 flip 함정** (학습/추론 문제가 아니라 **어떤 env 객체를 넘기냐** 문제)
> — `envs/metaworld.py` 의 wrapper 는 `render()`(149행)·`_format_raw_obs()`(172행) **두 곳 모두**에서
> `camera_name=="corner2"` 면 **이미** `np.flip(image, (0,1))` 을 한다
> (*"The corner2 camera outputs images with both axes inverted"*). 우리 `render_frame` **도** flip 한다.
> → rollout 에서 **wrapper 를 넘기면 이중 flip = 거꾸로 된 그림으로 추론**. Phase 1 수집이
> `render_frame(wrapper._env, 240)` (**내부** env) 였으므로 **rollout 도 똑같이 `wrapper._env` 를 넘겨야** 한다.
> 같은 이유로 wrapper 의 `pixels`(480, 이미 un-flip 됨)를 쓰면 안 된다 — train≠inference.
> 참고: wrapper 의 `agent_pos = raw_obs[:4]`(`metaworld.py:173`)은 우리 `state4` 와 **동일 정의**.

### 세부 작업
1. [ ] `make_env`(metaworld gym) 구성 — **내장 `--env.type=metaworld` 사용**, 단 위의 `wrapper._env` 함정 유의
2. [ ] **env_processor**: Phase 1 의 **같은 매핑 함수**를 얇은 `ObservationProcessorStep` 로 감쌈 (정석 레퍼런스 = lerobot `LiberoProcessorStep`: `_process_observation` 에서 env obs→`observation.state` 조립)
   - ⚠ **factory 우회 필수(무수정)**: lerobot `envs/factory.py` 의 `make_env_pre_post_processors` 는 LIBERO/Isaaclab 을 **하드코딩 if** 로 붙임 → metaworld 를 같은 식으로 넣으려면 factory 패치 필요 = **0-patch 위반**
   - → 우리 rollout 스크립트에서 **직접 조립**: `PolicyProcessorPipeline(steps=[MetaworldCanonicalStep()])` (공개 API만 사용). metaworld 는 factory 기본값이 어차피 identity 라 잃는 것 없음
3. [ ] rollout 루프: `env.step()→env_proc→policy_pre→policy→policy_post→**decode_policy_action**→env_proc→env.step()`
   - ★ **`select_action()` 을 쓰지 말 것** — 정책의 큐 stack 이 프로세서보다 뒤라 우리 relative step 이
     `(B,10)` 을 받아 **앵커를 못 만든다**(`ndim!=3` ValueError). 자체 히스토리 버퍼로 `(B,T,10)` 을 만들어
     `preprocessor` 에 넘기고 `policy.diffusion.generate_actions()` 를 직접 부른다. 근거·코드: **2-3 절**
   - ★ **역변환 필수**: 정책은 **relative 를 뱉는다**. `decode_policy_action(raw, anchor_state=canonical_window[OBS_STATE][-1], action_pose_repr=policy.config.action_pose_repr)` 로 **절대 canonical 로 되돌린 뒤** `canonical10_to_env_action` 에 넣는다. 빠뜨리면 **완전히 틀린 명령** (retargeting.md 6절)
   - 참고: lerobot_hong `rollout_metaworld_umidiffusion.py:98` 이 정확히 이 형태 — 그대로 따름
   - ★ **Phase 1 의 세 함수를 그대로 재사용**: `render_frame` / `state4_to_canonical10` / `canonical10_to_env_action` (`custom/envs/metaworld/canonical.py`)
   - ★ **Phase 1 과 같은 상수를 넘길 것** — 어긋나면 train≠inference (셋 다 조용히 실패한다):
     - `gripper_threshold = PICK_PLACE_GRIPPER_THRESHOLD` (0.7) — **태스크 의존**. 수집 때 데이터셋에 bake 된 값
     - `xyz_scale = ENV_XYZ_SCALE` (0.01) — **태스크 무관**(env 상수). **데이터 통계(p95=0.013 등)로 잡지 말 것** (retargeting.md 5절: 0.0155→35% 미달 / 0.004→항상 클립)
     - **카메라 = `corner2`** — **카메라 의존**(`FLIP_CAMERAS`). 넘기는 게 아니라 **같은 카메라로 env 를 만들면** 된다: `render_frame` 이 `env.camera_name` 을 읽어 스스로 가드하므로 desync 가 불가능하다. 다른 카메라로 만들면 flip 이 자동으로 안 걸리는데, 그건 맞는 동작이지만 **시점 자체가 학습 데이터와 달라진다**
   - ★ **`env.seeded_rand_vec = True` 를 반드시 켤 것** — 안 켜면 eval seed 가 **아무 일도 안 한다**.
     Meta-World 는 물체·목표 배치를 세 갈래로 뽑는데(`sawyer_xyz_env.py:697`), lerobot wrapper 가
     `_freeze_rand_vec=False` 만 켜고 `seeded_rand_vec` 는 안 켜서(`envs/metaworld.py:163`) **전역
     `np.random` 갈래**에 떨어진다. 거기선 `reset(seed=n)` 이 **설계상 버려지고**(Meta-World reset
     docstring: *"seed: The seed to use. **Ignored**, use `seed()` instead."*) `env.seed(n)` 조차 무력하다
     (`self.np_random` 을 안 읽으므로). 켜는 법은 `collect_metaworld.py` main() 참고 —
     **`env.seeded_rand_vec = True` + `env.seed(n)` + `env.reset()`** (reset 에 seed 를 넘기지 말 것).
   - eval seed 는 **0~9** 사용 — 수집이 `seed_base=100` 이므로 홀드아웃된다. **단 위의 `seeded_rand_vec` 가
     전제다**: 안 켜면 seed 가 장면을 정하지 않아 "홀드아웃" 이라는 개념 자체가 성립하지 않는다
     (실측: 안 켜면 같은 seed 로 두 번 돌려도 물체·목표가 매번 다름 → 데이터셋 재현 불가, 정책 A/B 를
     같은 장면에서 비교하는 것도 불가). information.md §1.3 "시딩 함정" 참고.
4. [ ] **검증**: 1 에피소드 rollout → 성공률/영상. **여기까지 = metaworld 코어 루프(데이터→학습→eval) 완성**
   - 정책의 그리퍼 출력은 **{0,1} 이진**이어야 자연스러움 (데이터가 이진) → `(0.5−o)×2` 로 `±1` 명령

> Robot 없음. 오프라인(Phase 1 수집)·온라인(rollout)이 **같은 canonical 매핑** → 자동 일치 (부록 D.1).

---

# ══════ UMI 확장 (metaworld 코어 검증 후) ══════

# Phase 6 — UMI 데이터 (raw → LeRobotDataset)

**목표**: UMI Record3D h5 를 **같은 canonical 10D** LeRobotDataset 으로 변환. metaworld 와 스키마 동일 → **Phase 2–5 정책·학습 그대로 재사용**. 단 UMI raw 는 지저분 → **인스펙터 검증 + 필터/리샘플/회전** 필요.

**차이(부록 D.3)**: metaworld 는 env↔canonical 자동일치였지만, UMI→franka 는 **cross-embodiment** → 컨버터(UMI→canonical)와 (나중)franka robot_processor 가 **같은 표현 codec**(Phase 2, 부록 D.5)을 써서 수동 일치.

### 세부 작업
1. [x] **raw 인스펙터** ✅ → `custom/scripts/data_processing/raw_inspect.py` (single file, `--format {record3d_h5, lerobot_dataset}`). **여기선 `record3d_h5`** 로 UMI raw 검사: fps 분포·차원·품질(non-mono/gap/**pose jump**) 리포트 + `outputs/.../*.json`(warning별 에피소드 번호). (`lerobot_dataset` 포맷은 Phase 1 metaworld 검증에서 재사용.) 상세: `information.md` §1.1
2. [ ] 인스펙터로 raw 검증 → **목표 fps 결정** + skip 목록 파악 (`move260626_preprocess.txt` 참고)
3. [ ] **umi2lerobot 컨버터 (config-driven)**: `reader/align/se3/build_dataset` + `UmiToLeRobotConfig`(dataclass↔YAML, draccus). 처리:
   - **UMI SLAM(quat/rpy) → transform → canonical pose9d** — 뒷단은 **Phase 2 의 표현 codec 을 import**(중복 금지, 부록 D.5). 앞단(quat/rpy 파싱)만 UMI 전용
   - **`pose_relative` 기준 = 촬영 시작 시점** (에피소드 시작이 아님). 촬영을 여러 번 나눠 해서 **기준점이 최소 19번 리셋**됨 — 대략 50ep 단위(실측: `|pos[0]|` 이 ep 0/50/102/151/… 에서 0 근처로 복귀)
     - **정책엔 영향 없음** — 입출력이 anchor-relative 라 기준이 상쇄됨(retargeting.md 6절). `pose_raw` 를 써도 동일
     - **lerobot_hong 도 `pose_relative` 사용** — config 기본값 + 변환본 이름 `_rel` + **값 비트 단위 일치**로 확인
     - 옛 변환본은 `n=951`(=raw 그대로) → **리샘플 안 된 상태**. fps mislabel 문제 잔존(부록 C.6)
   - 이 데이터셋의 **이름 함정 3연속**: `umi2lerobot_delta_pose`(실제 절대값) · `gripper_width`(실제 이진 개폐) · `pose_relative`(촬영 시작 기준) → **이름 믿지 말고 실측할 것**
   - **30fps 리샘플**(nearest RGB + interp/slerp pose) + **timestamp 정렬·중복제거**(non-mono 복구)
   - **RGB/depth CCW 90° 회전**(upright, shape 720×960 / 192×256) — `move260626_preprocess.txt`
   - **`skip_episodes`**(pose_jump ∪ rgb_nonmono)
4. [ ] **브릿지**: inspector `issues.json` → `config.yaml` 자동(`skip`, `target_fps`) → 사람 리뷰
5. [ ] 설치: `pip install h5py` + 컨버터 editable
6. [ ] 변환(1 에피소드부터) → **검증**: 로드, `meta.features`/`stats`, 샘플 gif(방향·pose 연속성)

> 결정 규칙(부록 C.6): 손상(pose_jump∪rgb_nonmono) **제외** / fps 이탈 **리샘플 회수** / depth·gripper non-mono **정렬 흡수** / gaps **관용**. move260626: 34개 제외 → 216개 @ 30fps.

---

# Phase 7 — UMI 오프라인 추론 검증 (실기·robot 불필요)

**목표**: 녹화 관측 시퀀스로 UMI 추론 경로를 검증(live 로봇 없이). 실시간 버퍼링·action chunking 이해.
> ⚠ 옛 폴더명 `umi2lerobot_delta_pose` 의 "delta" 는 오해 소지 — 데이터셋은 **절대 자세**를 담는다(실측 확인). retargeting.md 1절.

**참고**: 추론 관측 조립 = **학습 데이터와 같은 계약**(dt=1/학습fps, 정렬, canonical 표현) 재현. 오프라인 `align`↔온라인 `runtime_sync` 는 같은 math(slerp/interp/nearest). async(비동기 수집) vs sync 는 계약과 무관한 엔지니어링 선택.

### 세부 작업
1. [ ] `runtime_buffer`(history deque) + `runtime_sync`(비동기 multi-stream sync) 포팅
2. [ ] `real_inference_util`, `umi_fr3_transforms` 포팅
3. [ ] `offline_umidiffusion_inference` 포팅 (녹화 관측 재생)
4. [ ] **검증**: 녹화 시퀀스 오프라인 추론 vs GT(`dump_gt_actions`), **dt=1/학습fps 일치**(부록 C.6)

---

# Phase 8 — 정리 / 무수정 확인

### 세부 작업
1. [ ] **lerobot 무수정 확인**: `git -C lerobot diff` 비어있음 (패치 0파일)
2. [ ] depth ablation 회귀: `use_depth` on/off 학습·추론 (UMI 는 depth 있음, metaworld 는 없음)
3. [ ] `setup.sh` 최종화 (clone→v0.4.4→env→install, **패치 apply 없음**)

---

# ══════ 이후 (deferred) — robot(franka) + real-world env ══════

> **코어+UMI(Phase 0–8) 완료 후.** robot/teleop 패키지는 현재 빈 스텁, real 제어는 ROS2 스크립트뿐이라 전송 계층이 무거움.

| # | 층 | 할 일 | 참고 |
|---|---|---|---|
| **9** | robot(franka) 추상 | `lerobot_robot_franka`: `Robot` 10-메서드 + **robot_processor**(franka FK → transform → **Phase 2 표현 codec** → canonical). mock/sim 으로 단독 테스트 | 부록 D.2/D.5, `integrate_hardware.mdx` |
| **10** | real-world env | 배포 루프 `robot.get_observation()→policy→robot.send_action()`. Phase 7 오프라인을 live 로 승격 | `processors_robots_teleop.mdx` |
| **11** | ROS2 전송 이식 | `run_umidiffusion_ros2.py`(932줄)/`grace_fr3_bridge.py`(418줄) 로직을 franka `Robot` 내부 ROS2 I/O 로 | 별도 ROS2 레이어 |

**robot 층 = "인터페이스"와 "ROS2 전송" 분리**: `Robot` 10-메서드 계약이 이음새. `lerobot_robot_` 접두사 + `@RobotConfig.register_subclass` → 자동 탐지(무수정). robot_processor 는 **Phase 2 의 표현 codec 을 재사용**하고 앞단(FK)만 franka 용 → UMI 데이터와 canonical 로 일치 (부록 D.3/D.5).

---

# 부록 A — 아키텍처 (3 모듈 + observer 횡단)

lerobot 원칙: **"환경/로봇은 관측을 노출 → 프로세서가 표준화 → 정책이 소비. 각 층 하나의 책임."**

| 모듈 | lerobot 메커니즘 | 플러그인 규칙 |
|---|---|---|
| **policy** | Bring Your Own Policies | `lerobot_policy_<name>` |
| **env(metaworld)** | `make_env` + env_processor | `make_env()`→gym.VectorEnv |
| **robot(franka)** | `Robot` 10-메서드 + robot_processor | `lerobot_robot_<name>` |
| *(횡단) observer* | Processor 시스템 | `ProcessorStep`→`Pipeline` |

**참고 문서 (v0.4.4 동봉)**: policy=`bring_your_own_policies.mdx` / observer=`introduction_processors.mdx`·`implement_your_own_processor.mdx`·`processors_robots_teleop.mdx`·`env_processor.mdx` / dataset 변환=`porting_datasets_v3.mdx`(+`port_droid.py`) / env=`envhub.mdx`·`metaworld.mdx` / robot=`integrate_hardware.mdx`.

# 부록 B — 의존성 (ROS2 제외)

| 패키지 | 용도 | 설치 Phase |
|---|---|---|
| `torch numpy Pillow imageio` | 코어 (lerobot 제공) | 0 |
| `metaworld==3.0.0`(→mujoco) | metaworld 수집·eval | **1** |
| `scipy` | 프로세서(slerp 등) | 3 |
| `matplotlib` | 플롯/gif | 4 |
| `h5py` | UMI raw h5 | **6** |
| `pin`(pinocchio) | IK 시각화 | (선택) |

> ROS2(`rclpy`, msgs, `PyKDL`, `qpoases`, `trimesh` …)는 pip 아님 → Humble+colcon 별도 레이어(제외).

# 부록 C — 주요 발견 / 함정

### C.1 `~/.local` 오염 → 해결됨 (2026-07-15)
- `pip install --user` 로 `~/.local` 에 구 UMI 스택 → conda env 섀도잉 → lerobot import 깨짐. **해결**: `~/.local`(5.4G)→`lerobot_hong/_local_backup_20260715/` 이동. 복원: `mv` 원위치.

### C.2 diffusers 다운그레이드 불필요
- 깨끗한 env 의 `pip install -e lerobot` 이 정상 버전(0.35.x) 설치. umidiffusion 정상.

### C.3 git 구조 / 버전관리
- `lerobot_hong` 최상위 빈 repo + 하위 독립 clone 12개. ⚠️ `custom/` 버전관리 부재 → `lerobot_share` 는 git repo 로, `lerobot/` gitignore.

### C.4 lerobot = v0.4.4 (패치 없이 무수정 목표)
- clone+checkout 100% 재현. 패치 파일은 fallback 참고용만.

### C.5 robot/teleop 은 빈 스텁 (신규 구현)
- `custom/robots/*`, `teleoperators/inverse3` 2줄 placeholder. 실제 FR3 제어는 ROS2 스크립트에만 (Phase 9–11).

### C.6 UMI 데이터 fps mislabel (move260626) — Phase 6 처리
- raw 는 fps 혼재(30×144 + 32~60 산발), + non-mono 73(rgb 13), gap 다수, **pose jump 22**. `align` 리샘플 없이 fps=30 라벨 → 시간축 왜곡 → delta action 스케일 불일치.
- **대책(Phase 6)**: 인스펙터 검증 + `--tolerance-s` on + **명시 리샘플(30fps)** + 손상 제외(34) + 회전보정. → 216개 @ 30fps. 검증: 30fps-only ablation.

---

# 부록 D — 관측 어댑터 & canonical 계약 (env_processor vs robot_processor vs 컨버터)

## D.0 통일 원리
모든 경로엔 **"소스 ↔ canonical(=데이터셋 표현, 10D `[xyz, rot6d, gripper]`)" 어댑터**가 있고, **데이터 생산 시점 + 인퍼런스 시점 양쪽에서 같은 canonical 산출** = train==inference 계약.

### D.0.1 canonical 채널 규약 (모든 소스/싱크가 여기로 변환) — 정본: `schemas/canonical_ee10.py`
| 채널 | 의미 |
|---|---|
| `x,y,z` | EE 위치 [m] |
| `rot6d(6)` | 회전행렬 앞 두 열 flatten (Gram-Schmidt 로 복원, 3열=b1×b2). 회전 없음 = `IDENTITY_ROT6D`=`(1,0,0,0,1,0)` — **표현 상수**(어디서나 동일), *쓰느냐*만 embodiment 별 |
| `gripper` | **openness `[0,1]`, 0=닫힘 / 1=열림** (지배적 규약) |

- **통일은 canonical 층에서만.** 각 경계 규약은 우리가 못 바꾸므로 **어댑터가 번역**:

**그리퍼: 데이터셋·정책 입출력 전부 이진 `{0,1}`(0=닫힘, 1=열림), 변환 시 bake.** 런타임 프로세서 없음. 상세·근거: `retargeting.md` 4절.

> ⚠ `[-1,1]` 은 **우리 액션이 아니다** — metaworld `env.step()` 의 `spaces.Box(low=-1,high=1)` 강제 (lerobot `envs/metaworld.py:137`). **env 로 나가는 마지막 한 걸음에서만** 번역하며, 데이터셋에도 정책 입출력에도 그 값은 없다.

| 경계 | 원래 규약 | canonical(0=닫힘,1=열림) 대비 | 처리 |
|---|---|---|---|
| metaworld **obs**[3] | openness, **연속**(측정값), 1=열림 | **극성 일치**, 범위만 연속 | ★ **threshold 이진화**: `obs[3] >= thresh` → `1`, else `0`. 수집·rollout 이 **같은 함수**(`state4_to_canonical10`)를 지나 누락 불가 |
| metaworld **env.step() 입력** | closing effort `[-1,1]` (**API 강제**) | 극성 반대 | rollout 시 **마지막에만**: `effort = (0.5 − o) × 2` |
| **UMI raw** (`gripper/state/value`) | int32 **이진**, 파일 attr `state_rule: 0=close, 1=open` → **0=닫힘** | **완전 일치** | **그대로 통과**(float 캐스팅만, Phase 6) |

- ⚠ **threshold 는 태스크 의존적**: pick-place 실측 `obs[3]` 이중봉우리 ≈1.0(열림)/≈0.40~0.46(블록 쥠) → **0.7**. 물체 두께가 바뀌면 조용히 오분류 → 태스크별 실측 필수. **수집·rollout 동일 값** 아니면 train≠inference. 데이터셋에 bake 되므로 변경 시 **재수집**.
- ✅ `obs[3]` 은 **진짜 상태**라 rollout 에서 그대로 관측 가능 → 명령 추적·프레임 시프트 **불필요**. `action[t] = state[t+1]` 규칙 그대로.

- ⚠ **metaworld 는 obs 와 action 이 서로 반대**(비대칭). → canonical 이 특정 소스 종속이 아니라는 증거이기도 함: 3경계 중 2개가 뒤집기 필요하며, 어떤 규약을 골라도 일부 경계는 번역이 남는다.
- 뒤집기를 canonical 로 흘리면 그 소스의 규약이 데이터셋에 새어 다른 embodiment 와 어긋남 → **경계에 격리**.
### D.0.2 그리퍼 분포 실측 — ⚠ cross-embodiment 불일치
| 소스 | canonical openness 분포 | 근거 |
|---|---|---|
| **metaworld** (기존 pick_place_v3, **옛 방식**) | **연속** `[0.3955, 1.0]` — 고유값 611, 이중봉우리(≈1.0 열림 / ≈0.4 블록 쥠). obs[3] 을 이진화 없이 그대로 넣은 결과 | 데이터셋 state[9] 실측 |
| **UMI** (move260626) | **이진**, raw 가 전부 0(닫힘) → openness **전부 0** → 상수 채널(std=0) | h5 `gripper/state/value` 실측 |

- 이 불일치(연속 vs 이진)가 **이진화를 bake 하기로 한 이유**: 새 방식은 metaworld 도 `obs[3]` 을 threshold(0.7)로 이진화해 `{0,1}` 을 저장 → **두 소스가 디스크에서 일치**. 위 이중봉우리 실측이 곧 **threshold 근거**.
- → 기존 `pick_place_v3` 는 **옛 방식(연속) 산출물**이라 Step 3 에서 재수집 대상.
- 📌 metaworld 의 **rot6d 6채널 std=0** 실측 확인 — UMI 그리퍼 상수와 함께 **정보량 0 채널**. `meta.stats` 에서 std=0 이 나오는 건 정상(Phase 5 검증 시 참고).

| 인터페이스 | 어댑터 | canonical 산출 시점 | Robot 정의 |
|---|---|---|---|
| gym `Env` (metaworld) | **env_processor** | 수집 + rollout | ❌ |
| `Robot` (franka) | **robot_processor** | record + 인퍼런스 | ✅ |
| 혼합 (UMI→franka) | **컨버터 + robot_processor** | 변환 / 인퍼런스 | ✅(인퍼런스만) |

- 어댑터가 **한 개**(양쪽 동일)면 자동 일치, **두 개**(소스 다름)면 canonical 로 수동 일치.

## D.1 metaworld — env_processor (Robot 없음)
- gym Env(`make_env`), `robot_type: null`(Sawyer). 어댑터 = **`custom/envs/metaworld/canonical.py`** (`render_frame`/`state4_to_canonical10`/`canonical10_to_env_action`)
- 수집(Phase 1)·rollout(Phase 5) **같은 함수** → 자동 일치. `use_depth=False`. Robot 불필요.

## D.2 franka (real / sim-as-real) — robot_processor (Robot 정의)
- `Robot` 경로(실기 ROS2 or sim 을 Robot backend). 어댑터 = robot_processor(**robot↔canonical**, FK/IK)
- record·인퍼런스 같은 processor → 자동 일치. env_processor 아님.
- "sim franka 를 실제 인퍼런스처럼" = Robot 경로 → **Robot 정의 필요**(그게 목적).

## D.3 UMI → franka — 컨버터 + robot_processor (★ 다름)
cross-embodiment(소스=UMI, 배포=franka) → 어댑터 2개, canonical 로 수동 일치.
```
UMI 컨버터(오프라인)    = [UMI SLAM quat/rpy → transform] + [transform → canonical pose9d]
robot_processor(franka) = [franka FK        → transform] + [transform → canonical pose9d]
                              (embodiment별 앞단)              (뒷단 = 표현 codec, 공유)
```
- **변환(UMI)**: robot_processor **실행 안 함** — 컨버터는 **평범한 함수 + `add_frame`**(port_droid, Phase 1 에서 실증). 산출물이 인퍼런스가 재현할 계약.
- **인퍼런스(franka)**: robot_processor 가 `FK → canonical`.
- **일치 책임=우리** → 같은 표현 codec 을 쓰는 것으로 보장.

### ⚠ 폐기된 설계: `CanonicalPoseToState` 공유 step
초안에 있던 "컨버터와 robot_processor 가 `CanonicalPoseToState` **step** 을 공유" 는 **폐기**한다:
| 근거 | |
|---|---|
| lerobot_hong 에 **존재하지 않음** | `grep` 0건 — 우리가 만든 이름이었음 |
| **port_droid 패턴과 모순** | 컨버터는 ProcessorStep 을 **안 씀**(Phase 1 실증) → 오프라인/온라인이 step 을 공유할 수 없음 |
| **내용이 없음** | "pose+gripper → state" = `concatenate` 한 줄 |

**대체**: 공유 대상은 step 이 아니라 **표현의 codec(순수 함수)** — D.5. Phase 1 이 실증한 형태와 동일(`state4_to_canonical10` 은 함수이고, 프로세서는 그걸 감싸는 얇은 래퍼).

## D.4 Phase 매핑
| 항목 | Phase |
|---|---|
| env 어댑터 (metaworld) | 1(수집)·5(rollout) — ✅ 완료 |
| **표현 codec** (rot6d↔R, pose9d↔transform) | **2** (런타임 step 이 첫 소비자) |
| 런타임 relative step (`lerobot_policy_umidiffusion/steps.py`) | **2** (정책의 존재 이유 — dev_plan §12.1) |
| UMI 컨버터 (UMI→canonical) | 6 — codec **재사용** |
| robot_processor (franka FK/IK) | 9–11 (deferred) — codec 재사용 |

## D.5 표현 codec — 진짜 공유 지점 (lerobot_hong 의 중복을 고침)

**lerobot_hong 실측**: 같은 수학이 두 곳에 **다르게** 구현돼 있음
```python
# ↓ 아래 경로는 모두 lerobot_hong 의 것 — 우리 구조가 아니다. 개명 금지(기록 위조됨).
# custom/data_processing/umi2lerobot_delta_pose/se3.py (offline, numpy)
def relative_transform(base, target):  return np.linalg.inv(base) @ target   # 일반 역행렬

# custom/processors/robot_maps/.../steps.py:99 (online, torch)
def relative_transform(anchor, target): return invert_transform(anchor) @ target  # SE3 전용(R^T)
```
`transform_to_pose9d` · `rotation_matrix_to_rot6d` · `relative_transform` 3개가 중복이고, `raw_inspect.py` 의 `_rot6d_to_R` 까지 하면 **세 번째 복사본**. 게다가 se3.py 의 `transform_to_relative_pose9d`/`transform_to_delta_pose9d` 는 **pass-through 껍데기**(dev_plan §3.2 가 runtime 으로 옮기며 남은 잔해).
> 원인: dev_plan §8 이 *"runtime 에서 재사용할지는 **구현 시 결정**"* 으로 미뤘고 → 중복으로 귀결. **미루면 같은 결과가 난다.**

**우리 분해**:
| 무엇 | 어디 | 왜 |
|---|---|---|
| `rot6d↔R`, `pose9d↔transform`, `invert`, `relative` | **`schemas/` (표현 옆)** | "이 표현을 읽고 쓰는 법" = **표현의 일부**. 컨버터·런타임 step 둘 다 **대등한 소비자**라 어느 한쪽이 소유하면 나머지가 엉뚱한 패키지에 의존 |
| quaternion/rpy → transform | UMI 컨버터 | **UMI raw 파싱** = 소스 전용 |
| anchor 의미, relative/delta, Step 래퍼 | **`lerobot_policy_umidiffusion/steps.py`** | **런타임 윈도우 의미** = 런타임 전용 |

→ Phase 2 에서 codec 을 표현 옆에 두면 Phase 6 은 **import 만** 하면 되고 중복이 **구조적으로 불가능**. (torch 로 구현하면 오프라인에서도 CPU 로 그대로 사용 — lerobot 이 이미 torch 요구하므로 추가 비용 0)

## D.6 왜 `lerobot_canonical` 이 별도 배포판인가 (2026-07-17 결정)

**계기**: `lerobot-train --policy.type=mypolicy` 가 `invalid choice` 로 죽음. 진짜 원인은
`register_third_party_plugins()` 가 **import 에러를 삼켜서**(`except Exception: logging.exception`) 안 보였던
`ModuleNotFoundError: No module named 'custom'` — **설치된 배포판이 설치 안 된 경로(`custom/...`)를 import** 했다.

**강제되는 결론**: 정책은 배포판이어야 한다(`lerobot_policy_` 접두사 = 자동탐색 조건). 그러면
**정책이 import 하는 것도 전부 배포판이어야 한다.** 선택이 아니라 강제다.

**그럼 왜 정책 안에 넣지 않는가** — 반증 테스트:
> "데이터를 수집하려면 **정책을 설치**해야 하나?"

`collect_metaworld.py` 가 `from lerobot_policy_umidiffusion.schemas import ...` 를 하게 된다. 수집은 정책보다
**먼저** 존재하고, 같은 데이터셋에 ACT 를 붙이면 ACT 가 umidiffusion 을 거쳐 스키마를 가져온다. 화살표가 뒤집힌다.
스키마는 **쓰는 쪽(수집)과 읽는 쪽(정책)의 계약**이고, 계약이 당사자 중 한쪽 안에 살면 그건 계약이 아니다.

```
              lerobot_canonical          ← 아무에게도 의존 안 함 (lerobot, torch 만)
             ↗       ↑        ↖
        정책      env 어댑터    스크립트   ← 서로를 모른 채 여기서만 합의
```
실측 소비자 = 3범주 8곳 (`configuration_/processor_/steps` · `envs/metaworld/canonical.py` · `scripts/sim/collect_metaworld.py`).

**"정책도 env 도 robot 도 아닌데 뭐라 부르나"** → **lerobot 이 이미 `utils` 라 부른다.** 우리가 만든 범주가 아니다:
| 우리 | lerobot | 근거 |
|---|---|---|
| `keys.py` | `lerobot/utils/constants.py` | policies(41)·processor(9)·datasets(7)·scripts(6)·rl(6)·robots·envs… **10개 범주가 import** — 우리와 똑같은 사용 패턴 |
| `schemas/canonical_ee10_se3.py` | `lerobot/utils/rotation.py` | `class Rotation`, `from_matrix`, 쿼터니언 — 같은 종(種) |

**`common/` 을 안 쓰는 이유**: lerobot 도 `lerobot/common/` 이 있었고 **삭제했다**. 안에 datasets·envs·policies·
robot_devices·transport 가 **전부** 있었다 → 모든 걸 담으니 아무것도 구분 못 함 → `lerobot/common/*` → `lerobot/*` 로
평탄화. `custom/` 최상위를 `lerobot/` 최상위의 부분집합(`policies/` `envs/` `scripts/` `utils/`)으로 두면 읽는 사람이
새로 배울 게 없다.
> `utils/` = 잡동사니 아님. 이 코드베이스에서 `utils/constants.py` 는 정책 41개 파일이 물고 있는 **뼈대**다.
> 잡동사니화 위험도 우리가 더 낮다 — 여기 들어오려면 `.py` 가 아니라 **`pyproject.toml` 을 써야** 한다.

**이름이 `lerobot_policy_*` 가 아닌 것도 의도**: 자동탐색 접두사에 걸리면 라이브러리가 플러그인 행세를 한다.

**검증**(3중, 전부 통과): `/tmp` 에서 import ✓ · `lerobot-train --policy.type=umidiffusion` **완주**(263M) ✓ · 수동 import 불필요 ✓

### D.6.1 왜 `custom/envs/` 만 배포판이 아닌가 (의도된 예외)

`custom/envs/metaworld/canonical.py` 는 **모듈**이고 앞으로도 그렇다. 근거 3개:

1. **env 는 자동탐색 접두사가 아예 없다.** `prefixes = ("lerobot_robot_", "lerobot_camera_", "lerobot_teleoperator_", "lerobot_policy_")` — env 는 다른 메커니즘이다: `EnvConfig.register_subclass` + **`cfg.package_name` 을 `importlib.import_module`** (`envs/factory.py:200`) + gym registry. 즉 배포판으로 만들어도 자동으로 안 불린다.
2. **그럴 필요도 없다** — `@EnvConfig.register_subclass("metaworld")` 가 **lerobot 본체에 이미 있다**(`envs/configs.py:349`, `fps=80` 으로 우리 데이터셋과 일치). 우리 파일은 env **플러그인**이 아니라 **어댑터**다.
3. **아무 배포판도 이걸 import 하지 않는다.** 소비자는 스크립트(`scripts/sim/collect_metaworld.py`, 설치 안 됨 → `sys.path` 로 접근)와 Phase 5 rollout 스크립트뿐. D.6 의 규칙("설치된 배포판이 import 하는 건 전부 배포판")은 **위반되지 않는다**.

> 판정 기준: **설치된 배포판이 import 하면 배포판이어야 하고, 스크립트만 import 하면 모듈로 족하다.**
> Phase 5 에서 설치된 무언가가 이걸 물게 되면 그때 배포판으로 승격한다 (`pyproject.toml` 추가 = 5분).

## D.7 왜 diffusion 정책을 **통으로 복사**했나 (2026-07-17 결정)

**계기**: 2-2 를 설계하다가 `make_umidiffusion_pre_post_processors` 가 **한 번도 안 불린다**는 걸 발견.

### 원인 — 정책 발견과 프로세서 발견의 **비대칭**

| 무엇 | 어떻게 찾나 | 우리 결과 |
|---|---|---|
| **정책** `get_policy_class(name)` (`factory.py:61`) | `if name == "tdmpc" / elif "diffusion" / ...` = **이름 문자열** | `"umidiffusion" != "diffusion"` → 안 걸림 → `else: _get_policy_cls_from_policy_name` ✅ **2-1 이 성공한 이유** |
| **프로세서** `make_pre_post_processors` (`factory.py:296`) | `elif isinstance(policy_cfg, DiffusionConfig)` = **isinstance** | `UmiDiffusionConfig(DiffusionConfig)` → **True → 걸림** → lerobot 의 `make_diffusion_pre_post_processors` 실행 ✗ |

`else` 폴백(`_make_processors_from_policy_config`)에 **영원히 도달하지 못한다.**

**실측 증거**: 우리 함수 본문이 `...`(None 반환)인데도 `make_pre_post_processors(cfg)` 가
`[Rename, AddBatch, Device, Normalizer]` 를 **정상 반환**했다 → 우리 함수는 한 번도 안 불렸다.

### 왜 위험한가 — **조용하다**

2-2 를 구현하고 `--steps=1` 을 돌리면 **통과한다**(lerobot 의 diffusion 프로세서가 대신 일하므로).
2-3 에서 우리 step 을 넣어도 **안 불린다**. 2-5 에서 anchor-relative 수학을 채워도 **안 돈다**.
정책은 계속 **절대 canonical** 을 보며 학습하고 **아무 에러도 안 난다.**
`register_third_party_plugins` 가 import 에러를 삼킨 것, `reset(seed=)` 이 조용히 버려진 것과 **같은 종류**.

### lerobot_hong 의 답 = 패치 (같은 벽에 부딪혔다)

```python
# lerobot_hong  lerobot/src/lerobot/policies/factory.py:325  (v0.4.4, 우리와 동일 버전)
    if   isinstance(policy_cfg, TDMPCConfig): ...
    elif getattr(policy_cfg, "type", None) == "mypolicy":       # ← 13줄 삽입
             from lerobot_policy_mypolicy import make_mypolicy_pre_post_processors
             ...
    elif isinstance(policy_cfg, DiffusionConfig): ...           # ← 이 앞에 놓아야 이김
```
`MyPolicyConfig(DiffusionConfig)` 를 유지한 채 **isinstance 분기 앞에 자기 분기를 밀어넣어** 뚫었다.

**우리가 이 길을 안 가는 이유**:
1. **버전관리가 안 된다** — `.gitignore:4 /lerobot/`. 패치는 이 머신에만 산다
2. **새 머신에서 조용히 재발** — README 대로 `git clone && checkout v0.4.4` 하면 패치 없는 factory 가 오고,
   isinstance 가 다시 가로채고, **에러 없이 잘못된 정책이 학습된다**
3. **배포 불가** — 남이 `pip install` 해도 안 된다. "factory 도 패치하세요" 를 알려줘야 한다
4. `refactoring.md` Phase 8 의 **"`git -C lerobot diff` 비어있음"** 검증이 거짓이 된다

### 공식 규약이 답이었다 (검색·문서·예제 전부 일치)

| 출처 | 내용 |
|---|---|
| `docs/source/bring_your_own_policies.mdx` Step 2 | *"Create a configuration class that inherits from **`PreTrainedConfig`**"* |
| 같은 문서 Step 3 | *"inheriting from LeRobot's base **`PreTrainedPolicy`** class"* |
| 공식 예제 [lerobot_policy_ditflow](https://github.com/danielsanjosepro/lerobot_policy_ditflow) (문서가 직접 링크) | `class DiTFlowConfig(PreTrainedConfig)` — **DiT+flow = diffusion 계열인데도** DiffusionConfig 상속 안 함 |
| 웹 검색 | 이 문제의 이슈·질문이 **하나도 없다** = **아무도 기존 정책 config 를 상속하지 않는다**. 다들 규약대로 해서 `else` 폴백을 탄다 |

→ **lerobot 의 버그가 아니라 lerobot_hong 의 규약 이탈**이었고, 우리가 그걸 검토 없이 베꼈다.
`isinstance` 분기는 내장 정책용 **지름길**이고, 서드파티는 `else` 를 타는 게 설계 의도다
(*"LeRobot discovers your processor by name"*).

### 우리 결정: config **와** modeling 을 통으로 복사

```
configuration_umidiffusion.py   ← configuration_diffusion.py (259줄) 복사 + 개명 + 우리 필드
modeling_umidiffusion.py        ← modeling_diffusion.py     (784줄) 복사 + 개명 + depth 게이트
결과: UmiDiffusionConfig(PreTrainedConfig) + UmiDiffusionPolicy(PreTrainedPolicy)  = 공식 규약 100%
```

**config 복사는 버그가 강제한다**(안 하면 프로세서가 안 불림). **modeling 복사는 별개의 결정**이고
근거는 하나다 — **앞으로 정책 구조 자체를 수정할 계획**. 그 계획이 없다면 784줄은 부채일 뿐이었다
(지금 계획상 고칠 줄은 0이고, depth 게이트조차 config 로 처리된다).

**안전한 이유**(전부 실측):
| 확인 | 결과 |
|---|---|
| `isinstance(.*DiffusionConfig)` | lerobot 전체에서 **`factory.py:296` 딱 한 곳** (= 우리가 피하려는 그 줄) |
| 나머지 `DiffusionConfig` 참조 | import·재수출·**타입 힌트**·주석뿐 → 런타임 강제 없음 |
| `modeling_diffusion.py` 의 lerobot 의존 | `PreTrainedPolicy`, `policies.utils`, `utils.constants` — **전부 공개 API**, private 0개 |
| `config_class` | 존재 여부만 검사(`pretrained.py:65`), 타입 일치 검사 없음 |

**검증**(전부 통과):
```
① isinstance(cfg, lerobot.DiffusionConfig) == False       (MRO: UmiDiffusionConfig → PreTrainedConfig)
② make_pre_post_processors(cfg) → ★ TypeError 로 터짐      ← 우리 함수가 불렸다는 증거 (본문이 아직 `...`)
③ 필드 누락 0 · 기본값 불일치 0 (normalization_mapping 만 의도적 차이)
④ 파라미터 263,196,458 — 2-1 과 **한 자리도 안 틀림** = 복사가 충실
⑤ modeling diff 76줄 = 전부 의도한 것 (헤더·import·클래스명·depth 게이트·별칭)
```
**②가 핵심이다** — "안 터지면 성공"이 아니라 **"터져야 성공"**인 구간이 있어야 2-3~2-5 가 또 조용히 죽지 않는다.

### 대가와 관리법

- **1,043줄이 lerobot 과 중복**된다. 두 파일 맨 위에 **출처 블록**(원본 경로·`v0.4.4`·원본과의 차이 목록·재동기화법)을 박아뒀다
- lerobot 업그레이드 시 **우리가 수동으로 따라가야** 한다 → `v0.4.4` 핀(README)이 있으니 **우리가 정할 때만** 일어난다
- 재동기화: `diff <lerobot>/src/lerobot/policies/diffusion/modeling_diffusion.py <ours>` → 출처 블록의 "원본과의 차이" 만 나와야 정상
- **내부 부품 클래스는 원본 이름 유지**(`DiffusionModel`, `DiffusionRgbEncoder`, …) — 우리 모듈 안에만 살아 충돌이 없고,
  diff 를 최소로 유지해야 "원본 대비 내가 뭘 바꿨나" 가 보인다. 구조를 실제로 고칠 때 개명한다
