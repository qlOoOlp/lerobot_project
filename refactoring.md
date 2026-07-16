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
| **1** | 데이터 | ext_core(keys/dims) + metaworld→LeRobotDataset | 무수정 |
| **2** | 정책 | mypolicy config/modeling/processor + depth 게이트 | 무수정 |
| **3** | observer | processor pre/post + **CanonicalPoseToState 공유 step** | 무수정 |
| **4** | 학습 | lerobot-train (metaworld dataset) | 무수정 |
| **5** | eval | metaworld rollout (env_processor) | 무수정 |
| **── UMI 확장 (코어 검증 후) ──** |
| **6** | UMI 데이터 | raw 인스펙터(✅) + umi2lerobot 컨버터(공유 step 재사용) | 무수정 |
| **7** | UMI 오프라인 추론 | runtime buffer/sync, 녹화 관측 vs GT | 무수정 |
| **8** | 정리 | 무수정 확인·setup.sh 최종화 | 무수정 |
| **── 이후 (deferred) ──** |
| 9–11 | robot(franka) + real-world env + ROS2 | — | — |

> **lerobot 전 Phase 무수정.** 등록=플러그인 폴백, depth=`MyPolicyConfig.apply_depth_gate()` → `datasets/policies factory.py` 패치 **불필요**(Phase 2). 패치 파일(`patches/`)은 fallback 참고용.
> 관측 어댑터(env_processor vs robot_processor vs 컨버터)와 canonical 계약은 **부록 D** 참조 (핵심 설계).

---

# Phase 0 — 뼈대 (환경 부트스트랩) ✅ 완료

lerobot 을 clone 해 `v0.4.4` 로 고정하고, 깨끗한 conda env 만 세운다. (lerobot 패치 없음 — 전 Phase 무수정)

### 디렉토리 배치
```
lerobot_share/
├── lerobot/            # HF lerobot v0.4.4 (clone, detached HEAD)
├── custom/             # lerobot 플러그인/변환 코드 (ROS2 제외)
├── tools/              # 독립 유틸 (raw_inspect.py 등)
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
- [ ] `custom/{common,policies,processors,data_processing}` 빈 골격 (필요 Phase 에서)
- [ ] (선택) `setup.sh`

---

# ══════ 코어 (metaworld) ══════

# Phase 1 — 데이터 (metaworld → LeRobotDataset)

**목표**: metaworld env 데이터를 **LeRobotDataset(canonical 10D)** 으로 만들며 LeRobotDataset(create/add_frame/features/stats)을 이해. sim 이라 깨끗 → 필터/리샘플/회전 불필요.

**읽을 것**: `lerobot-dataset-v3.mdx`, **`porting_datasets_v3.mdx` + `examples/port_datasets/port_droid.py`**(로봇 없이 `create`+`add_frame` 변환 정석 ⭐) / 소스 `datasets/lerobot_dataset.py`(`create`/`add_frame`/`save_episode`/`meta`), `configs/types.py`.

**스키마 (canonical 10D, `use_depth=False`)**:
```
observation.images.rgb : (240,240,3)                       ← depth 없음
observation.state      : (10,) [x,y,z, rot6d(6), gripper]  ← UMI 와 동일
action                 : (10,) 같은 10D (절대 target = 다음 frame state)
fps 80
```

### 세부 작업
1. [x] **ext_core (최소)** ✅: `common/lerobot_ext_core/{keys,canonical_ee10}.py`
   - `keys.py`: lerobot `utils/constants.py`(`OBS_IMAGES`,`OBS_STATE`,`ACTION`) 위에 `image_key()`/`RGB_KEY`/`DEPTH_KEY`/`STATE_KEY`/`ACTION_KEY` — 전부 상수 **파생**(문자열 하드코딩 0). 표현 무관 → 모든 표현이 공유
   - `canonical_ee10.py`: dims/axes (`POSE_DIM=9`, `GRIPPER_DIM=1`, `STATE_DIM=10`, rot6d `STATE_AXES`). **표현 하나**임을 이름으로 명시(구 `schemas.py`)
     - schema 는 **로봇이 아니라 표현(EE-Cartesian pose+gripper)에 종속** → metaworld/Sawyer·franka·UMI 가 공유. **로봇 교체 = 재사용**, **표현 변경 = 새 모듈**
     - 새 표현(관절 등)은 **flat sibling** 추가(`canonical_joint7.py`), 기존 모듈 불변(open/closed). 표현이 여러 개로 번지면 그때 `schemas/` 서브패키지로 승격
     - 대외 계약(모든 표현 모듈 공통): `STATE_DIM`/`STATE_AXES`/`ACTION_DIM`/`ACTION_AXES` · 내부 사정: `POSE_DIM`/`POSE_AXES`/`GRIPPER_*` → 다운스트림은 `sch.STATE_DIM` 식으로 **모듈 import** 해 표현만 갈아끼움
   - ★ **feature 빌더·정책 feature 는 여기 두지 말 것** — 정책은 데이터셋에서 파생(`dataset_to_policy_features`), feature dict 는 변환기가 정의 (lerobot 정석, 부록 D)
2. [ ] **metaworld env↔canonical 매핑 포팅**: `metaworld_canonical.py`(`state4_to_canonical10`/`canonical10_to_env_action`) — 이게 **env_processor 역할**, 수집·rollout(Phase 5) 공유 (부록 D.1)
3. [ ] **수집/변환 → LeRobotDataset**: `collect_metaworld_canonical.py`(in-env 수집) 또는 `convert_metaworld_canonical.py`(`lerobot/metaworld_mt50` 변환). 수동 features(canonical 10D) + `LeRobotDataset.create` + `add_frame` 루프 + `save_episode` (**port_droid 패턴, Robot 없음**)
4. [ ] `pip install metaworld==3.0.0` (mujoco 자동)
5. [ ] **검증**: 로드 → `meta.fps==80`/`meta.features`/`meta.stats`, 샘플 gif, 기존 `metaworld_canonical/pick_place_v3` 와 스키마 대조
   - ★ **인스펙터 재사용**: `raw_inspect.py --format lerobot_dataset --raw-root <dataset> [--target-fps 80]` → 만든/기존 LeRobotDataset 을 UMI 와 **같은 리포트**(dim·fps·pose-jump)로 sanity 검사. (검증됨: `pick_place_v3` 50ep 80fps, 10D state/action, pose:state rot6d, no-warning). 상세: `information.md` §1.1

> **핵심**: metaworld 는 gym Env 경로 → **Robot·robot_processor 불필요.** env↔canonical 매핑이 계약이고 수집·rollout 양쪽에서 같은 함수 → 자동 일치 (부록 D.1).

---

# Phase 2 — 정책 (lerobot 무수정)

**목표**: mypolicy 를 BYO Policy 규칙대로 만들고 `make_policy(type="mypolicy")` 인스턴스화. lerobot 무수정(등록=플러그인 폴백, depth=config 게이트). **metaworld 는 `use_depth=False`** → depth 게이트 off 경로를 첫 테스트로.

**BYO Policy 규칙** (`lerobot_policy_mypolicy`):
- config `@PreTrainedConfig.register_subclass("mypolicy")`, `MyPolicyConfig(DiffusionConfig)`
- class `MyPolicyPolicy(DiffusionPolicy)`, `name="mypolicy"`
- processor `make_mypolicy_pre_post_processors`
- **자동 탐지(무수정)**: `_get_policy_cls_from_policy_name` + `_make_processors_from_policy_config` 폴백이 컨벤션으로 찾음.

### 세부 작업
1. [ ] `configuration_mypolicy.py`: register_subclass, `MyPolicyConfig`, `use_depth` + `apply_depth_gate()` (information.md §3.1)
2. [ ] `modeling_mypolicy.py`: `MyPolicyPolicy(DiffusionPolicy)`, `__init__` 에서 `super()` 전 `config.apply_depth_gate()`
3. [ ] `processor_mypolicy.py`: `make_mypolicy_pre_post_processors`(앞에 `apply_depth_gate()`) + `__init__` 노출
4. [ ] 설치: `pip install -e custom/policies/mypolicy/lerobot_policy_mypolicy`
5. [ ] **검증**: `make_policy(type="mypolicy", ds_meta=...)` + 더미 forward + `use_depth` on/off 둘 다

### 왜 lerobot 무수정 (patch 2파일 대체)
- 등록/프로세서 연결: 원래도 패치 불필요(플러그인 폴백)
- 패치 진짜 이유 = `use_depth` depth 필터 → **`apply_depth_gate()`(config)로 이전**: `use_depth=False` 면 `input_features` 에서 depth 제거 → 모델이 depth 인코더 안 만듦 → `policies/factory.py` 패치 불필요
- `datasets/factory.py` 패치는 "depth 미로드 효율"뿐 → 생략(로드되나 무시)
- → **패치 0파일 = 완전 무수정.** 상세: `information.md` §3.1

---

# Phase 3 — observer (processor) + 공유 step (lerobot 무수정)

**목표**: 정책 pre/post 프로세서를 `ProcessorStep`+`Pipeline` 로 구현. **`CanonicalPoseToState` 공유 step 을 여기서 정의** — metaworld env_processor·(나중)UMI 컨버터·franka robot_processor 가 **전부 재사용**하는 계약 (부록 D).

**쓰는 lerobot API** (v0.4.4 확인): `ProcessorStep`(6메서드: `__call__`/`transform_features`/`get_config`/`state_dict`/`load_state_dict`/`reset`), base 4종(`Observation/Action/RobotAction/ProcessorStep`), pipeline 2종(`PolicyProcessorPipeline`/`RobotProcessorPipeline`), feature contract(`create_initial_features`→`aggregate_pipeline_dataset_features`→`combine_feature_dicts`), 등록(`@ProcessorStepRegistry.register`).

### 세부 작업
1. [ ] **`CanonicalPoseToState` step** (`transform_features` 로 `observation.state`=(10,) 선언) — **공유 계약 step**(부록 D.3). metaworld/UMI/franka 앞단만 갈아끼움
2. [ ] `make_mypolicy_pre_post_processors`: pre `Rename→AddBatch→Device→Normalizer`, post `Unnormalizer→Device(cpu)` (앞에 `apply_depth_gate()`)
3. [ ] 커스텀 Step 에 `@ProcessorStepRegistry.register` 부여
4. [ ] `processors/{common,robot_maps,teleop_maps}` Step 포팅 (robot_maps 의 `CanonicalPoseTo...` 재사용)
5. [ ] `pip install scipy` + processor 패키지 editable
6. [ ] **검증**: `pre(sample)→select_action→post(action)` 오프라인 end-to-end

> 3층 파이프라인(부록 D): **Policy**(배치) · **Env**(metaworld, Phase 5) · **Robot**(franka, deferred). `CanonicalPoseToState` 는 셋의 공통 뒷단.

---

# Phase 4 — 학습 (metaworld dataset)

**목표**: mypolicy 를 `lerobot-train` 에 연결, metaworld 데이터로 소량 overfit 검증.

### 세부 작업
1. [ ] train config: `n_obs_steps`, `horizon`, `use_depth=False`, dataset root (delta_timestamps=`index/fps` 자동)
2. [ ] `lerobot-train --policy.type=mypolicy --dataset...`
3. [ ] 소량 overfit + `pip install matplotlib`
4. [ ] **검증**: loss 하강, 체크포인트 저장/로드

---

# Phase 5 — eval (metaworld rollout) — 코어 루프 완성

**목표**: 학습 정책을 metaworld 에서 rollout. **env_processor(=canonical 매핑)** 로 온라인 관측을 학습 데이터와 일치시킴.

### 세부 작업
1. [ ] `make_env`(metaworld gym) 구성
2. [ ] **env_processor**: `metaworld_canonical` 매핑(Phase 1 과 동일 함수) → env obs→canonical, canonical action→env 4D (`make_env_pre_post_processors` + `ObservationProcessorStep` 정석화)
3. [ ] rollout 루프: `env.step()→env_proc→policy_pre→policy→policy_post→env_proc→env.step()`
4. [ ] **검증**: 1 에피소드 rollout → 성공률/영상. **여기까지 = metaworld 코어 루프(데이터→학습→eval) 완성**

> Robot 없음. 오프라인(Phase 1 수집)·온라인(rollout)이 **같은 canonical 매핑** → 자동 일치 (부록 D.1).

---

# ══════ UMI 확장 (metaworld 코어 검증 후) ══════

# Phase 6 — UMI 데이터 (raw → LeRobotDataset)

**목표**: UMI Record3D h5 를 **같은 canonical 10D** LeRobotDataset 으로 변환. metaworld 와 스키마 동일 → **Phase 2–5 정책·학습 그대로 재사용**. 단 UMI raw 는 지저분 → **인스펙터 검증 + 필터/리샘플/회전** 필요.

**차이(부록 D.3)**: metaworld 는 env↔canonical 자동일치였지만, UMI→franka 는 **cross-embodiment** → 컨버터(UMI→canonical)와 (나중)franka robot_processor 가 **공유 `CanonicalPoseToState` step**(Phase 3)으로 수동 일치.

### 세부 작업
1. [x] **raw 인스펙터** ✅ → `tools/data_processing/raw_inspect.py` (single file, `--format {record3d_h5, lerobot_dataset}`). **여기선 `record3d_h5`** 로 UMI raw 검사: fps 분포·차원·품질(non-mono/gap/**pose jump**) 리포트 + `outputs/.../*.json`(warning별 에피소드 번호). (`lerobot_dataset` 포맷은 Phase 1 metaworld 검증에서 재사용.) 상세: `information.md` §1.1
2. [ ] 인스펙터로 raw 검증 → **목표 fps 결정** + skip 목록 파악 (`move260626_preprocess.txt` 참고)
3. [ ] **umi2lerobot 컨버터 (config-driven)**: `reader/align/se3/build_dataset` + `UmiToLeRobotConfig`(dataclass↔YAML, draccus). 처리:
   - **UMI SLAM → canonical pose → 공유 `CanonicalPoseToState` step** (Phase 3 재사용)
   - **30fps 리샘플**(nearest RGB + interp/slerp pose) + **timestamp 정렬·중복제거**(non-mono 복구)
   - **RGB/depth CCW 90° 회전**(upright, shape 720×960 / 192×256) — `move260626_preprocess.txt`
   - **`skip_episodes`**(pose_jump ∪ rgb_nonmono)
4. [ ] **브릿지**: inspector `issues.json` → `config.yaml` 자동(`skip`, `target_fps`) → 사람 리뷰
5. [ ] 설치: `pip install h5py` + 컨버터 editable
6. [ ] 변환(1 에피소드부터) → **검증**: 로드, `meta.features`/`stats`, 샘플 gif(방향·pose 연속성)

> 결정 규칙(부록 C.6): 손상(pose_jump∪rgb_nonmono) **제외** / fps 이탈 **리샘플 회수** / depth·gripper non-mono **정렬 흡수** / gaps **관용**. move260626: 34개 제외 → 216개 @ 30fps.

---

# Phase 7 — UMI 오프라인 추론 검증 (실기·robot 불필요)

**목표**: 녹화 관측 시퀀스로 UMI 추론 경로를 검증(live 로봇 없이). 실시간 버퍼링·action chunking·delta-pose 이해.

**참고**: 추론 관측 조립 = **학습 데이터와 같은 계약**(dt=1/학습fps, 정렬, canonical 표현) 재현. 오프라인 `align`↔온라인 `runtime_sync` 는 같은 math(slerp/interp/nearest). async(비동기 수집) vs sync 는 계약과 무관한 엔지니어링 선택.

### 세부 작업
1. [ ] `runtime_buffer`(history deque) + `runtime_sync`(비동기 multi-stream sync) 포팅
2. [ ] `real_inference_util`, `umi_fr3_transforms` 포팅
3. [ ] `offline_mypolicy_inference` 포팅 (녹화 관측 재생)
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
| **9** | robot(franka) 추상 | `lerobot_robot_franka`: `Robot` 10-메서드 + **robot_processor**(franka FK → canonical → **공유 `CanonicalPoseToState` step**). mock/sim 으로 단독 테스트 | 부록 D.2, `integrate_hardware.mdx` |
| **10** | real-world env | 배포 루프 `robot.get_observation()→policy→robot.send_action()`. Phase 7 오프라인을 live 로 승격 | `processors_robots_teleop.mdx` |
| **11** | ROS2 전송 이식 | `run_mypolicy_ros2.py`(932줄)/`grace_fr3_bridge.py`(418줄) 로직을 franka `Robot` 내부 ROS2 I/O 로 | 별도 ROS2 레이어 |

**robot 층 = "인터페이스"와 "ROS2 전송" 분리**: `Robot` 10-메서드 계약이 이음새. `lerobot_robot_` 접두사 + `@RobotConfig.register_subclass` → 자동 탐지(무수정). robot_processor 는 **Phase 3 의 공유 `CanonicalPoseToState` step 을 재사용**하고 앞단(FK)만 franka 용 → UMI 데이터와 canonical 로 일치 (부록 D.3).

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
- 깨끗한 env 의 `pip install -e lerobot` 이 정상 버전(0.35.x) 설치. mypolicy 정상.

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

| 인터페이스 | 어댑터 | canonical 산출 시점 | Robot 정의 |
|---|---|---|---|
| gym `Env` (metaworld) | **env_processor** | 수집 + rollout | ❌ |
| `Robot` (franka) | **robot_processor** | record + 인퍼런스 | ✅ |
| 혼합 (UMI→franka) | **컨버터 + robot_processor** | 변환 / 인퍼런스 | ✅(인퍼런스만) |

- 어댑터가 **한 개**(양쪽 동일)면 자동 일치, **두 개**(소스 다름)면 canonical 로 수동 일치.

## D.1 metaworld — env_processor (Robot 없음)
- gym Env(`make_env`), `robot_type: null`(Sawyer). 어댑터 = `metaworld_canonical.py`(env state↔canonical)
- 수집(Phase 1)·rollout(Phase 5) **같은 함수** → 자동 일치. `use_depth=False`. Robot 불필요.

## D.2 franka (real / sim-as-real) — robot_processor (Robot 정의)
- `Robot` 경로(실기 ROS2 or sim 을 Robot backend). 어댑터 = robot_processor(**robot↔canonical**, FK/IK)
- record·인퍼런스 같은 processor → 자동 일치. env_processor 아님.
- "sim franka 를 실제 인퍼런스처럼" = Robot 경로 → **Robot 정의 필요**(그게 목적).

## D.3 UMI → franka — 컨버터 + robot_processor (★ 다름)
cross-embodiment(소스=UMI, 배포=franka) → 어댑터 2개, canonical 로 수동 일치.
```
robot_processor(franka) = [franka FK → canonical pose] + [CanonicalPoseToState step]
UMI 컨버터              = [UMI SLAM  → canonical pose] + [CanonicalPoseToState step]  ← 공유!
                              (embodiment별 앞단만 다름)      (뒷단 = 공유 계약 step)
```
- **변환(UMI)**: robot_processor **실행 안 함** — UMI 컨버터가 `UMI SLAM→canonical`. 뒷단 `CanonicalPoseToState` 는 franka 와 **공유**. 컨버터 산출물 = 인퍼런스가 재현할 계약.
- **인퍼런스(franka)**: robot_processor(`franka FK→canonical`) + 공유 step.
- **일치 책임=우리** → 같은 canonical + 공유 step 으로 수동 보장.
- 정정: "변환 때 robot_processor 를 이어준다" ❌ → "변환은 UMI 컨버터로, 인퍼런스는 robot_processor 로, **각자 같은 canonical 생성 + 공유 step 재사용**" ✅

## D.4 Phase 매핑
| 항목 | Phase |
|---|---|
| env_processor (metaworld) | 1(수집)·5(rollout) |
| **`CanonicalPoseToState` 공유 step** | **3** (observer) — env/컨버터/robot 전부 재사용 |
| UMI 컨버터 (UMI→canonical) | 6 |
| robot_processor (franka FK/IK) | 9–11 (deferred) |
