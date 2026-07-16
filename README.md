# lerobot_share

[HuggingFace `lerobot`](https://github.com/huggingface/lerobot) 위에 올린 커스텀 로봇 학습 스택.
정책(policy) / 환경(env: metaworld·real) / 로봇(robot: franka) 을 **플러그인**으로 분리하고,
**lerobot 본체는 완전 무수정(0-patch)** 으로 유지한다. metaworld 로 코어 루프(데이터→학습→eval)를
먼저 완성하고 같은 canonical 스키마(10D `[xyz, rot6d, gripper]`)로 UMI 를 확장한다.

> 설계·Phase 계획: [refactoring.md](refactoring.md) · embodiment↔canonical 번역 계약: [retargeting.md](retargeting.md) · 구현 동작 상세: [information.md](information.md)

**진행 상황**: Phase 0(환경) ✅ · **Phase 1(데이터) ✅** (수집 코드 — canonical 10D, 이진 그리퍼, 80fps) ·
**Phase 2(정책) 진행 중** — 2-0/2-1 ✅, 다음 = 2-2 파이프라인 조립
> ⏳ 데이터셋은 **재수집 필요**: 옛 산출물이 시딩 함정(`reset(seed=)` 무시)으로 재현 불가라 삭제했다.
> 코드는 고쳐졌고 `--seed-base` 가 이제 실제로 작동한다 — 수집 명령은 refactoring.md Phase 1 참고.

## 디렉토리 구조

`custom/` 의 최상위는 **`lerobot/` 본체의 최상위를 그대로 거울처럼 따른다**(`policies/` `envs/` `scripts/` `utils/`)
— 읽는 사람이 새로 배울 구조가 없도록.

```
lerobot_share/
├── lerobot/                    # HF lerobot v0.4.4 — 별도 clone (git 무시, 벤더링 안 함)
├── custom/                     # lerobot 플러그인/확장 코드
│   ├── utils/lerobot_canonical/            ★ 배포판 — 공유 어휘 (lerobot/utils/ 대응)
│   │   └── src/lerobot_canonical/
│   │       ├── keys.py         #   데이터셋 키 (lerobot 상수에서 파생) ≈ utils/constants.py
│   │       └── schemas/        #   표현별 모듈 하나씩 (재노출 없음)
│   │           ├── canonical_ee10.py      #   EE-pose 10D [xyz, rot6d, gripper] — 치수·축
│   │           └── canonical_ee10_se3.py  #   그 표현의 codec (rot6d↔R, pose9d↔T) ≈ utils/rotation.py
│   ├── policies/umidiffusion/lerobot_policy_umidiffusion/   ★ 배포판 — 플러그인 (자동탐색)
│   │   └── src/lerobot_policy_umidiffusion/
│   │       ├── configuration_umidiffusion.py  #   UmiDiffusionConfig  @register_subclass
│   │       ├── modeling_umidiffusion.py       #   UmiDiffusionPolicy(DiffusionPolicy)
│   │       ├── steps.py                       #   런타임 anchor-relative 변환 (순수 로직)
│   │       └── processor_umidiffusion.py      #   pre/post 파이프라인 조립
│   ├── envs/metaworld/canonical.py   # env 어댑터 — 수집·rollout 공유(train==inference)
│   └── scripts/                # 실행 스크립트는 전부 여기
│       ├── data_processing/raw_inspect.py     #   raw 데이터 인스펙터 (자기완결)
│       └── sim/collect_metaworld.py           #   수집 (port_droid 패턴, Robot 없음)
├── patches/                    # fallback 참고용 (기본은 무수정 → apply 하지 않음)
├── outputs/                    # 실행 산출물 (로그·리포트) — git 무시
├── tmp/real/                   # 검증 산출물 (gif 등) — git 무시
├── refactoring.md              # 계획·Phase 순서·아키텍처
├── retargeting.md              # embodiment ↔ canonical 번역 계약
└── information.md              # 구현 동작 상세
```

**패키지 2개의 역할이 다르다**:
- `lerobot_canonical` = **라이브러리**. 정책·env·스크립트가 서로를 모른 채 합의하는 어휘. lerobot 의 자동탐색
  접두사에 **일부러 안 걸리게** 지었다(플러그인이 아니므로).
- `lerobot_policy_umidiffusion` = **플러그인**. `lerobot_policy_` 접두사 덕에 `register_third_party_plugins()`
  가 설치된 배포판을 훑어 **자동 import** → `--policy.type=umidiffusion` 이 lerobot 패치 없이 뜬다.

## 사전 준비

- Linux + conda (miniconda/anaconda)
- NVIDIA GPU + 드라이버(CUDA). 아래 명령은 **RTX 5090(Blackwell)** 기준 — 다른 GPU 는 torch 인덱스만 조정.

## 환경 세팅

### 1) 이 저장소 clone
```bash
git clone <this-repo-url> lerobot_share
cd lerobot_share
```

### 2) lerobot 본체 clone + 버전 고정 (v0.4.4)
lerobot 은 이 repo 에 **포함하지 않는다**(.gitignore). 아래로 직접 받는다.
```bash
git clone https://github.com/huggingface/lerobot.git
git -C lerobot checkout v0.4.4          # commit 8fff0fde
```
> **무수정 원칙**: lerobot 소스는 건드리지 않는다. 커스텀 정책/프로세서는 플러그인 규칙
> (`lerobot_policy_<name>` 접두사, `register_subclass` 폴백)으로 연결되므로 패치가 필요 없다.
> `patches/` 는 참고용이며 apply 하지 않는다.

### 3) conda 환경 생성
```bash
conda create -n lerobot_hong2 python=3.10 -y
conda activate lerobot_hong2
```

### 4) torch 설치 (GPU/CUDA 에 맞춰)
lerobot 은 `torch<2.11` 을 요구하고, RTX 5090(Blackwell)은 `cu128+` 이 필요하다 → 둘 다 만족:
```bash
pip install "torch==2.10.*" "torchvision==0.25.*" --index-url https://download.pytorch.org/whl/cu128
```
> 다른 GPU: `torch<2.11` 범위에서 자신의 CUDA 에 맞는 `--index-url` 을 선택.
> **bare `pip install torch` 금지** (버전 캡·CUDA 를 못 맞춤).

### 5) lerobot editable 설치
```bash
pip install -e lerobot
```

### 6) 커스텀 패키지 editable 설치
**의존 순서대로** (라이브러리 → 플러그인). `--no-deps` 는 이미 깐 torch 를 PyPI 판으로 덮어쓰지 않기 위함:
```bash
pip install -e custom/utils/lerobot_canonical --no-deps
pip install -e custom/policies/umidiffusion/lerobot_policy_umidiffusion --no-deps
```
> **editable 이라 코드 수정은 재설치 불필요.** 단 `pyproject.toml` 변경이나 **디렉토리 이동** 시에는
> 재설치해야 한다 — editable 은 설치 시점의 **절대경로**를 박아두므로 옮기면 옛 경로를 계속 가리킨다.

### 7) 검증
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.10.x+cu128  True
python -c "import lerobot; print(lerobot.__version__)"                          # 0.4.4
# 플러그인 자동탐색 — repo 밖(예: cd /tmp)에서 실행해도 떠야 정상
lerobot-train --help | grep -A2 "policy.type"                                   # umidiffusion 이 보임
```

## 커스텀 코드 실행

- **어느 디렉토리에서 실행해도 된다.** 두 패키지가 설치된 배포판이라 `cd` 위치와 무관하게 import 된다
  (스크립트는 필요 시 스스로 `PROJECT_ROOT` 를 `sys.path` 에 넣는다).
- 정책은 **이름만 대면 뜬다** — `lerobot_policy_` 접두사를 `register_third_party_plugins()` 가 자동 탐색:
  ```bash
  lerobot-train --policy.type=umidiffusion --policy.push_to_hub=false \
      --dataset.repo_id=local/x --dataset.root=~/datasets/metaworld_canonical/pick_place_v3_bin
  ```
  > `--policy.push_to_hub=false` 는 **필수**다 (`configs/train.py:138` — hub 업로드가 기본값).
- 추가 의존성은 Phase 별로 설치: `metaworld==3.0.0`(env), `scipy`(processor), `h5py`(UMI raw) 등 — refactoring.md 부록 B.

예) raw 데이터 인스펙터 / metaworld 수집:
```bash
python custom/scripts/data_processing/raw_inspect.py --raw-root <dataset> --format lerobot_dataset
python custom/scripts/sim/collect_metaworld.py --output-root ~/datasets/metaworld_canonical --num-episodes 300
```

## 트러블슈팅

- **`--policy.type=umidiffusion` 이 `invalid choice` 로 죽을 때** — ★ 진짜 원인은 화면에 안 나온다.
  `register_third_party_plugins()`(`lerobot/utils/import_utils.py:152`)가 플러그인 import 실패를
  **`except Exception: logging.exception` 으로 삼킨다.** 그래서 "정책이 없다"고만 보이고, 실제로는
  플러그인 안에서 `ModuleNotFoundError` 가 났을 뿐이다. 원인을 보려면 **직접 import** 해본다:
  ```bash
  cd /tmp && python -c "import lerobot_policy_umidiffusion"   # 여기서 진짜 예외가 뜬다
  ```
  가장 흔한 원인: **설치된 배포판이 설치 안 된 경로를 import** (`No module named 'custom'`).
  설치된 패키지가 import 하는 것은 **전부 설치된 배포판이어야 한다** — 이것이 `lerobot_canonical` 을
  별도 배포판으로 뺀 이유다(refactoring.md 부록 D.6).
- **디렉토리를 옮긴 뒤 옛 경로를 계속 가리킬 때**: editable 설치는 절대경로를 박아둔다 → 해당 패키지 재설치.
- **import 가 엉뚱하게 shadow 될 때**: `~/.local`(user-site) 오염 가능. `PYTHONNOUSERSITE=1` 로 무시하거나 user-site 를 정리.
- **CUDA 불일치**: `torch.cuda.is_available()` 가 False 면 `--index-url` 의 CUDA 를 드라이버에 맞춘다.

## 라이선스 / 출처

lerobot 은 별도 clone 이며 Apache-2.0 (HuggingFace). 이 repo 는 그 위의 커스텀 확장 코드만 포함한다.
