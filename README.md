# lerobot_share

[HuggingFace `lerobot`](https://github.com/huggingface/lerobot) 위에 올린 커스텀 로봇 학습 스택.
정책(policy) / 환경(env: metaworld·real) / 로봇(robot: franka) 을 **플러그인**으로 분리하고,
**lerobot 본체는 완전 무수정(0-patch)** 으로 유지한다. metaworld 로 코어 루프(데이터→학습→eval)를
먼저 완성하고 같은 canonical 스키마(10D `[xyz, rot6d, gripper]`)로 UMI 를 확장한다.

> 설계·단계별 계획: [refactoring.md](refactoring.md) · 구현 동작 상세: [information.md](information.md)

## 디렉토리 구조

```
lerobot_share/
├── lerobot/      # HF lerobot v0.4.4 — 별도 clone (git 무시, 벤더링 안 함)
├── custom/       # lerobot 플러그인/변환 코드 (policy·processor·data_processing)
├── tools/        # 독립 유틸 (raw_inspect.py 등)
├── patches/      # fallback 참고용 (기본은 무수정 → apply 하지 않음)
├── outputs/      # 실행 산출물 (로그·gif·체크포인트) — git 무시
├── refactoring.md
└── information.md
```

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

### 6) 검증
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.10.x+cu128  True
python -c "import lerobot; print(lerobot.__version__)"                          # 0.4.4
```

## 커스텀 코드 실행

- 스크립트는 **repo 루트에서** 실행한다(그래야 `custom...` / `tools...` import 가 잡힌다).
- 플러그인 패키지(정책 등)는 해당 Phase 에서 editable 설치: `pip install -e custom/policies/<pkg>` (계획은 refactoring.md).
- 추가 의존성은 Phase 별로 설치: `metaworld==3.0.0`(env), `scipy`(processor), `h5py`(UMI raw) 등 — refactoring.md 부록 B.

예) raw 데이터 인스펙터:
```bash
python tools/data_processing/raw_inspect.py --raw-root <dataset> --format lerobot_dataset
```

## 트러블슈팅

- **import 가 엉뚱하게 shadow 될 때**: `~/.local`(user-site) 오염 가능. `PYTHONNOUSERSITE=1` 로 무시하거나 user-site 를 정리.
- **CUDA 불일치**: `torch.cuda.is_available()` 가 False 면 `--index-url` 의 CUDA 를 드라이버에 맞춘다.

## 라이선스 / 출처

lerobot 은 별도 clone 이며 Apache-2.0 (HuggingFace). 이 repo 는 그 위의 커스텀 확장 코드만 포함한다.
