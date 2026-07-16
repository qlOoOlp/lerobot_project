# lerobot_share — 구현 상세 (information)

각 컴포넌트가 **실제로 어떻게 동작하는지**를 구현하면서 기록한다.
(전체 계획·순서는 `refactoring.md`, 이 문서는 "구현된 것의 동작·설계 상세".)

**컴포넌트 맵**
| 섹션 | 층 | 대응 lerobot 메커니즘 |
|---|---|---|
| 1. Dataset | 데이터 | LeRobotDataset / 변환 파이프라인 |
| 2. Observer | 프로세서(횡단) | ProcessorStep / Pipeline |
| 3. Policy | 정책 | Bring Your Own Policies |
| 4. Environment | sim / real | make_env / env processor |
| 5. Robot | 하드웨어 | Robot 추상 |

---

# 1. Dataset (데이터 층)

## 1.1 `raw_inspect.py` — raw 데이터 인스펙터 (변환 전 preflight)

`custom/scripts/data_processing/raw_inspect.py` — 단일 파일. UMI raw(`episode_*.h5`)를 LeRobotDataset 으로 변환하기 **전에** 데이터 상태(Hz·차원·품질)를 검사·리포트해 mislabel/오염을 조기 차단한다.

### 목적
- "이 데이터가 몇 fps 인지, 차원이 일관되는지, 품질 문제(순서 뒤바뀜/프레임 드랍/포즈 튐)가 있는지"를 **변환 전에** 파악
- 문제 에피소드 **번호 목록**을 산출 → 변환 시 `skip`/리샘플 근거로 사용

### 구조 (단일 파일 내부)
```
Stream / Episode              # 포맷무관 중간표현 (name, kind, timestamps, values, shape, dtype)
iter_record3d_h5(root)        # 어댑터: UMI episode_*.h5 → Episode 이터레이터
iter_lerobot_dataset(root)    # 어댑터: 기존 LeRobotDataset(repo_id/root) → Episode (metaworld 등)
analyze_episode / analyze_dataset   # 포맷무관 분석
print_episode / print_dataset / print_issues   # 리포트
main()                        # CLI + 자동 저장
```
- **두 포맷 (`--format`)**:
  - `record3d_h5` : 변환 전 UMI raw preflight (h5)
  - `lerobot_dataset` : **이미 만든 LeRobotDataset 검사** (metaworld 소스/변환물, 스키마·dim·range·pose-jump sanity)
- **새 포맷 확장 = `iter_<fmt>(root, **opts) -> Iterator[Episode]` 하나 추가 + `ITERATORS` 등록**. 분석/리포트 재사용.
- **anchor 자동**: `rgb` 있으면 그것, 없으면 첫 image 스트림. **pose-jump 자동 표현감지**: 6D=xyz+rpy(record3d) / ≥9D=xyz+rot6d(canonical, `_rot6d_to_R`).

### 동작 흐름 (record3d 예)
1. `iter_record3d_h5(root)` → `episode_*.h5` 정렬 수집
2. 각 파일 → **스트림 4종**: `rgb`, `depth`(옵션), `pose:<source>`(xyzrpy 6D), `gripper`
   - 이미지는 1프레임만 디코딩(shape/range), pose/gripper 는 전체 값 로드
   - **기준 클럭(anchor) = rgb**
   - (lerobot_dataset: parquet 컬럼만 읽음(비디오 디코드 X), state axes 가 x,y,z 로 시작하면 `pose:state` 노출 → pose-jump)
3. `analyze_episode` → 스트림별 Hz/차원/값범위 + 교차정렬(anchor 대비 gap) + pose-jump
4. `analyze_dataset` → fps 분포, 목표 이탈, 차원 일관성, 품질 경고, **warning별·스트림별 에피소드 번호(issues)**
5. 리포트 출력 + **자동 저장**

### 검사 항목 (warning) — 무엇을·왜

데이터의 **세 층위**(시간축 / 값 / 스키마)로 나뉜다.

| # | Warning | 무엇을 체크 | 검출 기준 | 원인 | 영향 | 처리 |
|---|---|---|---|---|---|---|
| 1 | **fps 이탈/MIXED** | 에피소드 rate 일관성 | 분포 2종↑, 또는 target 대비 >`fps-tol`(기본 10%) | Record3D achieved-fps 가 기기 부하로 변동 / 30·60fps 설정 혼용 | 한 fps 로 라벨 시 **시간축 왜곡 → delta action 스케일 불일치** | 목표 fps 로 리샘플 통일 또는 필터 |
| 2 | **non-monotonic** | 시각이 뒤로 가나 | `dt ≤ 0` 하나라도 | 멀티스레드 **순서 뒤바뀜**, 클럭 튐, 중복 ts | interp/Slerp/searchsorted 깨짐 + 프레임 시간역순 저장 → 잘못된 delta (rgb=grid 치명) | **ts 정렬**로 복구, 안 되면 제외 |
| 3 | **dropped-frame gap** | 프레임 누락/stall | `dt > 1.8×median` | 프레임 드랍/stall (기기 못 따라감, IO) | gap 지점 delta 커짐(대부분 2x=경미, 정상 움직임을 긴 dt로 샘플) | 대체로 관용/보간, 큰 gap 만 확인 |
| 4 | **pose jump** | 포즈 값 순간이동 | `>0.05m` OR `dpos/median>20x` OR geodesic `>10deg` | **SLAM/트래킹 오류** (정상 dt 에서 값만 튐) | 손상된 큰 action → 학습 오염 | 해당 에피소드 **제외** |
| 5 | **dim consistency** | 스키마 일관성 | 에피소드 간 shape/dtype 불일치 | 카메라 해상도/센서 구성 혼용 | 변환·feature 스키마 깨짐 | 표준화(리사이즈) 또는 제외 |

**원인의 갈래 정리**
```
캡처 타이밍(하드웨어/스레드)  →  ① fps 변동   ② 순서 뒤바뀜   ③ 프레임 드랍   [시간축]
포즈 추정(SLAM)             →  ④ teleport                                  [값]
센서 구성                   →  ⑤ 차원 불일치                                [스키마]
```
- ①②③ 은 "시간축이 언제 찍혔나" 계열(같은 뿌리, 다른 증상 → 따로 검출)
- ④ 는 시간축과 **독립**(실측: 22개 jump 전부 정상 dt 의 진짜 teleport, gap 과 무관)
- ⑤ 는 구조 문제
→ 5개를 **따로 검출·따로 처리**하는 게 맞다.

### pose-jump 상세 (④)
- `find_pose_jumps.py` 로직 이식 (3중 OR 기준). `_rpy_to_R`(Rz@Ry@Rx) + geodesic 으로 회전 delta 계산.
- **기존 도구와 결과 100% 일치** 검증 (move260626: 22/22 동일).
- 임계값 CLI 조정: `--pos-thresh`(0.05m) `--ratio-thresh`(20x) `--rot-thresh`(10deg).
- ratio 는 **비교 전 반올림 금지**(경계값 20.02x 누락 방지) — 표시할 때만 반올림.

### CLI 사용법
```bash
# fps 확인(발견)이 목적이면 target 없이
python custom/scripts/data_processing/raw_inspect.py --raw-root <ft_data 경로>

# 목표 fps 대비 이탈 검증 + 에피소드별 상세
python custom/scripts/data_processing/raw_inspect.py --raw-root <경로> --target-fps 30 --per-episode

# preflight 게이트 (이탈 시 exit 1)
python custom/scripts/data_processing/raw_inspect.py --raw-root <경로> --target-fps 30 --strict
```
| 옵션 | 뜻 |
|---|---|
| `--target-fps N` | N 대비 이탈 에피소드 검출 (없으면 분포만) |
| `--fps-tol` | 이탈 상대 허용오차 (기본 0.1) |
| `--pos/ratio/rot-thresh` | pose-jump 임계값 |
| `--per-episode` / `--max-episodes N` | 에피소드별 상세 / 앞 N개만 |
| `--out-dir` / `--no-save` | 저장 경로 변경 / 저장 끄기 |
| `--strict` | fps 이탈 있으면 exit 1 |

### 출력 (자동 저장)
- 실행 시 `outputs/data_processing/raw_inspect/<name>_<ts>.log`(사람용) + `.json`(구조화) 자동 기록. `<name>` = raw-root 경로 끝 2요소(예 `move260626_ft_data`).
- **JSON `dataset.issues`** = warning별·스트림별 에피소드 **번호** 목록:
  ```
  issues = {
    "fps_outliers": [번호...],
    "pose_jump":    [번호...],
    "non_monotonic": {"rgb": [...], "depth": [...], "gripper": [...]},
    "gaps":          {"rgb": [...], "depth": [...], "pose:...": [...], "gripper": [...]},
  }
  ```
- → 이 목록으로 **제외 리스트**를 만들어 변환에 넘긴다. (예: `rgb non_monotonic ∪ pose_jump` = 반드시 제외 후보)

### 실측 결과 (move260626, 250ep)
- fps: 30fps×144 + 32~60 산발 → **혼재**. target 30 이탈 **103개**
- non-monotonic: **rgb 13** / depth 68 / gripper 18
- gap: rgb 110 / depth 140 / pose 103 / gripper 85 (대부분 2x)
- **pose jump 22개** (전부 정상 dt 의 teleport)
- 차원: 전부 일관 ✓

### 설계 메모
- **자기완결**(lerobot_hong `reader.py` 미의존, h5 직접 읽음) → lerobot_share 독립성
- 의존: `numpy`, `h5py`, `pillow`
- 과설계 지양: 포맷 하나(record3d)뿐이라 단일 파일. 실제로 2번째 포맷 생기면 그때 분리 검토.

---

## 1.2 `lerobot_canonical` — 공유 어휘 (키 + 표현 + codec)

`custom/utils/lerobot_canonical/` — 정책·env·스크립트가 **서로를 모른 채 합의하는 어휘**.
독립 pip 배포판이다(이유는 refactoring.md 부록 D.6 — 요약: 설치된 배포판은 설치 안 된 경로를 import 할 수 없다).

```
custom/utils/lerobot_canonical/
├── pyproject.toml                 # name="lerobot_canonical", deps=["lerobot","torch"]
└── src/lerobot_canonical/
    ├── keys.py                    # 표현 무관 — 모든 표현이 공유
    └── schemas/
        ├── __init__.py            # "표현마다 모듈 하나" 규약 문서화. 재노출 없음
        ├── canonical_ee10.py      # EE-pose 10D 표현 하나 — 이름표와 치수 (로직 없음)
        └── canonical_ee10_se3.py  # 그 표현의 codec — 유일한 로직 (rot6d↔R, pose9d↔T)
```

**lerobot 본체의 대응물이 그대로 있다** — 우리가 발명한 범주가 아니다:

| 우리 | lerobot | 성격 |
|---|---|---|
| `keys.py` | `lerobot/utils/constants.py` | 키 문자열. lerobot 쪽은 policies·datasets·robots·envs 등 **10개 범주**가 import |
| `schemas/canonical_ee10_se3.py` | `lerobot/utils/rotation.py` | 회전 수학 (`class Rotation`, `from_matrix`, 쿼터니언) |

→ 그래서 디렉토리 이름이 `utils/` 다. lerobot 은 이 자리를 `utils` 라 부르고, 자기 `common/` 은
(datasets·envs·policies 를 **전부** 담고 있어 아무것도 구분하지 못했기에) **삭제**했다.
`custom/` 최상위 = `lerobot/` 최상위의 부분집합(`policies/` `envs/` `scripts/` `utils/`).

> ⚠ **플러그인 접두사에 일부러 안 걸린다.** `register_third_party_plugins()` 는
> `lerobot_robot_`/`camera_`/`teleoperator_`/`policy_` 만 자동 import 한다. 얘는 라이브러리이지
> 플러그인이 아니므로 그 목록에 없는 이름이어야 한다.

### 뭐가 배포판이고 뭐가 모듈인가 (판정 기준 하나)

> **설치된 배포판이 import 하면 → 배포판. 스크립트만 import 하면 → 모듈로 족하다.**

| 대상 | 형태 | 왜 |
|---|---|---|
| `lerobot_canonical` | **배포판** | 정책(배포판)이 import → 강제 |
| `lerobot_policy_umidiffusion` | **배포판** | `lerobot_policy_` 접두사 = 자동탐색 조건 |
| `custom/envs/metaworld/canonical.py` | **모듈** | 소비자가 스크립트뿐. 게다가 env 는 접두사가 아예 없고(`cfg.package_name` 방식), `@EnvConfig.register_subclass("metaworld")` 는 **lerobot 본체에 이미 있음**(`envs/configs.py:349`) |
| `custom/scripts/**` | **스크립트** | 아무도 import 안 함 (실행만) |

이 기준을 어기면 오늘의 그 에러가 난다 — 배포판이 비배포판을 import → `ModuleNotFoundError` → 자동탐색이 **삼킴** → `invalid choice`. 근거·검증은 refactoring.md 부록 D.6/D.6.1.

### `keys.py` — lerobot 상수에서 **파생**
`image_key(cam)` = `f"{OBS_IMAGES}.{cam}"` / `RGB_KEY`·`DEPTH_KEY` = 그 헬퍼로 조립 / `STATE_KEY`·`ACTION_KEY` = `OBS_STATE`·`ACTION` **별칭**.
→ 파일 안에 `"observation..."` 문자열 리터럴 **0개**. lerobot 이 규약을 바꿔도 자동 추종.

### `schemas/canonical_ee10.py` — 표현 **하나**
- 차원: `POSE_DIM=9`(xyz3+rot6d6), `GRIPPER_DIM=1`, `STATE_DIM`/`ACTION_DIM` = **파생**(10)
- 축: `POSE_AXES`(x,y,z,rot6d_0..5), `GRIPPER_AXES`, `STATE_AXES`/`ACTION_AXES` = **파생**
- 상수: `IDENTITY_ROT6D = (1,0,0,0,1,0)` — 항등회전의 rot6d 인코딩. **tuple** 로 둬 이 모듈을 numpy 의존 없이 유지(소비자가 `np.asarray`)
- **채널 의미**(정본): xyz [m, **절대**] / rot6d = 회전행렬 앞 두 열 flatten / **gripper = openness [0,1], 0=닫힘 1=열림**
  - **10개 숫자 = 7자유도** (위치3 + 회전3 + 그리퍼1). rot6d 가 3자유도를 6개로 **과표현**하는 건 연속성을 사기 위함 — 오일러(짐벌락)·쿼터니언(`q`/`−q` 이중덮개)은 불연속이라 신경망 학습이 찢어짐. 아무 6개 숫자든 Gram-Schmidt 가 유효 회전으로 만들어줌
  - `c3` 를 버려도 되는 이유: R 이 직교행렬이라 `c3 = c1 × c2` 로 복원 → **정보 손실 0**
  - **state·action 둘 다 절대값**(delta 아님). `action[t] = state[t+1]`. delta 는 env 경계에서만 → 근거·실측: `retargeting.md` 1·5절

### 설계 규칙 (실전에서 벼려진 것)
| 규칙 | 이유 |
|---|---|
| **primitive vs derived** — `STATE_DIM = POSE_DIM + GRIPPER_DIM` | 근원 한 곳만 고치면 파생이 따라옴. `10` 을 여섯 군데 하드코딩하면 표현 바꿀 때 조용한 불일치 |
| schema 는 **로봇이 아니라 표현**에 종속 | Sawyer·franka·UMI 가 같은 EE-pose 표현 공유 → 로봇 교체 시 재사용. **표현** 변경 시에만 새 모듈 |
| 새 표현은 `schemas/` 안 **sibling** 추가 | 기존 모듈 불변(open/closed). 대외 계약(`STATE_DIM`/`STATE_AXES`/`ACTION_*`)만 맞추면 다운스트림이 모듈만 갈아끼움 |
| **재노출 안 함** | import 가 항상 명시적 → 어떤 모듈도 "the schema" 행세 못 함 |
| **feature 빌더는 여기 두지 말 것** | 정책은 데이터셋에서 feature 파생(`dataset_to_policy_features`) → feature dict 는 **데이터 생산자(컨버터)** 책임 |

---

## 1.3 metaworld 수집 (`custom/envs/metaworld/` + `custom/scripts/sim/`)

Phase 1 산출물 = **코드**(어댑터 + 수집 스크립트). 데이터셋은 `~/datasets/metaworld_canonical/pick_place_v3_bin`
(canonical 10D, 이진 그리퍼, 80fps) — **2026-07-17 현재 재수집 필요**(아래 "시딩 함정"). 번역 계약 전반은 `retargeting.md`.

### `canonical.py` — env 어댑터 (경계를 넘는 것 **전부**)
| 함수 | 방향 | 핵심 |
|---|---|---|
| `render_frame(env, size)` | env 카메라 → 데이터셋 이미지 | corner2 **양축 flip 보정** + resize |
| `state4_to_canonical10(state4, thresh)` | env obs[:4] (abs) → canonical 10D (abs) | xyz 복사 · `IDENTITY_ROT6D` broadcast · **그리퍼 이진화** |
| `canonical10_to_env_action(t10, cur, scale)` | canonical (abs) → env 4D (**rel**) | `delta/scale` · 그리퍼 **극성 반전** `(0.5−o)×2` · `[-1,1]` 클립 |

**배치 기준**: 셋 다 **수집·rollout 이 공유** → train==inference 요구가 걸림 → **모듈**(`custom/envs/metaworld/canonical.py`).
수집 전용 로직(`action=다음 state` 라벨링)은 **스크립트**(`custom/scripts/sim/collect_metaworld.py`).
> 어댑터와 스크립트를 **가른 이유**: 어댑터는 Phase 5 rollout 이 다시 import 하지만, 스크립트는 아무도 import 하지 않는다.
> "재사용되는 것 = 모듈 / 실행되는 것 = `scripts/`" — 그래서 실행 스크립트는 전부 `custom/scripts/` 아래로 모았다.

**env 사실은 여기 / 표현 사실은 `lerobot_canonical`**:
```python
STATE4_DIM = ENV_ACTION_DIM = 4          # metaworld 의 사실
ENV_XYZ_SCALE = 0.01                     # env 의 action_scale        (태스크 무관)
PICK_PLACE_GRIPPER_THRESHOLD = 0.7       # obs[3] 이중봉우리 분리점    (태스크 의존 — 재실측 필요)
FLIP_CAMERAS = frozenset({"corner2"})    # 180° 굴러 장착된 카메라     (카메라 의존)
sch.STATE_DIM, sch.POSE_DIM, sch.IDENTITY_ROT6D   # lerobot_canonical 에서 import
```
> **상수의 세 축**이 각각 다르다 — 무엇이 바뀌면 재실측/재수집인지가 다르기 때문:
> 태스크 무관(env 상수) / 태스크 의존(물체 두께 바뀌면 재실측) / 카메라 의존(카메라 갈면 달라짐).

**`FLIP_CAMERAS` — flip 은 카메라의 성질이지 metaworld 의 성질이 아니다**
- 근거: `assets/objects/assets/xyz_base.xml` 의 `<camera name="corner2" ... euler="3.9 2.3 0.6"/>` — 180°를 넘겨 굴러 있어 mujoco 가 거꾸로 된 장면을 충실히 렌더한다. 실측: raw 는 **테이블이 천장에 매달림** (`tmp/real/corner2_{raw,flipped}.png`)
- `np.flip(img,(0,1))` 은 거울상이 아니라 **180° 회전** = 장착 각도 되돌리기. 그래서 **수집·rollout 양쪽**에 걸린다 — 카메라는 계속 뒤집힌 프레임을 뱉으니 읽을 때마다 같은 보정이 필요하다. *"데이터셋에 이미 flip 이 있으니 rollout 에선 건너뛰자"* 는 **똑바른 그림으로 학습한 정책에 거꾸로 된 그림을 먹이는 것**
- 형제 카메라는 정의가 다르다(`behindGripper quat="0 1 0 0"` 등) → **무조건 flip 은 버그**. lerobot 도 같은 가드를 함(`envs/metaworld.py:147`)
- 카메라 이름은 **인자가 아니라 `env.camera_name` 에서 읽는다** — env 가 이미 아는 사실이라 실제 렌더되는 카메라와 **desync 가 불가능**. (반면 `gripper_threshold`/`xyz_scale` 은 env 가 모르는 **우리 결정**이라 넘긴다.) 속성이 없으면 조용히 넘어가지 않고 `ValueError`
- ⚠ `corner2 ∈ FLIP_CAMERAS` 라 **기존 300ep 데이터셋은 재수집 불필요** — 픽셀 단위 동일 검증됨

### `scripts/sim/collect_metaworld.py` — port_droid 패턴 (Robot 없음)
`build_features` → `LeRobotDataset.create` → `collect_episode` → `to_canonical_and_actions` → `add_frame` 루프 → `save_episode` → `finalize`.

- **obs 39D 중 앞 4개만** 사용(`ee_xyz` + `gripper`). 나머지 35개는 **특권 정보**(물체·목표 좌표) → 버림. 정책은 **이미지로 추론**해야 실기 이식 가능. (반면 **expert 는 39D 전체**를 봄)
- **성공 에피소드만** 저장(`info["success"]`)
- **30% 에피소드에 xyz 노이즈**(`noise_std=0.15`) — 순수 데모는 최적경로만 보여줘 정책이 벗어나면 복구 불가. **그리퍼엔 절대 금지**(무작위 개폐 = 데모 파괴)
- `seed_base=100` — eval seed 0~9 홀드아웃
- `action[t] = state[t+1]` (절대 target). 마지막 프레임은 자기 복제

### 실전에서 걸린 함정
| 함정 | 증상 |
|---|---|
| 이미지 feature `names[2]` = `"channel"` | lerobot 이 `(H,W,C)→(C,H,W)` 변환 여부를 **이 값으로 결정**(`datasets/utils.py:724`). 오타 호환 경로라 **`"channels"`** 를 쓸 것. 아니면 정책이 240채널 이미지로 착각 |
| `env.render(mode=...)` | gymnasium 0.26+ 는 **인자 없음**. `render_mode`/`camera_name` 은 생성 시 고정 |
| `canonical[-1]` vs `canonical[-1:]` | 축이 사라져 `concatenate` 실패. `("gripper",)` 콤마, `state4[..., 3:4]` 와 같은 함정 |
| `xyz_scale` 을 데이터 통계로 잡기 | **env 상수(0.01)** 여야 함. 통계는 손의 **지연 응답**이라 gain 이 아님 → retargeting.md 5절 |
| `env.reset(seed=n)` 으로 장면 고정 | **Meta-World 가 설계상 버린다** → 아래 "시딩 함정" |
| `render_frame` 무조건 flip | flip 은 **corner2 카메라**의 성질. 카메라 바꾸면 조용히 틀림 → `FLIP_CAMERAS` 가드 |

### ★ 시딩 함정 — `reset(seed=)` 은 무시된다 (2026-07-17, 반나절 소모)

**증상**: 같은 명령으로 두 번 수집했는데 프레임 수가 달랐다(16,370 vs 16,291). **ep0 부터** 달랐다.

**원인**: Meta-World 는 물체·목표 배치(`rand_vec`)를 **세 갈래**로 뽑는다 — `sawyer_xyz_env.py:697`:
```python
if self._freeze_rand_vec:   return self._last_rand_vec           # 고정 (랜덤화 없음)
elif self.seeded_rand_vec:  rand_vec = self.np_random.uniform(...)   # env.seed(n) 이 제어 ✓
else:                       rand_vec = np.random.uniform(...)        # ★ 전역 np.random
```
lerobot wrapper 는 `_freeze_rand_vec = False` **만** 켜고 `seeded_rand_vec` 는 안 켠다(`envs/metaworld.py:163`)
→ **세 번째 갈래**. 여기선 우리가 뭘 넘겨도 장면을 제어할 수 없다:
- `env.reset(seed=n)` → **설계상 버려진다**. Meta-World reset docstring 이 직접 말한다:
  *"seed: The seed to use. **Ignored**, use `seed()` instead."* (`sawyer_xyz_env.py:670`).
  실제로 `reset()` 은 `reset_model()`(여기서 rand_vec 을 뽑음) 뒤에 `super().reset()` 을 **seed 없이** 부른다
  → gymnasium 의 재시드가 아예 안 일어난다.
- `env.seed(n)` → **이것도 무력**. `self.np_random` 을 seed 하지만 위 갈래는 그걸 안 읽는다.
- `np.random.seed(n)` → 먹히긴 하나 **전역 상태**를 오염시킨다.

**해법** (실측 검증): `env.seeded_rand_vec = True` + `env.seed(n)` + `env.reset()` (**reset 에 seed 금지**).
→ 같은 명령 두 번 = **완전히 동일한 데이터셋**.

**안 켰을 때의 실제 피해**:
- 데이터셋 **재현 불가** — 같은 실험을 다시 못 만든다
- **"eval seed 0~9 홀드아웃" 이 거짓말이 된다** — seed 가 장면을 안 정하니 홀드아웃 개념이 성립 안 함
  (다만 연속 분포에서 매번 무작위라 train/eval 이 겹칠 확률은 ≈0 → **일반화 평가 자체는 유효**했다)
- **정책 A/B 를 같은 장면에서 비교 불가** → Phase 5 에서 성공률 차이가 정책 탓인지 장면 운인지 구분 불가.
  이게 진짜 피해다.

### 검증 (Step 5) — 옛 300ep 산출물 기준 (2026-07-16)
```
raw_inspect   300ep 80fps 균일 · 10D 일관 · no-warning
이진 그리퍼    state[9]/action[9] 고유값 {0,1} · 열림 55.5%
meta.stats    rot6d std=0 (상수, 정상) · gripper mean .555/std .497
스키마 대조    옛 pick_place_v3_inenv 와 차이 1개: gripper_width → gripper (의도)
flip 방향     gif 육안 확인 (코드로 검증 불가) → tmp/real/
```
> ⚠ **위 수치는 재현 불가**(2026-07-17 확인). 시딩 함정 때문에 장면이 매번 달랐던 시절의 산출물이라,
> 같은 명령을 다시 돌려도 같은 숫자가 안 나온다. **성격 판정에는 여전히 유효**하다(80fps 균일, 10D 일관,
> 그리퍼 이진 {0,1}, rot6d std=0 은 구조적 사실이라 장면과 무관) — 재수집 후 바뀌는 건 프레임 수·평균값뿐.
> 검증 항목 자체는 재수집 때 그대로 다시 돌리면 된다.

> **데이터셋 현황(2026-07-17)**: `~/datasets/metaworld_canonical/` 전체 삭제됨. 시딩 수정 전 수집분이라
> 재현이 안 돼 남길 가치가 없었다(3.6GB). **재수집 필요** — 고쳐진 `collect_metaworld.py` 로 돌리면
> `--seed-base` 가 실제로 작동해 이후 모든 실험이 재현·비교 가능하다.
> 옛 `pick_place_v3`·`pick_place_v3_inenv` 는 그리퍼가 **연속**이라 어차피 새 규약과 달랐다(참고용 기록).

---

# 2. Observer (processor 층)

_(구현 예정 — ProcessorStep / Pipeline / 커스텀 Step 동작 기록)_

---

# 3. Policy (정책 층)

_(구현 예정 — umidiffusion config/modeling/processor. 아래는 확정된 설계.)_

## 3.1 depth ablation = `use_depth` 단일 스위치 (lerobot 무수정 설계)

**목표**: 한 데이터셋(depth 포함)으로 `use_depth` on/off 를 바꿔가며 학습·비교. lerobot 패치 없이.

**원리**: depth 를 **`input_features` 에서 빼면** 모델(DiffusionPolicy)이 depth 인코더를 안 만들고 배치의 depth 를 **무시**. → 별도 필터/패치 불필요.

**hook 위치**: `make_policy` 는 `input_features` 를 채운 뒤(factory.py:517) `validate_features` 를 **안 부르고** 바로 정책을 생성 → 필터는 **`UmiDiffusionPolicy.__init__` 의 `super()` 직전**이 정답(이때 input_features 세팅됨, 모델 미생성).

**3 조각**
```python
# configuration_umidiffusion.py — 필터 로직 한 곳 (idempotent)
DEPTH_KEY = "observation.images.depth"
class UmiDiffusionConfig(DiffusionConfig):
    use_depth: bool = True
    def apply_depth_gate(self):
        if not self.use_depth and self.input_features and DEPTH_KEY in self.input_features:
            self.input_features = {k: v for k, v in self.input_features.items() if k != DEPTH_KEY}

# modeling_umidiffusion.py — 모델 빌드 전 적용
class UmiDiffusionPolicy(DiffusionPolicy):
    def __init__(self, config, dataset_stats=None):
        config.apply_depth_gate()      # super() 전!
        super().__init__(config, dataset_stats)

# processor_umidiffusion.py — 프로세서도 동일 게이트 (순서 무관)
def make_umidiffusion_pre_post_processors(config, dataset_stats=None):
    config.apply_depth_gate()
    ...  # Normalizer features={**input_features, **output_features} 가 자동으로 depth 제외
```

**효과**
| use_depth | 데이터셋 | 모델 input_features | lerobot |
|---|---|---|---|
| True | depth 있음 | rgb+depth 인코더 | 무수정 |
| False | depth 있음(무시) | rgb 인코더 | 무수정 |

**주의**
- idempotent → policy/processor 어느 쪽이 먼저 config 를 만져도 동일. 저장/로드도 일관(체크포인트 config 에 필터된 input_features 저장).
- `DropObservationKeys` 스텝은 이제 **정확성엔 불필요**(모델이 depth 무시). depth 를 GPU 로 안 올리려면 Device 전에 두는 효율 옵션.
- `datasets/factory.py` 패치(depth 미로드)는 효율일 뿐 → 없어도 됨.
- 추론: 체크포인트의 `use_depth` 가 preprocessor·모델을 자동 일치시킴 → 실기에선 `runtime buffer` 의 `include_depth` 만 맞추면 됨.

→ 이 설계로 **lerobot 패치 2파일 모두 제거 가능**. lerobot_hong 실측: `datasets/factory.py`(+15줄) · `policies/factory.py`(+45줄) = **60줄이 전부 depth 필터**였음 (refactoring.md Phase 2 / 부록 D.5).

> ⚠ 정정: 위 "DropObservationKeys 는 정확성엔 불필요" 는 **효율 관점**이고, lerobot_hong 은 실제로 **두 겹**을 다 씀 — config 게이트(모델이 인코더를 안 만들게) + `DropObservationKeysProcessorStep`(관측 dict 에서 실제 제거). Phase 2 에선 둘 다 유지한다.

## 3.2 런타임 프로세서 — 정책이 **앵커 기준 relative** 를 본다 ★

_(구현 예정 — Phase 2. 아래는 lerobot_hong 실측 + dev_plan 근거로 확정된 설계.)_

**custom policy 의 존재 이유** (dev_plan §12.1): *"`policy.type=diffusion` 을 그대로 쓰면 factory 가 기본 processor 를 만든다. 그 기본 processor 에는 **runtime relative/delta pose step 이 없다**."* → 정책과 프로세서는 분리 불가.

### 파이프라인 (lerobot_hong 구조 그대로)
```python
input_steps = [Rename, AddBatch,
               CanonicalPoseToActionPoseReprStep(action_pose_repr),  # 액션 → relative|delta
               CanonicalPoseToRelativeObservationStep(),             # 관측 → anchor-relative
               Device, Normalizer]
# use_depth=False 면 index 1 에 DropObservationKeysProcessorStep 삽입
output_steps = [Unnormalizer, Device(cpu)]
```

### 왜 오프라인에 못 굽나 (dev_plan §3.2)
> *"relative/delta 는 frame 의 고정 속성이 아니라 **sample window 의 anchor 에 종속된 표현**"*

앵커 = **샘플된 윈도우의 마지막 관측**. 학습 때 윈도우를 어디서 자르냐에 따라 달라지므로 **오프라인 고정 저장이 원천 불가**. → 반드시 런타임.
> **대비**: 그리퍼 이진화는 threshold 가 **고정 상수**라 오프라인 bake 가능(§1.3). **pose 는 앵커 의존이라 불가.** 같은 10D 안에서도 채널마다 처리 원칙이 다른 이유.

### ★ 데이터 비의존성의 수학적 근거
```
relative_transform(anchor, state) = anchor⁻¹ @ state
좌표계 F → T@F 로 바뀌어도:
  (T@anchor)⁻¹ @ (T@state) = anchor⁻¹ @ T⁻¹ @ T @ state = anchor⁻¹ @ state   ← 동일
```
**오프라인 좌표계가 상쇄**되므로 metaworld(월드 절대)·UMI(episode-start 상대)가 **정책에겐 같은 의미**. "정책이 데이터에 의존하면 안 된다"가 이 한 줄로 보장됨.

### 정규화가 IDENTITY 인 이유 (dev_plan §11)
> *"dataset stats 는 canonical 기준인데 런타임이 relative 로 바꾼다 → canonical stats 를 그대로 쓰면 **표현 공간이 안 맞을 수 있다**"*

→ `STATE=IDENTITY`, `ACTION=IDENTITY`(1차 전략). relative 값은 이미 0 근처라 무방. 필요 시 relative 기준 stats 를 따로 계산해 확장.

> ⚠ **"rot6d std=0 나눗셈 회피"는 근거가 아니다** — 2-0 실측으로 반증됨. lerobot 이 `denom = std + eps`(eps=1e-8)로 이미 막는다(`processor/normalize_processor.py:94, :335`). MEAN_STD 여도 NaN 안 남(상수 채널은 `0/1e-8=0` → 죽은 채로 들어갈 뿐).
> → **표현공간 불일치(dev_plan §11)가 IDENTITY 의 유일한 근거.**

### 세부 규약
- **pose 9D 만 변환**, gripper 1D 는 그대로 이어붙임 / **차원 유지** 10D→10D (dev_plan §9.4)
- **`action` 없으면 skip** — eval/추론 raw 관측엔 action 이 없음 (dev_plan §9.3)
- `obs_pose_repr` 은 `"relative"` 만, `action_pose_repr` 은 `{"relative", "delta"}` (기본 `relative`)

### ★ 역변환 (`decode_policy_action`) — 추론의 필수 절반
정책은 **relative 를 뱉는다.** 그대로 쓰면 안 되고 **절대로 되돌려야** 한다:
```python
canonical_action = decode_policy_action(
    predicted_action,                                    # relative
    anchor_state=canonical_window[OBS_STATE][-1],        # 앵커 = 현재 관측
    action_pose_repr=policy.config.action_pose_repr)     # forward 와 같은 값!
env_action = canonical_action_to_env_action(canonical_action)
```
- `relative`: `base @ action` / `delta`: 누적 적분
- **파이프라인 밖**: `policy_post` 는 `PolicyAction` 만 받아 **앵커에 접근 불가** → 원본·lerobot_hong 모두 추론 루프가 직접 호출
- **원본 대조**: UMI `get_real_umi_action` 의 `convert_pose_mat_rep(..., backward=True)` 와 수식 동일

### 두 config 의 역할은 **비대칭**
| config | 담당 | 방향 |
|---|---|---|
| `obs_pose_repr` | 관측만 | **forward 뿐** (정책이 관측을 만들지 않으니 backward 없음) |
| **`action_pose_repr`** | 액션만 | **forward(학습) + backward(추론) 양쪽에 같은 값** |

forward/backward 가 **같은 repr + 같은 앵커**(`state[:,-1]`)여야 정확한 역함수 → train==inference.

### `relative` vs `delta`
| | 오차 특성 |
|---|---|
| **`relative`**(기본) | 전부 **같은 앵커** 기준 → 한 스텝 틀려도 전파 안 됨 |
| `delta` | 액션끼리 **연쇄**(backward=누적합) → 한 스텝 틀리면 **이후 전부 오염** |

> ⚠ **원본 UMI 의 버그**: `umi_dataset.py:349` 가 action 변환에 `pose_rep=self.obs_pose_repr` 을 넘긴다(`action_pose_repr` 이어야 함). 기본값이 둘 다 `relative` 라 안 드러나지만 `delta` 로 두면 **학습=relative / 추론=delta** 로 조용히 갈라짐. **lerobot_hong 은 이미 올바르게 고침** → 우리도 그쪽을 따른다.

### 채택하지 않은 것: `_wrt_start` 채널 (원본 UMI 에는 있음)

원본 UMI 의 `shape_meta.obs` 에는 우리에게 없는 채널이 하나 있다:
```yaml
robot0_eef_rot_axis_angle_wrt_start:  shape: [6]   # 에피소드 시작 기준 rot6d
# robot0_eef_pos_wrt_start:           ← 주석 처리(의도적)
```
- **무엇**: `relative(episode_start_pose, state)` 의 **회전만**. 학습 시 시작 자세에 노이즈(`N(0, 0.05)`) 주입.
- **왜 있나**: anchor-relative 는 좌표계를 상쇄하면서 **"지금 세상에서 어느 방향을 보고 있나"** 도 함께 잃는다(앵커 기준으론 항상 항등). `wrt_start` 가 **중력·접근각 같은 절대 방향 기준**을 회전에 한해 되돌려준다.
- **왜 위치는 빼나**: 위치 `wrt_start` 는 절대 위치 과적합 → 물체가 다른 데 있으면 깨짐. 회전만 주는 게 의도.

**우리는 구현하지 않는다** (metaworld·UMI 둘 다):
- metaworld 는 **회전 자체가 없어**(Sawyer, rot6d std=0) `wrt_start` 도 항등 상수 → 정보량 0
- UMI 에서도 **영향이 크지 않다고 판단**
- lerobot_hong 도 이 채널이 없다(10D) — 이미 떨어뜨린 상태를 물려받음

**나중에 필요해지면**: `observation.environment_state` 슬롯을 쓴다(lerobot 정식 지원 — `global_cond_dim += env_state_feature.shape[0]`, `global_cond_feats.append(batch[OBS_ENV_STATE])`, 없으면 자동 skip). 스키마 분기가 아니라 **옵셔널 채널**이라 데이터셋마다 있어도/없어도 되고, 정책 코드는 그대로다.
- ⚠ 제약: **저장 차원 = 정책이 보는 차원**. `input_features` 는 `dataset_to_policy_features(ds_meta.features)` 에서 오고 프로세서의 `transform_features` 는 **반영되지 않는다** → 6D 저장 → 6D 소비로 맞출 것.
- ⚠ `_wrt_start` 는 **앵커 의존이 아니다**(base=에피소드 시작). 따라서 relative pose 와 달리 **오프라인 bake 도 가능**하다 — 런타임인 유일한 이유는 노이즈 증강.
- ⚠ raw 에 `demo_start_pose` 필드는 **없다**. 컨버터가 에피소드 프레임 0 에서 **파생**해야 한다(원본 UMI 도 SLAM 궤적에서 파생해 zarr 에 넣음).

### 층 분리 (dev_plan §7.3/§9.5)
| 층 | 위치 | 패키지 | 내용 |
|---|---|---|---|
| **표현 codec** | `schemas/canonical_ee10_se3.py` | `lerobot_canonical` | `rot6d↔R`, `pose9d↔transform`, `invert`, `relative` — **컨버터·런타임 공용** (부록 D.5) |
| **런타임 step** | `steps.py` | `lerobot_policy_umidiffusion` | 앵커 의미 + relative/delta + Step 래퍼 |
| **조립** | `processor_umidiffusion.py` | `lerobot_policy_umidiffusion` | pipeline 에 끼우기만 |

> 층 분리는 **파일**로 유지하되 패키지는 2개다. codec 만 `lerobot_canonical` 인 이유: 컨버터(오프라인)와
> 런타임 step 이 **대등한 소비자**라 어느 한쪽이 소유하면 나머지가 엉뚱한 패키지에 의존하게 된다.
> 반면 step/조립은 **런타임 전용**이라 정책과 운명을 같이한다 → 같은 패키지.
> (lerobot_hong 은 이걸 `lerobot_processor_robot_maps` 라는 **별도 배포판**으로 뺐지만, 그 프로세서는
> 정책 하나만 쓰는데도 설치를 2번 하게 만들었다. 우리는 정책 안으로 흡수.)

> **lerobot_hong 의 실수**: codec 이 `se3.py`(numpy)와 `steps.py`(torch)에 **중복 구현**됐고 `relative_transform` 은 구현마저 다름(`np.linalg.inv` vs SE3 전치). dev_plan §8 이 *"구현 시 결정"* 으로 미룬 결과. → 우리는 **표현 옆에 codec 을 두어 구조적으로 차단**.

---

# 4. Environment (sim / real)

_(구현 예정 — make_env, env processor, sim eval / real 제어 루프 기록)_

---

# 5. Robot (franka)

_(구현 예정 — Robot 10-메서드 구현, ROS2 전송 기록)_
