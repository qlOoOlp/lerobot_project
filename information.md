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

`tools/data_processing/raw_inspect.py` — 단일 파일. UMI raw(`episode_*.h5`)를 LeRobotDataset 으로 변환하기 **전에** 데이터 상태(Hz·차원·품질)를 검사·리포트해 mislabel/오염을 조기 차단한다.

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
python tools/data_processing/raw_inspect.py --raw-root <ft_data 경로>

# 목표 fps 대비 이탈 검증 + 에피소드별 상세
python tools/data_processing/raw_inspect.py --raw-root <경로> --target-fps 30 --per-episode

# preflight 게이트 (이탈 시 exit 1)
python tools/data_processing/raw_inspect.py --raw-root <경로> --target-fps 30 --strict
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

# 2. Observer (processor 층)

_(구현 예정 — ProcessorStep / Pipeline / 커스텀 Step 동작 기록)_

---

# 3. Policy (정책 층)

_(구현 예정 — mypolicy config/modeling/processor. 아래는 확정된 설계.)_

## 3.1 depth ablation = `use_depth` 단일 스위치 (lerobot 무수정 설계)

**목표**: 한 데이터셋(depth 포함)으로 `use_depth` on/off 를 바꿔가며 학습·비교. lerobot 패치 없이.

**원리**: depth 를 **`input_features` 에서 빼면** 모델(DiffusionPolicy)이 depth 인코더를 안 만들고 배치의 depth 를 **무시**. → 별도 필터/패치 불필요.

**hook 위치**: `make_policy` 는 `input_features` 를 채운 뒤(factory.py:517) `validate_features` 를 **안 부르고** 바로 정책을 생성 → 필터는 **`MyPolicyPolicy.__init__` 의 `super()` 직전**이 정답(이때 input_features 세팅됨, 모델 미생성).

**3 조각**
```python
# configuration_mypolicy.py — 필터 로직 한 곳 (idempotent)
DEPTH_KEY = "observation.images.depth"
class MyPolicyConfig(DiffusionConfig):
    use_depth: bool = True
    def apply_depth_gate(self):
        if not self.use_depth and self.input_features and DEPTH_KEY in self.input_features:
            self.input_features = {k: v for k, v in self.input_features.items() if k != DEPTH_KEY}

# modeling_mypolicy.py — 모델 빌드 전 적용
class MyPolicyPolicy(DiffusionPolicy):
    def __init__(self, config, dataset_stats=None):
        config.apply_depth_gate()      # super() 전!
        super().__init__(config, dataset_stats)

# processor_mypolicy.py — 프로세서도 동일 게이트 (순서 무관)
def make_mypolicy_pre_post_processors(config, dataset_stats=None):
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

→ 이 설계로 **lerobot 패치 2파일(datasets/policies factory) 모두 제거 가능** (refactoring.md Phase 7).

---

# 4. Environment (sim / real)

_(구현 예정 — make_env, env processor, sim eval / real 제어 루프 기록)_

---

# 5. Robot (franka)

_(구현 예정 — Robot 10-메서드 구현, ROS2 전송 기록)_
