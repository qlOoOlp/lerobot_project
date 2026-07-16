# Retargeting — embodiment ↔ canonical 번역 계약

> **범위**: 각 소스/싱크(metaworld env, UMI raw, franka)를 **canonical(EE-pose 10D)** 로/에서 옮기는 규약과, 정책이 그 값을 **어떻게 소비**하는지.
> 정본 정의는 `custom/utils/lerobot_canonical/src/lerobot_canonical/schemas/canonical_ee10.py`, 설계 맥락은 `refactoring.md` 부록 D.

## 1. canonical 채널 규약 (정본)

```
canonical 10D = [  x,  y,  z  |  rot6d(6)  |  gripper  ]
                 └─ 위치 3D ─┘ └─ 자세 6D ─┘ └─ 개폐 1D ┘
```
자세(pose) = **위치**(어디) + **방향**(어느 쪽을 보는가). 10개 숫자에 **7자유도**(위치3 + 회전3 + 그리퍼1) — rot6d 가 3자유도를 6개로 **일부러 과표현**한다.

| 채널 | 의미 |
|---|---|
| `x, y, z` | EE 위치 [m], **절대**(소스의 월드/베이스 기준). 소스마다 기준이 달라 필요하면 **어댑터가 좌표계 변환**(metaworld 는 불필요, UMI/franka 는 필요) |
| `rot6d(6)` | 회전행렬 `R=[c1|c2|c3]` 의 **앞 두 열만** flatten. 회전 없음 = `IDENTITY_ROT6D=(1,0,0,0,1,0)` — **표현 상수**(어디서나 동일), *쓰느냐*만 embodiment 별 |
| `gripper` | **openness `[0,1]`, 0=닫힘 / 1=열림** (지배적 규약) |

### 왜 rot6d 인가 (오일러·쿼터니언 대신)
| 표현 | 개수 | 문제 |
|---|---|---|
| 오일러각 | 3 | **짐벌락** — 특정 자세에서 축이 겹쳐 표현 붕괴 |
| 쿼터니언 | 4 | **이중덮개** — `q` 와 `−q` 가 같은 회전인데 숫자는 정반대 → 학습이 찢어짐 |
| 회전행렬 | 9 | 중복 과다 + 직교성 제약을 신경망이 못 지킴 |
| **rot6d** | **6** | **없음(연속)** ← 채택 |

**핵심**: 신경망이 **아무 6개 숫자**를 뱉어도 Gram-Schmidt 가 **항상 유효한 회전**으로 만들고, 비슷한 회전 → 비슷한 6D. 불연속점이 없어 학습에 유리 (Zhou et al. 2019).

**c3 를 버려도 되는 이유**: R 은 직교행렬이라 `c3 = c1 × c2` 로 복원 가능 → **정보 손실 0**.
```
a1, a2 = 6개를 3개씩 자름
b1 = normalize(a1)
b2 = normalize(a2 − (b1·a2)·b1)     # a1 성분 제거 → 직교 보장
b3 = b1 × b2                        # 외적으로 3번째 열 복원
R  = [b1 | b2 | b3]
```
(구현: `raw_inspect.py` 의 `_rot6d_to_R`)

### 왜 **절대**인가 (delta 대신) ⚠ 폴더명 `umi2lerobot_delta_pose` 에 낚이지 말 것
`observation.state` 도 `action` 도 **절대 자세**다. `action[t] = state[t+1]` = "다음 순간 여기 있어라".

| | 저장값 | 오차 |
|---|---|---|
| **절대**(채택) | "(0.102, 0.60, 0.19) 로 가라" | 로봇이 예상 밖에 있어도 `목표−현재` 가 **자동 보정** |
| delta | "x 로 +2mm" | 오차가 **누적** → 드리프트 |

**실측 확인**: `action` 과 `state` 의 xyz 스케일이 동일 → 절대값
```
UMI 옛변환본  : state |mean|=0.1232   action |mean|=0.1232
metaworld_bin : state |mean|=0.2785   action |mean|=0.2798
```

> ⚠ **"절대"는 데이터셋 층 한정이다.** 정책이 보는 건 절대값이 **아니다** — 런타임 프로세서가 **앵커 기준 relative** 로 바꿔서 넣는다(**6절**). 데이터셋은 그 변환의 **원재료(canonical source)** 를 담을 뿐이다.

- canonical 은 **정의**이지 특정 embodiment 서술이 아니다. 소스는 얼마든지 불일치할 수 있고(대부분 그렇다), **각 경계의 어댑터가 번역**한다.
- 어떤 규약을 골라도 **일부 경계엔 뒤집기가 남는다** — 그게 canonical 의 본질. (근거: metaworld 는 obs 와 action 이 **서로** 반대라, 어느 쪽에 맞춰도 나머지가 어긋남)

## 2. 그리퍼 번역표

| 경계 | 원래 규약 | canonical 대비 | 처리 |
|---|---|---|---|
| metaworld **obs**[3] | openness `[0,1]`, 1=열림, **연속** | 일치 | **그대로 통과** |
| metaworld **action**[3] | closing effort `[-1,1]`, **+1=닫힘** | 극성 반대 | **뒤집기** `effort = (0.5 − openness) × 2` |
| **UMI raw** `gripper/state/value` | int32 binary, 파일 attr `state_rule: 0=close, 1=open` → **0=닫힘** | **완전 일치** | **그대로 통과** (float 캐스팅만) |
| franka (deferred) | TBD (FK/그리퍼 폭) | TBD | Phase 9 |

검산: `(0.5 − o) × 2` → `o=0`(닫힘)→`+1`(닫기) ✓ / `o=1`(열림)→`−1`(열기) ✓

## 3. 분포 실측 — cross-embodiment 불일치

| 소스 | canonical openness | 비고 |
|---|---|---|
| **metaworld** (pick_place_v3) | **연속** `[0.3955, 1.0]`, 고유값 611, 이중봉우리(≈1.0 열림 / ≈0.4 블록 쥠) | 완전폐쇄(0)에 도달 안 함 — 블록이 손가락을 막음. **정상** |
| **UMI** (move260626) | **이진**, raw 전부 0(닫힘) → openness **전부 0** | 상수 채널(std=0) |

→ **같은 채널, 전혀 다른 분포.** 정책이 그대로 학습하면 이식이 깨진다. → **런타임 이진화로 정렬**(아래 4절).

## 4. 그리퍼는 **이진**, 변환 시 bake — **그리퍼 채널 한정**

> ⚠ 이 절은 **그리퍼 1채널** 이야기다. **pose 9채널은 정반대로 런타임 변환**한다(6절) — 앵커가 sample window 에 종속돼 오프라인에 못 굽기 때문. 두 채널의 처리 원칙이 다른 이유:
> | | 오프라인 bake 가능? |
> |---|---|
> | **그리퍼** | ✅ threshold 는 **고정 상수** → 구워도 무방 |
> | **pose** | ❌ 앵커가 **윈도우마다 달라짐** → 원천적으로 불가 |

### 결정 — 데이터셋의 그리퍼는 **이진 `{0,1}`, 0=닫힘 / 1=열림**

| 채널 | 값 |
|---|---|
| `observation.state[9]` | **0=닫힘, 1=열림** |
| `action[9]` | **0=닫힘, 1=열림** |

그리퍼 채널은 런타임 가공 없이 **데이터 그대로** 정책에 전달된다(pose 와 달리 relative 변환 대상이 아님 — 6절 step 들은 `pose 9D` 만 변환하고 gripper 는 그대로 이어붙임).
→ 두 소스가 **디스크에서 이미 일치**하므로 3절 불일치 해소. 스키마는 **하나**(`canonical_ee10`) — 이진은 openness `[0,1]` 의 부분집합이라 차원·의미·범위 동일.

### ⚠ `[-1,1]` 은 우리 액션이 아니다 — **env API 강제**
metaworld `env.step()` 은 `spaces.Box(low=-1, high=1)` 를 요구한다 (lerobot `envs/metaworld.py:137`). 우리가 못 바꾼다. 그래서 **canonical 이 env 로 나가는 마지막 한 걸음에서만** `[-1,1]` closing effort 로 번역한다. **이 값은 데이터셋에도, 정책 입출력에도 존재하지 않는다.**

```
[데이터셋 · 정책 입력 · 정책 출력]  ─── 전부 이진 {0,1}, 0=닫힘 / 1=열림 ───
             │                                            ▲
             │ canonical10_to_env_action()                 │ gripper_cmd_to_openness()
             ▼  (마지막 경계에서만)                          │  (수집 시, env 명령 -> canonical)
[metaworld env.step() API]  ─── [-1,1] closing effort (시뮬레이터가 강제) ───
```

### 소스별 이진값 획득

| 소스 | 이진값을 어디서 | 처리 |
|---|---|---|
| **UMI** | raw 가 **이미 이진 + 극성도 일치** | **그대로 통과** (float 캐스팅만) |
| **metaworld** | **`obs[3]`(측정된 openness)** 를 **threshold 로 이진화** | `obs[3] >= thresh` → `1`(열림) / 아니면 `0`(닫힘) |

`obs[3]` 은 canonical 과 **극성이 같고**(1=열림) 범위만 연속이라, **이진화만** 하면 된다. 그리고 **진짜 상태**라서 rollout 에서 그대로 관측 가능 → **명령 추적·프레임 시프트 불필요**.

번역은 **env 경계 두 곳에서만**:

| 방향 | 함수 | 계산 |
|---|---|---|
| env obs → canonical (**수집·rollout 공통**) | `state4_to_canonical10(state4, thresh)` | `obs[3] >= thresh` → `1.0` / else `0.0` |
| canonical → env (**rollout**) | `canonical10_to_env_action(...)` | `effort = (0.5 − openness) × 2` → `0`→`+1`(닫기), `1`→`−1`(열기) |

> `[-1,1]` 이 보이는 건 마지막 줄 하나뿐 — **env API 강제**이며 데이터셋·정책 입출력엔 없다(위 다이어그램).

### ⚠ threshold 는 **태스크 의존적** — 호출부 하드코딩 금지
- pick-place 실측: `obs[3]` 이중봉우리 **≈1.0(열림) / ≈0.40~0.46(블록 쥠)** → **`0.7`** 로 여유있게 분리 (`PICK_PLACE_GRIPPER_THRESHOLD`)
- **두꺼운 물체**는 더 높은 openness 에서 쥐고, 얇으면 더 낮음 → **같은 값이 다른 태스크를 조용히 오분류**. 태스크마다 실측할 것.
- **수집과 rollout 에 같은 값**을 넘겨야 함 — 어긋나면 train != inference
- 값이 **데이터셋에 bake** 되므로, 바꾸려면 **재수집(env 재실행)** 필요

### 구현 위치
- **`envs/metaworld/canonical.py`**: `state4_to_canonical10(state4, gripper_threshold)` 가 **내부에서 이진화** (Step 2)
  - 왜 내부에서: 수집·rollout 이 **같은 함수**를 지나므로 이진화 누락이 구조적으로 불가능 → train==inference 보장
- **수집 스크립트** (Step 3): threshold 를 config 로 받아 넘김. **시프트 없음** — `action[t] = state[t+1]` 규칙 그대로
- **Phase 2**: 이진화 관련 config/ProcessorStep **불필요** (bake 했으므로). 단 pose 는 Phase 2 에서 런타임 relative 변환됨(6절)

### 대가 (감수하기로 함)
- **threshold 가 태스크마다 필요**하고 틀리면 조용히 깨짐
- 연속 vs 이진 **ablation 불가** — 원하면 재수집(env 재실행)
- metaworld 의 *연속* 그리퍼 정보(쥠 정도)는 데이터셋에 남지 않음

## 5. `xyz_scale` — env 상수이지 통계가 아님 ⚠

canonical 은 **절대 목표 위치**, metaworld 액션은 **상대 이동량**이라 경계에서 나눠줍니다:
```python
delta_xyz = (target_xyz − current_ee_xyz) / xyz_scale     # canonical10_to_env_action
```
`xyz_scale` = **"액션 1.0 이 몇 미터냐"**.

### ⚠ 프로세서가 **두 종류** — 서로 다른 변환을 한다
| 프로세서 | 하는 일 |
|---|---|
| **policy** pre/post (Phase 2) | Rename · AddBatch · **anchor-relative 변환** · Device · Normalize/Unnormalize |
| **env** processor (Phase 5) | canonical ↔ env (`state4_to_canonical10` / `canonical10_to_env_action`) — **여기서 절대→delta(m→액션단위)** |

```
학습:  dataset(절대) ─────────────► policy_pre[relative 변환 + 정규화] → policy → relative action

추론:  env → env_proc_pre(절대) ──► policy_pre[relative 변환 + 정규화] → policy → relative action
                                        → policy_post[역정규화]
                                        → **decode_policy_action[역변환: base @ action]** → 절대 canonical
                                        → env_proc_post[canonical10_to_env_action] → env 4D
```
→ **세 군데서 변환한다**. 특히 **역변환을 빠뜨리면** 정책의 relative 출력이 절대로 해석돼 **완전히 틀린 명령**이 된다(6절).

### 정답: `ENV_XYZ_SCALE = 0.01` (= env 의 `action_scale`)
env 소스에 박혀 있습니다:
```python
# metaworld/sawyer_xyz_env.py:327
pos_delta = np.clip(action, -1, 1) * self.action_scale
# :182   action_scale: float = 1.0 / 100
```
**실측 검증** (env 를 직접 구동, 정상상태 |dxyz|/step):
| action | 실측 | 기대 `action × 0.01` |
|---|---|---|
| 1.00 | **0.01003** | 0.01000 |
| 0.50 | **0.00513** | 0.00500 |
| 0.25 | **0.00261** | 0.00250 |

→ 정확히 선형. **태스크 무관**(env 상수)이라 그리퍼 threshold 와 성격이 다름.

### ⚠ 데이터 분포에 맞추지 말 것 — 두 번 낚인 지점
손은 mocap 을 **지연 추종**합니다(weld + `frame_skip=5`). 그래서 관측된 `|dxyz|` 는:
- 초기 ~10스텝 **램프업** (`0.0012 → 0.0028 → … → 0.01`)
- 뒤처졌다 **따라잡을 때 `action_scale` 을 초과** (pick-place 실측 mean 0.008, **max 0.016 > 0.01**)

이건 **응답(response)** 이지 **이득(gain)** 이 아닙니다. 통계로 잡으면:
| 잘못된 값 | 결과 |
|---|---|
| `0.0155` (p95) | 0.008m 원할 때 action 0.52 → 실제 0.005m → **매 스텝 35% 미달 → 누적 드리프트** |
| `0.004` (옛 메모) | 0.008m 원할 때 action 2.0 → **항상 클립 → 항상 최고속 → 미세조정 불가** (과거 grasp 실패와 일치) |

### 실측 통계의 올바른 용도 = **sanity check**
```
mean 0.008 / 0.01 = 0.8    → 전형 액션 ≈0.8, [-1,1] 범위를 잘 씀 ✓
max  0.016  / 0.01 = 1.6   → 상위 몇 %만 클립(따라잡기 구간) → 정상
```
이 비율이 0.1 이나 5.0 이면 뭔가 잘못된 것.

## 6. 런타임 표현 — 정책은 **앵커 기준 relative** 를 본다 ★

데이터셋은 절대 canonical 을 담지만(1절), **정책 입력은 런타임에 relative 로 변환**된다. 이게 정책이 **데이터셋·좌표계에 비의존**인 이유이자, custom policy 가 존재하는 이유다.

### 왜 오프라인에 못 굽나 (dev_plan §3.2)
> *"relative/delta 표현은 frame 자체의 고정 속성이 아니라 **sample window 의 anchor 에 종속된 표현**"*

**앵커 = 샘플된 관측 윈도우의 마지막 프레임**. 학습 때 윈도우를 어디서 자르냐에 따라 앵커가 달라지므로 **오프라인 고정 저장이 원천적으로 불가능**하다. → 반드시 런타임.

### 무엇을 하나
| step | 대상 | 동작 |
|---|---|---|
| `CanonicalPoseToRelativeObservationStep` | `observation.state` `(B,T,10)` | 앵커 = `state[:, -1]`(최신 관측) → 전 히스토리를 `anchor⁻¹ @ state` 로 |
| `CanonicalPoseToActionPoseReprStep` | `action` `(B,H,10)` | 같은 앵커로 `relative`, 또는 프레임간 `delta` |

- **그리퍼는 변환 안 함** — pose 9D 만 변환, gripper 1D 는 그대로 이어붙임
- **차원 유지** 10D→10D (dev_plan §9.4)
- **`action` 이 없으면 skip** — eval/추론 raw 관측엔 action 이 없음 (dev_plan §9.3)

### ★ 추론엔 **역변환이 필수** — 없으면 완전히 틀린 명령
정책은 **relative 를 뱉는다**. 그대로 env/로봇에 넣으면 안 되고, **절대로 되돌려야** 한다:
```
정책(relative) ──[역변환: base @ action, 앵커=현재 관측]──▶ 절대 canonical ──▶ env/로봇 명령
```
| 구현 | 위치 |
|---|---|
| **원본 UMI** | `convert_pose_mat_rep(..., backward=True)` in `get_real_umi_action` |
| **lerobot_hong** | `decode_policy_action` — 수식 동일, 호출부 5곳(`run_umidiffusion_inference`/`_ros2`/`offline_`/**`rollout_metaworld_`**/smoke) |

**파이프라인 밖에 두는 이유**: `policy_post` 는 `PolicyAction` 만 받아 **앵커(현재 관측)에 접근 불가**. 그래서 원본·lerobot_hong 모두 **추론 루프가 직접 호출**한다. → 우리 Phase 5 rollout 도 같은 형태.

### 두 config 의 역할은 **비대칭**
| config | 담당 | 방향 |
|---|---|---|
| `obs_pose_repr` | **관측**만 | **forward 뿐** — 정책이 관측을 만들지 않으니 backward 자체가 없음. lerobot_hong 은 `"relative"` 만 허용(`__post_init__` raise) |
| **`action_pose_repr`** | **액션**만 | **forward(학습) + backward(추론) 양쪽에 같은 값** |

```
[학습] dataset 절대 action ──forward(action_pose_repr)──▶ relative 정답   앵커=state[:,-1]
[추론] 정책 relative 출력   ──backward(action_pose_repr)──▶ 절대 action    앵커=obs[-1]
```
**forward/backward 가 같은 `action_pose_repr` + 같은 앵커** → 정확한 역함수 → train==inference. 하나라도 어긋나면 깨진다.

> ⚠ **원본 UMI 의 버그**: `umi_dataset.py:349` 가 action 변환에 `pose_rep=self.obs_pose_repr` 을 넘긴다(`action_pose_repr` 이어야 함). `self.action_pose_repr` 은 대입만 되고 **학습에서 미사용**. 기본값이 둘 다 `relative` 라 안 드러나지만 `action_pose_repr: delta` 로 두면 **학습=relative / 추론=delta** 로 조용히 갈라진다.
> → **lerobot_hong 은 이미 올바르게 고쳐놨다**(학습·추론 모두 `action_pose_repr` 사용). 우리도 그쪽을 따른다.

### `relative` vs `delta`
| | forward | backward | 오차 특성 |
|---|---|---|---|
| **`relative`** (기본) | `a_i' = inv(base) @ a_i` — 전부 **같은 앵커** | `a_i = base @ a_i'` | 한 스텝 틀려도 **다른 스텝에 전파 안 됨** |
| `delta` | `d_0 = a_0 − base`, `d_i = a_i − a_{i-1}` — **연쇄** | `a_i = base + cumsum(d)[i]` | 한 스텝 틀리면 **이후 전부 누적 오염** |

### ★ 데이터 비의존성의 수학적 근거
```
relative_transform(anchor, state) = anchor⁻¹ @ state
좌표계를 F → T@F 로 바꾸면:
  (T@anchor)⁻¹ @ (T@state) = anchor⁻¹ @ T⁻¹ @ T @ state = anchor⁻¹ @ state   ← 동일!
```
**오프라인 좌표계가 상쇄된다.** 그래서 metaworld(월드 절대)든 UMI(episode-start 상대)든 **정책이 보는 값은 같은 의미**가 된다. → "정책이 데이터에 의존하면 안 된다"는 요구가 이 한 줄로 보장됨.

### 정규화가 IDENTITY 인 이유 (dev_plan §11)
> *"dataset stats 는 canonical 기준인데 런타임이 relative 로 바꾼다 → canonical stats 를 그대로 쓰면 **표현 공간이 안 맞을 수 있다**"*

→ 1차 전략으로 `STATE=IDENTITY`, `ACTION=IDENTITY`(정규화 안 함). relative 값은 이미 0 근처 작은 범위라 무방.
→ 이게 **유일한 근거**다. 필요해지면 relative 기준 stats 를 따로 계산하는 방향으로 확장.

> ⚠ **"rot6d std=0 나눗셈 회피"는 근거가 아니다** (2-0 실측으로 반증). lerobot 이 이미 막는다:
> ```python
> # lerobot/processor/normalize_processor.py:94, :335
> eps: float = 1e-8
> denom = std + self.eps          # Avoid division by zero
> ```
> MEAN_STD 로 둬도 NaN 이 안 난다(상수 채널은 `0/1e-8 = 0` 이 됨 — 죽은 채로 들어갈 뿐 터지진 않음).

## 7. 회전 (rot6d)

| 소스 | rot6d 채널 |
|---|---|
| **metaworld** (Sawyer) | EE 가 회전 안 함(mocap weld) + obs 가 회전을 노출조차 안 함 → `IDENTITY_ROT6D` 로 채움. **std=0 실측 확인** (정보량 0, 정상) |
| **UMI** | SLAM 에서 온 **진짜 회전** |
| **franka** | FK 에서 온 **진짜 회전** |

→ **값(`IDENTITY_ROT6D`)은 표현 상수라 어디서나 동일**, *쓰느냐*만 embodiment 별.

## 8. 변경 이력

### 2026-07-16 — UMI raw 그리퍼 규약 재작성 (250 에피소드)
`/home/hong/datasets/umi_ft/move260626/ft_data/episode_*.h5`

| | 값 | `state_rule` attr | 물리적 의미 |
|---|---|---|---|
| 이전 | `1` | `0=open, 1=close` | 그리퍼 **닫힘** |
| **이후** | **`0`** | **`0=close, 1=open`** | 그리퍼 **닫힘** (동일) |

- **값과 attr 을 같이** 바꿔 파일이 여전히 자기를 정확히 문서화한다. 물리적 사실은 보존됨.
- 목적: raw 를 canonical 과 같은 극성으로 맞춰 **컨버터에서 뒤집기 없이 통과**시키기 위함.
- ⚠ 이 변경 이후로 raw 를 읽는 모든 코드는 **`0=닫힘`** 기준. 옛 변환본(`move260626_rel_clean_*` 등)은 **뒤집기 없던 시절 산출물이라 그리퍼가 전부 `1`** → 새 규약상 "열림"으로 오독됨. **Phase 6 에서 재변환 필요**(패치 아님).

## 9. 미해결 / TODO
- [x] `gripper_threshold` pick-place 실측 확정 → **0.7** (`PICK_PLACE_GRIPPER_THRESHOLD`). 다른 태스크는 재실측 필요
- [x] `xyz_scale` 확정 → **0.01** (`ENV_XYZ_SCALE`, env 상수 + 실측 검증). 5절
- [ ] UMI move260626 그리퍼가 **상수 0**(항상 닫힘) — 태스크 특성인지 기록 오류인지 Phase 6 에서 확인
- [ ] franka 그리퍼 번역 규약 (Phase 9)
- [ ] metaworld 연속 vs 이진 ablation 결과 기록
- [ ] Phase 5 rollout 이 `ENV_XYZ_SCALE`·`PICK_PLACE_GRIPPER_THRESHOLD` 를 **수집과 같은 값**으로 쓰는지 확인 (train==inference)
- [ ] **역변환 회귀 테스트**: `decode_policy_action(forward(a, anchor), anchor) == a` 왕복 검증 (`relative`/`delta` 둘 다) — 원본 UMI 가 여기서 버그를 냈으므로 필수
- [ ] Phase 2 에서 `action_pose_repr` 이 **학습·추론 양쪽에** 같은 값으로 흐르는지 확인 (원본 UMI 는 학습에서 `obs_pose_repr` 을 잘못 씀)
