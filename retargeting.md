# Retargeting — embodiment ↔ canonical 번역 계약

> **범위**: 각 소스/싱크(metaworld env, UMI raw, franka)를 **canonical(EE-pose 10D)** 로/에서 옮기는 규약과, 정책이 그 값을 **어떻게 소비**하는지.
> 정본 정의는 `custom/common/lerobot_ext_core/schemas/canonical_ee10.py`, 설계 맥락은 `refactoring.md` 부록 D.

## 1. canonical 채널 규약 (정본)

| 채널 | 의미 |
|---|---|
| `x, y, z` | EE 위치 [m] |
| `rot6d(6)` | 회전행렬 앞 두 열 flatten. Gram-Schmidt 로 복원(3열=b1×b2). 회전 없음 = `IDENTITY_ROT6D=(1,0,0,0,1,0)` — **표현 상수**(어디서나 동일), *쓰느냐*만 embodiment 별 |
| `gripper` | **openness `[0,1]`, 0=닫힘 / 1=열림** (지배적 규약) |

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

## 4. 그리퍼는 **이진**, 변환 시 bake (런타임 프로세서 ❌)

### 결정 — 우리 계약은 **state 도 action 도 이진 `{0,1}`, 0=닫힘 / 1=열림**

| 우리가 통제하는 것 | 값 | 누가 보나 |
|---|---|---|
| `observation.state[9]` | **0=닫힘, 1=열림** | 정책 **입력** |
| `action[9]` | **0=닫힘, 1=열림** | 정책 **출력** = 데이터셋 정답 |

정책이 보고 뱉는 건 **오직 이 이진값**. 런타임 가공 없이 데이터 그대로.
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
- **Phase 2–3**: 이진화 관련 config/ProcessorStep **불필요** (bake 했으므로)

### 대가 (감수하기로 함)
- **threshold 가 태스크마다 필요**하고 틀리면 조용히 깨짐
- 연속 vs 이진 **ablation 불가** — 원하면 재수집(env 재실행)
- metaworld 의 *연속* 그리퍼 정보(쥠 정도)는 데이터셋에 남지 않음

## 5. 회전 (rot6d)

| 소스 | rot6d 채널 |
|---|---|
| **metaworld** (Sawyer) | EE 가 회전 안 함(mocap weld) + obs 가 회전을 노출조차 안 함 → `IDENTITY_ROT6D` 로 채움. **std=0 실측 확인** (정보량 0, 정상) |
| **UMI** | SLAM 에서 온 **진짜 회전** |
| **franka** | FK 에서 온 **진짜 회전** |

→ **값(`IDENTITY_ROT6D`)은 표현 상수라 어디서나 동일**, *쓰느냐*만 embodiment 별.

## 6. 변경 이력

### 2026-07-16 — UMI raw 그리퍼 규약 재작성 (250 에피소드)
`/home/hong/datasets/umi_ft/move260626/ft_data/episode_*.h5`

| | 값 | `state_rule` attr | 물리적 의미 |
|---|---|---|---|
| 이전 | `1` | `0=open, 1=close` | 그리퍼 **닫힘** |
| **이후** | **`0`** | **`0=close, 1=open`** | 그리퍼 **닫힘** (동일) |

- **값과 attr 을 같이** 바꿔 파일이 여전히 자기를 정확히 문서화한다. 물리적 사실은 보존됨.
- 목적: raw 를 canonical 과 같은 극성으로 맞춰 **컨버터에서 뒤집기 없이 통과**시키기 위함.
- ⚠ 이 변경 이후로 raw 를 읽는 모든 코드는 **`0=닫힘`** 기준. 옛 변환본(`move260626_rel_clean_*` 등)은 **뒤집기 없던 시절 산출물이라 그리퍼가 전부 `1`** → 새 규약상 "열림"으로 오독됨. **Phase 6 에서 재변환 필요**(패치 아님).

## 7. 미해결 / TODO
- [ ] `gripper_threshold` 태스크별 실측값 확정 (pick-place 우선)
- [ ] UMI move260626 그리퍼가 **상수 0**(항상 닫힘) — 태스크 특성인지 기록 오류인지 Phase 6 에서 확인
- [ ] franka 그리퍼 번역 규약 (Phase 9)
- [ ] metaworld 연속 vs 이진 ablation 결과 기록
