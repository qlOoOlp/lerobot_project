# pose_math — canonical 10D 와 anchor-relative 표현의 수학

> 이 문서는 **왜 이 수식인가**를 공부하기 위한 것이다. 코드는
> `custom/utils/lerobot_canonical/src/lerobot_canonical/schemas/canonical_ee10_se3.py` (codec) 와
> `custom/policies/umidiffusion/.../steps.py` (앵커 의미) 에 있다.
> 설계 맥락은 `refactoring.md` 부록 D, 번역 계약은 `retargeting.md`.

---

## 0. 기호

| 기호 | 뜻 |
|---|---|
| $t \in \mathbb{R}^3$ | 위치 (translation) |
| $R \in SO(3)$ | 회전행렬. $R^\top R = I$, $\det R = +1$ |
| $T \in SE(3)$ | 동차변환행렬 (4×4) |
| $\mathbf{c}_1,\mathbf{c}_2,\mathbf{c}_3$ | $R$ 의 **열**벡터. $R = [\,\mathbf{c}_1 \mid \mathbf{c}_2 \mid \mathbf{c}_3\,]$ |
| $g \in \{0,1\}$ | 그리퍼 (0=닫힘, 1=열림) |
| $a \in \{0,\dots,H-1\}$ | 액션 인덱스 (horizon $H=16$) |
| $s \in \{0,\dots,T-1\}$ | 관측 인덱스 ($T = n\_obs\_steps = 2$) |

---

## 1. canonical 10D — 무엇을 10개 숫자로 적는가

$$
\mathbf{x} = \underbrace{[\,x,\ y,\ z,\ \underbrace{r_0, r_1, r_2, r_3, r_4, r_5}_{\text{rot6d}}\,]}_{\text{pose } 9\text{D}} \ \Vert\ \underbrace{[\,g\,]}_{1\text{D}}
$$

**10개 숫자인데 자유도는 7** 입니다:

$$
\underbrace{3}_{\text{위치}} + \underbrace{3}_{\text{회전}} + \underbrace{1}_{\text{그리퍼}} = 7 \quad\text{vs}\quad 10 \text{ 개 숫자}
$$

회전 3자유도를 **6개로 과표현**합니다. 왜 낭비를 하는가 — 다음 절.

### 1.1 왜 rot6d 인가 (연속성)

신경망이 회전을 **출력**하려면 그 표현이 연속이어야 합니다. 즉 "가까운 회전 ⇒ 가까운 숫자".

| 표현 | 개수 | 문제 |
|---|---|---|
| 오일러각 $(\alpha,\beta,\gamma)$ | 3 | **짐벌락**: $\beta = \pi/2$ 에서 $\alpha,\gamma$ 축이 겹쳐 자유도 1 상실. 그 근처에서 값이 발산 |
| 쿼터니언 $q$ | 4 | **이중덮개**: $q$ 와 $-q$ 가 **같은 회전**. 같은 자세인데 라벨이 두 개 → 신경망이 평균내면 $\frac{q + (-q)}{2} = 0$ (회전이 아님) |
| **rot6d** | 6 | 연속. 아래 |

> **수학적 사실** (Zhou et al. 2019): $SO(3)$ 를 $\mathbb{R}^d$ 로 **연속적으로** 매장하려면 $d \ge 5$ 가 필요합니다.
> $d=3$(오일러)·$d=4$(쿼터니언)로는 **원리적으로 불가능**합니다. rot6d 는 $d=6$ 으로 이를 만족합니다.

### 1.2 rot6d 의 정의 — 앞 두 열

$$
\text{rot6d}(R) = [\,\mathbf{c}_1^\top \ \Vert\ \mathbf{c}_2^\top\,] \in \mathbb{R}^6,
\qquad R = [\,\mathbf{c}_1 \mid \mathbf{c}_2 \mid \mathbf{c}_3\,]
$$

**$\mathbf{c}_3$ 를 버려도 정보 손실이 0** 입니다:

$$
\mathbf{c}_3 = \mathbf{c}_1 \times \mathbf{c}_2
$$

$R$ 이 직교행렬이고 $\det R = +1$ 이므로 세 열은 **오른손 정규직교기저**입니다. 두 개를 알면 세 번째는 외적으로 유일하게 결정됩니다.

> ⚠ **코드 함정**: `R[..., :, 0]` 이 첫 **열**입니다. `R[..., 0, :]` 은 첫 **행**이고 이건 $R^\top$ 의 열입니다.
> 마지막 두 축이 `(row, col)` 이라 헷갈리는데, **단위행렬로 테스트하면 행=열이라 안 잡힙니다.**
> 반드시 비대칭 회전으로 검증할 것. (`canonical_ee10_se3.py` 의 그 주석이 이 이야기)

### 1.3 복원 — Gram-Schmidt

신경망은 제약 없는 6개 숫자 $[\mathbf{a}_1 \Vert \mathbf{a}_2]$ 를 뱉습니다. 이게 정규직교일 리 없습니다. 그래서:

$$
\begin{aligned}
\mathbf{b}_1 &= \frac{\mathbf{a}_1}{\lVert \mathbf{a}_1 \rVert} \\[4pt]
\mathbf{b}_2 &= \frac{\mathbf{a}_2 - (\mathbf{b}_1 \cdot \mathbf{a}_2)\,\mathbf{b}_1}{\lVert \mathbf{a}_2 - (\mathbf{b}_1 \cdot \mathbf{a}_2)\,\mathbf{b}_1 \rVert} \\[4pt]
\mathbf{b}_3 &= \mathbf{b}_1 \times \mathbf{b}_2
\end{aligned}
\qquad\Longrightarrow\qquad
R = [\,\mathbf{b}_1 \mid \mathbf{b}_2 \mid \mathbf{b}_3\,]
$$

**둘째 줄의 의미**: $\mathbf{a}_2$ 에서 $\mathbf{b}_1$ **방향 성분을 빼면** 남는 건 $\mathbf{b}_1$ 과 직교입니다. $(\mathbf{b}_1 \cdot \mathbf{a}_2)\mathbf{b}_1$ 이 그 사영이죠.

**이게 rot6d 의 핵심 이점**입니다 — 신경망이 **무엇을 뱉든** 유효한 회전이 나옵니다. 쿼터니언은 정규화해도 $q/-q$ 문제가 남지만, rot6d 는 그런 게 없습니다.

**왕복 성질** (테스트 설계에 중요):
$$
R \to \text{rot6d} \to R \quad\text{은 항등} \qquad
\text{임의의 } \mathbf{v} \in \mathbb{R}^6 \to R \to \text{rot6d} \quad\text{은 항등이 아님}
$$
후자는 정규화가 개입하기 때문입니다. **테스트는 반드시 $R$ 기준 왕복으로** 짜야 합니다.

---

## 2. pose9d ↔ SE(3)

### 2.1 조립

$$
T(\mathbf{x}_{9}) =
\begin{bmatrix}
R & \mathbf{t} \\
\mathbf{0}^\top & 1
\end{bmatrix}
=
\begin{bmatrix}
r_{11} & r_{12} & r_{13} & x \\
r_{21} & r_{22} & r_{23} & y \\
r_{31} & r_{32} & r_{33} & z \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

- $R = \text{GramSchmidt}(r_0 \dots r_5)$
- $\mathbf{t} = (x, y, z)^\top$ — **마지막 열**의 앞 3개

> ⚠ `[..., 3, 3] = 1.0` 을 빠뜨리면 동차좌표가 깨져 **곱셈·역변환이 조용히 틀립니다.**
> `zeros_like` 로 만들면 그 자리가 0으로 남습니다.

### 2.2 왜 4×4 인가 — 회전과 이동을 한 번에

3×3 만으론 회전밖에 못 담습니다. 이동까지 담으려면 차원을 하나 늘려 **동차좌표**를 씁니다:

$$
\begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix}
\begin{bmatrix} \mathbf{p} \\ 1 \end{bmatrix}
=
\begin{bmatrix} R\mathbf{p} + \mathbf{t} \\ 1 \end{bmatrix}
$$

$R\mathbf{p} + \mathbf{t}$ — **회전시킨 뒤 이동**. 이게 한 번의 행렬곱이 됩니다. 그래서 변환의 **합성이 곱**이 됩니다:

$$
T_2 (T_1 \mathbf{p}) = (T_2 T_1)\mathbf{p}
$$

### 2.3 역변환 — $\det$ 없이 닫힌 해

$$
T^{-1} =
\begin{bmatrix}
R^\top & -R^\top \mathbf{t} \\
\mathbf{0}^\top & 1
\end{bmatrix}
$$

**유도**: $T^{-1}T = I$ 를 만족하는지 직접 곱해봅니다.

$$
\begin{bmatrix} R^\top & -R^\top \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix}
\begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix}
=
\begin{bmatrix} R^\top R & R^\top \mathbf{t} - R^\top \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix}
=
\begin{bmatrix} I & \mathbf{0} \\ \mathbf{0}^\top & 1 \end{bmatrix} \ \checkmark
$$

핵심은 $R^\top R = I$ — **회전행렬은 전치가 곧 역**입니다.

> ★ **`torch.linalg.inv` 를 쓰지 말 것.** 일반 역행렬은 LU 분해로 $O(n^3)$ 이고 수치적으로 불안정할 수 있습니다.
> 위 공식은 **전치 하나 + 행렬-벡터곱 하나**입니다. lerobot_hong 이 offline 에선 `np.linalg.inv`,
> online 에선 전치를 써서 **같은 이름의 함수 구현이 갈렸습니다** (refactoring.md 부록 D.5).

---

## 3. anchor-relative — 이 프로젝트의 심장

### 3.1 정의

관측 히스토리 $T_0, T_1, \dots, T_{T-1}$ 에 대해 **앵커는 마지막 프레임**:

$$
T_{\text{anchor}} := T_{T-1} \quad (\text{= "지금 내 자세"})
$$

각 프레임을 앵커 기준으로 다시 씁니다:

$$
\boxed{\ \tilde{T}_s = T_{\text{anchor}}^{-1} \, T_s\ }
$$

액션도 **같은 앵커**로:

$$
\tilde{A}_a = T_{\text{anchor}}^{-1} \, A_a
$$

### 3.2 앵커 자신은 항등이 된다

$$
\tilde{T}_{T-1} = T_{\text{anchor}}^{-1} T_{\text{anchor}} = I
\qquad\Longrightarrow\qquad
\mathbf{x}_9 = [\,0,0,0,\ \underbrace{1,0,0,\ 0,1,0}_{\text{IDENTITY\_ROT6D}}\,]
$$

**정보량이 0인 채널이 하나 생깁니다.** 낭비 같지만 정상입니다 —
정보는 **이전 히스토리 프레임**(내가 어디서 왔나)과 **이미지**(세상이 어떻게 생겼나)에 있습니다.

> 이것이 `canonical_ee10.py` 의 `IDENTITY_ROT6D = (1,0,0,0,1,0)` 이 나오는 자리입니다:
> $I_{3\times3}$ 의 첫 두 열 $\mathbf{c}_1 = (1,0,0)$, $\mathbf{c}_2 = (0,1,0)$.

### 3.3 ★ 데이터 비의존성 — 월드 프레임이 상쇄된다

**주장**: 데이터셋이 어느 좌표계를 쓰든 정책이 보는 값은 같다.

**증명**: 월드 프레임을 $W \in SE(3)$ 로 바꾸면 모든 자세가 $T \mapsto W T$ 로 변합니다. 그러면

$$
\begin{aligned}
\widetilde{(WT_s)} &= (W T_{\text{anchor}})^{-1} (W T_s) \\
&= T_{\text{anchor}}^{-1} \, \underbrace{W^{-1} W}_{= I} \, T_s \\
&= T_{\text{anchor}}^{-1} T_s \\
&= \tilde{T}_s \qquad \blacksquare
\end{aligned}
$$

**$W$ 가 사라집니다.** 그래서:

- metaworld 의 테이블 원점이든
- UMI 의 SLAM 원점이든 (에피소드마다 다름! 심지어 촬영 중 19번 리셋됨)
- franka 의 베이스 원점이든

**정책에겐 전부 같은 의미**입니다. *"정책이 데이터에 의존하면 안 된다"* 가 이 세 줄로 보장됩니다.

> **실측 검증** (2026-07-17): 랜덤 SE(3) 64개에 임의의 $W$ 를 곱해도 결과 동일 (`atol=1e-5`).

### 3.4 왜 오프라인에 못 굽나

$T_{\text{anchor}}$ 는 **샘플 윈도우의 마지막 프레임**입니다. 학습 시 DataLoader 가 궤적을 **어디서 자르냐**에 따라 달라집니다:

```
궤적:  ─────●─────●─────●─────●─────●─────
윈도우 A:      [  ●  ●  ]           앵커 = 두번째 ●
윈도우 B:            [  ●  ●  ]     앵커 = 네번째 ●   ← 같은 프레임이 다른 값을 갖는다
```

**같은 프레임이 어느 윈도우에 속하느냐에 따라 다른 값**이 됩니다. 그러니 "프레임마다 하나의 값"인 디스크에 미리 적을 수가 없습니다.

**대비 — 구울 수 있는 것들**:

| | 무엇에 의존하나 | 굽나 |
|---|---|---|
| 그리퍼 이진화 | 고정 상수 (`threshold=0.7`) | ✅ 수집 때 bake |
| 이미지 flip | 고정 (카메라 장착각) | ✅ 수집 때 bake |
| resize 240 | 고정 | ✅ 수집 때 bake |
| **anchor-relative** | **윈도우** | ❌ **런타임 필수** |

> **이것이 커스텀 정책이 존재하는 이유입니다.** `--policy.type=diffusion` 을 그대로 쓰면
> 기본 프로세서에 이 변환이 없습니다.

---

## 4. 정책 입력 — 단계별 수치 추적

로봇이 $x$ 축으로 이동 중, $n\_obs\_steps = 2$.

### 0단계 — 데이터셋 (절대 canonical)

```
state[0] = [0.28, 0.6, 0.1,  1,0,0, 0,1,0,  1.0]    ← 1프레임 전 (월드 좌표)
state[1] = [0.30, 0.6, 0.1,  1,0,0, 0,1,0,  1.0]    ← 지금
```

### 1단계 — pose / gripper 분리

$$
\mathbf{x}_{10} = [\underbrace{\mathbf{x}_9}_{\text{변환 대상}} \ \Vert\ \underbrace{g}_{\text{그대로}}]
$$

그리퍼는 **좌표계와 무관**합니다 — *"0.7만큼 열려라"* 는 내가 어디 있든 같은 뜻이니까요. 그래서 여기서 갈라져 끝까지 안 건드립니다.

> ⚠ 코드 함정: `x[..., 9:]` (슬라이스) 는 `(...,1)`, `x[..., 9]` (정수) 는 `(...)`. 후자는 **축이 사라져** 나중에 `cat` 이 깨집니다.

### 2단계 — SE(3) 로

$$
T_0 = \begin{bmatrix} I & (0.28, 0.6, 0.1)^\top \\ \mathbf{0}^\top & 1\end{bmatrix},
\qquad
T_1 = \begin{bmatrix} I & (0.30, 0.6, 0.1)^\top \\ \mathbf{0}^\top & 1\end{bmatrix}
$$

(metaworld 의 Sawyer 는 EE 가 회전하지 않아 $R = I$. UMI 는 진짜 회전이 들어갑니다.)

텐서 shape: $(B, T, 9) \to (B, T, 4, 4)$

### 3단계 — 앵커

$$
T_{\text{anchor}} = T_1
$$

> ⚠ 코드 함정: `state_transform[:, -1:]` (슬라이스, 축 유지) 후 `expand_as`.
> `[:, -1]` 이면 $T$ 축이 사라져 `(B,4,4)` 가 되고 broadcast 가 깨집니다.

### 4단계 — relative

$$
\tilde{T}_0 = T_1^{-1} T_0
= \begin{bmatrix} I & (-0.02, 0, 0)^\top \\ \mathbf{0}^\top & 1\end{bmatrix},
\qquad
\tilde{T}_1 = T_1^{-1} T_1 = I
$$

$R = I$ 라서 $T_1^{-1} T_0$ 의 이동 부분이 단순 뺄셈이 됐지만, **일반적으로는**

$$
T_{\text{anchor}}^{-1} T_s =
\begin{bmatrix}
R_{\text{a}}^\top R_s & R_{\text{a}}^\top (\mathbf{t}_s - \mathbf{t}_{\text{a}}) \\
\mathbf{0}^\top & 1
\end{bmatrix}
$$

**이동 부분에 $R_{\text{a}}^\top$ 가 붙습니다** — "앵커의 좌표계에서 본 변위" 이기 때문입니다. 이게 3.3 의 상쇄가 성립하는 이유이고, 뒤에 나올 `delta` 와 갈리는 지점입니다.

### 5–6단계 — 9D 로 되돌리고 gripper 재결합

```
state_rel[0] = [-0.02, 0, 0,  1,0,0, 0,1,0,  1.0]     "1프레임 전은 지금 나로부터 -2cm"
state_rel[1] = [ 0.00, 0, 0,  1,0,0, 0,1,0,  1.0]     "지금은 나 자신"
                                             └ 원본 그대로
```

**월드 좌표 0.28 / 0.30 은 사라졌습니다.** 이게 정책이 보는 값입니다.

---

## 5. 액션 — forward 와 backward

### 5.1 학습 (forward): 정답을 정책 언어로

$$
\tilde{A}_a = T_{\text{anchor}}^{-1} A_a
$$

```
action[0]  = [0.32, ...] 절대  →  [+0.02, 0, 0, ...]   "2cm 앞으로"
action[15] = [0.45, ...] 절대  →  [+0.15, 0, 0, ...]   "15cm 앞으로"
```

**앵커가 관측과 같습니다.** 그래서 입력 *"2cm 뒤에서 왔다"* ↔ 정답 *"2cm 앞으로 가라"* 로 짝이 맞습니다.

> ★ 앵커를 **액션의 첫 프레임**에서 뽑으면 안 됩니다 — 관측과 기준이 갈라져 정책이 배울 게 없어집니다.
> 그리고 **파이프라인에서 action step 이 obs step 보다 먼저** 와야 합니다:
> obs step 이 먼저 돌면 `state[:,-1]` 이 이미 $I$ 로 바뀌어, action step 이 **항등을 앵커로 삼습니다**
> → 액션이 전혀 변환되지 않고 **에러 없이 전부 망가집니다.**

### 5.2 추론 (backward): 정책 말을 세상 언어로

$$
A_a = T_{\text{anchor}} \, \tilde{A}_a
$$

**forward 의 정확한 역**입니다:

$$
T_{\text{anchor}} \left( T_{\text{anchor}}^{-1} A_a \right) = A_a \ \checkmark
$$

```
정책 출력  [+0.02, 0, 0, ...]   ← relative. 이대로 주면 "월드 원점에서 2cm" 로 오해
   ↓  decode_policy_action(action, anchor_state)
   ↓  T_anchor @ Ã
[0.32, 0.6, 0.1, ...]           ← 절대 canonical
   ↓  canonical10_to_env_action
env 4D
```

> **왜 파이프라인 밖인가**: post 파이프라인은 `PolicyAction`(텐서)만 받아 **앵커에 접근할 수 없습니다.**
> 그래서 rollout 루프가 직접 부릅니다. 원본 UMI(`get_real_umi_action`)·lerobot_hong(`decode_policy_action`)
> 모두 같은 구조입니다.
>
> **빠뜨리면**: 정책의 *"지금 기준 +2cm"* 를 *"월드 좌표 2cm 지점"* 으로 오해 → **완전히 틀린 명령**.

**실측 검증**: `decode(forward(a, anchor), anchor) == a`, 최대오차 $7.2 \times 10^{-7}$.

---

## 6. `delta` — 대안 표현과 그 함정

원본 UMI 는 `relative` 말고 `delta` 도 제공합니다 (`pose_repr_util.py:65-77`).

### 6.1 정의 — SE(3) 합성이 **아니다**

$$
\begin{aligned}
\text{위치:} \quad & \Delta\mathbf{t}_a = \mathbf{t}_a - \mathbf{t}_{a-1}, \qquad \mathbf{t}_{-1} := \mathbf{t}_{\text{anchor}} \\
\text{회전:} \quad & \Delta R_a = R_a R_{a-1}^{\top}, \qquad\quad\ \ R_{-1} := R_{\text{anchor}}
\end{aligned}
$$

**위치는 뺄셈, 회전은 곱셈** — 비대칭입니다. SE(3) 합성이라면 $\Delta T_a = T_{a-1}^{-1} T_a$ 여야 하는데, 그게 **아닙니다**.

### 6.2 역변환

$$
\begin{aligned}
\mathbf{t}_a &= \mathbf{t}_{\text{anchor}} + \sum_{j \le a} \Delta\mathbf{t}_j \qquad (\text{cumsum}) \\
R_a &= \Delta R_a R_{a-1} \qquad\qquad\quad\ (\text{순차 곱, 벡터화 불가})
\end{aligned}
$$

회전은 $R_a$ 가 $R_{a-1}$ 에 의존하는 **연쇄**라 루프를 돌아야 합니다. 이게 delta 가 **오차 누적**에 취약한 이유이기도 합니다 — relative 는 모든 항이 같은 앵커를 보므로 오차가 전파되지 않습니다.

### 6.3 ★ delta 는 프레임 비의존이 **아니다**

3.3 의 상쇄가 delta 엔 성립하지 않습니다. 월드 프레임 $W = (R_W, \mathbf{t}_W)$ 를 곱하면:

$$
\begin{aligned}
\mathbf{t}_a &\mapsto R_W \mathbf{t}_a + \mathbf{t}_W \\[2pt]
\Delta\mathbf{t}_a &\mapsto (R_W \mathbf{t}_a + \mathbf{t}_W) - (R_W \mathbf{t}_{a-1} + \mathbf{t}_W)
= R_W (\mathbf{t}_a - \mathbf{t}_{a-1}) = R_W \, \Delta\mathbf{t}_a
\end{aligned}
$$

**$\mathbf{t}_W$ 는 상쇄되지만 $R_W$ 는 남습니다.** 회전도 마찬가지:

$$
\Delta R_a \mapsto (R_W R_a)(R_W R_{a-1})^\top = R_W \, \Delta R_a \, R_W^\top \qquad (\text{켤레})
$$

| | 평행이동 | **회전** | 둘 다 |
|---|---|---|---|
| `relative` | 불변 | **불변** | **불변** ✅ |
| `delta` | 불변 | **변함** | **변함** ❌ |

**실측으로 확인했습니다** (2026-07-17). 구현 버그가 아니라 **정의 자체의 성질**입니다 — 위치를 월드 프레임에서 뺄셈하기 때문.

**우리에겐 무해합니다** — `obs_pose_repr` 은 `"relative"` 만 허용하고 `action_pose_repr` 기본값도 relative 입니다.
**단 `action_pose_repr="delta"` 로 바꾸는 순간 UMI↔metaworld 비의존성이 깨집니다.**

### 6.4 원본 UMI 의 버그

`umi_dataset.py:349` 가 **액션** 변환에 `pose_rep=self.obs_pose_repr` 을 넘깁니다 (`action_pose_repr` 이어야 함).
기본값이 둘 다 relative 라 안 드러나지만, `action_pose_repr="delta"` 로 두면
**학습=relative / 추론=delta** 로 조용히 갈라집니다.

→ 우리 구조에선 **forward step 과 `decode_policy_action` 이 같은 파일에** 있고 **같은 `action_pose_repr` 을 받습니다.**
그리고 `config.__post_init__` 과 step 의 `__post_init__` **양쪽**에서 값을 검증합니다.

---

## 7. 한눈에

```
                         ┌─────────── 어댑터 (embodiment 전용) ───────────┐
metaworld obs[:4]  ──────┤ state4_to_canonical10   (그리퍼 이진화, R = I) ├──┐
UMI h5 (xyz+rpy)   ──────┤ Phase 6 컨버터          (rpy → rot6d)         ├──┤
                         └──────────────────────────────────────────────┘  │
                                                                            ▼
                                            canonical 10D (절대) ── 디스크에 저장
                                                                            │
                         ┌────── 정책 프로세서 (공유, 학습·추론 동일) ──────┐│
                         │  T = pose9d_to_transform(x₉)                     ││
                         │  T̃ = T_anchor⁻¹ T        ← 월드 프레임 상쇄      │◀┘
                         │  x̃₉ = transform_to_pose9d(T̃)                     │
                         │  x̃₁₀ = [x̃₉ ‖ g]          ← 그리퍼 그대로          │
                         └──────────────────────────────────────────────────┘
                                                    │
                                                    ▼
                                                  정책
                                                    │  relative 액션
                                                    ▼
                         decode_policy_action:  A = T_anchor Ã     ← 파이프라인 밖
                                                    │  절대 canonical
                                                    ▼
                                       canonical10_to_env_action → env
```

**판정 기준 하나로 요약하면**:

> **오프라인에 구울 수 있나?**
> **예** (고정 상수) → 어댑터에서 bake. 수집 1번, 추론 1번. → flip, 이진화, resize
> **아니오** (윈도우 의존) → 정책 프로세서. 학습·추론 **양쪽**. → anchor-relative

---

## 8. 참고

| 무엇 | 어디 |
|---|---|
| codec 구현 | `custom/utils/lerobot_canonical/src/lerobot_canonical/schemas/canonical_ee10_se3.py` |
| 앵커 의미 + step | `custom/policies/umidiffusion/.../steps.py` |
| 표현 정의 (치수·축·상수) | `.../schemas/canonical_ee10.py` |
| 원본 UMI | `lerobot_hong/test/universal_manipulation_interface/diffusion_policy/common/pose_repr_util.py` |
| 설계 결정 기록 | `refactoring.md` 부록 D.5 (codec 중복), D.7 (벤더링) |
| 번역 계약 | `retargeting.md` |
| rot6d 원논문 | Zhou et al., *On the Continuity of Rotation Representations in Neural Networks*, CVPR 2019 |
