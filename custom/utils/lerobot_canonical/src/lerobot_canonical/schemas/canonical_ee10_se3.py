"""canonical_ee10 표현의 codec — "이 표현을 읽고 쓰는 법".

canonical_ee10.py 가 "무엇인가"(차원·축·상수)를 정의한다면, 이 모듈은 "어떻게
읽고 쓰는가"(pose9d <-> 4x4 변환행렬)를 담당한다. 둘 다 표현의 일부라 나란히 둔다.

■ 왜 여기(표현 옆)인가 — refactoring.md 부록 D.5
  컨버터(오프라인)와 런타임 step 은 **대등한 소비자**다. 어느 한쪽이 codec 을 소유하면
  나머지가 엉뚱한 패키지에 의존하게 된다(컨버터가 "processor" 패키지를 import 하는 꼴).
  표현 옆에 두면 둘 다 대등하게 import 한다. 새 표현(canonical_joint7)은 자기 codec 을 가져온다.

  ★ lerobot_hong 의 실패를 고치는 것:
      se3.py  (offline, numpy):  relative_transform = np.linalg.inv(base) @ target
      steps.py(online,  torch):  relative_transform = invert_transform(anchor) @ target
    같은 이름·같은 공식인데 **구현이 갈렸다**. raw_inspect 의 _rot6d_to_R 까지 하면 3중복.
    원인: dev_plan §8 이 "재사용할지는 구현 시 결정"으로 미룸 → 중복으로 귀결.
    => 미루지 말고 여기 한 곳에 둔다. Phase 6 UMI 컨버터는 이걸 import 만 한다.

■ torch 인 이유
  런타임 step 이 배치 텐서를 다루므로 torch 가 필수. 오프라인 컨버터도 torch 를 CPU 로
  쓰면 그만이다(lerobot 이 이미 torch 를 요구하므로 추가 비용 0). numpy 로 따로 짜면
  위의 중복이 재발한다.

■ 배치 규약
  모든 함수는 선행 축을 자유롭게 허용한다: (..., 9) / (..., 4, 4).
  (B, T, ...) 든 (B, H, ...) 든 단일 프레임이든 같은 코드로 처리.
"""
from __future__ import annotations

import torch

from lerobot_canonical.schemas import canonical_ee10 as sch


def rot6d_to_rotation_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """rot6d (..., 6) -> 회전행렬 (..., 3, 3). Gram-Schmidt.

    ■ 핵심: **아무 6개 숫자든 유효한 회전행렬**로 만든다.
      신경망이 제약 없이 6D 를 뱉어도 여기서 직교화되므로 항상 R^T R = I, det=+1.
      이게 rot6d 를 쓰는 이유(오일러=짐벌락, 쿼터니언=q/-q 이중덮개는 불연속).

    ■ 알고리즘
        a1, a2 = rot6d[..., :3], rot6d[..., 3:]
        b1 = normalize(a1)
        b2 = normalize(a2 - (b1·a2) b1)      # a1 성분 제거 -> b1 과 직교 보장
        b3 = cross(b1, b2)                   # 외적으로 3번째 열 복원
        R  = [b1 | b2 | b3]                  # 열벡터로 쌓기 (stack dim=-1)

    ■ 유의
      - b1, b2 를 **열**로 쌓아야 한다(dim=-1). 행으로 쌓으면 전치된 R 이 나와 조용히 틀림.
      - a1 이 영벡터거나 a1 ∥ a2 이면 정규화가 터진다. 학습 중엔 사실상 안 생기지만
        eps 를 둘지는 구현 판단.
      - 왕복: R -> rot6d -> R 은 **항등**(직교행렬이므로). 반대로 임의의 6D -> R -> 6D 는
        항등이 **아니다**(정규화되므로). 테스트는 R 기준 왕복으로 짤 것.
    """
    ...  # 구현 ①


def rotation_matrix_to_rot6d(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """회전행렬 (..., 3, 3) -> rot6d (..., 6).

    ■ 핵심: **앞 두 열만** 떼서 이어붙인다. c3 는 c1 × c2 로 복원되므로 정보 손실 0.
    ■ 유의
      - **열**을 떼야 한다: R[..., :, 0] 과 R[..., :, 1]. R[..., 0, :] (행)이 아니다.
        마지막 두 축이 (row, col) 이므로 헷갈리기 쉽다. 항등행렬로는 행/열이 같아서
        **테스트가 안 걸린다** — 비대칭 회전으로 검증할 것.
      - 결과 순서는 [c1(3), c2(3)] = rot6d_0..5, canonical_ee10.POSE_AXES 와 일치해야 함.
    """
    ...  # 구현 ②


def pose9d_to_transform(pose9d: torch.Tensor) -> torch.Tensor:
    """canonical pose9d (..., 9) -> 동차변환행렬 (..., 4, 4).

    pose9d = [x, y, z, rot6d(6)]  (canonical 10D 에서 gripper 를 뗀 앞 9개)

    ■ 만드는 것
        [ R(3x3)   t(3) ]
        [ 0  0  0   1   ]
      R = rot6d_to_rotation_matrix(pose9d[..., 3:]),  t = pose9d[..., :3]

    ■ 유의
      - trailing dim 이 sch.POSE_DIM(9) 인지 검증할 것. 10D(그리퍼 포함)를 그대로 넘기는
        실수가 잦다 — 호출 전에 _split_pose_and_gripper 로 잘라야 한다.
      - 마지막 행은 [0,0,0,1]. zeros_like 로 만들고 [..., 3, 3] = 1 을 빠뜨리면
        역변환·곱셈이 조용히 망가진다.
    """
    ...  # 구현 ③


def transform_to_pose9d(transform: torch.Tensor) -> torch.Tensor:
    """동차변환행렬 (..., 4, 4) -> canonical pose9d (..., 9).

    pose9d_to_transform 의 역. t = transform[..., :3, 3], rot6d = R 의 앞 두 열.

    ■ 유의
      - transform[..., :3, 3] 은 **열벡터 t**(마지막 열의 앞 3개). [..., 3, :3] 이 아니다.
      - 이 함수 + pose9d_to_transform 의 왕복이 항등이어야 한다 → 회귀 테스트 필수.
    """
    ...  # 구현 ④


def invert_transform(transform: torch.Tensor) -> torch.Tensor:
    """SE3 역변환 (..., 4, 4) -> (..., 4, 4).

    ■ ★ torch.linalg.inv 를 쓰지 말 것 — SE3 는 닫힌 해가 있다:
        R_inv = R^T                    (직교행렬이므로 전치 = 역)
        t_inv = -R^T @ t
        [ R^T   -R^T t ]
        [ 0  0  0   1  ]
      일반 역행렬보다 빠르고 **수치적으로 안정**하다. lerobot_hong 의 offline se3.py 가
      np.linalg.inv 를 써서 online 과 구현이 갈렸다(부록 D.5) — 같은 실수 반복 금지.

    ■ 유의
      - t_inv 계산 시 t 를 (..., 3, 1) 로 unsqueeze 해서 행렬곱한 뒤 squeeze.
      - zeros_like 로 만들고 [..., 3, 3] = 1.0 을 반드시 채울 것.
      - 검증: invert_transform(T) @ T == I (배치 전체에서)
    """
    ...  # 구현 ⑤


def relative_transform(anchor_transform: torch.Tensor, target_transform: torch.Tensor) -> torch.Tensor:
    """target 을 anchor 기준으로 표현: inv(anchor) @ target.

    ■ 원본 UMI 와 동일한 정의 (pose_repr_util.py:64  `'relative'`):
        out = np.linalg.inv(base_pose_mat) @ pose_mat
      (원본의 `'rel'` 은 pos/rot 을 따로 빼는 **legacy buggy** 구현 — 쓰지 말 것)

    ■ ★ 이게 데이터 비의존성의 수학적 근거
        좌표계를 F -> T@F 로 바꿔도:
          (T@anchor)⁻¹ @ (T@target) = anchor⁻¹ @ T⁻¹ @ T @ target = anchor⁻¹ @ target
        => 오프라인 좌표계가 **상쇄**된다. metaworld(월드 절대)든 UMI(SLAM 기준)든
           정책이 보는 값은 같은 의미가 된다. 테스트로 반드시 검증할 것.

    ■ 유의
      - anchor 와 target 의 shape 이 같아야 함(호출부에서 expand 로 맞춰 넣는다).
      - anchor == target 이면 결과는 **항등**. 그래서 관측 히스토리의 마지막 프레임은
        변환 후 항상 (0,0,0, 1,0,0,0,1,0) 이 된다 — 정보량 0이지만 정상.
    """
    ...  # 구현 ⑥


__all__ = [
    "rot6d_to_rotation_matrix",
    "rotation_matrix_to_rot6d",
    "pose9d_to_transform",
    "transform_to_pose9d",
    "invert_transform",
    "relative_transform",
]
