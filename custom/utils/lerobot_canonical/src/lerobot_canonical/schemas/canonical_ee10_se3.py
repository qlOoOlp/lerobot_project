"""canonical_ee10 표현의 codec — 그 표현을 읽고 쓰는 법.

canonical_ee10.py 가 표현이 무엇인지(차원·축·상수)를 정의한다면, 이 모듈은 그것을 어떻게
읽고 쓰는지(pose9d <-> 4x4 변환행렬)를 담당한다. 둘 다 표현의 일부라 나란히 둔다.

표현 옆에 두는 이유는 소비자가 여럿이기 때문이다. 오프라인 컨버터와 런타임 step 은 대등한
소비자라서, 어느 한쪽이 codec 을 소유하면 나머지가 엉뚱한 패키지를 import 하게 된다. 여기
두면 둘 다 대등하게 가져다 쓰고, 새 표현(canonical_joint7 등)은 자기 codec 을 따로 가진다.

torch 로 쓴 이유는 런타임 step 이 배치 텐서를 다루기 때문이다. 오프라인 컨버터도 CPU 로 쓰면
그만이고(lerobot 이 이미 torch 를 요구한다), numpy 판을 따로 만들면 같은 수학이 두 벌로
갈라진다.

모든 함수는 선행 축을 자유롭게 허용한다: (..., 9) / (..., 4, 4). (B, T, ...) 든 단일
프레임이든 같은 코드로 처리된다.
"""
from __future__ import annotations

import torch

from lerobot_canonical.schemas import canonical_ee10 as sch


def rot6d_to_rotation_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """rot6d (..., 6) -> 회전행렬 (..., 3, 3). Gram-Schmidt.

    아무 6개 숫자든 유효한 회전행렬이 된다. 신경망이 제약 없이 뱉어도 여기서 직교화되므로
    항상 R^T R = I, det = +1 이다. rot6d 를 쓰는 이유가 이것이다 — 오일러는 짐벌락이,
    쿼터니언은 q/-q 이중덮개가 불연속을 만든다.

        b1 = normalize(a1)
        b2 = normalize(a2 - (b1·a2) b1)   a1 성분을 빼면 b1 과 직교
        b3 = cross(b1, b2)                외적으로 3번째 열 복원
        R  = [b1 | b2 | b3]

    왕복 성질에 주의: R -> rot6d -> R 은 항등이지만, 임의의 6D -> R -> 6D 는 정규화 때문에
    항등이 아니다. 테스트는 R 기준으로 짤 것.
    """
    if rot6d.shape[-1] != 6:
        raise ValueError(f"Expected rot6d trailing dim 6, got {tuple(rot6d.shape)}.")

    a1, a2 = rot6d[..., :3], rot6d[..., 3:]

    # eps 를 둔다: 학습 중 a1≈0 이나 a1∥a2 는 사실상 안 생기지만, 생기면 NaN 이 되어
    # loss 전체를 오염시키고 원인 추적이 매우 어렵다. eps 는 정상 입력엔 영향이 없다.
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-8)
    # Gram-Schmidt: a2 에서 b1 성분을 빼면 b1 과 직교. keepdim 으로 (..., 1) 을 유지해야
    # broadcast 가 맞는다 — (...,) 로 줄면 마지막 축이 3과 곱해져 조용히 틀린다.
    b2 = torch.nn.functional.normalize(
        a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1, eps=1e-8
    )
    b3 = torch.cross(b1, b2, dim=-1)

    # dim=-1 로 쌓아야 b1,b2,b3 가 열이 된다. dim=-2 면 행이 되어 전치된 R 이 나오는데,
    # 단위행렬 테스트로는 행과 열이 같아 안 잡힌다. 비대칭 회전으로 검증할 것.
    return torch.stack((b1, b2, b3), dim=-1)


def rotation_matrix_to_rot6d(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """회전행렬 (..., 3, 3) -> rot6d (..., 6).

    앞 두 열만 떼서 이어붙인다. c3 는 c1 x c2 로 복원되므로 정보 손실이 없다.
    결과 순서 [c1(3), c2(3)] 는 canonical_ee10.POSE_AXES 의 rot6d_0..5 와 일치한다.
    """
    if rotation_matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrix (..., 3, 3), got {tuple(rotation_matrix.shape)}.")

    # [..., :, 0] 은 0번째 열이다. [..., 0, :] 은 행이라 전치된 값이 나오는데, 마지막 두 축이
    # (row, col) 이라 헷갈리기 쉽고 단위행렬로는 구분되지 않는다.
    c1 = rotation_matrix[..., :, 0]
    c2 = rotation_matrix[..., :, 1]
    # c3 는 버린다 — R 이 직교행렬이라 c1 x c2 로 복원된다. 정보 손실 0.
    return torch.cat((c1, c2), dim=-1)


def pose9d_to_transform(pose9d: torch.Tensor) -> torch.Tensor:
    """canonical pose9d (..., 9) -> 동차변환행렬 (..., 4, 4).

    pose9d = [x, y, z, rot6d(6)]  (canonical 10D 에서 gripper 를 뗀 앞 9개)

        [ R(3x3)   t(3) ]
        [ 0  0  0    1   ]

    10D(그리퍼 포함)를 그대로 넘기는 실수가 잦다. 호출 전에 pose 와 gripper 를 갈라야 한다.
    """
    if pose9d.shape[-1] != sch.POSE_DIM:
        raise ValueError(
            f"Expected pose9d trailing dim {sch.POSE_DIM}, got {tuple(pose9d.shape)}. "
            f"Passing the full {sch.STATE_DIM}D canonical state (with gripper) is a common mistake — "
            f"split it first."
        )

    rotation = rot6d_to_rotation_matrix(pose9d[..., sch.POSE_DIM - 6 :])
    translation = pose9d[..., :3]

    # new_zeros: dtype/device 를 입력에서 그대로 물려받는다 (torch.zeros 는 CPU/float32 로 새로 만듦).
    transform = pose9d.new_zeros((*pose9d.shape[:-1], 4, 4))
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translation   # t 는 마지막 열의 앞 3개
    transform[..., 3, 3] = 1.0            # 빠뜨리면 동차좌표가 깨져 곱셈이 조용히 틀린다
    return transform


def transform_to_pose9d(transform: torch.Tensor) -> torch.Tensor:
    """동차변환행렬 (..., 4, 4) -> canonical pose9d (..., 9).

    pose9d_to_transform 의 역. t = transform[..., :3, 3] (마지막 열의 앞 3개), rot6d = R 의 앞 두 열.
    """
    if transform.shape[-2:] != (4, 4):
        raise ValueError(f"Expected transform (..., 4, 4), got {tuple(transform.shape)}.")

    translation = transform[..., :3, 3]
    rot6d = rotation_matrix_to_rot6d(transform[..., :3, :3])
    return torch.cat((translation, rot6d), dim=-1)


def invert_transform(transform: torch.Tensor) -> torch.Tensor:
    """SE3 역변환 (..., 4, 4) -> (..., 4, 4).

    torch.linalg.inv 를 쓰지 않는다. SE3 는 닫힌 해가 있어 더 빠르고 수치적으로 안정하다:

        R_inv = R^T          회전행렬은 전치가 곧 역
        t_inv = -R^T @ t
    """
    if transform.shape[-2:] != (4, 4):
        raise ValueError(f"Expected transform (..., 4, 4), got {tuple(transform.shape)}.")

    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]

    # R^T. transpose(-1, -2) 는 마지막 두 축만 바꾸므로 선행 배치축은 그대로 산다.
    rotation_inv = rotation.transpose(-1, -2)
    # t_inv = -R^T @ t. t 를 (..., 3, 1) 로 세워 행렬곱한 뒤 다시 (..., 3) 으로 눕힌다.
    translation_inv = -torch.matmul(rotation_inv, translation.unsqueeze(-1)).squeeze(-1)

    inverse = transform.new_zeros(transform.shape)
    inverse[..., :3, :3] = rotation_inv
    inverse[..., :3, 3] = translation_inv
    inverse[..., 3, 3] = 1.0
    return inverse


def relative_transform(anchor_transform: torch.Tensor, target_transform: torch.Tensor) -> torch.Tensor:
    """target 을 anchor 기준으로 표현: inv(anchor) @ target.

    원본 UMI 의 'relative' 와 같은 정의다 (pose_repr_util.py:64). 원본의 'rel' 은 pos/rot 을
    따로 빼는 legacy 구현이라 쓰지 않는다.

    이것이 데이터 비의존성의 근거다. 좌표계를 F -> W@F 로 바꿔도

        (W@anchor)^-1 @ (W@target) = anchor^-1 @ W^-1 @ W @ target = anchor^-1 @ target

    로 W 가 상쇄된다. metaworld 의 월드 원점이든 UMI 의 SLAM 원점이든 정책이 보는 값은 같다.

    anchor 와 target 은 shape 이 같아야 하며, 호출부가 expand 로 맞춰 넣는다. anchor == target
    이면 결과는 항등이라 관측 히스토리의 마지막 프레임은 늘 (0,0,0, 1,0,0,0,1,0) 이 된다.
    """
    if anchor_transform.shape != target_transform.shape:
        raise ValueError(
            f"anchor {tuple(anchor_transform.shape)} and target {tuple(target_transform.shape)} "
            f"must have the same shape — expand the anchor at the call site."
        )
    return torch.matmul(invert_transform(anchor_transform), target_transform)


__all__ = [
    "rot6d_to_rotation_matrix",
    "rotation_matrix_to_rot6d",
    "pose9d_to_transform",
    "transform_to_pose9d",
    "invert_transform",
    "relative_transform",
]
