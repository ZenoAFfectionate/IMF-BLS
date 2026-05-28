# -*- coding: utf-8 -*-
"""utils/linalg.py 数值原语单元测试。

覆盖（对应论文 Algorithms / 公式）::

    Algorithm 1 forward_substitution      → R^T K = V
    Algorithm 2 backward_substitution     → R W = K
    Eq. 5       solve_sne                 → R^T R W = V
    QR          qr_R                      → 上三角 + 正对角
    Eq. 7       incremental_qr_update     → 与全量 QR 等价
    Section 2.4 tsqr_R                    → 与 qr_R 数学等价
    Remark 2.2  cholesky_lower            → L L^T = M
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import (
    assert_R_diag_positive,
    assert_R_is_upper_triangular,
    random_full_rank_matrix,
    random_well_conditioned_R,
)
from utils.linalg import (
    backward_substitution,
    cholesky_lower,
    forward_substitution,
    incremental_qr_update,
    qr_R,
    solve_sne,
    tsqr_R,
)


# ===========================================================================
# 替换法
# ===========================================================================


@pytest.mark.parametrize("p", [3, 8, 30, 64])
def test_forward_substitution_matches_solve(p: int) -> None:
    """forward_substitution(L, V) 应与 numpy.linalg.solve(L, V) 数值一致。"""
    rng = np.random.default_rng(p)
    L = np.tril(rng.standard_normal((p, p)))
    np.fill_diagonal(L, np.abs(np.diag(L)) + 1.0)
    V = rng.standard_normal((p, 4))

    K_ours = forward_substitution(L, V)
    K_ref = np.linalg.solve(L, V)
    assert np.allclose(K_ours, K_ref, atol=1e-10)


@pytest.mark.parametrize("p", [3, 8, 30, 64])
def test_backward_substitution_matches_solve(p: int) -> None:
    R = random_well_conditioned_R(p, seed=p)
    rng = np.random.default_rng(p + 1)
    K = rng.standard_normal((p, 5))

    W_ours = backward_substitution(R, K)
    W_ref = np.linalg.solve(R, K)
    assert np.allclose(W_ours, W_ref, atol=1e-10)


def test_substitution_supports_1d_rhs() -> None:
    """替换法应正确处理一维右端项。"""
    R = random_well_conditioned_R(10, seed=0)
    v = np.arange(10, dtype=np.float64)

    K = forward_substitution(R.T, v)
    assert K.shape == (10,)
    assert np.allclose(R.T @ K, v, atol=1e-10)

    W = backward_substitution(R, K)
    assert W.shape == (10,)
    assert np.allclose(R @ W, K, atol=1e-10)


def test_solve_sne_matches_normal_equation() -> None:
    """solve_sne(R, V) 应等价于求解 (R^T R) W = V。"""
    R = random_well_conditioned_R(20, seed=11)
    rng = np.random.default_rng(12)
    V = rng.standard_normal((20, 3))

    W_ours = solve_sne(R, V)
    W_ref = np.linalg.solve(R.T @ R, V)
    assert np.allclose(W_ours, W_ref, atol=1e-9)


def test_substitution_raises_on_shape_mismatch() -> None:
    R = random_well_conditioned_R(5)
    with pytest.raises(ValueError):
        forward_substitution(R, np.zeros((4, 2)))  # 行数不匹配
    with pytest.raises(ValueError):
        backward_substitution(np.zeros((4, 5)), np.zeros((4, 2)))  # R 非方阵


# ===========================================================================
# QR
# ===========================================================================


@pytest.mark.parametrize("m,n", [(100, 8), (50, 20), (200, 5)])
def test_qr_R_basic_properties(m: int, n: int) -> None:
    """qr_R(A) 必须返回上三角、正对角、且 R^T R = A^T A。"""
    A = random_full_rank_matrix(m, n, seed=m * n)
    R = qr_R(A)

    assert R.shape == (n, n)
    assert_R_is_upper_triangular(R)
    assert_R_diag_positive(R)
    # 数值不变量：R^T R = A^T A
    assert np.allclose(R.T @ R, A.T @ A, atol=1e-7)


def test_qr_R_rejects_wide_matrix() -> None:
    with pytest.raises(ValueError):
        qr_R(np.zeros((3, 5)))


# ===========================================================================
# 增量 R-factor 更新（Theorem 3.1）
# ===========================================================================


def test_incremental_qr_initial_matches_qr() -> None:
    """首次调用时 incremental_qr_update(None, A) 应等同于 qr_R(A)。"""
    A = random_full_rank_matrix(80, 10, seed=1)
    assert np.allclose(incremental_qr_update(None, A), qr_R(A))


def test_incremental_qr_two_step_equals_full_qr() -> None:
    """[R0; A1] 增量更新与 [A0; A1] 全量 QR 在符号约束下等价。"""
    A0 = random_full_rank_matrix(120, 10, seed=2)
    A1 = random_full_rank_matrix(50, 10, seed=3)

    R_full = qr_R(np.vstack([A0, A1]))
    R_inc = incremental_qr_update(qr_R(A0), A1)
    assert np.allclose(R_full, R_inc, atol=1e-8)


def test_incremental_qr_multi_step() -> None:
    """连续 5 个 batch 的增量结果与全量 QR 一致。"""
    rng = np.random.default_rng(42)
    p = 12
    blocks = [random_full_rank_matrix(int(rng.integers(40, 80)), p, seed=i) for i in range(5)]

    R = None
    for blk in blocks:
        R = incremental_qr_update(R, blk)
    R_full = qr_R(np.vstack(blocks))
    assert np.allclose(R, R_full, atol=1e-7)


def test_incremental_qr_preserves_inner_product() -> None:
    """增量更新后应保持 R^T R = A^T A（关键不变量）。"""
    A0 = random_full_rank_matrix(150, 8, seed=10)
    A1 = random_full_rank_matrix(60, 8, seed=11)

    R = incremental_qr_update(qr_R(A0), A1)
    A_total = np.vstack([A0, A1])
    assert np.allclose(R.T @ R, A_total.T @ A_total, atol=1e-7)


def test_incremental_qr_shape_mismatch_raises() -> None:
    R = qr_R(random_full_rank_matrix(50, 6, seed=0))
    with pytest.raises(ValueError):
        incremental_qr_update(R, random_full_rank_matrix(20, 7, seed=0))


# ===========================================================================
# Tall-Skinny QR
# ===========================================================================


@pytest.mark.parametrize("blocks", [1, 2, 4, 8])
def test_tsqr_matches_qr(blocks: int) -> None:
    A = random_full_rank_matrix(500, 15, seed=blocks)
    R_tsqr = tsqr_R(A, n_blocks=blocks)
    R_ref = qr_R(A)
    assert np.allclose(R_tsqr, R_ref, atol=1e-7)


def test_tsqr_rejects_wide_matrix() -> None:
    with pytest.raises(ValueError):
        tsqr_R(np.zeros((4, 5)))


def test_tsqr_inner_product_invariant() -> None:
    """TSQR 的 R 同样满足 R^T R = A^T A。"""
    A = random_full_rank_matrix(800, 10, seed=99)
    R = tsqr_R(A, n_blocks=4)
    assert np.allclose(R.T @ R, A.T @ A, atol=1e-7)


# ===========================================================================
# Cholesky
# ===========================================================================


def test_cholesky_basic() -> None:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((50, 8))
    M = A.T @ A + 1e-3 * np.eye(8)

    L = cholesky_lower(M)
    assert np.allclose(L @ L.T, M, atol=1e-10)
    assert (np.diag(L) > 0).all()


def test_cholesky_jitter_recovers_singular() -> None:
    """对接近奇异的矩阵，jitter 路径应能完成分解。"""
    M = np.zeros((6, 6))
    M[:3, :3] = np.eye(3)  # rank-deficient
    L = cholesky_lower(M, jitter=1e-6)
    assert np.allclose(L @ L.T, M + 1e-6 * np.eye(6), atol=1e-3)
