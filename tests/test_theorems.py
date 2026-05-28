# -*- coding: utf-8 -*-
"""论文核心定理的工程验证。

  * Theorem 3.1 (R-factor uniqueness, Eq. 7)
        增量更新 R_k 与全量 QR 的 R-factor 在符号约束下完全相同。
  * Theorem 3.2 (Bridge L_k = R_k^T)
        当 R_k^T R_k = A^T A + λI 时，L_k = R_k^T 满足 Cholesky 分解。
"""

from __future__ import annotations

import numpy as np

from tests.conftest import random_full_rank_matrix
from utils.linalg import cholesky_lower, incremental_qr_update, qr_R


# ===========================================================================
# Theorem 3.1
# ===========================================================================


def test_theorem_3_1_two_blocks() -> None:
    """Theorem 3.1（基础情形）: A_{0:1} 全量 QR ↔ [R_0; A_1] 增量 QR。"""
    A0 = random_full_rank_matrix(120, 10, seed=1)
    A1 = random_full_rank_matrix(60, 10, seed=2)

    R_full = qr_R(np.vstack([A0, A1]))
    R_inc = incremental_qr_update(qr_R(A0), A1)
    assert np.allclose(R_full, R_inc, atol=1e-8)


def test_theorem_3_1_induction_six_blocks() -> None:
    """Theorem 3.1 归纳情形：连续 6 个 batch 增量结果与全量 QR 完全等价。"""
    rng = np.random.default_rng(2024)
    p = 12
    blocks = [
        random_full_rank_matrix(int(rng.integers(40, 100)), p, seed=int(s))
        for s in rng.integers(0, 10000, size=6)
    ]

    R = None
    for blk in blocks:
        R = incremental_qr_update(R, blk)
    R_full = qr_R(np.vstack(blocks))
    assert np.allclose(R, R_full, atol=1e-7)


def test_theorem_3_1_inner_product_preservation() -> None:
    """A_{0:k}^T A_{0:k} = R_k^T R_k （Theorem 3.1 的代数推论）。"""
    rng = np.random.default_rng(7)
    p = 10
    blocks = [random_full_rank_matrix(int(rng.integers(50, 90)), p, seed=int(s))
              for s in rng.integers(0, 10000, size=4)]

    R = None
    for blk in blocks:
        R = incremental_qr_update(R, blk)

    A_total = np.vstack(blocks)
    assert np.allclose(R.T @ R, A_total.T @ A_total, atol=1e-7)


# ===========================================================================
# Theorem 3.2 — Bridge
# ===========================================================================


def test_theorem_3_2_bridge_l_equals_r_transpose() -> None:
    """Theorem 3.2: 维护 R^T R = A^T A + λI 时，Cholesky 因子 L = R^T。

    工程实现方式：将 ``[sqrt(λ) I; A]`` 做增量 QR，自然保证 ``R^T R = A^T A + λI``。
    """
    rng = np.random.default_rng(31)
    p = 8
    lam = 1e-3
    A_full = random_full_rank_matrix(200, p, seed=0)
    blocks = np.array_split(A_full, 4)

    # 用 sqrt(λ) I 作为初始"虚拟数据"
    R = qr_R(np.sqrt(lam) * np.eye(p))
    for blk in blocks:
        R = incremental_qr_update(R, blk)

    # 等价的 Cholesky 因子
    M = A_full.T @ A_full + lam * np.eye(p)
    L = cholesky_lower(M)

    # ⭐ Bridge: L = R^T
    assert np.allclose(L, R.T, atol=1e-7), "Theorem 3.2 违反"
    # 同时验证 R^T R = M
    assert np.allclose(R.T @ R, M, atol=1e-7)


def test_theorem_3_2_holds_after_node_increment_setup() -> None:
    """模拟节点增量场景：在已有 R 的基础上扩展，仍满足 R*^T R* = M*。"""
    rng = np.random.default_rng(0)
    p = 6
    n_new = 4
    lam = 1e-3

    # 已有 R: R^T R = A^T A + λI
    A = random_full_rank_matrix(80, p, seed=0)
    R = incremental_qr_update(qr_R(np.sqrt(lam) * np.eye(p)), A)

    # 节点增量"虚拟"扩展：构造 H_new (即论文中的新 enhancement 节点输出)
    H = rng.standard_normal((80, n_new))

    # 用 IMF-BLS 节点增量公式计算 R_star
    from utils.linalg import forward_substitution
    cross = A.T @ H
    E_T = forward_substitution(R.T, cross)            # (p, n_new)
    schur = H.T @ H + lam * np.eye(n_new) - E_T.T @ E_T
    G = cholesky_lower(schur, jitter=1e-12)

    R_star = np.zeros((p + n_new, p + n_new))
    R_star[:p, :p] = R
    R_star[:p, p:] = E_T
    R_star[p:, p:] = G.T

    # 验证关键不变量：R_star^T R_star = [A, H]^T [A, H] + λI
    A_aug = np.concatenate([A, H], axis=1)
    M_aug = A_aug.T @ A_aug + lam * np.eye(p + n_new)
    assert np.allclose(R_star.T @ R_star, M_aug, atol=1e-8)


# ===========================================================================
# 补充：Theorem 3.1 在多次 (>= 5 次) 增量后仍精确成立（数值稳健性）
# ===========================================================================


def test_theorem_3_1_holds_after_many_increments() -> None:
    """对 8 个 batch 累积 incremental QR 后，R 仍等价于全量 QR。"""
    from utils.linalg import incremental_qr_update, qr_R

    rng = np.random.default_rng(2025)
    n_per_batch = 80
    p = 30
    batches = [rng.standard_normal((n_per_batch, p)) for _ in range(8)]

    R = None
    for A_k in batches:
        R = incremental_qr_update(R, A_k)

    A_full = np.vstack(batches)
    R_full = qr_R(A_full)
    # 浮点误差容忍：8 步累积仍在 1e-9 以内
    assert np.allclose(R, R_full, atol=1e-9)


def test_theorem_3_2_bridge_after_node_increment() -> None:
    """节点增量后：L*_k = (R*_k)^T 仍是 M*_k 的 Cholesky 因子。"""
    import numpy as np
    from src.imf_bls import IMFBLS
    from src.bls_base import BLSConfig

    rng = np.random.default_rng(7)
    X = rng.standard_normal((300, 6))
    Y = rng.standard_normal((300, 2))

    cfg = BLSConfig(n_mapping_per_window=5, n_mapping_windows=4,
                    n_enhancement=20, reg_lambda=1e-3, seed=11)
    m = IMFBLS(config=cfg).fit_initial(X, Y)
    m.add_nodes(X, Y, n_new=12)

    A = m.feature_layer.transform(X)
    p = A.shape[1]
    M_star = A.T @ A + cfg.reg_lambda * np.eye(p)
    L_star = m.R.T  # 桥
    err = np.max(np.abs(L_star @ L_star.T - M_star))
    assert err < 1e-9
