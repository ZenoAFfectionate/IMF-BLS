# -*- coding: utf-8 -*-
"""数值线性代数原语 — IMF-BLS 算法的全部底层运算。

严格对应论文 Section 2 与 Section 2.4：

    Algorithm 1   forward_substitution     R^T K = V    (lower triangular)
    Algorithm 2   backward_substitution    R W = K      (upper triangular)
    Eq. 5         solve_sne                R^T R W = V  (substitution-only)
    QR            qr_R                     A → R        (R-factor only)
    Eq. 7         incremental_qr_update    [R_{k-1}; A_k] → R_k
    Section 2.4   tsqr_R                   分块并行 TSQR
    Remark 2.2    cholesky_lower           A^T A + λI = L L^T

设计准则：
  * 所有公开函数有完整 docstring 与论文引用
  * 默认 float64，避免精度退化
  * 强制 R-factor 正对角（保证 QR 唯一性，与 Theorem 3.1 严格一致）
  * 不依赖 scipy（仅用 numpy）
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# 替换法 — 论文 Algorithms 1, 2
# ---------------------------------------------------------------------------


def forward_substitution(L: np.ndarray, V: np.ndarray) -> np.ndarray:
    """求解下三角线性方程组 ``L K = V``（论文 Algorithm 1）。

    对应论文中的算子 :math:`\\mathcal{F}(L, V)`，在 IMF-BLS 中用作:
        L = R^T  (上三角矩阵 R 的转置)
        V        右端项

    Args:
        L: 非奇异下三角矩阵 ``(p, p)``。
        V: 右端项 ``(p, c)`` 或 ``(p,)``。

    Returns:
        K: ``L K = V`` 的解，shape 与 ``V`` 相同。

    Notes:
        复杂度 ``c * p^2`` flops，远小于矩阵求逆的 ``O(p^3)``。
    """
    L = np.asarray(L, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    p = L.shape[0]
    if L.shape != (p, p):
        raise ValueError(f"L 必须是方阵，收到 shape={L.shape}")
    if V.shape[0] != p:
        raise ValueError(f"V 行数 {V.shape[0]} ≠ L 阶数 {p}")

    one_d = V.ndim == 1
    V2 = V.reshape(-1, 1) if one_d else V

    K = np.zeros_like(V2)
    K[0] = V2[0] / L[0, 0]
    for i in range(1, p):
        K[i] = (V2[i] - L[i, :i] @ K[:i]) / L[i, i]
    return K.ravel() if one_d else K


def backward_substitution(R: np.ndarray, K: np.ndarray) -> np.ndarray:
    """求解上三角线性方程组 ``R W = K``（论文 Algorithm 2）。

    Args:
        R: 非奇异上三角矩阵 ``(p, p)``。
        K: 右端项 ``(p, c)`` 或 ``(p,)``。

    Returns:
        W: ``R W = K`` 的解。
    """
    R = np.asarray(R, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    p = R.shape[0]
    if R.shape != (p, p):
        raise ValueError(f"R 必须是方阵，收到 shape={R.shape}")
    if K.shape[0] != p:
        raise ValueError(f"K 行数 {K.shape[0]} ≠ R 阶数 {p}")

    one_d = K.ndim == 1
    K2 = K.reshape(-1, 1) if one_d else K

    W = np.zeros_like(K2)
    W[p - 1] = K2[p - 1] / R[p - 1, p - 1]
    for i in range(p - 2, -1, -1):
        W[i] = (K2[i] - R[i, i + 1 :] @ W[i + 1 :]) / R[i, i]
    return W.ravel() if one_d else W


def solve_sne(R: np.ndarray, V: np.ndarray) -> np.ndarray:
    """求解半正规方程 ``R^T R W = V`` —— 论文 Eq. (5)。

    实现两步：

        1. forward:  R^T K = V
        2. backward: R W = K

    完全规避矩阵求逆，将 ``O(p^3)`` 求逆退化为两次 ``O(p^2)`` 替换。
    """
    K = forward_substitution(R.T, V)
    W = backward_substitution(R, K)
    return W


# ---------------------------------------------------------------------------
# QR 分解（仅返回 R-factor）
# ---------------------------------------------------------------------------


def _normalize_R_sign(R: np.ndarray) -> np.ndarray:
    """强制 R 的对角线为正，使 R-factor 唯一（与 Theorem 3.1 严格匹配）。"""
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    return R * signs[:, None]


def qr_R(A: np.ndarray) -> np.ndarray:
    """对矩阵 ``A`` 做 reduced QR 分解，仅返回上三角因子 ``R``。

    适用于 ``A.shape[0] >= A.shape[1]``（高瘦矩阵，论文默认场景，``l >> p``）。

    Returns:
        R: 上三角矩阵，shape ``(p, p)``，对角线为正。
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A 必须是二维矩阵")
    if A.shape[0] < A.shape[1]:
        raise ValueError(
            f"qr_R 仅适用于 m >= n 的矩阵，收到 {A.shape}；rank-deficient 请用 cholesky_lower"
        )
    _, R = np.linalg.qr(A, mode="reduced")
    return _normalize_R_sign(R)


def incremental_qr_update(R_prev: Optional[np.ndarray], A_new: np.ndarray) -> np.ndarray:
    """增量 R-factor 更新 — 论文 Eq. (7)。

    对应论文中：

        :math:`\\begin{bmatrix} R_{k-1} \\\\ A_k \\end{bmatrix} = \\tilde Q_k R_k`

    Theorem 3.1 保证：通过此公式得到的 ``R_k`` 与对全量数据 ``[A_0; ...; A_k]`` 整体
    做 QR 得到的 R-factor 完全相同（对角线符号约束下唯一）。

    Args:
        R_prev: 上一步的 R 因子 ``(p, p)``；首次调用传 ``None``。
        A_new:  新数据块 ``(l_k, p)``。

    Returns:
        R_new: 新的上三角因子 ``(p, p)``。
    """
    A_new = np.asarray(A_new, dtype=np.float64)
    if A_new.ndim != 2:
        raise ValueError("A_new 必须是二维矩阵")

    if R_prev is None:
        return qr_R(A_new) if A_new.shape[0] >= A_new.shape[1] else _qr_padded(A_new)

    R_prev = np.asarray(R_prev, dtype=np.float64)
    if R_prev.shape[1] != A_new.shape[1]:
        raise ValueError(
            f"R_prev 列数 {R_prev.shape[1]} 与 A_new 列数 {A_new.shape[1]} 不一致"
        )
    stacked = np.vstack([R_prev, A_new])
    return qr_R(stacked)


def _qr_padded(A: np.ndarray) -> np.ndarray:
    """对 m < n 的矩阵 ``A``：通过堆叠零行扩到方阵后做 QR。"""
    m, n = A.shape
    padded = np.vstack([A, np.zeros((n - m, n))])
    _, R = np.linalg.qr(padded, mode="reduced")
    return _normalize_R_sign(R)


# ---------------------------------------------------------------------------
# Tall-Skinny QR — 论文 Section 2.4
# ---------------------------------------------------------------------------


def tsqr_R(A: np.ndarray, n_blocks: int = 4) -> np.ndarray:
    """Tall-Skinny QR 分解 ``A → R``（论文 Section 2.4 / Fig. 7）。

    沿行方向把 ``A`` 切成 ``n_blocks`` 个块，对每块做局部 QR 得 ``R_i``；再把相邻
    ``R_i`` 上下拼接继续 QR，自底向上归约直到剩一个 R。

    在分布式/并行环境下，每块的 QR 可独立计算，仅需通信 R 因子，
    极大降低通信开销与内存（论文核心动机之一）。

    Args:
        A: ``(m, n)`` 矩阵，要求 ``m >= n``。
        n_blocks: 初始分块数，``n_blocks=1`` 退化为标准 QR。

    Returns:
        R: 上三角因子，与 ``qr_R(A)`` 在数学上完全等价（对角符号约束下唯一）。
    """
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    n_blocks = max(1, int(n_blocks))
    if m < n:
        raise ValueError(f"tsqr_R 要求 m >= n，收到 {A.shape}")
    if n_blocks == 1 or m < 2 * n:
        return qr_R(A)

    block_size = max(n, m // n_blocks)
    Rs = []
    for start in range(0, m, block_size):
        block = A[start : start + block_size]
        if block.shape[0] >= n:
            Rs.append(qr_R(block))
        elif block.shape[0] > 0:
            Rs.append(_qr_padded(block))

    while len(Rs) > 1:
        next_level = []
        for i in range(0, len(Rs), 2):
            if i + 1 == len(Rs):
                next_level.append(Rs[i])
            else:
                next_level.append(qr_R(np.vstack([Rs[i], Rs[i + 1]])))
        Rs = next_level
    return Rs[0]


# ---------------------------------------------------------------------------
# Cholesky 分解 — 论文 Eq. 15 / Remark 2.2
# ---------------------------------------------------------------------------


def cholesky_lower(M: np.ndarray, jitter: float = 0.0) -> np.ndarray:
    """对正定矩阵 ``M`` 做 Cholesky 分解，返回下三角 ``L`` 满足 ``L L^T = M``。

    用于：
      * Remark 2.2：``l_0 <= p`` 时通过 ``M = A^T A + λI`` 求 ``L = R_0^T``
      * Eq. 12：节点增量阶段计算 Schur 补 ``G G^T = H^T H − E E^T + λI``

    Args:
        M: ``(p, p)`` 对称正定矩阵。
        jitter: 若分解失败则给对角加 ``jitter * I`` 重试。

    Returns:
        L: 下三角矩阵，对角线为正。
    """
    M = np.asarray(M, dtype=np.float64)
    if jitter > 0.0:
        M = M + jitter * np.eye(M.shape[0])
    try:
        return np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        eps = max(np.trace(M) / max(M.shape[0], 1), 1.0) * 1e-10
        return np.linalg.cholesky(M + eps * np.eye(M.shape[0]))
