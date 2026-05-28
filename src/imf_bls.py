# -*- coding: utf-8 -*-
"""IMF-BLS: Inverse Matrix-Free Broad Learning System.

论文::

    G.-Z. Chen et al. "Efficient incremental learning for Inverse Matrix-Free
    broad learning system", Information Fusion 127 (2026) 103842.

注：``IMF-BLS`` 是本仓库使用的简称（与原论文中的 ``InvF-BLS`` 完全等价，
仅命名风格不同）。

算法概述
========

记忆模块 ``(R, V)``：

    R^T R = A^T A + λI       (大小 p×p 的上三角因子)
    V     = A^T Y            (右端项 p×c)

权重通过半正规方程 :math:`R^T R W = V` 解出（论文 Eq. 5）::

    K = forward(R^T, V),  W = backward(R, K)

完全规避矩阵求逆，把 ``O(p^3)`` 退化为两次 ``O(p^2)`` 替换。

三个阶段（论文 Section 2）
==========================

Phase 1 — 初始训练 (Section 2.1)
--------------------------------
对初始数据 ``A_0``::

    R_0^T R_0 = A_0^T A_0 + λI
    V_0       = A_0^T Y_0

实现：将 ``[sqrt(λ) I; A_0]`` 做 reduced QR，自然得到 ``R_0`` 满足
``R_0^T R_0 = λI + A_0^T A_0``。该方法对 ``l_0 ≤ p`` 与 ``l_0 > p`` **统一**适用，
比 Remark 2.2 的 Cholesky 路径更稳定（避免显式构造 ``A^T A``）。

Phase 2 — 加数据 (Section 2.2)
------------------------------
对新数据块 ``(A_k, Y_k)``::

    R_k = qr_R([R_{k-1}; A_k])         (Eq. 7)
    V_k = V_{k-1} + A_k^T Y_k          (Eq. 8)
    W_k = solve_sne(R_k, V_k)          (Eq. 9)

由 Theorem 3.1（增量 R-factor 唯一性），``R_k^T R_k = A_{0:k}^T A_{0:k} + λI``。

Phase 3 — 加节点 (Section 2.3)
------------------------------
当新增 enhancement 节点 ``H_new``（在已观测的所有样本上的输出 ``A_{0:k}^T H_new``）::

    L_k = R_k^T                          (桥; Theorem 3.2)
    L_k E^T = A_{0:k}^T H_new            ⇒ E^T = forward(L_k, ·)  (Eq. 12)
    G G^T = H_new^T H_new + λI - E E^T   ⇒ G = chol(...)          (Eq. 12)
    R*    = [[R_k,  E^T],
             [0,    G^T]]
    V*    = [V_k;
             H_new^T Y_{0:k}]            (Eq. 13)
    W     = solve_sne(R*, V*)            (Eq. 14)

注：Phase 3 需要历史输入 ``X_all`` 用于计算 ``H_new``（这是论文公式所必需的，
因为 ``A_{0:k}^T H_new`` 关联到所有历史样本）。两次节点增量之间不要求保留 X。
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.bls_base import BLSBase, BLSConfig  # noqa: E402
from utils.linalg import (  # noqa: E402
    cholesky_lower,
    forward_substitution,
    incremental_qr_update,
    qr_R,
    solve_sne,
    tsqr_R,
)


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------


class IMFBLS(BLSBase):
    """Inverse Matrix-Free Broad Learning System（论文主算法）。

    Args:
        config:      :class:`BLSConfig` 通用配置。
        use_tsqr:    Phase 1 / Phase 2 是否启用 Tall-Skinny QR 加速 (Section 2.4)。
        tsqr_blocks: TSQR 分块数。
    """

    def __init__(
        self,
        config: Optional[BLSConfig] = None,
        use_tsqr: bool = False,
        tsqr_blocks: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self.use_tsqr = use_tsqr
        self.tsqr_blocks = tsqr_blocks

        # 记忆模块
        self.R: Optional[np.ndarray] = None  # (p, p) 上三角，R^T R = A^T A + λI
        self.V: Optional[np.ndarray] = None  # (p, c) 右端项

    # ==================================================================== #
    # Phase 1: 初始训练 (Section 2.1)
    # ==================================================================== #

    def fit_initial(self, X: np.ndarray, Y: np.ndarray) -> "IMFBLS":
        """用首批数据训练初始模型。"""
        Y = self._ensure_2d_target(Y)
        A = self._build_features(X, fit=True)
        p = A.shape[1]

        self.R = self._build_initial_R(A, p)
        self.V = A.T @ Y
        self.W = solve_sne(self.R, self.V)
        self._is_fitted = True
        return self

    def _build_initial_R(self, A: np.ndarray, p: int) -> np.ndarray:
        """构造满足 ``R^T R = A^T A + λI`` 的 R。

        统一处理 ``l_0 > p`` 与 ``l_0 ≤ p``：
            将 ``[sqrt(λ) I_p; A]`` 做 QR，得到的 R 自然满足

                R^T R = λ I + A^T A

            因为 ``[sqrt(λ) I; A]^T [sqrt(λ) I; A] = λI + A^T A``。

        优势:
          * 与论文 Eq. 4 完全一致
          * Theorem 3.2（``L_k = R_k^T``）在所有 k 上恒成立，
            节点增量阶段不再需要重新引入 λ
          * ``l_0`` 任意大小均适用，不依赖 Cholesky 显式构造 ``A^T A``
        """
        lam = self.config.reg_lambda
        sqrt_lam_I = np.sqrt(max(lam, 0.0)) * np.eye(p)
        stacked = np.vstack([sqrt_lam_I, A])

        if self.use_tsqr and stacked.shape[0] >= 2 * p:
            return tsqr_R(stacked, n_blocks=self.tsqr_blocks)
        return qr_R(stacked)

    # ==================================================================== #
    # Phase 2: 加数据 (Section 2.2)
    # ==================================================================== #

    def add_data(self, X_new: np.ndarray, Y_new: np.ndarray) -> "IMFBLS":
        """加入新数据块进行增量训练。

        Mathematical update::

            R_k = qr_R([R_{k-1}; A_k])         (Eq. 7)
            V_k = V_{k-1} + A_k^T Y_k          (Eq. 8)
            W_k = solve_sne(R_k, V_k)          (Eq. 9)
        """
        if not self._is_fitted:
            raise RuntimeError("请先 fit_initial")

        Y_new = self._ensure_2d_target(Y_new)
        A_new = self.feature_layer.transform(X_new)

        # Eq. 7：增量 R-factor 更新
        if self.use_tsqr:
            self.R = tsqr_R(np.vstack([self.R, A_new]), n_blocks=self.tsqr_blocks)
        else:
            self.R = incremental_qr_update(self.R, A_new)

        # Eq. 8：右端项更新
        self.V = self.V + A_new.T @ Y_new

        # Eq. 9：替换法求解
        self.W = solve_sne(self.R, self.V)
        return self

    # ==================================================================== #
    # Phase 3: 加节点 (Section 2.3)
    # ==================================================================== #

    def add_nodes(
        self,
        X_all: np.ndarray,
        Y_all: np.ndarray,
        n_new: int,
    ) -> "IMFBLS":
        """新增 enhancement 节点（论文 Section 2.3, Eq. 12-14）。

        数学步骤::

            L_k = R_k^T                          (Bridge, Theorem 3.2)
            E^T = forward(L_k, A_{0:k}^T H_new)  (Eq. 12)
            G G^T = H_new^T H_new + λI − E E^T   (Eq. 12, Schur 补)
            R*    = [[R_k, E^T], [0, G^T]]       (Eq. 12)
            V*    = [V_k; H_new^T Y_{0:k}]       (Eq. 13)
            W     = solve_sne(R*, V*)            (Eq. 14)

        Args:
            X_all: 截至当前的全部历史输入 ``(N_total, d)``。
                   用于计算新节点对历史样本的输出 ``H_new = ξ(Z_all W_h + b_h)``。
            Y_all: 对应标签 ``(N_total, c)``。
            n_new: 新增 enhancement 节点数。
        """
        if not self._is_fitted:
            raise RuntimeError("请先 fit_initial")
        Y_all = self._ensure_2d_target(Y_all)
        if X_all.shape[0] != Y_all.shape[0]:
            raise ValueError("X_all 与 Y_all 行数不一致")

        # 当前已有的广义特征 A_all 与新节点输出 H_new
        Z_all = self.feature_layer.transform_mapping(X_all)
        A_old = np.concatenate(
            [Z_all, self.feature_layer.transform_enhancement(Z_all)], axis=1
        )
        win_idx = self.feature_layer.add_enhancement_window(n_new)
        H_new = self.feature_layer.transform_window(Z_all, win_idx)

        # ----- 构造扩展后的 R*  (Eq. 12) -----
        # 桥：L_k = R_k^T （Theorem 3.2 在 R^T R = A^T A + λI 时严格成立）
        L_k = self.R.T  # (p_old, p_old)

        # 解 L_k E^T = A_all^T H_new  ⇒  E^T = forward(L_k, A_all^T H_new)
        cross = A_old.T @ H_new                         # (p_old, n_new)
        E_T = forward_substitution(L_k, cross)          # (p_old, n_new)
        E = E_T.T                                       # (n_new, p_old)

        # Schur 补：G G^T = H_new^T H_new + λI - E E^T
        lam = self.config.reg_lambda
        schur = H_new.T @ H_new + lam * np.eye(n_new) - E @ E.T
        G = cholesky_lower(schur, jitter=1e-12)         # (n_new, n_new)

        # 拼装 R* = [[R, E^T], [0, G^T]]
        p_old = self.R.shape[0]
        p_new = p_old + n_new
        R_star = np.zeros((p_new, p_new), dtype=np.float64)
        R_star[:p_old, :p_old] = self.R
        R_star[:p_old, p_old:] = E_T
        R_star[p_old:, p_old:] = G.T

        # ----- 扩展 V* (Eq. 13) -----
        V_star = np.concatenate([self.V, H_new.T @ Y_all], axis=0)

        # ----- 提交记忆模块并求解 (Eq. 14) -----
        self.R = R_star
        self.V = V_star
        self.W = solve_sne(self.R, self.V)
        return self

    # ==================================================================== #
    # 工具方法
    # ==================================================================== #

    def memory_module(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回当前记忆模块 ``(R, V)``。"""
        return self.R, self.V

    def memory_footprint_bytes(self) -> int:
        """返回 ``(R, V, W)`` 总字节数 —— 对应论文 Table 2 的空间复杂度 ``O(p² + pc)``。"""
        return sum(arr.nbytes for arr in (self.R, self.V, self.W) if arr is not None)

    @property
    def width(self) -> int:
        """当前广义特征维度 ``p = mapping_dim + enhancement_dim``。"""
        return int(self.R.shape[0]) if self.R is not None else 0
