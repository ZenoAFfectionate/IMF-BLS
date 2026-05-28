# -*- coding: utf-8 -*-
"""论文 Section 4 中所有对比方法的实现。

================================================================
方法                   公式                              依赖求逆
----------------------------------------------------------------
IncrementalBLS         Greville 伪逆增量 (论文 1.2)      ✓
RIBLS [Zhong 2024]     U = A^T A + λI, V = A^T Y         ✓
TiBLS [Fu 2022]        Sherman-Morrison-Woodbury         ✓
ApproximationMethodBLS Ridge 平均 [Zhang 2012]           ✓
================================================================

所有方法均与 :class:`IMFBLS` 共享 :class:`FeatureLayer`，便于公平比较。
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.bls_base import BLSBase, BLSConfig  # noqa: E402


# ============================================================================
# 通用工具：兼容宽 / 高矩阵的 ridge 伪逆
# ============================================================================


def _ridge_pinv(A: np.ndarray, lam: float) -> np.ndarray:
    """计算 ridge 伪逆 ``A^+``，正确处理宽矩阵 (n < p) 与高矩阵 (n >= p)。

    数学等价但数值更稳::

        n >= p:  A^+ = (A^T A + λI)^{-1} A^T
        n <  p:  A^+ = A^T (A A^T + λI)^{-1}
    """
    n, p = A.shape
    if n >= p:
        return np.linalg.solve(A.T @ A + lam * np.eye(p), A.T)
    return A.T @ np.linalg.solve(A @ A.T + lam * np.eye(n), np.eye(n))


# ============================================================================
# 1. Incremental BLS (Greville 伪逆增量, 论文 Section 1.2)
# ============================================================================


class IncrementalBLS(BLSBase):
    """原版 BLS 的伪逆增量算法 (Chen & Liu, 2018)。

    使用 Greville 公式更新 ``A^+``::

        D^T = A_new A^+
        C   = A_new - D^T A
        B   = C^+                              (||C|| > 0)
            = A^+ D (I + D^T D)^{-1}            (||C|| = 0)
        new A^+ = [A^+ - B D^T | B]
        W       = A^+ Y_all

    对于 rank-deficient 阶段（``A.shape[0] < A.shape[1]``，常见于初始 batch 较小的场景），
    Greville 公式数值不稳，回退到对累积数据重算 ridge 伪逆。
    """

    def __init__(self, config: Optional[BLSConfig] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self._A: Optional[np.ndarray] = None
        self._A_pinv: Optional[np.ndarray] = None
        self._Y: Optional[np.ndarray] = None

    def fit_initial(self, X: np.ndarray, Y: np.ndarray) -> "IncrementalBLS":
        Y = self._ensure_2d_target(Y)
        A = self._build_features(X, fit=True)
        self._A = A
        self._A_pinv = _ridge_pinv(A, self.config.reg_lambda)
        self._Y = Y
        self.W = self._A_pinv @ Y
        self._is_fitted = True
        return self

    def add_data(self, X_new: np.ndarray, Y_new: np.ndarray) -> "IncrementalBLS":
        if not self._is_fitted:
            raise RuntimeError("请先 fit_initial")
        Y_new = self._ensure_2d_target(Y_new)
        A_new = self.feature_layer.transform(X_new)
        A_old = self._A

        # rank-deficient 时 Greville 数值不稳，重算 ridge 伪逆
        if A_old.shape[0] < A_old.shape[1]:
            self._A = np.vstack([A_old, A_new])
            self._Y = np.vstack([self._Y, Y_new])
            self._A_pinv = _ridge_pinv(self._A, self.config.reg_lambda)
            self.W = self._A_pinv @ self._Y
            return self

        # 标准 Greville 路径
        A_pinv = self._A_pinv
        D_T = A_new @ A_pinv
        C = A_new - D_T @ A_old

        if np.linalg.norm(C) > 1e-9:
            B = C.T @ np.linalg.inv(C @ C.T + self.config.reg_lambda * np.eye(C.shape[0]))
        else:
            inner = np.eye(D_T.shape[0]) + D_T @ D_T.T
            B = A_pinv @ D_T.T @ np.linalg.inv(inner)

        new_pinv_left = A_pinv - B @ D_T
        self._A_pinv = np.concatenate([new_pinv_left, B], axis=1)
        self._A = np.vstack([A_old, A_new])
        self._Y = np.vstack([self._Y, Y_new])
        self.W = self._A_pinv @ self._Y
        return self

    def add_nodes(self, X_all: np.ndarray, Y_all: np.ndarray, n_new: int) -> "IncrementalBLS":
        """节点增量：直接重算 ridge 伪逆（与论文 IncrementalBLS 节点增量等价）。"""
        if not self._is_fitted:
            raise RuntimeError("请先 fit_initial")
        Y_all = self._ensure_2d_target(Y_all)
        Z_all = self.feature_layer.transform_mapping(X_all)
        win = self.feature_layer.add_enhancement_window(n_new)
        H_new = self.feature_layer.transform_window(Z_all, win)
        A_new = np.concatenate([self._A, H_new], axis=1)

        self._A = A_new
        self._A_pinv = _ridge_pinv(A_new, self.config.reg_lambda)
        self._Y = Y_all
        self.W = self._A_pinv @ Y_all
        return self


# ============================================================================
# 2. RI-BLS (Robust Incremental BLS, Zhong 2024)
# ============================================================================


class RIBLS(BLSBase):
    """Robust Incremental BLS。

    维护两个记忆矩阵::

        U = A^T A + λI    (p × p)
        V = A^T Y         (p × c)
        W = U^{-1} V

    每个增量步只需 ``U += A_new^T A_new``、``V += A_new^T Y_new``。
    空间复杂度 ``O(p^2 + pc)``，与 IMF-BLS 相同；但仍依赖矩阵求逆。
    """

    def __init__(self, config: Optional[BLSConfig] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.U: Optional[np.ndarray] = None
        self.V: Optional[np.ndarray] = None

    def fit_initial(self, X: np.ndarray, Y: np.ndarray) -> "RIBLS":
        Y = self._ensure_2d_target(Y)
        A = self._build_features(X, fit=True)
        p = A.shape[1]
        self.U = A.T @ A + self.config.reg_lambda * np.eye(p)
        self.V = A.T @ Y
        self.W = np.linalg.solve(self.U, self.V)
        self._is_fitted = True
        return self

    def add_data(self, X_new: np.ndarray, Y_new: np.ndarray) -> "RIBLS":
        Y_new = self._ensure_2d_target(Y_new)
        A_new = self.feature_layer.transform(X_new)
        self.U = self.U + A_new.T @ A_new
        self.V = self.V + A_new.T @ Y_new
        self.W = np.linalg.solve(self.U, self.V)
        return self

    def add_nodes(self, X_all: np.ndarray, Y_all: np.ndarray, n_new: int) -> "RIBLS":
        Y_all = self._ensure_2d_target(Y_all)
        Z_all = self.feature_layer.transform_mapping(X_all)
        # 在新增窗口前先取得 A_old（旧 enhancement）
        A_old = np.concatenate(
            [Z_all, self.feature_layer.transform_enhancement(Z_all)], axis=1
        )
        win = self.feature_layer.add_enhancement_window(n_new)
        H_new = self.feature_layer.transform_window(Z_all, win)

        cross = A_old.T @ H_new                                          # (p_old, n_new)
        bottom = H_new.T @ H_new + self.config.reg_lambda * np.eye(n_new)

        p_old = self.U.shape[0]
        new_U = np.zeros((p_old + n_new, p_old + n_new), dtype=np.float64)
        new_U[:p_old, :p_old] = self.U
        new_U[:p_old, p_old:] = cross
        new_U[p_old:, :p_old] = cross.T
        new_U[p_old:, p_old:] = bottom

        self.U = new_U
        self.V = np.concatenate([self.V, H_new.T @ Y_all], axis=0)
        self.W = np.linalg.solve(self.U, self.V)
        return self


# ============================================================================
# 3. TI-BLS (Task-Incremental BLS, Fu et al. 2022)
# ============================================================================


class TiBLS(BLSBase):
    """Task-Incremental BLS。

    使用 Sherman-Morrison-Woodbury (SMW) 直接维护 ``(A^T A + λI)^{-1}``::

        (M + B^T B)^{-1} = M^{-1} - M^{-1} B^T (I + B M^{-1} B^T)^{-1} B M^{-1}

    论文 Section 4.1 注明：TI-BLS 严格要求每批样本数一致，故只支持 equal-scale 场景。
    """

    def __init__(self, config: Optional[BLSConfig] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self._inv_M: Optional[np.ndarray] = None
        self._V: Optional[np.ndarray] = None
        self._batch_size: Optional[int] = None

    def fit_initial(self, X: np.ndarray, Y: np.ndarray) -> "TiBLS":
        Y = self._ensure_2d_target(Y)
        A = self._build_features(X, fit=True)
        p = A.shape[1]
        self._inv_M = np.linalg.inv(A.T @ A + self.config.reg_lambda * np.eye(p))
        self._V = A.T @ Y
        self.W = self._inv_M @ self._V
        self._batch_size = X.shape[0]
        self._is_fitted = True
        return self

    def add_data(self, X_new: np.ndarray, Y_new: np.ndarray) -> "TiBLS":
        if X_new.shape[0] != self._batch_size:
            raise ValueError(
                f"TiBLS 要求 batch 大小恒定 ({self._batch_size})，收到 {X_new.shape[0]}"
            )
        Y_new = self._ensure_2d_target(Y_new)
        A_new = self.feature_layer.transform(X_new)

        BMinv = A_new @ self._inv_M
        inner = np.eye(A_new.shape[0]) + BMinv @ A_new.T
        self._inv_M = self._inv_M - BMinv.T @ np.linalg.inv(inner) @ BMinv
        self._V = self._V + A_new.T @ Y_new
        self.W = self._inv_M @ self._V
        return self


# ============================================================================
# 4. Approximation Method (Ridge 平均, Zhang et al. NeurIPS 2012)
# ============================================================================


class ApproximationMethodBLS(BLSBase):
    """基于 ridge 平均的近似分布式估计。

    每个 batch 单独求 ridge 解 ``W_k = (A_k^T A_k + λI)^{-1} A_k^T Y_k``，
    最终 ``W = (1/K) Σ_k W_k``。简单但精度通常较低。
    """

    def __init__(self, config: Optional[BLSConfig] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self._W_sum: Optional[np.ndarray] = None
        self._n_batches: int = 0

    def _ridge_solve(self, A: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return _ridge_pinv(A, self.config.reg_lambda) @ Y

    def fit_initial(self, X: np.ndarray, Y: np.ndarray) -> "ApproximationMethodBLS":
        Y = self._ensure_2d_target(Y)
        A = self._build_features(X, fit=True)
        W_first = self._ridge_solve(A, Y)
        self._W_sum = W_first.copy()
        self._n_batches = 1
        self.W = W_first
        self._is_fitted = True
        return self

    def add_data(self, X_new: np.ndarray, Y_new: np.ndarray) -> "ApproximationMethodBLS":
        Y_new = self._ensure_2d_target(Y_new)
        A_new = self.feature_layer.transform(X_new)
        W_new = self._ridge_solve(A_new, Y_new)
        self._W_sum = self._W_sum + W_new
        self._n_batches += 1
        self.W = self._W_sum / self._n_batches
        return self
