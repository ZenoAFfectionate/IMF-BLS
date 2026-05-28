# -*- coding: utf-8 -*-
"""BLS 通用基类与非增量 BLS（联合学习上界）。

设计原则：
  * 各 BLS 变种共享 :class:`FeatureLayer`、:meth:`predict`、:meth:`score_*` 实现
  * 子类只需实现 :meth:`fit_initial` 与可选的 :meth:`add_data` / :meth:`add_nodes`
  * :class:`NonIncrementalBLS` 用 Cholesky+替换法求解（与论文 IMF-BLS 数学等价），
    作为联合训练的标准答案与等价性验证基准
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

# 让 src 包可以独立运行（``python -m src.imf_bls`` 等）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.feature_layer import FeatureLayer  # noqa: E402
from utils.linalg import (  # noqa: E402
    backward_substitution,
    cholesky_lower,
    forward_substitution,
)
from utils.metrics import classification_accuracy, regression_rmse  # noqa: E402


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class BLSConfig:
    """BLS 通用超参数。

    Attributes:
        n_mapping_per_window: ``N1`` —— 每个 mapping 窗口节点数。
        n_mapping_windows:    ``N2`` —— mapping 窗口数。
        n_enhancement:        ``N3`` —— 初始 enhancement 节点数。
        activation:           激活函数（``tanh``/``sigmoid``/``relu``）。
        enh_scale:            enhancement 权重缩放（防止 tanh 饱和）。
        reg_lambda:           Tikhonov 正则化参数 ``λ``。
        seed:                 随机种子。
    """

    n_mapping_per_window: int = 10
    n_mapping_windows: int = 10
    n_enhancement: int = 1000
    activation: str = "tanh"
    enh_scale: float = 0.8
    reg_lambda: float = 1e-6
    seed: int = 0

    def copy(self) -> "BLSConfig":
        return replace(self)


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BLSBase(ABC):
    """所有 BLS 变种的公共基类。"""

    def __init__(self, config: Optional[BLSConfig] = None, **kwargs) -> None:
        if config is None:
            config = BLSConfig(**kwargs)
        elif kwargs:
            for k, v in kwargs.items():
                setattr(config, k, v)
        self.config = config

        self.feature_layer = FeatureLayer(
            n_mapping_per_window=config.n_mapping_per_window,
            n_mapping_windows=config.n_mapping_windows,
            n_enhancement=config.n_enhancement,
            activation=config.activation,
            enh_scale=config.enh_scale,
            seed=config.seed,
        )
        self.W: Optional[np.ndarray] = None
        self._target_dim: Optional[int] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------ helpers

    def _ensure_2d_target(self, Y: np.ndarray) -> np.ndarray:
        """保证 ``Y`` 为 ``(n, c)``，并校验目标维度一致。"""
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if self._target_dim is None:
            self._target_dim = Y.shape[1]
        elif Y.shape[1] != self._target_dim:
            raise ValueError(
                f"目标维度不一致：已学习 {self._target_dim}，本次 {Y.shape[1]}"
            )
        return Y

    def _build_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """前向广义特征矩阵 ``A = [Z | H]``（论文 Eq. 1）。"""
        if fit:
            self.feature_layer.fit_random_weights(input_dim=X.shape[1])
        return self.feature_layer.transform(X)

    # ------------------------------------------------------------------ abstract

    @abstractmethod
    def fit_initial(self, X: np.ndarray, Y: np.ndarray) -> "BLSBase":
        """用首批数据训练初始模型。"""

    def add_data(self, X_new: np.ndarray, Y_new: np.ndarray) -> "BLSBase":
        raise NotImplementedError(f"{type(self).__name__} 不支持 add_data")

    def add_nodes(self, X_all: np.ndarray, Y_all: np.ndarray, n_new: int) -> "BLSBase":
        raise NotImplementedError(f"{type(self).__name__} 不支持 add_nodes")

    # ------------------------------------------------------------------ inference

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self.W is None:
            raise RuntimeError("模型尚未训练，请先调用 fit_initial")
        return self.feature_layer.transform(X) @ self.W

    def score_classification(self, X: np.ndarray, Y_one_hot: np.ndarray) -> float:
        return classification_accuracy(Y_one_hot, self.predict(X))

    def score_rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        return regression_rmse(y, self.predict(X))


# ---------------------------------------------------------------------------
# 联合训练上界
# ---------------------------------------------------------------------------


class NonIncrementalBLS(BLSBase):
    """非增量 BLS（联合学习上界）。

    在所有可见数据上一次性求 ridge 解::

        (A^T A + λI) W = A^T Y

    采用 Cholesky + forward/backward 替换法，与 :class:`IMFBLS` 数学等价；
    用作 IMF-BLS 等价性单元测试的"金标准"。
    """

    def fit_initial(self, X: np.ndarray, Y: np.ndarray) -> "NonIncrementalBLS":
        Y = self._ensure_2d_target(Y)
        A = self._build_features(X, fit=True)
        p = A.shape[1]
        M = A.T @ A + self.config.reg_lambda * np.eye(p)
        L = cholesky_lower(M)            # M = L L^T
        K = forward_substitution(L, A.T @ Y)
        self.W = backward_substitution(L.T, K)
        self._is_fitted = True
        return self

    # 别名：与"联合训练"语义对应
    fit_all = fit_initial
