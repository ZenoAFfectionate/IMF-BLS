# -*- coding: utf-8 -*-
"""BLS 特征层 — 论文 Eq. (1)。

数学定义::

    Z_n = [phi(X W_e1 + b_e1), ..., phi(X W_en + b_en)]   (mapping)
    H_m = [xi(Z_n W_h1 + b_h1), ..., xi(Z_n W_hm + b_hm)] (enhancement)
    A   = [Z_n | H_m]                                      (broad feature)

支持论文 Section 2.3 的节点增量：
  * 已有 mapping/enhancement 权重保持不变（保证记忆模块的正确性）
  * 新增 enhancement 节点时，仅追加新随机权重并提供 ``transform_window`` API
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 辅助：归一化
# ---------------------------------------------------------------------------


def standardize_minmax(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    eps: float = 1e-12,
):
    """按训练集 min-max 归一化到 ``[0, 1]``，测试集套用同一仿射。"""
    X_train = np.asarray(X_train, dtype=np.float64)
    x_min = X_train.min(axis=0, keepdims=True)
    x_max = X_train.max(axis=0, keepdims=True)
    scale = np.where(x_max - x_min < eps, 1.0, x_max - x_min)
    Xs_train = (X_train - x_min) / scale
    if X_test is None:
        return Xs_train, (x_min, scale)
    Xs_test = (np.asarray(X_test, dtype=np.float64) - x_min) / scale
    return Xs_train, Xs_test, (x_min, scale)


# ---------------------------------------------------------------------------
# 激活函数
# ---------------------------------------------------------------------------


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


_ACTIVATIONS = {"tanh": _tanh, "sigmoid": _sigmoid, "relu": _relu}


# ---------------------------------------------------------------------------
# 内部数据类
# ---------------------------------------------------------------------------


@dataclass
class _Window:
    """单个 mapping / enhancement 窗口的随机权重。"""

    W: np.ndarray   # (in_dim, out_dim)
    b: np.ndarray   # (1, out_dim)


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------


@dataclass
class FeatureLayer:
    """BLS 广义特征层。

    论文符号::

        N1 = n_mapping_per_window      每个 mapping 窗口节点数
        N2 = n_mapping_windows         mapping 窗口数
        N3 = n_enhancement             enhancement 节点数（初始）
        d  = 输入特征维度
        p  = N1 * N2 + Σ enh_window_size

    Attributes:
        activation: ``tanh`` / ``sigmoid`` / ``relu``
        enh_scale:  enhancement 层权重缩放（避免 tanh 饱和；BLS 文献常用 0.8）
        seed:       随机种子，保证可复现
    """

    n_mapping_per_window: int = 10
    n_mapping_windows: int = 10
    n_enhancement: int = 1000
    activation: str = "tanh"
    enh_scale: float = 0.8
    seed: int = 0

    _mapping: List[_Window] = field(default_factory=list, init=False)
    _enh: List[_Window] = field(default_factory=list, init=False)
    _input_dim: Optional[int] = field(default=None, init=False)
    _z_dim: Optional[int] = field(default=None, init=False)
    _rng: np.random.Generator = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"未知激活函数: {self.activation}")
        self._rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------ build

    def fit_random_weights(self, input_dim: int) -> "FeatureLayer":
        """初始化 mapping + enhancement 随机权重（论文 Algorithm 3 第 1-10 行）。"""
        self._input_dim = int(input_dim)
        self._mapping = []
        self._enh = []

        # mapping 层
        for _ in range(self.n_mapping_windows):
            W = 2.0 * self._rng.random((input_dim, self.n_mapping_per_window)) - 1.0
            b = 2.0 * self._rng.random((1, self.n_mapping_per_window)) - 1.0
            self._mapping.append(_Window(W=W, b=b))

        self._z_dim = self.n_mapping_per_window * self.n_mapping_windows

        # 初始 enhancement 层（一个大窗口）
        if self.n_enhancement > 0:
            self._enh.append(self._make_enh_window(self.n_enhancement))
        return self

    def _make_enh_window(self, n_new: int) -> _Window:
        W = 2.0 * self._rng.random((self._z_dim, n_new)) - 1.0
        b = 2.0 * self._rng.random((1, n_new)) - 1.0
        # 列归一化 + 缩放（防 tanh 饱和；与论文常见实现一致）
        norms = np.linalg.norm(W, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        W = self.enh_scale * W / norms
        return _Window(W=W, b=b)

    # ------------------------------------------------------------------ forward

    def transform_mapping(self, X: np.ndarray) -> np.ndarray:
        """计算 mapping 特征 ``Z_n``。"""
        self._check_ready()
        X = np.asarray(X, dtype=np.float64)
        act = _ACTIVATIONS[self.activation]
        return np.concatenate([act(X @ w.W + w.b) for w in self._mapping], axis=1)

    def transform_enhancement(self, Z: np.ndarray) -> np.ndarray:
        """计算 enhancement 特征 ``H_m``（拼接所有窗口的输出）。"""
        if not self._enh:
            return np.zeros((Z.shape[0], 0), dtype=np.float64)
        act = _ACTIVATIONS[self.activation]
        return np.concatenate([act(Z @ w.W + w.b) for w in self._enh], axis=1)

    def transform_window(self, Z: np.ndarray, window_idx: int) -> np.ndarray:
        """仅用第 ``window_idx`` 个 enhancement 窗口前向（节点增量场景使用）。"""
        if not 0 <= window_idx < len(self._enh):
            raise IndexError(f"window_idx={window_idx} 越界（共 {len(self._enh)} 窗口）")
        act = _ACTIVATIONS[self.activation]
        w = self._enh[window_idx]
        return act(Z @ w.W + w.b)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """计算完整广义特征矩阵 ``A = [Z | H]``（论文 Eq. 1）。"""
        Z = self.transform_mapping(X)
        H = self.transform_enhancement(Z)
        return np.concatenate([Z, H], axis=1)

    # ------------------------------------------------------------------ growth

    def add_enhancement_window(self, n_new: int) -> int:
        """新增一个 enhancement 节点窗口；返回新窗口的索引。"""
        if n_new <= 0:
            raise ValueError("n_new 必须 > 0")
        self._check_ready()
        self._enh.append(self._make_enh_window(n_new))
        return len(self._enh) - 1

    # ------------------------------------------------------------------ properties

    @property
    def feature_dim(self) -> int:
        return self.mapping_dim + self.enhancement_dim

    @property
    def mapping_dim(self) -> int:
        return self._z_dim or 0

    @property
    def enhancement_dim(self) -> int:
        return sum(w.W.shape[1] for w in self._enh)

    # ------------------------------------------------------------------ misc

    def _check_ready(self) -> None:
        if self._input_dim is None:
            raise RuntimeError("请先调用 fit_random_weights(input_dim)")
