# -*- coding: utf-8 -*-
"""数据集加载与 batch 切分。

提供：
  * 合成数据生成（无外部依赖，单元测试与默认实验均使用）
  * sklearn 内置数据集（digits / iris / california housing，可选依赖）
  * IDX 格式 MNIST / Fashion-MNIST 加载（论文主要数据集）
  * 等量 / 不定 scale 数据流切分（论文 Section 4.1 / 4.2）

约定：
  * 分类: ``y`` 为整数标签，调用方用 :func:`one_hot_encode` 转 ±1 编码
  * 回归: ``y`` 为浮点 1D
  * 所有训练 / 测试数据返回 ``(X_train, y_train, X_test, y_test)``
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 标签编码
# ---------------------------------------------------------------------------


def one_hot_encode(y: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """将整数标签 ``y`` 编码为 ±1 矩阵（BLS 文献常用编码）。

    Args:
        y:           ``(n,)`` 整数标签。
        num_classes: 类别数；缺省自动推断。

    Returns:
        ``(n, c)`` 矩阵：第 i 行第 y[i] 列为 +1，其余为 -1。
    """
    y = np.asarray(y).astype(int).ravel()
    if num_classes is None:
        num_classes = int(y.max()) + 1
    Y = -np.ones((len(y), num_classes), dtype=np.float64)
    Y[np.arange(len(y)), y] = 1.0
    return Y


# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------


def make_synthetic_classification(
    n_train: int = 2000,
    n_test: int = 500,
    n_features: int = 20,
    n_classes: int = 3,
    noise: float = 0.4,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成可分性中等、带噪声的多分类合成数据（用于增量学习仿真）。

    每个类别在特征空间中有一个随机中心，样本 = 中心 + 高斯噪声。
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_classes, n_features)) * 2.0

    def _sample(n: int):
        labels = rng.integers(0, n_classes, size=n)
        X = centers[labels] + noise * rng.standard_normal((n, n_features))
        return X.astype(np.float64), labels

    X_tr, y_tr = _sample(n_train)
    X_te, y_te = _sample(n_test)
    return X_tr, y_tr, X_te, y_te


def make_synthetic_regression(
    n_train: int = 2000,
    n_test: int = 500,
    n_features: int = 10,
    noise: float = 0.1,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成带非线性项 (sin / 二次项) 的回归合成数据。"""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n_features)

    def _sample(n: int):
        X = rng.standard_normal((n, n_features))
        y = X @ w + 0.5 * np.sin(X[:, 0] * 2.0) + 0.3 * X[:, 1] ** 2
        y = y + noise * rng.standard_normal(n)
        return X.astype(np.float64), y.astype(np.float64)

    X_tr, y_tr = _sample(n_train)
    X_te, y_te = _sample(n_test)
    return X_tr, y_tr, X_te, y_te


# ---------------------------------------------------------------------------
# sklearn 数据集（可选）
# ---------------------------------------------------------------------------


def _require_sklearn(name: str) -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as e:
        raise ImportError(f"加载 {name} 需要 sklearn：pip install scikit-learn") from e


def _load_digits():
    _require_sklearn("digits")
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    data = load_digits()
    X = data.data.astype(np.float64) / 16.0
    y = data.target.astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)[:4][::1]


def _load_iris():
    _require_sklearn("iris")
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    data = load_iris()
    X = data.data.astype(np.float64)
    y = data.target.astype(int)
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)[:4][::1]


def _load_california():
    _require_sklearn("california_housing")
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    data = fetch_california_housing()
    X = data.data.astype(np.float64)
    y = data.target.astype(np.float64)
    return train_test_split(X, y, test_size=0.2, random_state=42)[:4][::1]


# ---------------------------------------------------------------------------
# IDX (MNIST / Fashion-MNIST) 加载
# ---------------------------------------------------------------------------


def _read_idx(path: str) -> np.ndarray:
    """读取 LeCun IDX 二进制格式（支持 .gz）。"""
    import gzip

    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        ndim = magic & 0xFF
        shape = [int.from_bytes(f.read(4), "big") for _ in range(ndim)]
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(shape)


def _load_mnist_like(path: str):
    """加载 MNIST / Fashion-MNIST 等 IDX 数据集。

    要求 ``path`` 下含 4 个文件 (gz 或解压版)::

        train-images-idx3-ubyte[.gz]
        train-labels-idx1-ubyte[.gz]
        t10k-images-idx3-ubyte[.gz]
        t10k-labels-idx1-ubyte[.gz]
    """

    def _find(keys: List[str]) -> str:
        for k in keys:
            p = os.path.join(path, k)
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"在 {path} 下找不到 {keys}")

    X_tr = _read_idx(_find(["train-images-idx3-ubyte", "train-images-idx3-ubyte.gz"])).astype(np.float64) / 255.0
    y_tr = _read_idx(_find(["train-labels-idx1-ubyte", "train-labels-idx1-ubyte.gz"])).astype(int)
    X_te = _read_idx(_find(["t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte.gz"])).astype(np.float64) / 255.0
    y_te = _read_idx(_find(["t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte.gz"])).astype(int)
    return X_tr.reshape(X_tr.shape[0], -1), y_tr, X_te.reshape(X_te.shape[0], -1), y_te


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------


def load_classification_dataset(name: str = "synthetic", path: Optional[str] = None, **kwargs):
    """加载分类数据集。

    支持:
        合成: ``synthetic``
        sklearn: ``digits`` / ``iris``
        IDX: ``mnist`` / ``fashion_mnist``（需要 ``path`` 指向 IDX 目录）
        UCI: ``pendigits`` / ``letter`` / ``shuttle`` / ``waveform`` / ``led``
    """
    name = name.lower().replace("-", "_")
    if name == "synthetic":
        return make_synthetic_classification(**kwargs)
    if name == "digits":
        return _load_digits()
    if name == "iris":
        return _load_iris()
    if name in {"mnist", "fashion_mnist"}:
        if not path or not os.path.isdir(path):
            raise FileNotFoundError(f"加载 {name} 需要 path 指向 IDX 数据目录")
        return _load_mnist_like(path)
    # UCI 数据集
    try:
        from utils.uci_loader import _CLASSIFICATION_LOADERS
    except ImportError:
        from .uci_loader import _CLASSIFICATION_LOADERS
    if name in _CLASSIFICATION_LOADERS:
        return _CLASSIFICATION_LOADERS[name](data_dir=path)
    raise ValueError(f"未知分类数据集: {name}")


def load_regression_dataset(name: str = "synthetic", path: Optional[str] = None, **kwargs):
    """加载回归数据集。

    支持:
        合成: ``synthetic``
        sklearn: ``california`` / ``california_housing``
        UCI: ``abalone`` / ``bodyfat`` / ``energy_efficiency`` / ``appliances_energy``
            / ``weather_izmir``
    """
    name = name.lower().replace("-", "_")
    if name == "synthetic":
        return make_synthetic_regression(**kwargs)
    if name in {"california", "california_housing"}:
        return _load_california()
    try:
        from utils.uci_loader import _REGRESSION_LOADERS
    except ImportError:
        from .uci_loader import _REGRESSION_LOADERS
    if name in _REGRESSION_LOADERS:
        return _REGRESSION_LOADERS[name](data_dir=path)
    raise ValueError(f"未知回归数据集: {name}")


# ---------------------------------------------------------------------------
# 数据流切分（增量学习场景）
# ---------------------------------------------------------------------------


def split_into_batches(
    X: np.ndarray,
    Y: np.ndarray,
    n_batches: int,
    shuffle: bool = True,
    seed: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """将 ``(X, Y)`` 等量切成 ``n_batches`` 份（论文 Section 4.1）。"""
    if n_batches < 1:
        raise ValueError("n_batches 至少 1")
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    parts = np.array_split(idx, n_batches)
    return [(X[p], Y[p]) for p in parts]


def split_random_batches(
    X: np.ndarray,
    Y: np.ndarray,
    n_batches: int,
    seed: int = 0,
    alpha: float = 2.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """模拟"不定 scale 数据流"（论文 Section 4.2），通过 Dirichlet 切分大小不均的块。

    保证：
        * 共生成 ``n_batches`` 个块
        * 每块至少 1 个样本（要求 ``n_batches <= n``）
        * 全部块大小之和等于 ``n``（无样本丢失或重复）

    Args:
        alpha: Dirichlet 浓度参数；越小越不均匀，默认 2.0 给出适度变化。

    Raises:
        ValueError: 当 ``n_batches < 1`` 或 ``n_batches > n``。
    """
    if n_batches < 1:
        raise ValueError("n_batches 至少 1")
    n = X.shape[0]
    if n_batches > n:
        raise ValueError(f"n_batches ({n_batches}) 不能大于样本数 ({n})")

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    # 1. Dirichlet 采样 + 至少 1 个样本
    fractions = rng.dirichlet(alpha=np.ones(n_batches) * alpha)
    sizes = np.maximum(1, np.round(fractions * n).astype(int))

    # 2. 修正总和到 n：多则从最大块扣，少则加到最大块。
    #    保证不会让任何块 size 变 0。
    while sizes.sum() != n:
        diff = n - sizes.sum()
        if diff > 0:
            sizes[np.argmax(sizes)] += diff
        else:
            # diff < 0：从最大块扣，但每块仍保 >= 1
            i = int(np.argmax(sizes))
            take = min(-diff, sizes[i] - 1)
            if take == 0:
                # 所有块都已经只剩 1 个，理论上 n < n_batches，前面已经检查过
                break
            sizes[i] -= take

    splits = np.cumsum(sizes)[:-1]
    parts = np.array_split(idx, splits)
    return [(X[p], Y[p]) for p in parts]
