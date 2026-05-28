# -*- coding: utf-8 -*-
"""共享测试夹具 (pytest fixtures) 与工具函数。"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def random_well_conditioned_R(p: int, seed: int = 0) -> np.ndarray:
    """生成对角元 ≥ 1 的良态上三角矩阵。"""
    rng = np.random.default_rng(seed)
    R = np.triu(rng.standard_normal((p, p)))
    np.fill_diagonal(R, np.abs(np.diag(R)) + 1.0)
    return R


def random_full_rank_matrix(m: int, n: int, seed: int = 0) -> np.ndarray:
    """生成行 m、列 n 的（数值上）满列秩矩阵。"""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    # 加一点对角扰动以避免极端病态
    if m >= n:
        A[:n, :] += 0.1 * np.eye(n)
    return A


# ---------------------------------------------------------------------------
# 数据集 fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def small_classification():
    """小规模分类数据：X(600,15) / Y(600,3 one-hot)。"""
    from utils.data import make_synthetic_classification, one_hot_encode
    from utils.feature_layer import standardize_minmax

    X_tr, y_tr, X_te, y_te = make_synthetic_classification(
        n_train=600, n_test=200, n_features=15, n_classes=3, seed=0
    )
    X_tr_s, X_te_s, _ = standardize_minmax(X_tr, X_te)
    Y_tr = one_hot_encode(y_tr, num_classes=3)
    Y_te = one_hot_encode(y_te, num_classes=3)
    return X_tr_s, Y_tr, X_te_s, Y_te


@pytest.fixture
def medium_classification():
    """中等规模分类数据：X(1500,20) / Y(1500,5 one-hot)。"""
    from utils.data import make_synthetic_classification, one_hot_encode
    from utils.feature_layer import standardize_minmax

    X_tr, y_tr, X_te, y_te = make_synthetic_classification(
        n_train=1500, n_test=400, n_features=20, n_classes=5, seed=42
    )
    X_tr_s, X_te_s, _ = standardize_minmax(X_tr, X_te)
    Y_tr = one_hot_encode(y_tr, num_classes=5)
    Y_te = one_hot_encode(y_te, num_classes=5)
    return X_tr_s, Y_tr, X_te_s, Y_te


@pytest.fixture
def small_regression():
    """小规模回归数据：X(800,10) / y(800,)。"""
    from utils.data import make_synthetic_regression
    from utils.feature_layer import standardize_minmax

    X_tr, y_tr, X_te, y_te = make_synthetic_regression(
        n_train=800, n_test=200, n_features=10, seed=7
    )
    X_tr_s, X_te_s, _ = standardize_minmax(X_tr, X_te)
    return X_tr_s, y_tr.reshape(-1, 1), X_te_s, y_te.reshape(-1, 1)


# ---------------------------------------------------------------------------
# 通用断言
# ---------------------------------------------------------------------------


def assert_R_is_upper_triangular(R: np.ndarray, atol: float = 1e-12) -> None:
    """检查 R 严格上三角（下三角部分应为 0）。"""
    below = np.tril(R, k=-1)
    assert np.max(np.abs(below)) <= atol, f"R 不是上三角，max|tril|={np.max(np.abs(below))}"


def assert_R_diag_positive(R: np.ndarray) -> None:
    """检查 R 的对角线全为正（保证 QR 因子唯一）。"""
    diag = np.diag(R)
    assert (diag > 0).all(), f"R 对角应全正，得到 min={diag.min()}"
