# -*- coding: utf-8 -*-
"""utils/data.py 测试。

覆盖：
  * one_hot_encode：±1 编码、自动推断 num_classes、形状
  * make_synthetic_classification / regression：可重现性、形状、类别覆盖
  * split_into_batches：等量切分、不丢失样本、shuffle 行为
  * split_random_batches：不均匀切分、总数守恒
  * load_classification_dataset / load_regression_dataset：参数校验
"""

from __future__ import annotations

import numpy as np
import pytest

from utils.data import (
    load_classification_dataset,
    load_regression_dataset,
    make_synthetic_classification,
    make_synthetic_regression,
    one_hot_encode,
    split_into_batches,
    split_random_batches,
)


# ===========================================================================
# one_hot_encode
# ===========================================================================


def test_one_hot_encode_basic() -> None:
    y = np.array([0, 1, 2, 1, 0])
    Y = one_hot_encode(y, num_classes=3)

    assert Y.shape == (5, 3)
    # ±1 编码：每行只有一个 +1，其余 -1
    assert np.all((Y == 1.0) | (Y == -1.0))
    assert (Y.sum(axis=1) == -1.0).all(), "每行应有 1 个 +1，2 个 -1，和 = -1"
    # 索引位置正确
    for i, label in enumerate(y):
        assert Y[i, label] == 1.0


def test_one_hot_encode_auto_num_classes() -> None:
    """num_classes=None 时自动推断为 max(y) + 1。"""
    y = np.array([0, 2, 4])
    Y = one_hot_encode(y)
    assert Y.shape == (3, 5)


def test_one_hot_encode_with_extra_classes() -> None:
    """num_classes 大于实际最大值时，多出的列全为 -1。"""
    y = np.array([0, 1, 2])
    Y = one_hot_encode(y, num_classes=5)
    assert Y.shape == (3, 5)
    # 第 3、4 列从未被任何样本激活
    assert (Y[:, 3] == -1).all()
    assert (Y[:, 4] == -1).all()


def test_one_hot_encode_handles_1d_input() -> None:
    Y = one_hot_encode([0, 1, 0], num_classes=2)
    assert Y.shape == (3, 2)


# ===========================================================================
# make_synthetic_classification
# ===========================================================================


def test_synthetic_classification_shapes_and_classes() -> None:
    X_tr, y_tr, X_te, y_te = make_synthetic_classification(
        n_train=200, n_test=50, n_features=10, n_classes=4, seed=0
    )
    assert X_tr.shape == (200, 10)
    assert y_tr.shape == (200,)
    assert X_te.shape == (50, 10)
    assert y_te.shape == (50,)
    # 训练集应覆盖所有类别（高概率）
    assert set(np.unique(y_tr).tolist()) <= {0, 1, 2, 3}


def test_synthetic_classification_reproducible() -> None:
    a = make_synthetic_classification(seed=42)
    b = make_synthetic_classification(seed=42)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_synthetic_classification_different_seeds_differ() -> None:
    X1, _, _, _ = make_synthetic_classification(seed=0, n_train=100)
    X2, _, _, _ = make_synthetic_classification(seed=1, n_train=100)
    assert not np.allclose(X1, X2)


# ===========================================================================
# make_synthetic_regression
# ===========================================================================


def test_synthetic_regression_shapes_and_dtype() -> None:
    X_tr, y_tr, X_te, y_te = make_synthetic_regression(
        n_train=300, n_test=80, n_features=8, seed=1
    )
    assert X_tr.shape == (300, 8)
    assert y_tr.shape == (300,)
    assert X_te.shape == (80, 8)
    assert y_te.dtype == np.float64


def test_synthetic_regression_reproducible() -> None:
    a = make_synthetic_regression(seed=7)
    b = make_synthetic_regression(seed=7)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


# ===========================================================================
# split_into_batches
# ===========================================================================


def test_split_into_batches_equal_size() -> None:
    X = np.arange(100).reshape(100, 1)
    Y = np.arange(100).reshape(100, 1)
    batches = split_into_batches(X, Y, n_batches=5, shuffle=False)

    assert len(batches) == 5
    sizes = [len(b[0]) for b in batches]
    assert sum(sizes) == 100
    # 等量切分：差距最多 1
    assert max(sizes) - min(sizes) <= 1


def test_split_into_batches_no_data_loss() -> None:
    """切分后所有样本必须被恰好覆盖一次。"""
    X = np.arange(50).reshape(-1, 1).astype(float)
    Y = X.copy()
    batches = split_into_batches(X, Y, n_batches=4, shuffle=True, seed=42)
    all_x = np.vstack([b[0] for b in batches])

    assert all_x.shape == X.shape
    assert sorted(all_x.ravel().tolist()) == sorted(X.ravel().tolist())


def test_split_into_batches_shuffle_changes_order() -> None:
    """shuffle=True 时两个不同 seed 的结果应该不同。"""
    X = np.arange(100).reshape(-1, 1)
    Y = X.copy()
    a = split_into_batches(X, Y, n_batches=5, shuffle=True, seed=0)
    b = split_into_batches(X, Y, n_batches=5, shuffle=True, seed=1)
    assert not np.array_equal(a[0][0], b[0][0])


def test_split_into_batches_invalid_n_batches() -> None:
    X = np.zeros((10, 2))
    Y = np.zeros((10, 1))
    with pytest.raises(ValueError):
        split_into_batches(X, Y, n_batches=0)


# ===========================================================================
# split_random_batches
# ===========================================================================


def test_split_random_batches_total_count_preserved() -> None:
    X = np.arange(200).reshape(-1, 1)
    Y = X.copy()
    batches = split_random_batches(X, Y, n_batches=8, seed=0)

    assert len(batches) == 8
    total = sum(len(b[0]) for b in batches)
    assert total == 200


def test_split_random_batches_size_variability() -> None:
    """Dirichlet 切分应该产生真正不均匀的 batch（max ≠ min）。"""
    X = np.arange(500).reshape(-1, 1)
    Y = X.copy()
    batches = split_random_batches(X, Y, n_batches=10, seed=0, alpha=1.0)
    sizes = [len(b[0]) for b in batches]
    # 至少 1 个 batch 大小与最大值差距 > 5
    assert max(sizes) - min(sizes) > 5


def test_split_random_batches_each_size_at_least_1() -> None:
    """即使 alpha 极小、随机性极强，每个 batch 也至少有 1 个样本。"""
    X = np.arange(50).reshape(-1, 1)
    Y = X.copy()
    batches = split_random_batches(X, Y, n_batches=10, seed=99, alpha=0.5)
    assert all(len(b[0]) >= 1 for b in batches)


def test_split_random_batches_no_overlap_no_duplicate() -> None:
    """所有切分块应恰好覆盖原始 X 一次（不重不漏）。"""
    X = np.arange(120).reshape(-1, 1)
    Y = X.copy()
    batches = split_random_batches(X, Y, n_batches=7, seed=10)
    all_x = np.vstack([b[0] for b in batches])
    assert all_x.shape == X.shape
    assert sorted(all_x.ravel().tolist()) == sorted(X.ravel().tolist())


def test_split_random_batches_too_many_batches_raises() -> None:
    """n_batches > n 时应抛 ValueError（无法保证每块 ≥ 1）。"""
    X = np.zeros((5, 2))
    Y = np.zeros((5, 1))
    with pytest.raises(ValueError):
        split_random_batches(X, Y, n_batches=10)


def test_split_random_batches_extreme_alpha() -> None:
    """alpha 极小（接近 one-hot 分布）时仍应满足约束。"""
    X = np.arange(30).reshape(-1, 1)
    Y = X.copy()
    batches = split_random_batches(X, Y, n_batches=5, seed=0, alpha=0.01)
    sizes = [len(b[0]) for b in batches]
    assert all(s >= 1 for s in sizes)
    assert sum(sizes) == 30


# ===========================================================================
# 统一入口
# ===========================================================================


def test_load_classification_dataset_synthetic() -> None:
    X_tr, y_tr, X_te, y_te = load_classification_dataset(
        name="synthetic", n_train=100, n_test=30, n_features=5, n_classes=3, seed=0
    )
    assert X_tr.shape == (100, 5)
    assert y_tr.shape == (100,)


def test_load_classification_dataset_unknown_raises() -> None:
    with pytest.raises(ValueError):
        load_classification_dataset(name="not_a_real_dataset")


def test_load_classification_dataset_mnist_without_path_raises() -> None:
    """加载 mnist 时必须提供有效 path。"""
    with pytest.raises(FileNotFoundError):
        load_classification_dataset(name="mnist", path=None)


def test_load_regression_dataset_synthetic() -> None:
    X_tr, y_tr, X_te, y_te = load_regression_dataset(name="synthetic", n_train=80, n_test=20)
    assert X_tr.shape[0] == 80
    assert y_tr.shape == (80,)


def test_load_regression_dataset_unknown_raises() -> None:
    with pytest.raises(ValueError):
        load_regression_dataset(name="bogus_regression")
