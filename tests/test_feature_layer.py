# -*- coding: utf-8 -*-
"""FeatureLayer 单元测试。

覆盖：
  * 随机权重确定性
  * Z / H / A 维度正确
  * 节点增量保持已有列不变（增量学习的关键性质）
  * 单一窗口前向 (transform_window) 与拼接结果一致
"""

from __future__ import annotations

import numpy as np
import pytest

from utils.feature_layer import FeatureLayer, standardize_minmax


def _make_layer(seed: int = 0, n_enh: int = 30) -> FeatureLayer:
    fl = FeatureLayer(
        n_mapping_per_window=5, n_mapping_windows=4,
        n_enhancement=n_enh, seed=seed,
    )
    fl.fit_random_weights(input_dim=12)
    return fl


def test_layer_deterministic_under_same_seed() -> None:
    """相同 seed → 完全相同的特征输出。"""
    fl1 = _make_layer(seed=42)
    fl2 = _make_layer(seed=42)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 12))
    assert np.allclose(fl1.transform(X), fl2.transform(X))


def test_layer_dimensions() -> None:
    fl = _make_layer(seed=0, n_enh=30)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((7, 12))

    Z = fl.transform_mapping(X)
    H = fl.transform_enhancement(Z)
    A = fl.transform(X)

    assert Z.shape == (7, 5 * 4)
    assert H.shape == (7, 30)
    assert A.shape == (7, 5 * 4 + 30)
    assert fl.feature_dim == A.shape[1]
    assert fl.mapping_dim == 20
    assert fl.enhancement_dim == 30


def test_add_enhancement_window_appends_columns() -> None:
    """节点增量后：原有列必须保持不变（保证记忆模块语义正确）。"""
    fl = _make_layer(seed=1, n_enh=10)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 12))

    A_before = fl.transform(X)
    fl.add_enhancement_window(7)
    A_after = fl.transform(X)

    assert A_after.shape[1] == A_before.shape[1] + 7
    # ⭐ 关键：旧列完全一致
    assert np.allclose(A_after[:, : A_before.shape[1]], A_before)


def test_transform_window_consistent_with_full_transform() -> None:
    """transform_window 与 transform_enhancement 拼接结果应完全一致。"""
    fl = _make_layer(seed=2, n_enh=15)
    fl.add_enhancement_window(8)  # 现在有 2 个窗口

    rng = np.random.default_rng(7)
    X = rng.standard_normal((5, 12))
    Z = fl.transform_mapping(X)

    H_full = fl.transform_enhancement(Z)
    H_concat = np.concatenate(
        [fl.transform_window(Z, 0), fl.transform_window(Z, 1)], axis=1
    )
    assert np.allclose(H_full, H_concat)


def test_transform_window_index_validation() -> None:
    fl = _make_layer(seed=0, n_enh=5)
    Z = np.zeros((1, 20))
    with pytest.raises(IndexError):
        fl.transform_window(Z, 99)


def test_layer_requires_fit_before_transform() -> None:
    fl = FeatureLayer(seed=0)  # 未 fit
    with pytest.raises(RuntimeError):
        fl.transform(np.zeros((3, 4)))


def test_unknown_activation_raises() -> None:
    with pytest.raises(ValueError):
        FeatureLayer(activation="not_a_real_activation").fit_random_weights(input_dim=5)


def test_standardize_minmax_train_test() -> None:
    rng = np.random.default_rng(0)
    X_tr = rng.standard_normal((100, 5))
    X_te = rng.standard_normal((40, 5))

    Xs_tr, Xs_te, _ = standardize_minmax(X_tr, X_te)
    assert Xs_tr.min() >= -1e-9 and Xs_tr.max() <= 1 + 1e-9
    assert Xs_te.shape == X_te.shape  # 形状不变


def test_standardize_handles_constant_column() -> None:
    """常量列应被安全处理（避免除零）。"""
    X = np.ones((10, 3))
    X[:, 1] = np.arange(10)
    Xs, _ = standardize_minmax(X)
    assert np.isfinite(Xs).all()
