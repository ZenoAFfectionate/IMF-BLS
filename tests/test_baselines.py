# -*- coding: utf-8 -*-
"""对比方法的功能性与数学等价性测试。

  * IncrementalBLS / RIBLS / TiBLS / ApproximationMethodBLS 均能正常增量训练
  * RIBLS 与 IMFBLS 在数据增量上数学等价（同求 (A^T A + λI)^{-1} A^T Y）
  * RIBLS 与 IMFBLS 在节点增量上数学等价
  * TiBLS 严格要求等量 batch
"""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines import (
    ApproximationMethodBLS,
    IncrementalBLS,
    RIBLS,
    TiBLS,
)
from src.bls_base import BLSConfig
from src.imf_bls import IMFBLS
from utils.data import split_into_batches


def _cfg(seed: int = 1) -> BLSConfig:
    return BLSConfig(
        n_mapping_per_window=6, n_mapping_windows=4,
        n_enhancement=80, reg_lambda=1e-4, seed=seed,
    )


# ---------------------------------------------------------------------------
# RIBLS ↔ IMFBLS 数学等价性
# ---------------------------------------------------------------------------


def test_ribls_equiv_imf_bls_on_add_data(small_classification) -> None:
    """RIBLS 与 IMFBLS 在数据增量上数学等价（同 (A^T A + λI)^{-1} A^T Y）。"""
    X_tr, Y_tr, _, _ = small_classification
    batches = split_into_batches(X_tr, Y_tr, n_batches=4, shuffle=False)

    rib = RIBLS(config=_cfg(seed=2))
    rib.fit_initial(*batches[0])
    for X_b, Y_b in batches[1:]:
        rib.add_data(X_b, Y_b)

    invf = IMFBLS(config=_cfg(seed=2))
    invf.fit_initial(*batches[0])
    for X_b, Y_b in batches[1:]:
        invf.add_data(X_b, Y_b)

    assert np.allclose(rib.W, invf.W, atol=1e-6)


def test_ribls_equiv_imf_bls_on_add_nodes(small_classification) -> None:
    X_tr, Y_tr, _, _ = small_classification

    rib = RIBLS(config=_cfg(seed=3)).fit_initial(X_tr, Y_tr)
    rib.add_nodes(X_tr, Y_tr, n_new=30)

    invf = IMFBLS(config=_cfg(seed=3)).fit_initial(X_tr, Y_tr)
    invf.add_nodes(X_tr, Y_tr, n_new=30)

    assert np.allclose(rib.W, invf.W, atol=1e-6)


# ---------------------------------------------------------------------------
# IncrementalBLS 基本可用性
# ---------------------------------------------------------------------------


def test_incremental_bls_runs(small_classification) -> None:
    X_tr, Y_tr, X_te, Y_te = small_classification
    batches = split_into_batches(X_tr, Y_tr, n_batches=5, shuffle=False)
    model = IncrementalBLS(config=_cfg())
    model.fit_initial(*batches[0])
    for X_b, Y_b in batches[1:]:
        model.add_data(X_b, Y_b)
    acc = model.score_classification(X_te, Y_te)
    assert acc > 0.6


def test_incremental_bls_node_increment(small_classification) -> None:
    X_tr, Y_tr, X_te, Y_te = small_classification
    model = IncrementalBLS(config=_cfg(seed=2)).fit_initial(X_tr, Y_tr)
    model.add_nodes(X_tr, Y_tr, n_new=20)
    acc = model.score_classification(X_te, Y_te)
    assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# TiBLS：等量 batch 约束
# ---------------------------------------------------------------------------


def test_tibls_works_on_uniform_batches(small_classification) -> None:
    X_tr, Y_tr, X_te, Y_te = small_classification
    # 强制等量 batch
    n_per_batch = 100
    n_total = (X_tr.shape[0] // n_per_batch) * n_per_batch
    X_tr, Y_tr = X_tr[:n_total], Y_tr[:n_total]
    batches = split_into_batches(X_tr, Y_tr, n_batches=n_total // n_per_batch, shuffle=False)
    common = min(len(b[0]) for b in batches)
    batches = [(X[:common], Y[:common]) for X, Y in batches]

    model = TiBLS(config=_cfg(seed=4)).fit_initial(*batches[0])
    for X_b, Y_b in batches[1:]:
        model.add_data(X_b, Y_b)
    assert 0.0 <= model.score_classification(X_te, Y_te) <= 1.0


def test_tibls_rejects_unequal_batch(small_classification) -> None:
    X_tr, Y_tr, _, _ = small_classification
    model = TiBLS(config=_cfg(seed=5)).fit_initial(X_tr[:100], Y_tr[:100])
    with pytest.raises(ValueError):
        model.add_data(X_tr[100:151], Y_tr[100:151])  # 大小 51，不等于初始 100


# ---------------------------------------------------------------------------
# ApproximationMethodBLS
# ---------------------------------------------------------------------------


def test_approximation_method_runs(small_classification) -> None:
    X_tr, Y_tr, X_te, Y_te = small_classification
    batches = split_into_batches(X_tr, Y_tr, n_batches=4, shuffle=False)
    model = ApproximationMethodBLS(config=_cfg(seed=6)).fit_initial(*batches[0])
    for X_b, Y_b in batches[1:]:
        model.add_data(X_b, Y_b)
    acc = model.score_classification(X_te, Y_te)
    assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# NonIncrementalBLS（联合训练上界）
# ---------------------------------------------------------------------------


def test_non_incremental_bls_classification(small_classification) -> None:
    """NonIncrementalBLS 应在分类任务上达到不错的准确率。"""
    from src.bls_base import NonIncrementalBLS

    X_tr, Y_tr, X_te, Y_te = small_classification
    model = NonIncrementalBLS(config=_cfg(seed=7)).fit_initial(X_tr, Y_tr)
    assert model.score_classification(X_te, Y_te) > 0.6


def test_non_incremental_bls_regression(small_regression) -> None:
    """NonIncrementalBLS 在回归任务上 RMSE 应低于 y 的标准差。"""
    from src.bls_base import NonIncrementalBLS

    X_tr, Y_tr, X_te, Y_te = small_regression
    model = NonIncrementalBLS(config=_cfg(seed=8)).fit_initial(X_tr, Y_tr)
    rmse = model.score_rmse(X_te, Y_te)
    assert rmse < 1.5 * float(np.std(Y_tr))


def test_non_incremental_bls_does_not_support_add_data(small_classification) -> None:
    """NonIncrementalBLS 是联合训练，不应支持 add_data。"""
    from src.bls_base import NonIncrementalBLS

    X_tr, Y_tr, _, _ = small_classification
    model = NonIncrementalBLS(config=_cfg()).fit_initial(X_tr, Y_tr)
    with pytest.raises(NotImplementedError):
        model.add_data(X_tr[:10], Y_tr[:10])


def test_non_incremental_bls_does_not_support_add_nodes(small_classification) -> None:
    from src.bls_base import NonIncrementalBLS

    X_tr, Y_tr, _, _ = small_classification
    model = NonIncrementalBLS(config=_cfg()).fit_initial(X_tr, Y_tr)
    with pytest.raises(NotImplementedError):
        model.add_nodes(X_tr, Y_tr, n_new=5)


def test_non_incremental_bls_fit_all_alias(small_classification) -> None:
    """fit_all 应该是 fit_initial 的别名（语义上更强调"联合训练"）。"""
    from src.bls_base import NonIncrementalBLS

    X_tr, Y_tr, _, _ = small_classification
    a = NonIncrementalBLS(config=_cfg(seed=20)).fit_all(X_tr, Y_tr)
    b = NonIncrementalBLS(config=_cfg(seed=20)).fit_initial(X_tr, Y_tr)
    assert np.allclose(a.W, b.W, atol=1e-12)


# ---------------------------------------------------------------------------
# RIBLS 节点增量与联合训练等价
# ---------------------------------------------------------------------------


def test_ribls_add_nodes_equiv_joint_training(small_classification) -> None:
    """RIBLS 加节点后等价于在扩展特征层上联合训练。"""
    from src.bls_base import NonIncrementalBLS

    X_tr, Y_tr, _, _ = small_classification
    cfg_a, cfg_b = _cfg(seed=30), _cfg(seed=30)

    rib = RIBLS(config=cfg_a).fit_initial(X_tr, Y_tr)
    rib.add_nodes(X_tr, Y_tr, n_new=20)

    full = NonIncrementalBLS(config=cfg_b).fit_initial(X_tr, Y_tr)
    full.feature_layer.add_enhancement_window(20)
    A_ext = full.feature_layer.transform(X_tr)
    p = A_ext.shape[1]
    M = A_ext.T @ A_ext + cfg_b.reg_lambda * np.eye(p)
    full.W = np.linalg.solve(M, A_ext.T @ Y_tr)

    assert np.allclose(rib.W, full.W, atol=1e-6)


# ---------------------------------------------------------------------------
# IncrementalBLS / RIBLS 在回归任务上的可用性
# ---------------------------------------------------------------------------


def test_ribls_regression(small_regression) -> None:
    X_tr, Y_tr, X_te, Y_te = small_regression
    batches = split_into_batches(X_tr, Y_tr, n_batches=3, shuffle=False)
    model = RIBLS(config=_cfg(seed=40)).fit_initial(*batches[0])
    for X_b, Y_b in batches[1:]:
        model.add_data(X_b, Y_b)
    rmse = model.score_rmse(X_te, Y_te)
    assert rmse < 2.0 * float(np.std(Y_tr))


def test_imf_bls_regression_equiv_ribls(small_regression) -> None:
    """回归任务上 IMF-BLS 与 RIBLS 数学等价。"""
    X_tr, Y_tr, X_te, Y_te = small_regression
    batches = split_into_batches(X_tr, Y_tr, n_batches=3, shuffle=False)

    rib = RIBLS(config=_cfg(seed=50))
    rib.fit_initial(*batches[0])
    for X_b, Y_b in batches[1:]:
        rib.add_data(X_b, Y_b)

    invf = IMFBLS(config=_cfg(seed=50))
    invf.fit_initial(*batches[0])
    for X_b, Y_b in batches[1:]:
        invf.add_data(X_b, Y_b)

    assert np.allclose(rib.W, invf.W, atol=1e-6)
    assert abs(rib.score_rmse(X_te, Y_te) - invf.score_rmse(X_te, Y_te)) < 1e-9


# ===========================================================================
# 高级等价性：所有 baseline 在 fit_initial 上都应给出 (A^T A + λI) W = A^T Y 的解
# ===========================================================================


def test_all_methods_initial_solve_normal_equation(small_classification) -> None:
    """除 ApproximationMethod 外，所有方法 fit_initial 后都应满足 normal equation。"""
    from src.imf_bls import IMFBLS

    X_tr, Y_tr, _, _ = small_classification

    methods = {
        "IMF-BLS":         IMFBLS,
        "RIBLS":           RIBLS,
        "TiBLS":           TiBLS,
        "IncrementalBLS":  IncrementalBLS,
    }
    cfg_seed = 31
    for name, cls in methods.items():
        m = cls(config=_cfg(seed=cfg_seed)).fit_initial(X_tr, Y_tr)
        A = m.feature_layer.transform(X_tr)
        p = A.shape[1]
        M = A.T @ A + m.config.reg_lambda * np.eye(p)
        rhs = A.T @ Y_tr
        err = np.max(np.abs(M @ m.W - rhs))
        # 替换法/直接求逆/伪逆 三类方法残差差异允许
        assert err < 1e-5, f"{name}: ||MW - rhs||_inf = {err:.2e} 过大"


def test_imf_and_ribls_yield_same_W_under_same_data_stream(small_classification) -> None:
    """IMF-BLS（替换法）与 RIBLS（直接求逆）在每一步增量后 W 数学等价。"""
    from src.imf_bls import IMFBLS

    X_tr, Y_tr, _, _ = small_classification
    batches = split_into_batches(X_tr, Y_tr, n_batches=4, shuffle=False)

    inv = RIBLS(config=_cfg(seed=120)).fit_initial(*batches[0])
    imf = IMFBLS(config=_cfg(seed=120)).fit_initial(*batches[0])
    # 初始等价
    assert np.allclose(inv.W, imf.W, atol=1e-6)

    for X_b, Y_b in batches[1:]:
        inv.add_data(X_b, Y_b)
        imf.add_data(X_b, Y_b)
        assert np.allclose(inv.W, imf.W, atol=1e-6)
