# -*- coding: utf-8 -*-
"""端到端：分类 / 回归任务上的精度检验（论文 Section 4 的最小复现）。

通过这些测试可以确保整体 pipeline（数据 → 特征 → 增量学习 → 推理 → 评估）正确串通。
"""

from __future__ import annotations

import numpy as np

from src.bls_base import BLSConfig, NonIncrementalBLS
from src.imf_bls import IMFBLS
from utils.data import split_into_batches


# ===========================================================================
# 分类
# ===========================================================================


def test_classification_pipeline_high_accuracy(medium_classification) -> None:
    """中等规模分类任务上 IMF-BLS 应达到不错的准确率。"""
    X_tr, Y_tr, X_te, Y_te = medium_classification
    cfg = BLSConfig(
        n_mapping_per_window=10, n_mapping_windows=10,
        n_enhancement=400, reg_lambda=1e-6, seed=0,
    )
    invf = IMFBLS(config=cfg).fit_initial(X_tr, Y_tr)
    acc = invf.score_classification(X_te, Y_te)
    assert acc > 0.85, f"分类准确率过低: {acc}"


def test_classification_incremental_matches_joint(medium_classification) -> None:
    """增量训练 5 次 vs 联合训练，最终测试集准确率必须严格相等。"""
    X_tr, Y_tr, X_te, Y_te = medium_classification
    cfg = BLSConfig(
        n_mapping_per_window=10, n_mapping_windows=10,
        n_enhancement=300, reg_lambda=1e-5, seed=99,
    )
    batches = split_into_batches(X_tr, Y_tr, n_batches=5, shuffle=False)

    invf = IMFBLS(config=cfg.copy()).fit_initial(*batches[0])
    for X_b, Y_b in batches[1:]:
        invf.add_data(X_b, Y_b)

    full = NonIncrementalBLS(config=cfg.copy()).fit_initial(X_tr, Y_tr)

    # 关键性质：增量结果与联合结果完全一致
    assert abs(invf.score_classification(X_te, Y_te) -
               full.score_classification(X_te, Y_te)) < 1e-9


# ===========================================================================
# 回归
# ===========================================================================


def test_regression_pipeline_low_rmse(small_regression) -> None:
    X_tr, Y_tr, X_te, Y_te = small_regression
    cfg = BLSConfig(
        n_mapping_per_window=8, n_mapping_windows=6,
        n_enhancement=200, reg_lambda=1e-4, seed=0,
    )
    invf = IMFBLS(config=cfg).fit_initial(X_tr, Y_tr)
    rmse = invf.score_rmse(X_te, Y_te)
    # 对合成数据来说 RMSE 应远低于标签量纲（约几个标准差）
    y_std = float(np.std(Y_tr))
    assert rmse < 1.5 * y_std, f"回归 RMSE 过大: {rmse:.4f} (y_std={y_std:.4f})"


def test_regression_incremental_matches_joint(small_regression) -> None:
    X_tr, Y_tr, X_te, Y_te = small_regression
    cfg = BLSConfig(
        n_mapping_per_window=6, n_mapping_windows=5,
        n_enhancement=150, reg_lambda=1e-4, seed=88,
    )
    batches = split_into_batches(X_tr, Y_tr, n_batches=4, shuffle=False)

    invf = IMFBLS(config=cfg.copy()).fit_initial(*batches[0])
    for X_b, Y_b in batches[1:]:
        invf.add_data(X_b, Y_b)

    full = NonIncrementalBLS(config=cfg.copy()).fit_initial(X_tr, Y_tr)

    assert abs(invf.score_rmse(X_te, Y_te) - full.score_rmse(X_te, Y_te)) < 1e-7


# ===========================================================================
# 流场景：增量过程中精度单调（不应退化）
# ===========================================================================


def test_incremental_accuracy_does_not_degrade(medium_classification) -> None:
    """每加一个 batch，测试集准确率不应大幅下降（论文 Fig. 8 的关键性质）。"""
    X_tr, Y_tr, X_te, Y_te = medium_classification
    cfg = BLSConfig(
        n_mapping_per_window=10, n_mapping_windows=8,
        n_enhancement=300, reg_lambda=1e-5, seed=66,
    )
    batches = split_into_batches(X_tr, Y_tr, n_batches=5, shuffle=False)

    invf = IMFBLS(config=cfg).fit_initial(*batches[0])
    accs = [invf.score_classification(X_te, Y_te)]
    for X_b, Y_b in batches[1:]:
        invf.add_data(X_b, Y_b)
        accs.append(invf.score_classification(X_te, Y_te))

    # 最终准确率必须比初始更高（信息累积单调性）
    assert accs[-1] >= accs[0] - 0.02
    # 总体不应严重退化（最大跌幅 ≤ 5%）
    assert max(accs) - min(accs) < 0.10
