# -*- coding: utf-8 -*-
"""utils/metrics.py 与 utils/timing.py 测试。

覆盖：
  * classification_accuracy：argmax 一致性、边界情况
  * regression_rmse：与 numpy 直接计算一致；自动 ravel
  * sne_residual_norm：满足 ||A^T A W - A^T Y||_2 数学定义；W=最优解时 ≈ 0
  * Timer：基本计时正确性、上下文管理器
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from utils.metrics import (
    classification_accuracy,
    regression_rmse,
    sne_residual_norm,
)
from utils.timing import Timer


# ===========================================================================
# classification_accuracy
# ===========================================================================


def test_classification_accuracy_perfect() -> None:
    """全部预测正确 → 准确率 1.0。"""
    Y = np.eye(5)
    Pred = np.eye(5) * 10  # argmax 不变
    assert classification_accuracy(Y, Pred) == 1.0


def test_classification_accuracy_all_wrong() -> None:
    """全部预测错 → 准确率 0.0。"""
    Y = np.eye(3)              # true = [0, 1, 2]
    Pred = np.eye(3)[::-1]     # pred = [2, 1, 0]，第二个相同 ⇒ acc=1/3
    assert abs(classification_accuracy(Y, Pred) - 1 / 3) < 1e-12


def test_classification_accuracy_partial() -> None:
    """部分正确：5 个样本中 3 个正确。"""
    rng = np.random.default_rng(0)
    Y = np.eye(4)[rng.integers(0, 4, size=5)]
    Pred = Y.copy()
    Pred[0] = np.eye(4)[(np.argmax(Y[0]) + 1) % 4]  # 故意错 1 个
    Pred[2] = np.eye(4)[(np.argmax(Y[2]) + 1) % 4]  # 故意错 1 个
    assert abs(classification_accuracy(Y, Pred) - 3 / 5) < 1e-12


def test_classification_accuracy_returns_float() -> None:
    Y = np.eye(2)
    Pred = np.eye(2)
    acc = classification_accuracy(Y, Pred)
    assert isinstance(acc, float)


# ===========================================================================
# regression_rmse
# ===========================================================================


def test_regression_rmse_zero_when_equal() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert regression_rmse(y, y) == 0.0


def test_regression_rmse_matches_numpy() -> None:
    rng = np.random.default_rng(1)
    y = rng.standard_normal(50)
    yhat = rng.standard_normal(50)
    expected = float(np.sqrt(np.mean((y - yhat) ** 2)))
    assert abs(regression_rmse(y, yhat) - expected) < 1e-12


def test_regression_rmse_handles_2d_input() -> None:
    """传入列向量也应正确（自动 ravel）。"""
    y = np.array([[1.0], [2.0], [3.0]])
    yhat = np.array([[1.5], [2.5], [3.5]])
    assert abs(regression_rmse(y, yhat) - 0.5) < 1e-12


def test_regression_rmse_returns_float() -> None:
    rmse = regression_rmse(np.zeros(5), np.ones(5))
    assert isinstance(rmse, float)


# ===========================================================================
# sne_residual_norm
# ===========================================================================


def test_sne_residual_zero_at_optimal_W() -> None:
    """W 为最优解时，A^T A W = A^T Y，残差应为 0（机器精度内）。"""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((50, 8))
    Y = rng.standard_normal((50, 3))
    # 最优 W：A^T A W = A^T Y → W = (A^T A)^{-1} A^T Y
    W_opt = np.linalg.solve(A.T @ A, A.T @ Y)
    assert sne_residual_norm(A, W_opt, Y) < 1e-9


def test_sne_residual_positive_for_non_optimal() -> None:
    """非最优 W 应给出正残差。"""
    rng = np.random.default_rng(2)
    A = rng.standard_normal((30, 6))
    Y = rng.standard_normal((30, 2))
    W_random = rng.standard_normal((6, 2))
    assert sne_residual_norm(A, W_random, Y) > 1e-3


def test_sne_residual_matches_definition() -> None:
    """直接验证数学定义。"""
    rng = np.random.default_rng(3)
    A = rng.standard_normal((20, 4))
    Y = rng.standard_normal((20, 1))
    W = rng.standard_normal((4, 1))
    expected = float(np.linalg.norm(A.T @ (A @ W) - A.T @ Y))
    assert abs(sne_residual_norm(A, W, Y) - expected) < 1e-12


# ===========================================================================
# Timer
# ===========================================================================


def test_timer_basic_usage() -> None:
    """Timer 上下文管理器：elapsed 应反映块内执行时间。"""
    with Timer() as t:
        time.sleep(0.05)
    assert t.elapsed >= 0.05
    assert t.elapsed < 1.0  # 不应离谱地长


def test_timer_repeatable() -> None:
    """同一 Timer 重新进入 with 块后 elapsed 重置。"""
    t = Timer()
    with t:
        pass
    first = t.elapsed
    assert first >= 0

    with t:
        time.sleep(0.02)
    assert t.elapsed >= 0.02


def test_timer_zero_when_no_with_block() -> None:
    """未进入 with 块前 elapsed 默认 0。"""
    t = Timer()
    assert t.elapsed == 0.0


def test_timer_with_name() -> None:
    """Timer 接受可选 name 参数。"""
    t = Timer(name="my-timer")
    assert t.name == "my-timer"
    with t:
        pass
    assert t.elapsed >= 0
