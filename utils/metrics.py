# -*- coding: utf-8 -*-
"""评估指标。"""

from __future__ import annotations

import numpy as np


def classification_accuracy(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """多分类准确率（取 argmax 最大值的列作为预测类别）。"""
    return float((np.argmax(Y_pred, axis=1) == np.argmax(Y_true, axis=1)).mean())


def regression_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差（自动 ravel 到 1D 比较）。"""
    a = np.asarray(y_true, dtype=np.float64).ravel()
    b = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((a - b) ** 2)))


def sne_residual_norm(A: np.ndarray, W: np.ndarray, Y: np.ndarray) -> float:
    """半正规方程残差 ``||A^T A W - A^T Y||_2``（论文 Section 3.2）。

    用于直接比较 IMF-BLS 与求逆方法的数值稳定性。
    """
    return float(np.linalg.norm(A.T @ (A @ W) - A.T @ Y))
