# -*- coding: utf-8 -*-
"""IMF-BLS utilities：纯数值原语与数据/日志辅助工具。

本包按"职责单一"原则组织：

    linalg.py        — 替换法、增量 QR、TSQR、Cholesky
    feature_layer.py — BLS 特征层（Eq. 1）+ 节点增量
    data.py          — 数据加载 / batch 切分
    metrics.py       — accuracy / RMSE / SNE 残差
    timing.py        — 高精度计时器
    logger.py        — 统一日志 + 结构化实验记录器
"""

from .linalg import (
    forward_substitution,
    backward_substitution,
    solve_sne,
    qr_R,
    incremental_qr_update,
    tsqr_R,
    cholesky_lower,
)
from .feature_layer import FeatureLayer, standardize_minmax
from .data import (
    one_hot_encode,
    make_synthetic_classification,
    make_synthetic_regression,
    load_classification_dataset,
    load_regression_dataset,
    split_into_batches,
    split_random_batches,
)
from .metrics import classification_accuracy, regression_rmse, sne_residual_norm
from .timing import Timer
from .logger import (
    get_logger,
    ExperimentRecorder,
    log_array_stats,
    ColorFormatter,
)
from .paper_presets import DatasetPreset, get_preset, list_presets
from .uci_loader import load_uci_classification, load_uci_regression

__all__ = [
    # linalg
    "forward_substitution", "backward_substitution", "solve_sne",
    "qr_R", "incremental_qr_update", "tsqr_R", "cholesky_lower",
    # feature
    "FeatureLayer", "standardize_minmax",
    # data
    "one_hot_encode", "make_synthetic_classification", "make_synthetic_regression",
    "load_classification_dataset", "load_regression_dataset",
    "split_into_batches", "split_random_batches",
    # metrics
    "classification_accuracy", "regression_rmse", "sne_residual_norm",
    # timing
    "Timer",
    # logger
    "get_logger", "ExperimentRecorder", "log_array_stats", "ColorFormatter",
    # paper presets & uci
    "DatasetPreset", "get_preset", "list_presets",
    "load_uci_classification", "load_uci_regression",
]
