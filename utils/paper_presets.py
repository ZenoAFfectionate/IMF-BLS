# -*- coding: utf-8 -*-
"""论文 Table 3 / Table 4 的超参数预设。

为各个公开数据集提供论文中使用的精确超参数，保证复现实验时配置一致。

使用::

    from utils.paper_presets import get_preset
    preset = get_preset("mnist")
    cfg = BLSConfig(**preset.bls_kwargs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DatasetPreset:
    """单个数据集的论文实验配置。

    Attributes:
        name: 数据集名（与 loader / paper 完全一致）。
        task: ``classification`` 或 ``regression``。
        bls_kwargs: 喂给 :class:`BLSConfig` 的字典。
        equal_scale_n_batches: 论文 Section 4.1 等量数据流 batch 数（Table 3 末列）。
        uncertain_scale_n_batches: 论文 Section 4.2 不定数据流 batch 数选项。
        uncertain_scale_repeats: 不定数据流重复次数（论文为 10）。
        n_train: 论文 Table 3/4 训练集大小（仅供文档记录）。
        n_test: 论文 Table 3/4 测试集大小。
        notes: 备注（譬如数据来源链接）。
    """

    name: str
    task: str
    bls_kwargs: Dict
    equal_scale_n_batches: int = 6
    uncertain_scale_n_batches: List[int] = field(
        default_factory=lambda: [5, 10, 15, 20, 25]
    )
    uncertain_scale_repeats: int = 10
    n_train: Optional[int] = None
    n_test: Optional[int] = None
    notes: str = ""


# ---------------------------------------------------------------------------
# 论文 Table 3：分类
# ---------------------------------------------------------------------------
# 论文 N1=10, N2=10, N3=enhancement_nodes，配置中 (1, 5000) 等代表
# 一组 mapping 窗口 + 一个 enhancement 窗口大小


_PRESETS: Dict[str, DatasetPreset] = {
    # ============== 论文主图像数据集 ==============
    "mnist": DatasetPreset(
        name="mnist",
        task="classification",
        bls_kwargs=dict(
            n_mapping_per_window=10,
            n_mapping_windows=10,
            n_enhancement=5000,
            reg_lambda=1e-6,
            activation="tanh",
            seed=0,
        ),
        equal_scale_n_batches=6,
        n_train=60000,
        n_test=10000,
        notes="论文 Table 3 配置 (1, 5000), λ=10^-6, batch=6",
    ),
    "fashion_mnist": DatasetPreset(
        name="fashion_mnist",
        task="classification",
        bls_kwargs=dict(
            n_mapping_per_window=10,
            n_mapping_windows=10,
            n_enhancement=5000,
            reg_lambda=1e-6,
            activation="tanh",
            seed=0,
        ),
        equal_scale_n_batches=6,
        n_train=60000,
        n_test=10000,
        notes="论文 Table 3",
    ),
    "norb": DatasetPreset(
        name="norb",
        task="classification",
        bls_kwargs=dict(
            n_mapping_per_window=10,
            n_mapping_windows=10,
            n_enhancement=3000,
            reg_lambda=1e-3,
            activation="tanh",
            seed=0,
        ),
        equal_scale_n_batches=6,
        n_train=24300,
        n_test=24300,
        notes="论文 Table 3 (N3=3000, λ=10^-3)",
    ),
    "emnist": DatasetPreset(
        name="emnist",
        task="classification",
        bls_kwargs=dict(
            n_mapping_per_window=10,
            n_mapping_windows=10,
            n_enhancement=5000,
            reg_lambda=1e-6,
            activation="tanh",
            seed=0,
        ),
        equal_scale_n_batches=6,
        n_train=240000,
        n_test=40000,
        notes="论文 Table 3",
    ),
    # ============== 论文 UCI 分类数据集 ==============
    "shuttle": DatasetPreset(
        name="shuttle",
        task="classification",
        bls_kwargs=dict(
            n_mapping_per_window=10,
            n_mapping_windows=10,
            n_enhancement=3000,
            reg_lambda=1e-6,
            activation="tanh",
            seed=0,
        ),
        equal_scale_n_batches=5,
        n_train=43500,
        n_test=14500,
    ),
    "letter": DatasetPreset(
        name="letter",
        task="classification",
        bls_kwargs=dict(
            n_mapping_per_window=10,
            n_mapping_windows=10,
            n_enhancement=3000,
            reg_lambda=1e-6,
            activation="tanh",
            seed=0,
        ),
        equal_scale_n_batches=5,
        n_train=16000,
        n_test=4000,
    ),
    "pendigits": DatasetPreset(
        name="pendigits",
        task="classification",
        bls_kwargs=dict(
            n_mapping_per_window=10,
            n_mapping_windows=10,
            n_enhancement=1000,
            reg_lambda=1e-6,
            activation="tanh",
            seed=0,
        ),
        equal_scale_n_batches=5,
        n_train=8000,
        n_test=2992,
    ),
    "led": DatasetPreset(
        name="led",
        task="classification",
        bls_kwargs=dict(
            n_mapping_per_window=10,
            n_mapping_windows=10,
            n_enhancement=5000,
            reg_lambda=1e-6,
            activation="tanh",
            seed=0,
        ),
        equal_scale_n_batches=5,
        n_train=160000,
        n_test=40000,
    ),
    "waveform": DatasetPreset(
        name="waveform",
        task="classification",
        bls_kwargs=dict(
            n_mapping_per_window=10,
            n_mapping_windows=10,
            n_enhancement=600,
            reg_lambda=1e-6,
            activation="tanh",
            seed=0,
        ),
        equal_scale_n_batches=5,
        n_train=4200,
        n_test=800,
    ),
    # ============== 论文 Table 4：回归 ==============
    "abalone": DatasetPreset(
        name="abalone",
        task="regression",
        bls_kwargs=dict(
            n_mapping_per_window=10,
            n_mapping_windows=10,
            n_enhancement=600,
            reg_lambda=1e-6,
            activation="tanh",
            seed=0,
        ),
        equal_scale_n_batches=5,
        n_train=2784,
        n_test=1393,
    ),
    "bodyfat": DatasetPreset(
        name="bodyfat",
        task="regression",
        bls_kwargs=dict(
            n_mapping_per_window=10,
            n_mapping_windows=10,
            n_enhancement=200,
            reg_lambda=1e-6,
            activation="tanh",
            seed=0,
        ),
        equal_scale_n_batches=4,
        n_train=168,
        n_test=84,
    ),
    "weather_izmir": DatasetPreset(
        name="weather_izmir",
        task="regression",
        bls_kwargs=dict(
            n_mapping_per_window=10,
            n_mapping_windows=10,
            n_enhancement=600,
            reg_lambda=1e-6,
            activation="tanh",
            seed=0,
        ),
        equal_scale_n_batches=5,
        n_train=974,
        n_test=487,
    ),
    "energy_efficiency": DatasetPreset(
        name="energy_efficiency",
        task="regression",
        bls_kwargs=dict(
            n_mapping_per_window=10,
            n_mapping_windows=10,
            n_enhancement=600,
            reg_lambda=1e-6,
            activation="tanh",
            seed=0,
        ),
        equal_scale_n_batches=5,
        n_train=614,
        n_test=154,
    ),
    "appliances_energy": DatasetPreset(
        name="appliances_energy",
        task="regression",
        bls_kwargs=dict(
            n_mapping_per_window=10,
            n_mapping_windows=10,
            n_enhancement=600,
            reg_lambda=1e-6,
            activation="tanh",
            seed=0,
        ),
        equal_scale_n_batches=5,
        n_train=15788,
        n_test=3947,
    ),
    # ============== smoke 测试用合成预设（轻量，无外部依赖） ==============
    "synthetic_classification": DatasetPreset(
        name="synthetic_classification",
        task="classification",
        bls_kwargs=dict(
            n_mapping_per_window=5,
            n_mapping_windows=5,
            n_enhancement=100,
            reg_lambda=1e-6,
            activation="tanh",
            seed=0,
        ),
        equal_scale_n_batches=4,
        uncertain_scale_n_batches=[3, 5],
        uncertain_scale_repeats=2,
        n_train=1000,
        n_test=200,
        notes="用于 smoke 测试，无外部依赖",
    ),
    "synthetic_regression": DatasetPreset(
        name="synthetic_regression",
        task="regression",
        bls_kwargs=dict(
            n_mapping_per_window=5,
            n_mapping_windows=5,
            n_enhancement=100,
            reg_lambda=1e-6,
            activation="tanh",
            seed=0,
        ),
        equal_scale_n_batches=3,
        n_train=600,
        n_test=200,
        notes="用于 smoke 测试，无外部依赖",
    ),
}


def get_preset(name: str) -> DatasetPreset:
    """获取数据集的论文超参数预设。"""
    name = name.lower().replace("-", "_")
    if name not in _PRESETS:
        raise KeyError(
            f"未知数据集 {name}；支持: {sorted(_PRESETS)}"
        )
    return _PRESETS[name]


def list_presets() -> List[str]:
    return sorted(_PRESETS)


__all__ = ["DatasetPreset", "get_preset", "list_presets"]
