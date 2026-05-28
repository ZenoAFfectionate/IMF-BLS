# -*- coding: utf-8 -*-
"""海量数据可扩展性测试 — 验证论文 Section 3.3 + Table 2 的核心声明。

这些测试是 IMF-BLS 算法**核心 motivation** 的工程证明：

  C1. 空间复杂度 O(p² + pc)，与累积样本数 N 无关
  C2. 每步时间复杂度 O(N* p²)，仅依赖新批次大小 N*
  C3. R 矩阵在所有阶段都保持上三角 + 对角全正
  C4. 不变量 R^T R = A^T A + λI 在 N 累积过程中精确成立

注：这些测试是 ``slow`` 标记，使用 ``pytest -m "not slow"`` 默认跳过；
   使用 ``pytest -m slow`` 显式运行。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bls_base import BLSConfig
from src.imf_bls import IMFBLS


pytestmark = pytest.mark.slow


def _gen_batch(n: int, d: int = 50, c: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = rng.integers(0, c, size=n)
    Y = -np.ones((n, c))
    Y[np.arange(n), y] = 1.0
    return X, Y


# ===========================================================================
# C1: 空间复杂度恒定（与 N 无关）
# ===========================================================================


def test_memory_constant_under_streaming_increment() -> None:
    """流式增量 10 个 batch，IMF-BLS 内部字节数应全程不变。"""
    cfg = BLSConfig(
        n_mapping_per_window=5, n_mapping_windows=5,
        n_enhancement=200, reg_lambda=1e-6, seed=0,
    )
    batch_size = 1000
    n_steps = 10

    m = None
    sizes = []
    for step in range(n_steps):
        X, Y = _gen_batch(batch_size, seed=step)
        if m is None:
            m = IMFBLS(config=cfg).fit_initial(X, Y)
        else:
            m.add_data(X, Y)
        sizes.append(m.memory_footprint_bytes())

    # 全部 step 的 footprint 应完全相同（R 与 V 形状不变）
    assert len(set(sizes)) == 1, \
        f"Memory footprint changed across steps: {sizes}"


def test_R_shape_constant_under_streaming() -> None:
    """流式增量 N 步，R 形状应恒定为 (p, p)。"""
    cfg = BLSConfig(
        n_mapping_per_window=5, n_mapping_windows=5,
        n_enhancement=300, reg_lambda=1e-6, seed=0,
    )
    p_expected = 5 * 5 + 300

    m = None
    for step in range(8):
        X, Y = _gen_batch(2000, seed=step)
        if m is None:
            m = IMFBLS(config=cfg).fit_initial(X, Y)
        else:
            m.add_data(X, Y)
        assert m.R.shape == (p_expected, p_expected), \
            f"step {step}: R.shape = {m.R.shape}, 期望 ({p_expected}, {p_expected})"


# ===========================================================================
# C2: 每步耗时仅依赖 N*（新批次大小），与累积 N 无关
# ===========================================================================


def test_per_step_time_does_not_grow_with_cumulative_N() -> None:
    """累积样本数从 1k 增长到 50k，单步耗时应在 2 倍内（即不随 N 增长）。"""
    import time

    cfg = BLSConfig(
        n_mapping_per_window=5, n_mapping_windows=5,
        n_enhancement=500, reg_lambda=1e-6, seed=0,
    )
    batch_size = 5000
    n_steps = 10  # 累积 50k

    m = None
    times = []
    for step in range(n_steps):
        X, Y = _gen_batch(batch_size, seed=step)
        t0 = time.perf_counter()
        if m is None:
            m = IMFBLS(config=cfg).fit_initial(X, Y)
        else:
            m.add_data(X, Y)
        times.append(time.perf_counter() - t0)

    # 跳过 step 0（fit_initial 包含初始化随机权重等开销）
    add_data_times = times[1:]
    fastest = min(add_data_times)
    slowest = max(add_data_times)
    # 最慢/最快比例应在 3 倍内（CPU 抖动允许；理论上应当几乎恒定）
    assert slowest / fastest < 3.0, \
        f"add_data 时间显著增长（最快 {fastest:.3f}s / 最慢 {slowest:.3f}s）→ 不符合 O(N*) 规律"


# ===========================================================================
# C3: R 上三角 + 对角全正（Theorem 3.1）
# ===========================================================================


def test_R_remains_upper_triangular_with_positive_diag_at_scale() -> None:
    """累积 30k 样本后 R 仍严格上三角 + 对角全正。"""
    cfg = BLSConfig(
        n_mapping_per_window=5, n_mapping_windows=5,
        n_enhancement=200, reg_lambda=1e-6, seed=0,
    )

    m = None
    for step in range(30):
        X, Y = _gen_batch(1000, seed=step)
        if m is None:
            m = IMFBLS(config=cfg).fit_initial(X, Y)
        else:
            m.add_data(X, Y)

    # 严格上三角
    below = np.tril(m.R, k=-1)
    assert np.max(np.abs(below)) < 1e-9, \
        f"R 下三角部分不为零，max |L| = {np.max(np.abs(below)):.2e}"
    # 对角全正
    assert (np.diag(m.R) > 0).all(), \
        "R 对角线包含非正值"


# ===========================================================================
# C4: R^T R = A^T A + λI 不变量在大 N 下精确成立
# ===========================================================================


def test_invariant_holds_after_30k_streaming() -> None:
    """累积 30k 样本后 R^T R 仍精确等于 (A^T A + λI)。"""
    cfg = BLSConfig(
        n_mapping_per_window=5, n_mapping_windows=5,
        n_enhancement=150, reg_lambda=1e-6, seed=0,
    )

    seen_X = None
    seen_Y = None
    m = None
    for step in range(30):
        X, Y = _gen_batch(1000, seed=step)
        if m is None:
            m = IMFBLS(config=cfg).fit_initial(X, Y)
        else:
            m.add_data(X, Y)
        seen_X = X if seen_X is None else np.vstack([seen_X, X])
        seen_Y = Y if seen_Y is None else np.vstack([seen_Y, Y])

    A = m.feature_layer.transform(seen_X)
    p = A.shape[1]
    err = np.max(np.abs(m.R.T @ m.R - A.T @ A - cfg.reg_lambda * np.eye(p)))
    # 30k 累积后浮点累积误差应仍 < 1e-7
    assert err < 1e-7, f"30k 后不变量误差 {err:.2e} 过大"


def test_V_AT_Y_invariant_at_scale() -> None:
    """累积 30k 样本后 V = A^T Y 仍精确成立。"""
    cfg = BLSConfig(
        n_mapping_per_window=5, n_mapping_windows=5,
        n_enhancement=150, reg_lambda=1e-6, seed=0,
    )

    seen_X = None
    seen_Y = None
    m = None
    for step in range(30):
        X, Y = _gen_batch(1000, seed=step)
        if m is None:
            m = IMFBLS(config=cfg).fit_initial(X, Y)
        else:
            m.add_data(X, Y)
        seen_X = X if seen_X is None else np.vstack([seen_X, X])
        seen_Y = Y if seen_Y is None else np.vstack([seen_Y, Y])

    A = m.feature_layer.transform(seen_X)
    err = np.max(np.abs(m.V - A.T @ seen_Y))
    assert err < 1e-7, f"V invariant 误差 {err:.2e} 过大"
