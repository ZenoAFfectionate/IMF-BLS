# -*- coding: utf-8 -*-
"""IMF-BLS 端到端正确性测试。

⭐ 论文核心算法的工程化等价性验证 ⭐

  * 初始训练等价于 NonIncrementalBLS （论文 Eq. 4 ↔ Eq. 5 解一致）
  * 加数据 N 步 ↔ 在全部数据上联合训练（Theorem 3.1 + Eq. 9）
  * 加节点 ↔ 扩展特征层后联合训练（Section 2.3, Eq. 14）
  * 数据 + 节点交替增量任意顺序（论文 Section 2.3 灵活顺序声明）
  * 病态矩阵下残差 ≤ RIBLS （论文 Section 3.2 数值稳定性）
  * 内存模块大小恒定 (与 N 无关，论文 Table 2)
  * use_tsqr=True 与 use_tsqr=False 结果一致
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bls_base import BLSConfig, NonIncrementalBLS
from src.imf_bls import IMFBLS
from utils.data import (
    make_synthetic_classification,
    one_hot_encode,
    split_into_batches,
)
from utils.feature_layer import standardize_minmax
from utils.metrics import sne_residual_norm


# ---------------------------------------------------------------------------
# Helper：构造对照实验中"扩展特征层后联合训练"的参考权重
# ---------------------------------------------------------------------------


def _ridge_solve(A: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """直接求解 (A^T A + λI) W = A^T Y，作为联合训练的参考解。"""
    p = A.shape[1]
    M = A.T @ A + lam * np.eye(p)
    return np.linalg.solve(M, A.T @ Y)


def _make_cfg(n_enh: int = 200, seed: int = 11) -> BLSConfig:
    return BLSConfig(
        n_mapping_per_window=8, n_mapping_windows=5,
        n_enhancement=n_enh, activation="tanh",
        reg_lambda=1e-4, seed=seed,
    )


# ===========================================================================
# 1. 初始训练等价性
# ===========================================================================


def test_initial_train_matches_non_incremental(small_classification) -> None:
    """初始训练后 IMF-BLS 与 NonIncrementalBLS 权重严格相等。"""
    X_tr, Y_tr, X_te, Y_te = small_classification
    cfg_a, cfg_b = _make_cfg(seed=11), _make_cfg(seed=11)

    invf = IMFBLS(config=cfg_a).fit_initial(X_tr, Y_tr)
    full = NonIncrementalBLS(config=cfg_b).fit_initial(X_tr, Y_tr)

    assert np.allclose(invf.W, full.W, atol=1e-8)
    assert abs(invf.score_classification(X_te, Y_te) -
               full.score_classification(X_te, Y_te)) < 1e-9


def test_R_satisfies_invariant(small_classification) -> None:
    """初始训练后 R^T R 应等于 A^T A + λI（关键不变量）。"""
    X_tr, Y_tr, _, _ = small_classification
    cfg = _make_cfg(seed=22)
    invf = IMFBLS(config=cfg).fit_initial(X_tr, Y_tr)

    A = invf.feature_layer.transform(X_tr)
    p = A.shape[1]
    expected = A.T @ A + cfg.reg_lambda * np.eye(p)
    assert np.allclose(invf.R.T @ invf.R, expected, atol=1e-6)


# ===========================================================================
# 2. 加数据等价性 (Theorem 3.1 工程证明)
# ===========================================================================


def test_add_data_5_batches_equiv_joint(small_classification) -> None:
    X_tr, Y_tr, X_te, Y_te = small_classification
    cfg_inc, cfg_full = _make_cfg(seed=33), _make_cfg(seed=33)
    batches = split_into_batches(X_tr, Y_tr, n_batches=5, shuffle=False)

    invf = IMFBLS(config=cfg_inc).fit_initial(*batches[0])
    for X_b, Y_b in batches[1:]:
        invf.add_data(X_b, Y_b)

    full = NonIncrementalBLS(config=cfg_full).fit_initial(X_tr, Y_tr)

    # ⭐ 关键等价性：W 应数值相等
    assert np.allclose(invf.W, full.W, atol=1e-7), (
        f"加数据后差异 {np.max(np.abs(invf.W - full.W))}"
    )
    # 测试集准确率必须严格相同
    assert abs(invf.score_classification(X_te, Y_te) -
               full.score_classification(X_te, Y_te)) < 1e-9


def test_add_data_irregular_batches(small_classification) -> None:
    """非等量 batch 同样保持等价性。"""
    X_tr, Y_tr, _, _ = small_classification
    cfg_inc, cfg_full = _make_cfg(seed=44), _make_cfg(seed=44)

    sizes = [80, 30, 150, 100, 240]  # 总和 = 600
    indices = np.cumsum(sizes)[:-1]
    parts_X = np.split(X_tr, indices)
    parts_Y = np.split(Y_tr, indices)

    invf = IMFBLS(config=cfg_inc).fit_initial(parts_X[0], parts_Y[0])
    for X_b, Y_b in zip(parts_X[1:], parts_Y[1:]):
        invf.add_data(X_b, Y_b)

    full = NonIncrementalBLS(config=cfg_full).fit_initial(X_tr, Y_tr)
    assert np.allclose(invf.W, full.W, atol=1e-7)


def test_add_data_invariant_after_each_step(small_classification) -> None:
    """每一步 add_data 后，R^T R = A_累计^T A_累计 + λI 应严格成立。"""
    X_tr, Y_tr, _, _ = small_classification
    cfg = _make_cfg(seed=55)
    batches = split_into_batches(X_tr, Y_tr, n_batches=4, shuffle=False)

    invf = IMFBLS(config=cfg).fit_initial(*batches[0])
    X_seen, _ = batches[0]

    for X_b, Y_b in batches[1:]:
        invf.add_data(X_b, Y_b)
        X_seen = np.vstack([X_seen, X_b])
        A_seen = invf.feature_layer.transform(X_seen)
        p = A_seen.shape[1]
        expected = A_seen.T @ A_seen + cfg.reg_lambda * np.eye(p)
        assert np.allclose(invf.R.T @ invf.R, expected, atol=1e-5)


# ===========================================================================
# 3. 加节点等价性 (Section 2.3, Eq. 14)
# ===========================================================================


def test_add_nodes_equiv_extended_joint(small_classification) -> None:
    """新增 enhancement 节点后等价于在扩展特征层上联合训练。"""
    X_tr, Y_tr, _, _ = small_classification
    cfg_a, cfg_b = _make_cfg(n_enh=100, seed=77), _make_cfg(n_enh=100, seed=77)

    invf = IMFBLS(config=cfg_a).fit_initial(X_tr, Y_tr)
    invf.add_nodes(X_tr, Y_tr, n_new=50)

    # 参考：用相同 RNG 扩展特征层后联合求解
    full = NonIncrementalBLS(config=cfg_b).fit_initial(X_tr, Y_tr)
    full.feature_layer.add_enhancement_window(50)
    A_ext = full.feature_layer.transform(X_tr)
    full.W = _ridge_solve(A_ext, Y_tr, cfg_b.reg_lambda)

    assert invf.W.shape == full.W.shape
    assert np.allclose(invf.W, full.W, atol=1e-6), (
        f"加节点后差异 {np.max(np.abs(invf.W - full.W))}"
    )


def test_add_nodes_multiple_times(small_classification) -> None:
    """连续多次节点增量都应保持等价。"""
    X_tr, Y_tr, _, _ = small_classification
    cfg_a, cfg_b = _make_cfg(n_enh=80, seed=88), _make_cfg(n_enh=80, seed=88)

    invf = IMFBLS(config=cfg_a).fit_initial(X_tr, Y_tr)
    for n_new in [30, 20, 40]:
        invf.add_nodes(X_tr, Y_tr, n_new=n_new)

    full = NonIncrementalBLS(config=cfg_b).fit_initial(X_tr, Y_tr)
    for n_new in [30, 20, 40]:
        full.feature_layer.add_enhancement_window(n_new)
    A_ext = full.feature_layer.transform(X_tr)
    full.W = _ridge_solve(A_ext, Y_tr, cfg_b.reg_lambda)

    assert np.allclose(invf.W, full.W, atol=1e-5)


# ===========================================================================
# 4. 数据 + 节点交替增量，任意顺序均等价 (Section 2.3 灵活顺序)
# ===========================================================================


def test_data_then_nodes_then_data(medium_classification) -> None:
    X_tr, Y_tr, _, _ = medium_classification
    cfg_a, cfg_b = _make_cfg(n_enh=100, seed=101), _make_cfg(n_enh=100, seed=101)

    # ⭐ 数据 → 节点 → 数据 的顺序
    invf = IMFBLS(config=cfg_a).fit_initial(X_tr[:600], Y_tr[:600])
    invf.add_data(X_tr[600:1000], Y_tr[600:1000])
    invf.add_nodes(X_tr[:1000], Y_tr[:1000], n_new=40)
    invf.add_data(X_tr[1000:], Y_tr[1000:])

    # 联合训练参考
    full = NonIncrementalBLS(config=cfg_b).fit_initial(X_tr[:600], Y_tr[:600])
    full.feature_layer.add_enhancement_window(40)
    A_ext = full.feature_layer.transform(X_tr)
    full.W = _ridge_solve(A_ext, Y_tr, cfg_b.reg_lambda)

    assert np.allclose(invf.W, full.W, atol=1e-5)


def test_nodes_then_data_then_nodes(medium_classification) -> None:
    """另一种顺序：先加节点，再加数据，再加节点。结果同样等价。"""
    X_tr, Y_tr, _, _ = medium_classification
    cfg_a, cfg_b = _make_cfg(n_enh=80, seed=202), _make_cfg(n_enh=80, seed=202)

    invf = IMFBLS(config=cfg_a).fit_initial(X_tr[:500], Y_tr[:500])
    invf.add_nodes(X_tr[:500], Y_tr[:500], n_new=30)
    invf.add_data(X_tr[500:1100], Y_tr[500:1100])
    invf.add_nodes(X_tr[:1100], Y_tr[:1100], n_new=20)
    invf.add_data(X_tr[1100:], Y_tr[1100:])

    full = NonIncrementalBLS(config=cfg_b).fit_initial(X_tr[:500], Y_tr[:500])
    full.feature_layer.add_enhancement_window(30)
    full.feature_layer.add_enhancement_window(20)
    A_ext = full.feature_layer.transform(X_tr)
    full.W = _ridge_solve(A_ext, Y_tr, cfg_b.reg_lambda)

    assert np.allclose(invf.W, full.W, atol=1e-5)


# ===========================================================================
# 5. 数值稳定性：病态特征矩阵下 IMF-BLS 残差不大于 RIBLS (Section 3.2)
# ===========================================================================


def test_imf_residual_not_worse_than_ridge_inverse() -> None:
    """构造病态 A 验证论文 Section 3.2.4 的结论：
    IMF-BLS 通过替换法获得的解残差不应大于 RIBLS（直接求逆）。"""
    from src.baselines import RIBLS

    rng = np.random.default_rng(123)
    n = 400
    X = rng.standard_normal((n, 6))
    X[:, 5] = X[:, 0] + 1e-7 * rng.standard_normal(n)  # 接近共线
    Y = rng.standard_normal((n, 1))

    cfg = BLSConfig(
        n_mapping_per_window=4, n_mapping_windows=3,
        n_enhancement=50, reg_lambda=1e-10, seed=0,
    )
    invf = IMFBLS(config=cfg.copy()).fit_initial(X, Y)
    rib = RIBLS(config=cfg.copy()).fit_initial(X, Y)

    A = invf.feature_layer.transform(X)
    res_invf = sne_residual_norm(A, invf.W, Y)
    res_rib = sne_residual_norm(A, rib.W, Y)

    # IMF-BLS 残差应不显著大于 RIBLS（理论上更小或同量级）
    assert res_invf <= res_rib * 5 + 1e-9, (
        f"IMF-BLS 残差 {res_invf:.3e} > RIBLS {res_rib:.3e} * 5"
    )


# ===========================================================================
# 6. 内存恒定性 (Table 2)
# ===========================================================================


def test_memory_constant_with_data_growth(small_classification) -> None:
    """加入 N 个 batch 数据后，记忆模块大小不应随 N 增长。"""
    X_tr, Y_tr, _, _ = small_classification
    cfg = _make_cfg(seed=303)
    batches = split_into_batches(X_tr, Y_tr, n_batches=10, shuffle=False)

    invf = IMFBLS(config=cfg).fit_initial(*batches[0])
    init_bytes = invf.memory_footprint_bytes()

    for X_b, Y_b in batches[1:]:
        invf.add_data(X_b, Y_b)
        # ⭐ 内存恒定（数据增量过程中 R, V, W 形状不变）
        assert invf.memory_footprint_bytes() == init_bytes


# ===========================================================================
# 7. use_tsqr 选项不影响数值结果
# ===========================================================================


def test_tsqr_option_produces_same_weights(small_classification) -> None:
    """use_tsqr=True 与 False 应给出相同权重（最多到浮点误差）。"""
    X_tr, Y_tr, _, _ = small_classification
    cfg_a, cfg_b = _make_cfg(seed=404), _make_cfg(seed=404)
    batches = split_into_batches(X_tr, Y_tr, n_batches=4, shuffle=False)

    a = IMFBLS(config=cfg_a, use_tsqr=False).fit_initial(*batches[0])
    b = IMFBLS(config=cfg_b, use_tsqr=True, tsqr_blocks=4).fit_initial(*batches[0])
    for X_b, Y_b in batches[1:]:
        a.add_data(X_b, Y_b)
        b.add_data(X_b, Y_b)

    assert np.allclose(a.W, b.W, atol=1e-7)


# ===========================================================================
# 8. 错误处理
# ===========================================================================


def test_predict_before_fit_raises() -> None:
    invf = IMFBLS()
    with pytest.raises(RuntimeError):
        invf.predict(np.zeros((1, 5)))


def test_add_data_before_fit_raises() -> None:
    invf = IMFBLS()
    with pytest.raises(RuntimeError):
        invf.add_data(np.zeros((1, 5)), np.zeros((1, 1)))


def test_add_nodes_target_dim_mismatch_raises(small_classification) -> None:
    X_tr, Y_tr, _, _ = small_classification
    invf = IMFBLS(config=_make_cfg()).fit_initial(X_tr, Y_tr)
    # 错误的标签维度
    with pytest.raises(ValueError):
        invf.add_data(X_tr[:5], np.zeros((5, Y_tr.shape[1] + 1)))


def test_add_nodes_x_y_shape_mismatch_raises(small_classification) -> None:
    """X_all 与 Y_all 行数不一致时应抛 ValueError。"""
    X_tr, Y_tr, _, _ = small_classification
    invf = IMFBLS(config=_make_cfg()).fit_initial(X_tr, Y_tr)
    with pytest.raises(ValueError):
        invf.add_nodes(X_tr[:50], Y_tr[:30], n_new=10)


# ===========================================================================
# 9. rank-deficient 路径 (l_0 ≤ p, Remark 2.2)
# ===========================================================================


def test_rank_deficient_initial_works() -> None:
    """l_0 ≤ p 时仍能正确 fit_initial（依赖 sqrt(λ)I 拼接的统一路径）。"""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 5))         # 仅 30 个样本
    y = rng.integers(0, 3, size=30)
    Y = -np.ones((30, 3))
    Y[np.arange(30), y] = 1.0

    cfg = BLSConfig(
        n_mapping_per_window=10, n_mapping_windows=10,  # mapping_dim = 100
        n_enhancement=50,                                 # 总 p = 150 > l_0 = 30
        reg_lambda=1e-3, seed=0,
    )
    model = IMFBLS(config=cfg).fit_initial(X, Y)

    # 应能正常预测
    pred = model.predict(X)
    assert pred.shape == (30, 3)
    # 不变量 R^T R = A^T A + λI 在此路径上仍成立
    A = model.feature_layer.transform(X)
    p = A.shape[1]
    err = np.max(np.abs(model.R.T @ model.R - A.T @ A - cfg.reg_lambda * np.eye(p)))
    assert err < 1e-8


def test_rank_deficient_then_add_data_recovers() -> None:
    """先用少量数据初始化（rank-deficient），再加入更多数据后 R 应正确累积。"""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 8))
    y = rng.integers(0, 3, size=300)
    Y = -np.ones((300, 3))
    Y[np.arange(300), y] = 1.0

    cfg = BLSConfig(
        n_mapping_per_window=8, n_mapping_windows=6,
        n_enhancement=50, reg_lambda=1e-3, seed=0,
    )
    # 先用 30 个样本（rank-deficient）→ 再加入剩余
    model = IMFBLS(config=cfg).fit_initial(X[:30], Y[:30])
    model.add_data(X[30:], Y[30:])

    # 联合训练对照
    full = NonIncrementalBLS(config=cfg.copy()).fit_initial(X, Y)
    assert np.allclose(model.W, full.W, atol=1e-6)


# ===========================================================================
# 10. 不同激活函数下 fit/predict 都正常
# ===========================================================================


@pytest.mark.parametrize("activation", ["tanh", "sigmoid", "relu"])
def test_different_activations(activation, small_classification) -> None:
    X_tr, Y_tr, X_te, Y_te = small_classification
    cfg = BLSConfig(
        n_mapping_per_window=6, n_mapping_windows=4,
        n_enhancement=80, activation=activation,
        reg_lambda=1e-4, seed=0,
    )
    model = IMFBLS(config=cfg).fit_initial(X_tr, Y_tr)
    acc = model.score_classification(X_te, Y_te)
    # 至少应优于随机猜测 (1/3 = 0.333 for 3-class)
    assert acc > 0.4, f"activation={activation} 准确率 {acc:.4f} 过低"


# ===========================================================================
# 11. 确定性：相同 seed → 相同输出
# ===========================================================================


def test_deterministic_fit_under_same_seed(small_classification) -> None:
    X_tr, Y_tr, _, _ = small_classification
    cfg_a, cfg_b = _make_cfg(seed=12345), _make_cfg(seed=12345)
    a = IMFBLS(config=cfg_a).fit_initial(X_tr, Y_tr)
    b = IMFBLS(config=cfg_b).fit_initial(X_tr, Y_tr)
    assert np.allclose(a.W, b.W, atol=1e-12)
    assert np.allclose(a.R, b.R, atol=1e-12)


def test_predict_deterministic_on_repeated_input(small_classification) -> None:
    """同一 X 多次 predict 应返回完全相同结果（无随机性）。"""
    X_tr, Y_tr, X_te, _ = small_classification
    model = IMFBLS(config=_make_cfg()).fit_initial(X_tr, Y_tr)
    pred1 = model.predict(X_te)
    pred2 = model.predict(X_te)
    assert np.array_equal(pred1, pred2)


# ===========================================================================
# 12. memory_module API
# ===========================================================================


def test_memory_module_returns_R_V(small_classification) -> None:
    X_tr, Y_tr, _, _ = small_classification
    model = IMFBLS(config=_make_cfg()).fit_initial(X_tr, Y_tr)
    R, V = model.memory_module()
    assert R is model.R
    assert V is model.V
    assert R.shape == (model.width, model.width)
    assert V.shape == (model.width, Y_tr.shape[1])


def test_memory_module_R_is_upper_triangular(small_classification) -> None:
    """R 始终是上三角矩阵（论文 Eq. 4 的关键结构）。"""
    X_tr, Y_tr, _, _ = small_classification
    model = IMFBLS(config=_make_cfg()).fit_initial(X_tr, Y_tr)
    R = model.R
    # 下三角部分应为 0
    below = np.tril(R, k=-1)
    assert np.max(np.abs(below)) < 1e-12


def test_memory_module_R_diag_positive_after_increments(small_classification) -> None:
    """连续 add_data 后 R 对角线仍全正（保证 R-factor 唯一）。"""
    X_tr, Y_tr, _, _ = small_classification
    batches = split_into_batches(X_tr, Y_tr, n_batches=4, shuffle=False)
    model = IMFBLS(config=_make_cfg(seed=7)).fit_initial(*batches[0])
    for X_b, Y_b in batches[1:]:
        model.add_data(X_b, Y_b)
    assert (np.diag(model.R) > 0).all()


def test_memory_grows_only_on_add_nodes(small_classification) -> None:
    """add_data 不应改变 R 的形状；add_nodes 必须扩大形状。"""
    X_tr, Y_tr, _, _ = small_classification
    model = IMFBLS(config=_make_cfg(seed=8)).fit_initial(X_tr, Y_tr)
    p_init = model.width

    # add_data 不改变形状
    model.add_data(X_tr[:50], Y_tr[:50])
    assert model.width == p_init

    # add_nodes 增加 30 个 enhancement 节点 → R 扩大 30
    model.add_nodes(X_tr, Y_tr, n_new=30)
    assert model.width == p_init + 30


# ===========================================================================
# 13. reg_lambda = 0 极限情况（无正则化）
# ===========================================================================


def test_zero_regularization_classification(small_classification) -> None:
    """λ=0 时 IMF-BLS 应仍能工作，与无正则化的最小二乘解一致。"""
    X_tr, Y_tr, _, _ = small_classification
    cfg = BLSConfig(
        n_mapping_per_window=6, n_mapping_windows=4,
        n_enhancement=50,        # 让 p = 24 + 50 = 74 < n=600，A 满列秩
        reg_lambda=0.0, seed=11,
    )
    model = IMFBLS(config=cfg).fit_initial(X_tr, Y_tr)
    A = model.feature_layer.transform(X_tr)
    # 不变量退化为 R^T R = A^T A
    assert np.allclose(model.R.T @ model.R, A.T @ A, atol=1e-7)

    # 与 numpy.linalg.lstsq 比较
    W_lstsq, *_ = np.linalg.lstsq(A, Y_tr, rcond=None)
    assert np.allclose(model.W, W_lstsq, atol=1e-6)


# ===========================================================================
# 14. 多次重复 add_data：与单次大 batch 等价
# ===========================================================================


def test_many_small_batches_equiv_one_big_batch(small_classification) -> None:
    """加入 N 个小 batch 后 W 应等于一次性加入大 batch 的结果。"""
    X_tr, Y_tr, _, _ = small_classification
    cfg = _make_cfg(seed=99)

    # 分 20 个小 batch
    invf_small = IMFBLS(config=cfg.copy()).fit_initial(X_tr[:50], Y_tr[:50])
    for i in range(50, 600, 50):
        invf_small.add_data(X_tr[i : i + 50], Y_tr[i : i + 50])

    # 直接联合训练
    full = NonIncrementalBLS(config=cfg.copy()).fit_initial(X_tr, Y_tr)

    assert np.allclose(invf_small.W, full.W, atol=1e-6)


# ===========================================================================
# 15. width 属性
# ===========================================================================


def test_width_zero_before_fit() -> None:
    model = IMFBLS()
    assert model.width == 0


def test_width_matches_R_shape(small_classification) -> None:
    X_tr, Y_tr, _, _ = small_classification
    model = IMFBLS(config=_make_cfg()).fit_initial(X_tr, Y_tr)
    assert model.width == model.R.shape[0] == model.R.shape[1]


# ===========================================================================
# 16. memory_footprint_bytes API
# ===========================================================================


def test_memory_footprint_includes_R_V_W(small_classification) -> None:
    """memory_footprint_bytes 应等于 R, V, W 的字节数之和。"""
    X_tr, Y_tr, _, _ = small_classification
    model = IMFBLS(config=_make_cfg()).fit_initial(X_tr, Y_tr)
    expected = model.R.nbytes + model.V.nbytes + model.W.nbytes
    assert model.memory_footprint_bytes() == expected


# ===========================================================================
# 17. 论文级深度等价性 — 多次连续 add_nodes
# ===========================================================================


def test_three_consecutive_add_nodes_equiv_joint_training(small_classification) -> None:
    """连续 3 次 add_nodes（不同 n_new）后 W 应等于一次性扩展同样多节点的联合训练。"""
    X_tr, Y_tr, _, _ = small_classification
    cfg_a, cfg_b = _make_cfg(seed=42), _make_cfg(seed=42)

    model = IMFBLS(config=cfg_a).fit_initial(X_tr, Y_tr)
    model.add_nodes(X_tr, Y_tr, n_new=10)
    model.add_nodes(X_tr, Y_tr, n_new=15)
    model.add_nodes(X_tr, Y_tr, n_new=8)

    full = NonIncrementalBLS(config=cfg_b).fit_initial(X_tr, Y_tr)
    full.feature_layer.add_enhancement_window(10)
    full.feature_layer.add_enhancement_window(15)
    full.feature_layer.add_enhancement_window(8)
    A_ext = full.feature_layer.transform(X_tr)
    W_ref = _ridge_solve(A_ext, Y_tr, cfg_b.reg_lambda)

    assert np.allclose(model.W, W_ref, atol=1e-6)


def test_data_node_data_node_data_chain(small_classification) -> None:
    """data → node → data → node → data 5 步完整生命周期等价性。"""
    X_tr, Y_tr, _, _ = small_classification
    batches = split_into_batches(X_tr, Y_tr, n_batches=4, shuffle=False)
    cfg_a, cfg_b = _make_cfg(seed=99), _make_cfg(seed=99)

    model = IMFBLS(config=cfg_a).fit_initial(*batches[0])
    seen_X = batches[0][0].copy()
    seen_Y = batches[0][1].copy()

    model.add_data(*batches[1])
    seen_X = np.vstack([seen_X, batches[1][0]])
    seen_Y = np.vstack([seen_Y, batches[1][1]])
    model.add_nodes(seen_X, seen_Y, n_new=12)
    model.add_data(*batches[2])
    seen_X = np.vstack([seen_X, batches[2][0]])
    seen_Y = np.vstack([seen_Y, batches[2][1]])
    model.add_nodes(seen_X, seen_Y, n_new=20)
    model.add_data(*batches[3])
    seen_X = np.vstack([seen_X, batches[3][0]])
    seen_Y = np.vstack([seen_Y, batches[3][1]])

    full = NonIncrementalBLS(config=cfg_b).fit_initial(seen_X, seen_Y)
    full.feature_layer.add_enhancement_window(12)
    full.feature_layer.add_enhancement_window(20)
    A_ext = full.feature_layer.transform(seen_X)
    W_ref = _ridge_solve(A_ext, seen_Y, cfg_b.reg_lambda)

    assert np.allclose(model.W, W_ref, atol=1e-6)



# ===========================================================================
# 18. R^T R = A^T A + λI 不变量在每一步都成立
# ===========================================================================


def test_invariant_holds_after_every_phase(small_classification) -> None:
    """整个生命周期内：每次操作后 R^T R = A^T A + λI 都精确成立。"""
    X_tr, Y_tr, _, _ = small_classification
    batches = split_into_batches(X_tr, Y_tr, n_batches=4, shuffle=False)
    cfg = _make_cfg(seed=33)
    model = IMFBLS(config=cfg).fit_initial(*batches[0])
    seen_X = batches[0][0].copy()
    seen_Y = batches[0][1].copy()

    def _check_invariant(label: str) -> None:
        A = model.feature_layer.transform(seen_X)
        p = A.shape[1]
        err = np.max(np.abs(model.R.T @ model.R - A.T @ A - cfg.reg_lambda * np.eye(p)))
        assert err < 1e-8, f"{label}: invariant violated, err={err:.2e}"

    _check_invariant("init")
    model.add_data(*batches[1])
    seen_X = np.vstack([seen_X, batches[1][0]])
    seen_Y = np.vstack([seen_Y, batches[1][1]])
    _check_invariant("after add_data")

    model.add_nodes(seen_X, seen_Y, n_new=15)
    _check_invariant("after add_nodes")

    model.add_data(*batches[2])
    seen_X = np.vstack([seen_X, batches[2][0]])
    seen_Y = np.vstack([seen_Y, batches[2][1]])
    _check_invariant("after add_data #2")

    model.add_nodes(seen_X, seen_Y, n_new=10)
    _check_invariant("after add_nodes #2")


# ===========================================================================
# 19. V = A^T Y 不变量
# ===========================================================================


def test_V_equals_AT_Y_invariant(small_classification) -> None:
    """整个生命周期内 V = A^T Y 应保持成立。"""
    X_tr, Y_tr, _, _ = small_classification
    batches = split_into_batches(X_tr, Y_tr, n_batches=3, shuffle=False)
    cfg = _make_cfg(seed=55)
    model = IMFBLS(config=cfg).fit_initial(*batches[0])
    seen_X = batches[0][0].copy()
    seen_Y = batches[0][1].copy()

    def _check() -> None:
        A = model.feature_layer.transform(seen_X)
        err = np.max(np.abs(model.V - A.T @ seen_Y))
        assert err < 1e-9, f"V invariant violated, err={err:.2e}"

    _check()
    model.add_data(*batches[1])
    seen_X = np.vstack([seen_X, batches[1][0]])
    seen_Y = np.vstack([seen_Y, batches[1][1]])
    _check()

    model.add_nodes(seen_X, seen_Y, n_new=8)
    _check()


# ===========================================================================
# 20. 论文 Section 3.2.4 替换法残差严格 ≤ ridge inverse
# ===========================================================================


def test_substitution_residual_not_worse_than_inverse_on_ill_conditioned() -> None:
    """病态情形下替换法残差不应明显大于直接求逆（论文 Theorem 3.6）。"""
    rng = np.random.default_rng(2024)
    p = 60
    U, _ = np.linalg.qr(rng.standard_normal((300, p)))
    s = np.logspace(0, -6, p)
    Vt, _ = np.linalg.qr(rng.standard_normal((p, p)))
    A = U @ np.diag(s) @ Vt
    Y = rng.standard_normal((300, 4))
    lam = 1e-8
    M = A.T @ A + lam * np.eye(p)
    rhs = A.T @ Y

    from utils.linalg import qr_R, solve_sne
    sqrt_lam = np.sqrt(lam)
    R = qr_R(np.vstack([sqrt_lam * np.eye(p), A]))
    W_sub = solve_sne(R, rhs)
    res_sub = np.linalg.norm(M @ W_sub - rhs)

    W_inv = np.linalg.inv(M) @ rhs
    res_inv = np.linalg.norm(M @ W_inv - rhs)

    # 替换法残差不应明显大于求逆（允许 1.5 倍裕度）
    assert res_sub <= res_inv * 1.5 + 1e-12


# ===========================================================================
# 21. tsqr_blocks 参数化对结果无影响（论文 Section 2.4）
# ===========================================================================


@pytest.mark.parametrize("n_blocks", [1, 2, 4, 8])
def test_tsqr_blocks_param_equivalent(small_classification, n_blocks) -> None:
    """不同 tsqr_blocks 应产生同一权重（数学等价）。"""
    X_tr, Y_tr, _, _ = small_classification
    base = IMFBLS(config=_make_cfg(seed=77), use_tsqr=False).fit_initial(X_tr, Y_tr)
    tsqr = IMFBLS(config=_make_cfg(seed=77), use_tsqr=True,
                  tsqr_blocks=n_blocks).fit_initial(X_tr, Y_tr)
    assert np.allclose(base.W, tsqr.W, atol=1e-9)


# ===========================================================================
# 22. R 因子上三角且对角全正（论文 Theorem 3.1 唯一性）
# ===========================================================================


def test_R_upper_triangular_with_positive_diag_throughout(small_classification) -> None:
    """整个生命周期内 R 始终是上三角矩阵且对角线全正。"""
    X_tr, Y_tr, _, _ = small_classification
    batches = split_into_batches(X_tr, Y_tr, n_batches=3, shuffle=False)
    model = IMFBLS(config=_make_cfg(seed=88)).fit_initial(*batches[0])
    seen_X = batches[0][0].copy()
    seen_Y = batches[0][1].copy()

    def _check(label: str) -> None:
        below = np.tril(model.R, k=-1)
        assert np.max(np.abs(below)) < 1e-12, f"{label}: not upper triangular"
        assert (np.diag(model.R) > 0).all(), f"{label}: diag not all positive"

    _check("init")
    model.add_data(*batches[1])
    seen_X = np.vstack([seen_X, batches[1][0]])
    seen_Y = np.vstack([seen_Y, batches[1][1]])
    _check("after add_data")

    model.add_nodes(seen_X, seen_Y, n_new=10)
    _check("after add_nodes")


# ===========================================================================
# 23. 解析正确性：W 严格满足 (A^T A + λI) W = A^T Y
# ===========================================================================


def test_solution_exactly_satisfies_normal_equation(small_classification) -> None:
    """fit_initial 后 W 应满足 (A^T A + λI) W = A^T Y。"""
    X_tr, Y_tr, _, _ = small_classification
    cfg = _make_cfg(seed=44)
    model = IMFBLS(config=cfg).fit_initial(X_tr, Y_tr)
    A = model.feature_layer.transform(X_tr)
    p = A.shape[1]
    M = A.T @ A + cfg.reg_lambda * np.eye(p)
    rhs = A.T @ Y_tr
    err = np.max(np.abs(M @ model.W - rhs))
    # 替换法解极精确
    assert err < 1e-7


# ===========================================================================
# 24. seed 隔离：不同 seed 的两个模型不互相干扰
# ===========================================================================


def test_seed_isolation_between_models(small_classification) -> None:
    """两个不同 seed 的 IMF-BLS 实例应有不同 R 但都正确。"""
    X_tr, Y_tr, _, _ = small_classification
    a = IMFBLS(config=_make_cfg(seed=1)).fit_initial(X_tr, Y_tr)
    b = IMFBLS(config=_make_cfg(seed=2)).fit_initial(X_tr, Y_tr)
    # 不同 seed → 不同 mapping/enh 权重 → 不同 R
    assert not np.allclose(a.R, b.R)
    # 但各自的解都满足各自的 normal equation
    for m in (a, b):
        A = m.feature_layer.transform(X_tr)
        p = A.shape[1]
        M = A.T @ A + m.config.reg_lambda * np.eye(p)
        assert np.max(np.abs(M @ m.W - A.T @ Y_tr)) < 1e-7
