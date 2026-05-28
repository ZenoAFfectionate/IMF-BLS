# -*- coding: utf-8 -*-
"""utils/paper_presets.py 与 utils/uci_loader.py 的单元测试。

注意：UCI 数据下载可能依赖网络，故下载相关测试默认跳过 / 仅做 mock。
"""

from __future__ import annotations

import numpy as np
import pytest

from utils.paper_presets import DatasetPreset, get_preset, list_presets


# ===========================================================================
# paper_presets
# ===========================================================================


def test_list_presets_contains_all_paper_datasets() -> None:
    presets = set(list_presets())
    expected = {
        # 论文 Table 3 主图像
        "mnist", "fashion_mnist", "norb", "emnist",
        # 论文 Table 3 UCI 分类
        "shuttle", "letter", "pendigits", "led", "waveform",
        # 论文 Table 4 回归
        "abalone", "bodyfat", "weather_izmir",
        "energy_efficiency", "appliances_energy",
    }
    missing = expected - presets
    assert not missing, f"缺少论文数据集预设: {missing}"


@pytest.mark.parametrize("name", [
    "mnist", "fashion_mnist", "pendigits", "abalone",
])
def test_get_preset_returns_dataset_preset(name) -> None:
    p = get_preset(name)
    assert isinstance(p, DatasetPreset)
    assert p.name == name
    assert p.task in {"classification", "regression"}


def test_preset_classification_has_required_bls_kwargs() -> None:
    p = get_preset("mnist")
    for k in ("n_mapping_per_window", "n_mapping_windows",
              "n_enhancement", "reg_lambda", "activation", "seed"):
        assert k in p.bls_kwargs


def test_preset_mnist_matches_paper_table3() -> None:
    """论文 Table 3 中 MNIST: N3=5000, λ=10^-6, batch=6。"""
    p = get_preset("mnist")
    assert p.bls_kwargs["n_enhancement"] == 5000
    assert p.bls_kwargs["reg_lambda"] == pytest.approx(1e-6)
    assert p.equal_scale_n_batches == 6
    assert p.task == "classification"


def test_preset_norb_matches_paper_table3() -> None:
    """论文 Table 3 中 NORB: N3=3000, λ=10^-3。"""
    p = get_preset("norb")
    assert p.bls_kwargs["n_enhancement"] == 3000
    assert p.bls_kwargs["reg_lambda"] == pytest.approx(1e-3)


def test_preset_pendigits_matches_paper_table3() -> None:
    """论文 Table 3 中 Pendigits: N3=1000, λ=10^-6。"""
    p = get_preset("pendigits")
    assert p.bls_kwargs["n_enhancement"] == 1000


def test_preset_unknown_dataset_raises() -> None:
    with pytest.raises(KeyError):
        get_preset("definitely_not_a_dataset")


def test_preset_uncertain_scale_default_batches() -> None:
    """论文 Section 4.2: 5/10/15/20/25 个 batch。"""
    p = get_preset("mnist")
    assert p.uncertain_scale_n_batches == [5, 10, 15, 20, 25]
    assert p.uncertain_scale_repeats == 10


def test_preset_dash_form_supported() -> None:
    """``fashion-mnist`` 与 ``fashion_mnist`` 应都被支持。"""
    p1 = get_preset("fashion_mnist")
    p2 = get_preset("fashion-mnist")
    assert p1.name == p2.name == "fashion_mnist"


# ===========================================================================
# uci_loader（不发起真实下载，仅测试本地工具函数）
# ===========================================================================


def test_uci_loader_split_train_test() -> None:
    from utils.uci_loader import _split_train_test

    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5))
    y = rng.standard_normal(100)
    X_tr, y_tr, X_te, y_te = _split_train_test(X, y, test_size=20, seed=42)
    assert len(X_tr) == 80
    assert len(X_te) == 20
    assert X_tr.shape[1] == X.shape[1]


def test_uci_loader_libsvm_parser(tmp_path) -> None:
    """验证 LIBSVM 文本解析正确（label idx:val 格式）。"""
    from utils.uci_loader import _read_libsvm

    text = "1.0 1:0.5 3:1.5\n-1.0 2:2.0\n"
    p = tmp_path / "demo.libsvm"
    p.write_text(text)

    X, y = _read_libsvm(str(p), n_features=4)
    assert X.shape == (2, 4)
    assert y.tolist() == [1.0, -1.0]
    np.testing.assert_allclose(X[0], [0.5, 0.0, 1.5, 0.0])
    np.testing.assert_allclose(X[1], [0.0, 2.0, 0.0, 0.0])


def test_uci_loader_classification_loaders_dict() -> None:
    """所有论文 UCI 分类加载器都已注册。"""
    from utils.uci_loader import _CLASSIFICATION_LOADERS

    expected = {"pendigits", "letter", "shuttle", "waveform", "led"}
    assert expected.issubset(set(_CLASSIFICATION_LOADERS.keys()))


def test_uci_loader_regression_loaders_dict() -> None:
    from utils.uci_loader import _REGRESSION_LOADERS

    expected = {"abalone", "bodyfat", "energy_efficiency",
                "appliances_energy", "weather_izmir"}
    assert expected.issubset(set(_REGRESSION_LOADERS.keys()))


def test_uci_loader_unknown_dataset_raises() -> None:
    from utils.uci_loader import (
        load_uci_classification, load_uci_regression,
    )
    with pytest.raises(ValueError):
        load_uci_classification("definitely_not_a_dataset")
    with pytest.raises(ValueError):
        load_uci_regression("definitely_not_a_dataset")


# ===========================================================================
# led 数据集是合成的，不依赖网络，可以测真实加载
# ===========================================================================


def test_led_loader_produces_correct_shape() -> None:
    from utils.uci_loader import load_led

    X_tr, y_tr, X_te, y_te = load_led(n_samples=2000, seed=0)
    # 共 2000，按 20% 切：1600 / 400
    assert X_tr.shape == (1600, 24)
    assert X_te.shape == (400, 24)
    # 标签 0..9
    assert int(y_tr.min()) >= 0 and int(y_tr.max()) <= 9
    assert int(y_te.min()) >= 0 and int(y_te.max()) <= 9


def test_led_loader_deterministic_under_same_seed() -> None:
    from utils.uci_loader import load_led

    a = load_led(n_samples=500, seed=7)
    b = load_led(n_samples=500, seed=7)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


# ===========================================================================
# load_classification_dataset / load_regression_dataset 接入 UCI
# ===========================================================================


def test_load_classification_dispatches_to_led() -> None:
    """``load_classification_dataset('led')`` 应当走 UCI loader。"""
    from utils.data import load_classification_dataset

    # 替换 led loader 防止外部下载（led 本身合成不下载，正好可测）
    X_tr, y_tr, X_te, y_te = load_classification_dataset("led")
    # 默认 200000 太慢，跳过 default；验证至少不报错
    assert X_tr.ndim == 2
    assert len(X_tr) > 0


def test_load_regression_unknown_raises() -> None:
    from utils.data import load_regression_dataset

    with pytest.raises(ValueError):
        load_regression_dataset("definitely_not_a_dataset")


# ===========================================================================
# 增补：paper_presets 中的 synthetic 预设（用于 smoke 测试）
# ===========================================================================


def test_synthetic_classification_preset() -> None:
    """``synthetic_classification`` 预设可获取且任务正确。"""
    p = get_preset("synthetic_classification")
    assert p.task == "classification"
    assert p.bls_kwargs["n_enhancement"] > 0


def test_synthetic_regression_preset() -> None:
    p = get_preset("synthetic_regression")
    assert p.task == "regression"


def test_all_presets_have_consistent_task_and_bls_kwargs() -> None:
    """每个 preset 的 task 与 bls_kwargs 都应合法。"""
    for name in list_presets():
        p = get_preset(name)
        assert p.task in {"classification", "regression"}, \
            f"{name}: task={p.task} 非法"
        for k in ("n_mapping_per_window", "n_mapping_windows",
                  "n_enhancement", "reg_lambda", "activation"):
            assert k in p.bls_kwargs, f"{name}: 缺少 {k}"
        assert p.bls_kwargs["n_enhancement"] > 0, \
            f"{name}: n_enhancement 非正"
        assert p.equal_scale_n_batches >= 1


# ===========================================================================
# 增补：uci_loader 工具函数
# ===========================================================================


def test_uci_loader_ensure_dir(tmp_path) -> None:
    from utils.uci_loader import _ensure_dir

    target = tmp_path / "a" / "b" / "c"
    _ensure_dir(str(target))
    assert target.is_dir()
    # 重复调用不报错
    _ensure_dir(str(target))


def test_uci_loader_split_train_test_float_size() -> None:
    """``test_size`` 是 float 时按比例切。"""
    from utils.uci_loader import _split_train_test

    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    y = rng.standard_normal(200)
    X_tr, y_tr, X_te, y_te = _split_train_test(X, y, test_size=0.25, seed=42)
    assert len(X_te) == 50
    assert len(X_tr) == 150


def test_uci_loader_split_train_test_deterministic_under_same_seed() -> None:
    from utils.uci_loader import _split_train_test

    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3))
    y = rng.standard_normal(50)
    a = _split_train_test(X, y, test_size=10, seed=99)
    b = _split_train_test(X, y, test_size=10, seed=99)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_uci_loader_download_skips_existing_file(tmp_path) -> None:
    """``_download`` 对已存在文件应直接返回（不发起网络请求）。"""
    from utils.uci_loader import _download

    target = tmp_path / "exists.txt"
    target.write_text("local content")
    # url 故意写错，但因为 target 已存在，函数不应触发下载
    out = _download("http://invalid-domain-zzz.example/x", str(target))
    assert out == str(target)
    assert target.read_text() == "local content"


def test_uci_loader_libsvm_empty_file(tmp_path) -> None:
    from utils.uci_loader import _read_libsvm

    p = tmp_path / "empty.libsvm"
    p.write_text("")
    X, y = _read_libsvm(str(p), n_features=3)
    assert X.shape == (0, 3)
    assert y.shape == (0,)


def test_uci_loader_libsvm_with_blank_lines(tmp_path) -> None:
    from utils.uci_loader import _read_libsvm

    p = tmp_path / "demo.libsvm"
    p.write_text("1.0 1:0.5\n\n\n2.0 2:1.0\n")
    X, y = _read_libsvm(str(p), n_features=3)
    assert X.shape == (2, 3)
    assert y.tolist() == [1.0, 2.0]
