# -*- coding: utf-8 -*-
"""包结构、模块路径、命名一致性测试。

本测试文件确保 IMF-BLS 项目重命名后所有公开 API 保持自洽：

  * 主类必须命名为 ``IMFBLS``，不得保留旧名 ``InvFBLS``
  * 主算法模块路径必须为 ``src.imf_bls``，不得保留旧路径 ``src.invf_bls``
  * ``src.__init__`` 与 ``utils.__init__`` 的 ``__all__`` 完整且与实际导出一致
  * 关键类的 MRO（继承）正确
  * 文档字符串与论文公式编号引用未丢失
"""

from __future__ import annotations

import importlib
import inspect

import pytest


# ===========================================================================
# 1. 模块路径
# ===========================================================================


def test_imf_bls_module_path_is_imf_bls() -> None:
    """主算法模块路径必须为 ``src.imf_bls`` —— 旧路径必须不可导入。"""
    mod = importlib.import_module("src.imf_bls")
    assert mod.__name__ == "src.imf_bls"

    # 旧路径已删除，不应可导入
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.invf_bls")


def test_main_class_name_is_imfbls() -> None:
    """主类必须叫 IMFBLS（非 InvFBLS）。"""
    from src.imf_bls import IMFBLS

    assert IMFBLS.__name__ == "IMFBLS"

    # 旧名 InvFBLS 不应存在
    mod = importlib.import_module("src.imf_bls")
    assert not hasattr(mod, "InvFBLS"), "残留旧类名 InvFBLS"


# ===========================================================================
# 2. 包导出 __all__
# ===========================================================================


def test_src_package_all_export() -> None:
    """src.__all__ 必须列出所有 BLS 类，且每个类都能从包顶层访问。"""
    import src

    expected = {
        "BLSBase", "BLSConfig", "NonIncrementalBLS",
        "IMFBLS",
        "IncrementalBLS", "RIBLS", "TiBLS", "ApproximationMethodBLS",
    }
    assert set(src.__all__) == expected, (
        f"src.__all__ 不一致: 期望 {expected}, 实际 {set(src.__all__)}"
    )

    for name in src.__all__:
        assert hasattr(src, name), f"src 未导出 {name}"
        # InvFBLS 也不应通过包顶层访问
        assert not hasattr(src, "InvFBLS")


def test_utils_package_all_export() -> None:
    """utils.__all__ 应覆盖核心数值原语 + 数据 + 指标。"""
    import utils

    must_have = {
        "forward_substitution", "backward_substitution", "solve_sne",
        "qr_R", "incremental_qr_update", "tsqr_R", "cholesky_lower",
        "FeatureLayer", "standardize_minmax",
        "one_hot_encode", "split_into_batches", "split_random_batches",
        "classification_accuracy", "regression_rmse", "sne_residual_norm",
        "Timer",
        # logger
        "get_logger", "ExperimentRecorder", "log_array_stats", "ColorFormatter",
    }
    actual = set(utils.__all__)
    missing = must_have - actual
    assert not missing, f"utils.__all__ 缺少: {missing}"

    for name in must_have:
        assert hasattr(utils, name), f"utils 未导出 {name}"


# ===========================================================================
# 3. 类继承与抽象方法
# ===========================================================================


def test_imfbls_inherits_from_blsbase() -> None:
    from src.bls_base import BLSBase
    from src.imf_bls import IMFBLS

    assert issubclass(IMFBLS, BLSBase)


def test_all_bls_methods_share_blsbase() -> None:
    """所有 BLS 方法必须继承自同一个 BLSBase（保证 predict/score 一致）。"""
    from src.bls_base import BLSBase
    from src.baselines import (
        ApproximationMethodBLS, IncrementalBLS, RIBLS, TiBLS,
    )
    from src.imf_bls import IMFBLS

    for cls in (IMFBLS, IncrementalBLS, RIBLS, TiBLS, ApproximationMethodBLS):
        assert issubclass(cls, BLSBase), f"{cls.__name__} 未继承 BLSBase"


def test_blsbase_is_abstract() -> None:
    """BLSBase 不能直接实例化。"""
    from src.bls_base import BLSBase

    with pytest.raises(TypeError):
        BLSBase()  # type: ignore[abstract]


def test_imfbls_signatures() -> None:
    """IMFBLS 必须实现 fit_initial / add_data / add_nodes 三个核心方法。"""
    from src.imf_bls import IMFBLS

    for name in ("fit_initial", "add_data", "add_nodes", "predict",
                 "memory_module", "memory_footprint_bytes"):
        assert callable(getattr(IMFBLS, name, None)), f"IMFBLS 缺方法: {name}"


# ===========================================================================
# 4. 文档字符串：论文引用未丢失
# ===========================================================================


def test_imf_bls_module_docstring_references_paper() -> None:
    """imf_bls.py 的模块 docstring 必须引用论文，并保留 IMF-BLS ↔ InvF-BLS 等价说明。"""
    from src import imf_bls

    doc = imf_bls.__doc__ or ""
    assert "IMF-BLS" in doc
    assert "Inverse Matrix-Free" in doc
    assert "Information Fusion" in doc, "未引用论文期刊"
    # 关键算法记忆模块说明应该存在
    assert "R^T R" in doc and "λI" in doc


def test_imfbls_method_docstrings_reference_equations() -> None:
    """add_data / add_nodes 必须在 docstring 中引用论文公式编号。"""
    from src.imf_bls import IMFBLS

    add_data_doc = (IMFBLS.add_data.__doc__ or "")
    add_nodes_doc = (IMFBLS.add_nodes.__doc__ or "")

    assert "Eq." in add_data_doc, "add_data 未引用 Eq."
    assert "Eq." in add_nodes_doc, "add_nodes 未引用 Eq."


# ===========================================================================
# 5. main.py 输出标签
# ===========================================================================


def test_main_uses_imf_bls_label() -> None:
    """main.py 中的方法标签必须为 'IMF-BLS (Ours)'，不应残留 'InvF-BLS'。"""
    import os

    main_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py"
    )
    with open(main_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "IMF-BLS (Ours)" in content
    assert "InvF-BLS (Ours)" not in content, "main.py 残留旧标签"


# ===========================================================================
# 6. 没有循环依赖
# ===========================================================================


def test_no_circular_imports() -> None:
    """连续导入所有模块不报错（循环依赖检查）。"""
    modules = [
        "utils.linalg", "utils.feature_layer", "utils.data",
        "utils.metrics", "utils.timing",
        "src.bls_base", "src.imf_bls", "src.baselines",
    ]
    for m in modules:
        importlib.import_module(m)


# ===========================================================================
# .gitignore 配置正确性
# ===========================================================================


def test_gitignore_excludes_paper_files() -> None:
    """.gitignore 应包含 paper.txt 与 paper.pdf 条目，避免论文原文入库。"""
    import os
    root = os.path.join(os.path.dirname(__file__), "..")
    gitignore_path = os.path.join(root, ".gitignore")
    assert os.path.exists(gitignore_path), ".gitignore 文件应该存在"

    with open(gitignore_path, "r", encoding="utf-8") as f:
        lines = {line.strip() for line in f if line.strip() and not line.startswith("#")}

    assert "paper.txt" in lines, ".gitignore 应忽略 paper.txt"
    assert "paper.pdf" in lines, ".gitignore 应忽略 paper.pdf"


def test_license_file_exists_and_is_mit() -> None:
    """LICENSE 文件应存在且是 MIT 许可。"""
    import os
    root = os.path.join(os.path.dirname(__file__), "..")
    license_path = os.path.join(root, "LICENSE")
    assert os.path.exists(license_path)
    with open(license_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "MIT License" in content
