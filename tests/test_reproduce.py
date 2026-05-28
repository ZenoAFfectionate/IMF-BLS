# -*- coding: utf-8 -*-
"""``reproduce.py`` 的单元 + 集成测试。

测试覆盖：
    1. 模块可导入、关键函数可调用
    2. 内部辅助：`_make_config` / `_eval_classification` / `_eval_regression` /
       `_prepare_classification` / `_prepare_regression`
    3. 报告生成：`_save_metrics` / `_write_table5_report` / `_write_table6_report` /
       `_write_table7_report` / `write_summary_md`
    4. 端到端：用 ``synthetic_classification`` / ``synthetic_regression`` 预设
       跑通 ``reproduce_table5`` / ``reproduce_table6`` / ``reproduce_table7``
    5. CLI：argparse 参数解析 + dataset list 解析
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


# ===========================================================================
# 1. 模块可导入
# ===========================================================================


def test_reproduce_module_imports() -> None:
    """``reproduce`` 顶层符号应可正常导入。"""
    import reproduce as r

    for name in [
        "reproduce_table5", "reproduce_table6", "reproduce_table7",
        "write_summary_md",
        "DEFAULT_TABLE5_DATASETS",
        "DEFAULT_TABLE6_DATASETS",
        "DEFAULT_TABLE7_DATASETS",
        "_make_config", "_prepare_classification", "_prepare_regression",
        "_eval_classification", "_eval_regression",
        "_save_metrics", "_parse_dataset_list",
    ]:
        assert hasattr(r, name), f"reproduce.py 缺少符号 {name}"


def test_default_dataset_lists_have_correct_paper_datasets() -> None:
    import reproduce as r

    assert {"pendigits", "letter", "shuttle", "waveform", "led"}.issubset(
        set(r.DEFAULT_TABLE5_DATASETS)
    )
    assert {"abalone", "bodyfat"}.issubset(set(r.DEFAULT_TABLE6_DATASETS))
    assert {"mnist", "fashion_mnist"}.issubset(set(r.DEFAULT_TABLE7_DATASETS))


# ===========================================================================
# 2. 辅助函数
# ===========================================================================


def test_make_config_uses_preset_kwargs() -> None:
    from reproduce import _make_config
    from utils.paper_presets import get_preset

    p = get_preset("pendigits")
    cfg = _make_config(p, seed=99)
    assert cfg.n_enhancement == p.bls_kwargs["n_enhancement"]
    assert cfg.reg_lambda == p.bls_kwargs["reg_lambda"]
    assert cfg.seed == 99


def test_make_config_default_seed_uses_preset() -> None:
    from reproduce import _make_config
    from utils.paper_presets import get_preset

    p = get_preset("pendigits")
    cfg = _make_config(p)  # seed 不指定 → 使用 preset 默认
    assert cfg.seed == p.bls_kwargs["seed"]


def test_parse_dataset_list_default() -> None:
    from reproduce import _parse_dataset_list

    assert _parse_dataset_list(None, ["a", "b"]) == ["a", "b"]


def test_parse_dataset_list_explicit_empty_returns_empty() -> None:
    """显式传 ``""`` 应返回空列表（用于跳过该 table）。"""
    from reproduce import _parse_dataset_list

    assert _parse_dataset_list("", ["a", "b"]) == []


def test_parse_dataset_list_csv() -> None:
    from reproduce import _parse_dataset_list

    assert _parse_dataset_list("foo,bar,baz", ["x"]) == ["foo", "bar", "baz"]
    # 去除空白
    assert _parse_dataset_list(" foo , bar ", []) == ["foo", "bar"]


# ===========================================================================
# 3. _prepare_classification / _prepare_regression（合成预设）
# ===========================================================================


def test_prepare_classification_synthetic() -> None:
    from reproduce import _prepare_classification

    X_tr, Y_tr, X_te, Y_te, n_classes, preset = _prepare_classification(
        "synthetic_classification", mnist_path=None
    )
    # 数据应已被 min-max 归一化到 [0, 1]
    assert X_tr.min() >= -1e-9 and X_tr.max() <= 1.0 + 1e-9
    # one-hot ±1
    assert set(np.unique(Y_tr).tolist()).issubset({-1.0, 1.0})
    assert preset.task == "classification"
    assert n_classes >= 2


def test_prepare_classification_subsample() -> None:
    from reproduce import _prepare_classification

    X_tr, Y_tr, X_te, Y_te, _, _ = _prepare_classification(
        "synthetic_classification", mnist_path=None,
        subsample_train=200, subsample_test=50,
    )
    assert len(X_tr) == 200
    assert len(X_te) == 50


def test_prepare_regression_synthetic() -> None:
    from reproduce import _prepare_regression

    X_tr, Y_tr, X_te, Y_te, preset = _prepare_regression("synthetic_regression")
    assert Y_tr.shape[1] == 1
    assert preset.task == "regression"
    # 归一化到 [0, 1]
    assert X_tr.min() >= -1e-9 and X_tr.max() <= 1.0 + 1e-9


# ===========================================================================
# 4. eval functions
# ===========================================================================


def test_eval_classification_returns_accuracy() -> None:
    from reproduce import _eval_classification, _make_config, _prepare_classification
    from src.imf_bls import IMFBLS

    X_tr, Y_tr, X_te, Y_te, _, preset = _prepare_classification(
        "synthetic_classification", mnist_path=None
    )
    m = IMFBLS(config=_make_config(preset, seed=0)).fit_initial(X_tr, Y_tr)
    acc = _eval_classification(m, X_te, Y_te)
    assert 0.0 <= acc <= 1.0


def test_eval_regression_returns_rmse() -> None:
    from reproduce import _eval_regression, _make_config, _prepare_regression
    from src.imf_bls import IMFBLS

    X_tr, Y_tr, X_te, Y_te, preset = _prepare_regression("synthetic_regression")
    m = IMFBLS(config=_make_config(preset, seed=0)).fit_initial(X_tr, Y_tr)
    rmse = _eval_regression(m, X_te, Y_te)
    assert rmse >= 0.0


# ===========================================================================
# 5. _run_method
# ===========================================================================


def test_run_method_returns_step_metrics(tmp_path) -> None:
    """``_run_method`` 应返回 per-step metric / time + final / total。"""
    from reproduce import (_run_method, _make_config, _prepare_classification,
                            _eval_classification)
    from src.imf_bls import IMFBLS
    from utils.data import split_into_batches

    X_tr, Y_tr, X_te, Y_te, _, preset = _prepare_classification(
        "synthetic_classification", mnist_path=None
    )
    batches = split_into_batches(X_tr, Y_tr, n_batches=3, shuffle=False)
    info = _run_method(IMFBLS, _make_config(preset), batches, X_te, Y_te,
                       _eval_classification)
    assert "metric_per_step" in info
    assert "time_per_step" in info
    assert len(info["metric_per_step"]) == 3
    assert info["final_metric"] == info["metric_per_step"][-1]
    assert info["total_time"] == pytest.approx(sum(info["time_per_step"]))


# ===========================================================================
# 6. 报告生成
# ===========================================================================


def test_save_metrics_creates_json(tmp_path) -> None:
    from reproduce import _save_metrics

    target = tmp_path / "out"
    data = {"foo": {"acc": 0.95, "time": 1.23}}
    _save_metrics(target, data)
    assert (target / "metrics.json").exists()
    loaded = json.loads((target / "metrics.json").read_text())
    assert loaded["foo"]["acc"] == 0.95


def test_reproduce_table5_synthetic_e2e(tmp_path) -> None:
    """端到端：用 synthetic 预设跑 reproduce_table5。"""
    from reproduce import reproduce_table5

    results = reproduce_table5("synthetic_classification", tmp_path, seed=0)

    # 每个方法都应有 final_metric
    expected_methods = {
        "Non-Incremental BLS", "Incremental BLS", "TiBLS",
        "Approximation Method", "RI-BLS", "IMF-BLS (Ours)",
    }
    assert expected_methods.issubset(set(results.keys()))

    # IMF-BLS 应跑通
    info = results["IMF-BLS (Ours)"]
    assert "error" not in info
    assert 0.0 <= info["final_metric"] <= 1.0

    # 应生成 report.md
    target = tmp_path / "table5" / "synthetic_classification"
    assert (target / "metrics.json").exists()
    assert (target / "report.md").exists()


def test_reproduce_table6_synthetic_e2e(tmp_path) -> None:
    from reproduce import reproduce_table6

    summary = reproduce_table6("synthetic_regression", tmp_path,
                               repeats=2, seed=0)

    expected_methods = {
        "Non-Incremental BLS", "Incremental BLS",
        "Approximation Method", "TiBLS", "RI-BLS", "IMF-BLS (Ours)",
    }
    assert expected_methods.issubset(set(summary.keys()))

    info = summary["IMF-BLS (Ours)"]
    assert "error" not in info
    assert info["rmse_mean"] >= 0
    assert "rmse_std" in info

    target = tmp_path / "table6" / "synthetic_regression"
    assert (target / "report.md").exists()


def test_reproduce_table7_synthetic_e2e(tmp_path) -> None:
    from reproduce import reproduce_table7

    results = reproduce_table7("synthetic_classification", tmp_path,
                                seed=0, n_batches_choices=[3, 5],
                                n_repeats=2)

    # 应包含 reference + 各 batch 配置 + Overall
    assert "_non_incremental_reference" in results
    assert "n_batches=3" in results
    assert "n_batches=5" in results
    assert "Overall" in results

    # IMF-BLS 在 Overall 中存在
    assert "IMF-BLS (Ours)" in results["Overall"]

    target = tmp_path / "table7" / "synthetic_classification"
    assert (target / "report.md").exists()


# ===========================================================================
# 7. write_summary_md
# ===========================================================================


def test_write_summary_md_table5(tmp_path) -> None:
    """从 reproduce_table5 输出生成跨数据集汇总。"""
    from reproduce import reproduce_table5, write_summary_md

    reproduce_table5("synthetic_classification", tmp_path, seed=0)
    write_summary_md("table5", tmp_path, ["synthetic_classification"])

    out = tmp_path / "table5" / "summary.md"
    assert out.exists()
    text = out.read_text()
    assert "Paper Table 5" in text
    assert "synthetic_classification" in text
    assert "IMF-BLS" in text


def test_write_summary_md_table6(tmp_path) -> None:
    from reproduce import reproduce_table6, write_summary_md

    reproduce_table6("synthetic_regression", tmp_path, repeats=2, seed=0)
    write_summary_md("table6", tmp_path, ["synthetic_regression"])

    out = tmp_path / "table6" / "summary.md"
    text = out.read_text()
    assert "Paper Table 6" in text
    assert "synthetic_regression" in text


def test_write_summary_md_table7(tmp_path) -> None:
    from reproduce import reproduce_table7, write_summary_md

    reproduce_table7("synthetic_classification", tmp_path,
                     seed=0, n_batches_choices=[3], n_repeats=2)
    write_summary_md("table7", tmp_path, ["synthetic_classification"])

    out = tmp_path / "table7" / "summary.md"
    text = out.read_text()
    assert "Paper Table 7" in text


def test_write_summary_md_handles_missing_dataset(tmp_path) -> None:
    """若数据集目录不存在不应崩溃，而是跳过。"""
    from reproduce import write_summary_md

    write_summary_md("table5", tmp_path, ["nonexistent_dataset"])
    # 应至少创建 summary.md（即使没数据）
    assert (tmp_path / "table5" / "summary.md").exists()


def test_summary_md_no_consecutive_blank_lines(tmp_path) -> None:
    """汇总 md 不应包含连续空行（避免 markdown 表格中断）。"""
    from reproduce import reproduce_table5, write_summary_md

    reproduce_table5("synthetic_classification", tmp_path, seed=0)
    write_summary_md("table5", tmp_path, ["synthetic_classification"])

    text = (tmp_path / "table5" / "summary.md").read_text()
    assert "\n\n\n" not in text, "summary.md 含连续 3 个换行（空行）"


# ===========================================================================
# 8. CLI
# ===========================================================================


def test_cli_help_works() -> None:
    """``python reproduce.py --help`` 不应报错。"""
    import subprocess
    import sys
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [sys.executable, "reproduce.py", "--help"],
        capture_output=True, text=True, cwd=root, timeout=30,
    )
    assert result.returncode == 0
    assert "table5" in result.stdout or "table5" in result.stderr


@pytest.mark.parametrize("subcmd", ["table5", "table6", "table7", "all"])
def test_cli_subcommand_help(subcmd) -> None:
    """每个子命令的 --help 都应有效。"""
    import subprocess
    import sys
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [sys.executable, "reproduce.py", subcmd, "--help"],
        capture_output=True, text=True, cwd=root, timeout=30,
    )
    assert result.returncode == 0


def test_cli_table5_synthetic(tmp_path) -> None:
    """通过 CLI 跑 table5 synthetic 应成功。"""
    import subprocess
    import sys
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [sys.executable, "reproduce.py", "table5",
         "--dataset", "synthetic_classification",
         "--output_dir", str(tmp_path)],
        capture_output=True, text=True, cwd=root, timeout=60,
    )
    assert result.returncode == 0, f"CLI 失败:\n{result.stderr}"
    assert (tmp_path / "table5" / "synthetic_classification" / "metrics.json").exists()
