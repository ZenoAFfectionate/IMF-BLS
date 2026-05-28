# -*- coding: utf-8 -*-
"""``scripts/*.sh`` 复现脚本的可运行性测试。

测试范围：
    1. 所有脚本存在且可执行
    2. bash 语法检查（``bash -n``）
    3. ``--help`` / 文档说明可用
    4. 关键脚本中调用的 reproduce.py 子命令存在
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


SCRIPT_NAMES = [
    "run_all.sh",
    "table5.sh",
    "table6.sh",
    "table7.sh",
    "single.sh",
    "download_mnist.sh",
]


# ===========================================================================
# 1. 文件存在 + 可执行
# ===========================================================================


@pytest.mark.parametrize("script", SCRIPT_NAMES)
def test_script_file_exists(script) -> None:
    p = SCRIPTS_DIR / script
    assert p.exists(), f"缺少脚本: {p}"


@pytest.mark.parametrize("script", SCRIPT_NAMES)
def test_script_is_executable(script) -> None:
    p = SCRIPTS_DIR / script
    mode = p.stat().st_mode
    # 至少 owner 有执行权
    assert mode & stat.S_IXUSR, f"{p} 没有执行权"


@pytest.mark.parametrize("script", SCRIPT_NAMES)
def test_script_has_shebang(script) -> None:
    """所有脚本第一行应是 ``#!/usr/bin/env bash`` 或 ``#!/bin/bash``。"""
    p = SCRIPTS_DIR / script
    first_line = p.read_text().splitlines()[0]
    assert first_line.startswith("#!"), f"{p} 缺少 shebang"
    assert "bash" in first_line, f"{p} shebang 不是 bash"


# ===========================================================================
# 2. bash 语法检查
# ===========================================================================


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash 不可用")
@pytest.mark.parametrize("script", SCRIPT_NAMES)
def test_script_bash_syntax_valid(script) -> None:
    """``bash -n`` 检查脚本语法是否合法（不实际执行）。"""
    p = SCRIPTS_DIR / script
    result = subprocess.run(
        ["bash", "-n", str(p)],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, \
        f"{p} 语法错误:\n{result.stderr}"


# ===========================================================================
# 3. 文档与 --help
# ===========================================================================


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash 不可用")
def test_run_all_help() -> None:
    """``run_all.sh --help`` 应输出说明。"""
    p = SCRIPTS_DIR / "run_all.sh"
    result = subprocess.run(
        ["bash", str(p), "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert len(result.stdout) > 0


@pytest.mark.parametrize("script", SCRIPT_NAMES)
def test_script_docstring_present(script) -> None:
    """每个脚本应有顶部注释（说明用法）。"""
    p = SCRIPTS_DIR / script
    text = p.read_text()
    # 第二/三行应为注释（Shebang 之后的注释块）
    head = text.splitlines()[1:6]
    has_comment = any(line.startswith("#") for line in head)
    assert has_comment, f"{p} 缺少头部用法注释"


# ===========================================================================
# 4. 引用一致性：脚本中调用的 reproduce.py 子命令存在
# ===========================================================================


def test_run_all_invokes_reproduce_correctly() -> None:
    """``run_all.sh`` 应调用 ``reproduce.py table5/6/7``。"""
    text = (SCRIPTS_DIR / "run_all.sh").read_text()
    assert "reproduce.py table5" in text
    assert "reproduce.py table6" in text
    assert "reproduce.py table7" in text


def test_table_scripts_invoke_reproduce_all() -> None:
    """``table5.sh`` / ``table6.sh`` 应通过 ``reproduce.py all`` 派发。"""
    for n in ("table5.sh", "table6.sh"):
        text = (SCRIPTS_DIR / n).read_text()
        assert "reproduce.py all" in text, \
            f"{n} 未通过 reproduce.py all 调用"


def test_single_script_uses_reproduce() -> None:
    text = (SCRIPTS_DIR / "single.sh").read_text()
    assert "reproduce.py" in text


def test_download_mnist_creates_correct_dirs() -> None:
    """``download_mnist.sh`` 应下载到 data/mnist/ 与 data/fashion_mnist/。"""
    text = (SCRIPTS_DIR / "download_mnist.sh").read_text()
    assert "data/mnist" in text
    assert "data/fashion_mnist" in text


# ===========================================================================
# 5. 集成：用合成预设跑 single.sh（最快路径）
# ===========================================================================


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash 不可用")
def test_single_script_runs_synthetic(tmp_path) -> None:
    """通过 ``single.sh`` 跑 synthetic 数据集应成功。"""
    project_root = SCRIPTS_DIR.parent
    p = SCRIPTS_DIR / "single.sh"
    env = os.environ.copy()
    env["PYTHON"] = env.get("PYTHON", "python3")

    result = subprocess.run(
        ["bash", str(p), "table5", "synthetic_classification",
         "--output_dir", str(tmp_path)],
        capture_output=True, text=True, cwd=str(project_root),
        env=env, timeout=120,
    )
    assert result.returncode == 0, \
        f"single.sh table5 synthetic 失败:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert (tmp_path / "table5" / "synthetic_classification" / "metrics.json").exists()
