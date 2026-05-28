# -*- coding: utf-8 -*-
"""utils/logger.py 的完整测试。

覆盖：
    - get_logger：handler 去重 / level / 文件输出 / propagate / 多次调用同名
    - ColorFormatter：着色 / 关闭着色
    - log_array_stats：基本输出 / 边界（空数组、含 NaN）
    - ExperimentRecorder：log_step / 记录恢复 / overwrite / filter / summarize /
      stage 上下文 / config / save_summary / attach_logger / numpy 类型转换 / CSV 表头扩展
"""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from utils.logger import (
    ColorFormatter,
    ExperimentRecorder,
    get_logger,
    log_array_stats,
)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _close_logger(logger: logging.Logger) -> None:
    """安全关闭 logger 的所有 handler，避免文件被占用。"""
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)


@pytest.fixture(autouse=True)
def _isolate_logging():
    """每个测试前后清空 imf_bls* logger 的 handler，避免相互影响。"""
    yield
    for name, lg in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(lg, logging.Logger) and (
            name.startswith("imf_bls") or name.startswith("test_logger.")
        ):
            _close_logger(lg)


# ===========================================================================
# 1. get_logger
# ===========================================================================


def test_get_logger_returns_logger_instance() -> None:
    lg = get_logger("test_logger.basic")
    assert isinstance(lg, logging.Logger)
    assert lg.name == "test_logger.basic"


def test_get_logger_level_int() -> None:
    lg = get_logger("test_logger.lvl_int", level=logging.DEBUG)
    assert lg.level == logging.DEBUG


def test_get_logger_level_str() -> None:
    lg = get_logger("test_logger.lvl_str", level="warning")
    assert lg.level == logging.WARNING


def test_get_logger_unknown_level_falls_back_to_info() -> None:
    lg = get_logger("test_logger.lvl_unknown", level="not_a_level")
    assert lg.level == logging.INFO


def test_get_logger_handler_not_duplicated_on_repeated_call() -> None:
    lg1 = get_logger("test_logger.dup")
    n1 = len(lg1.handlers)
    lg2 = get_logger("test_logger.dup")
    assert lg1 is lg2
    assert len(lg2.handlers) == n1, "重复调用同名 logger 不应重复挂载 handler"


def test_get_logger_writes_to_file(tmp_path: Path) -> None:
    log_file = tmp_path / "run.log"
    lg = get_logger("test_logger.file", log_file=log_file, use_color=False)
    lg.warning("hello-file-output")
    _close_logger(lg)

    text = log_file.read_text(encoding="utf-8")
    assert "hello-file-output" in text
    assert "WARNING" in text


def test_get_logger_does_not_attach_same_file_twice(tmp_path: Path) -> None:
    log_file = tmp_path / "run.log"
    get_logger("test_logger.file_dup", log_file=log_file)
    lg = get_logger("test_logger.file_dup", log_file=log_file)
    file_handlers = [h for h in lg.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1


def test_get_logger_creates_log_dir_if_missing(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested" / "dir" / "run.log"
    lg = get_logger("test_logger.mkdir", log_file=nested)
    lg.info("ok")
    _close_logger(lg)
    assert nested.exists()


def test_get_logger_propagate_default_false() -> None:
    lg = get_logger("test_logger.prop")
    assert lg.propagate is False


def test_get_logger_propagate_true() -> None:
    lg = get_logger("test_logger.prop_true", propagate=True)
    assert lg.propagate is True


def test_get_logger_use_color_explicit_false(tmp_path: Path) -> None:
    """显式 use_color=False 时控制台 handler 不应着色。"""
    lg = get_logger("test_logger.no_color", use_color=False)
    console = next(h for h in lg.handlers if isinstance(h, logging.StreamHandler)
                   and not isinstance(h, logging.FileHandler))
    assert isinstance(console.formatter, ColorFormatter)
    assert console.formatter.use_color is False


# ===========================================================================
# 2. ColorFormatter
# ===========================================================================


def test_color_formatter_with_color() -> None:
    fmt = ColorFormatter("%(message)s", use_color=True)
    rec = logging.LogRecord("x", logging.WARNING, "f", 1, "hello", None, None)
    out = fmt.format(rec)
    assert "\033[" in out      # 含 ANSI 转义
    assert "hello" in out
    assert out.endswith("\033[0m")


def test_color_formatter_without_color() -> None:
    fmt = ColorFormatter("%(message)s", use_color=False)
    rec = logging.LogRecord("x", logging.ERROR, "f", 1, "hi", None, None)
    out = fmt.format(rec)
    assert "\033[" not in out
    assert out == "hi"


def test_color_formatter_unknown_level_no_color() -> None:
    fmt = ColorFormatter("%(message)s", use_color=True)
    # 自定义级别（25 不在 _COLORS 中）
    rec = logging.LogRecord("x", 25, "f", 1, "msg", None, None)
    out = fmt.format(rec)
    # 未注册级别，不应着色
    assert out == "msg"


# ===========================================================================
# 3. log_array_stats
# ===========================================================================


def _capture_log(level: int = logging.DEBUG) -> tuple:
    """返回 (logger, handler, stream) 三元组用于捕获日志。"""
    stream = io.StringIO()
    h = logging.StreamHandler(stream)
    h.setFormatter(logging.Formatter("%(message)s"))
    h.setLevel(level)
    lg = logging.Logger("test_logger.array", level=level)
    lg.addHandler(h)
    return lg, h, stream


def test_log_array_stats_normal_array() -> None:
    lg, _, stream = _capture_log()
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    log_array_stats(lg, "M", arr, level=logging.INFO)
    out = stream.getvalue()
    assert "M:" in out
    assert "shape=(2, 2)" in out
    assert "min=1" in out
    assert "max=4" in out


def test_log_array_stats_empty_array() -> None:
    lg, _, stream = _capture_log(level=logging.INFO)
    log_array_stats(lg, "Empty", np.zeros((0, 3)), level=logging.INFO)
    assert "[empty]" in stream.getvalue()


def test_log_array_stats_with_nan() -> None:
    lg, _, stream = _capture_log(level=logging.INFO)
    arr = np.array([1.0, np.nan, 3.0])
    log_array_stats(lg, "WithNaN", arr, level=logging.INFO)
    assert "nan=1" in stream.getvalue()


def test_log_array_stats_non_ndarray() -> None:
    lg, _, stream = _capture_log(level=logging.INFO)
    log_array_stats(lg, "scalar", 3.14, level=logging.INFO)  # type: ignore[arg-type]
    assert "not ndarray" in stream.getvalue()


def test_log_array_stats_int_array() -> None:
    """整数数组无 NaN/Inf 检查路径。"""
    lg, _, stream = _capture_log(level=logging.INFO)
    log_array_stats(lg, "I", np.array([1, 2, 3], dtype=np.int64), level=logging.INFO)
    out = stream.getvalue()
    assert "shape=(3,)" in out
    assert "nan" not in out  # 整型不会触发 nan 警告


# ===========================================================================
# 4. ExperimentRecorder：基础生命周期
# ===========================================================================


def test_recorder_creates_directory(tmp_path: Path) -> None:
    out = tmp_path / "exp1"
    rec = ExperimentRecorder(out_dir=out)
    assert out.exists() and out.is_dir()
    assert rec.jsonl_path == out / "metrics.jsonl"
    assert rec.csv_path == out / "metrics.csv"


def test_recorder_log_step_writes_jsonl_and_csv(tmp_path: Path) -> None:
    rec = ExperimentRecorder(out_dir=tmp_path / "exp")
    rec.log_step(method="IMF-BLS", step=0, accuracy=0.9, train_time=1.23)
    rec.log_step(method="IMF-BLS", step=1, accuracy=0.95, train_time=1.4)

    # JSONL 内容
    lines = rec.jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    assert rec0["method"] == "IMF-BLS"
    assert rec0["step"] == 0
    assert rec0["accuracy"] == 0.9
    assert "timestamp" in rec0 and "elapsed_s" in rec0

    # CSV 内容
    with open(rec.csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["method"] == "IMF-BLS"
    assert float(rows[1]["accuracy"]) == 0.95


def test_recorder_log_config(tmp_path: Path) -> None:
    rec = ExperimentRecorder(out_dir=tmp_path / "exp", experiment="my_exp")
    rec.log_config({"lambda": 1e-3, "n_batches": 5, "seed": 0})
    payload = json.loads(rec.config_path.read_text(encoding="utf-8"))
    assert payload["experiment"] == "my_exp"
    assert payload["config"]["lambda"] == 1e-3
    assert payload["config"]["n_batches"] == 5


def test_recorder_numpy_types_serialized(tmp_path: Path) -> None:
    """numpy.float64 / int64 / ndarray 都应被 json 安全序列化。"""
    rec = ExperimentRecorder(out_dir=tmp_path / "exp_np")
    rec.log_step(
        method="X",
        step=np.int64(3),
        score=np.float64(0.987),
        weights=np.array([1.0, 2.0, 3.0]),
    )
    line = rec.jsonl_path.read_text(encoding="utf-8").strip()
    obj = json.loads(line)
    assert obj["step"] == 3
    assert abs(obj["score"] - 0.987) < 1e-9
    assert obj["weights"] == [1.0, 2.0, 3.0]


def test_recorder_records_in_memory(tmp_path: Path) -> None:
    rec = ExperimentRecorder(out_dir=tmp_path / "exp_mem")
    rec.log_step(method="A", step=0, m=1.0)
    rec.log_step(method="B", step=0, m=2.0)
    assert len(rec.records) == 2
    # 返回副本，外部修改不应影响内部
    rec.records.append({"x": 1})
    assert len(rec.records) == 2


# ===========================================================================
# 5. ExperimentRecorder：恢复/overwrite
# ===========================================================================


def test_recorder_recovers_existing_records(tmp_path: Path) -> None:
    out = tmp_path / "exp_recover"
    rec1 = ExperimentRecorder(out_dir=out)
    rec1.log_step(method="A", step=0, m=0.5)
    rec1.log_step(method="A", step=1, m=0.7)

    # 重新初始化（不 overwrite）应能加载已有记录
    rec2 = ExperimentRecorder(out_dir=out)
    assert len(rec2.records) == 2
    assert rec2.records[0]["method"] == "A"
    # 继续追加
    rec2.log_step(method="A", step=2, m=0.8)
    assert len(rec2.records) == 3

    # 文件中也确实是 3 行
    lines = rec2.jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


def test_recorder_overwrite_clears_old_files(tmp_path: Path) -> None:
    out = tmp_path / "exp_ow"
    r1 = ExperimentRecorder(out_dir=out)
    r1.log_step(method="A", step=0, m=1.0)
    assert r1.jsonl_path.exists()

    r2 = ExperimentRecorder(out_dir=out, overwrite=True)
    assert r2.records == []
    # JSONL/CSV 应被清空（log_step 还未调用）
    assert (not r2.jsonl_path.exists()) or r2.jsonl_path.read_text() == ""


# ===========================================================================
# 6. ExperimentRecorder：CSV 列扩展 (新键自动加入表头)
# ===========================================================================


def test_recorder_csv_header_grows_with_new_keys(tmp_path: Path) -> None:
    rec = ExperimentRecorder(out_dir=tmp_path / "exp_csv")
    rec.log_step(method="A", step=0, accuracy=0.5)
    rec.log_step(method="A", step=1, accuracy=0.6, extra="foo")  # 新键 extra

    with open(rec.csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    assert "extra" in fieldnames
    assert len(rows) == 2
    # 第 1 行 extra 列为空
    assert rows[0].get("extra", "") == ""
    assert rows[1]["extra"] == "foo"


# ===========================================================================
# 7. ExperimentRecorder：filter / summarize / save_summary
# ===========================================================================


def test_recorder_filter(tmp_path: Path) -> None:
    rec = ExperimentRecorder(out_dir=tmp_path / "exp_filter")
    rec.log_step(method="A", step=0, m=1.0)
    rec.log_step(method="A", step=1, m=2.0)
    rec.log_step(method="B", step=0, m=10.0)

    a = rec.filter(method="A")
    assert len(a) == 2
    assert all(r["method"] == "A" for r in a)

    b = rec.filter(method="B")
    assert len(b) == 1


def test_recorder_summarize_basic(tmp_path: Path) -> None:
    rec = ExperimentRecorder(out_dir=tmp_path / "exp_sum")
    for i, v in enumerate([0.5, 0.7, 0.9]):
        rec.log_step(method="IMF", step=i, accuracy=v, train_time=0.1 * (i + 1))
    rec.log_step(method="RIB", step=0, accuracy=0.4, train_time=0.05)

    summary = rec.summarize(group_by="method")
    assert set(summary) == {"IMF", "RIB"}
    assert summary["IMF"]["count"] == 3
    assert abs(summary["IMF"]["accuracy"]["mean"] - 0.7) < 1e-9
    assert abs(summary["IMF"]["accuracy"]["max"] - 0.9) < 1e-9
    assert abs(summary["IMF"]["accuracy"]["last"] - 0.9) < 1e-9
    assert summary["RIB"]["count"] == 1


def test_recorder_summarize_skips_step_field(tmp_path: Path) -> None:
    """step 不应被聚合（discard）。"""
    rec = ExperimentRecorder(out_dir=tmp_path / "exp_sum2")
    rec.log_step(method="A", step=0, m=1.0)
    rec.log_step(method="A", step=1, m=2.0)
    s = rec.summarize()
    assert "step" not in s["A"]


def test_recorder_save_summary_writes_json(tmp_path: Path) -> None:
    rec = ExperimentRecorder(out_dir=tmp_path / "exp_save_sum")
    rec.log_step(method="A", step=0, score=1.0)
    rec.log_step(method="A", step=1, score=2.0)
    summary = rec.save_summary()
    assert rec.summary_path.exists()
    payload = json.loads(rec.summary_path.read_text(encoding="utf-8"))
    assert payload["group_by"] == "method"
    assert "A" in payload["summary"]
    assert payload["summary"]["A"]["count"] == 2
    # 返回值与文件内容一致
    assert summary["A"]["count"] == 2


# ===========================================================================
# 8. ExperimentRecorder：stage 上下文管理器
# ===========================================================================


def test_recorder_stage_records_method_step_train_time(tmp_path: Path) -> None:
    rec = ExperimentRecorder(out_dir=tmp_path / "exp_stage")
    with rec.stage(method="A", step=0) as bag:
        bag["accuracy"] = 0.8
    assert len(rec.records) == 1
    r = rec.records[0]
    assert r["method"] == "A"
    assert r["step"] == 0
    assert r["accuracy"] == 0.8
    assert "train_time" in r and r["train_time"] >= 0.0


def test_recorder_stage_records_even_on_exception(tmp_path: Path) -> None:
    """stage 内抛异常时仍应记录已收集的指标 + train_time。"""
    rec = ExperimentRecorder(out_dir=tmp_path / "exp_stage_err")
    with pytest.raises(RuntimeError):
        with rec.stage(method="X", step=5) as bag:
            bag["partial"] = 1
            raise RuntimeError("boom")
    assert len(rec.records) == 1
    assert rec.records[0]["partial"] == 1
    assert rec.records[0]["method"] == "X"


def test_recorder_stage_logs_to_logger(tmp_path: Path) -> None:
    rec = ExperimentRecorder(out_dir=tmp_path / "exp_stage_log")
    stream = io.StringIO()
    lg = logging.Logger("test_logger.stage", level=logging.INFO)
    h = logging.StreamHandler(stream)
    h.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(h)

    with rec.stage(method="IMF", step=2, logger=lg) as bag:
        bag["accuracy"] = 0.99
    out = stream.getvalue()
    assert "IMF" in out
    assert "step=2" in out
    assert "accuracy=" in out


# ===========================================================================
# 9. ExperimentRecorder：attach_logger
# ===========================================================================


def test_recorder_attach_logger_writes_to_run_log(tmp_path: Path) -> None:
    rec = ExperimentRecorder(out_dir=tmp_path / "exp_attach", experiment="exp_attach")
    lg = rec.attach_logger()
    lg.info("test message inside experiment")
    _close_logger(lg)

    assert rec.log_path.exists()
    text = rec.log_path.read_text(encoding="utf-8")
    assert "test message inside experiment" in text


def test_recorder_attach_logger_default_name(tmp_path: Path) -> None:
    rec = ExperimentRecorder(out_dir=tmp_path / "exp_attach2", experiment="alpha")
    lg = rec.attach_logger()
    assert lg.name == "imf_bls.alpha"
    _close_logger(lg)


# ===========================================================================
# 10. 工具函数（_to_json_safe / _collect_numeric_keys / _fmt_value）
# ===========================================================================


def test_to_json_safe_handles_path_and_nested() -> None:
    from utils.logger import _to_json_safe
    obj = {
        "p": Path("/tmp/xxx"),
        "arr": np.array([1, 2]),
        "nested": [np.float64(1.5), {"x": np.int32(7)}],
        "tuple": (1, 2, 3),
    }
    safe = _to_json_safe(obj)
    # 应能被 json.dumps 序列化
    json.dumps(safe)
    assert safe["arr"] == [1, 2]
    assert safe["nested"][0] == 1.5
    assert safe["tuple"] == [1, 2, 3]


def test_fmt_value_and_collect_numeric_keys() -> None:
    from utils.logger import _collect_numeric_keys, _fmt_value

    # _fmt_value
    assert _fmt_value(0.123) == "0.1230"
    assert "e" in _fmt_value(1.23e10)
    assert _fmt_value("abc") == "abc"

    # _collect_numeric_keys：bool 不计为数值；step 被 discard
    keys = _collect_numeric_keys([
        {"step": 0, "acc": 0.9, "ok": True, "name": "A"},
        {"step": 1, "acc": 0.8, "ok": False, "extra": 3},
    ])
    assert "acc" in keys
    assert "extra" in keys
    assert "step" not in keys
    assert "ok" not in keys     # bool 不算
    assert "name" not in keys   # str 不算
