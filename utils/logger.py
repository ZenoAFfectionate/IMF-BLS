# -*- coding: utf-8 -*-
"""IMF-BLS 日志与实验记录工具。

本模块提供两类能力：

1. ``get_logger``    — 统一封装的 Python ``logging``：
   * 控制台彩色输出（自动检测 TTY）
   * 可选写入文件
   * 通过 ``name`` 复用 logger，避免重复挂载 handler

2. ``ExperimentRecorder`` — 结构化实验记录器：
   * 增量记录每步指标（``log_step``）
   * 自动落盘 ``metrics.jsonl`` + ``metrics.csv``
   * 支持读取已有记录、按方法/阶段聚合统计
   * 复用同一目录可继续追加（断点恢复）

3. ``log_array_stats`` — 记录 numpy 数组的诊断信息（形状/范数/极值）

模块仅依赖 Python 标准库与 numpy。
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Union

import numpy as np

__all__ = [
    "get_logger",
    "ExperimentRecorder",
    "log_array_stats",
    "ColorFormatter",
]


# ---------------------------------------------------------------------------
# 1. 彩色日志 Formatter
# ---------------------------------------------------------------------------


class ColorFormatter(logging.Formatter):
    """带 ANSI 颜色的日志 Formatter。

    - DEBUG / INFO / WARNING / ERROR / CRITICAL 各自不同颜色
    - 非 TTY 环境自动降级为无色
    """

    _COLORS: Dict[int, str] = {
        logging.DEBUG: "\033[37m",     # gray
        logging.INFO: "\033[36m",      # cyan
        logging.WARNING: "\033[33m",   # yellow
        logging.ERROR: "\033[31m",     # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }
    _RESET = "\033[0m"

    def __init__(self, fmt: str, datefmt: Optional[str] = None,
                 use_color: bool = True) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        msg = super().format(record)
        if not self.use_color:
            return msg
        color = self._COLORS.get(record.levelno, "")
        return f"{color}{msg}{self._RESET}" if color else msg


# ---------------------------------------------------------------------------
# 2. get_logger
# ---------------------------------------------------------------------------


_DEFAULT_FMT = "[%(asctime)s] %(levelname)-7s %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str = "imf_bls",
    *,
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    fmt: str = _DEFAULT_FMT,
    datefmt: str = _DEFAULT_DATEFMT,
    use_color: Optional[bool] = None,
    propagate: bool = False,
) -> logging.Logger:
    """获取（或创建）一个统一配置的 Logger。

    特性：
        * 同一 ``name`` 重复调用返回同一 Logger（不会重复挂载 handler）
        * 默认控制台输出 + 可选追加文件输出
        * 控制台 handler 自动彩色（可通过 ``use_color`` 强制开/关）

    Args:
        name:        Logger 名（建议形如 ``"imf_bls.experiment"``）
        level:       日志级别（int 或 "INFO"/"DEBUG" 等字符串）
        log_file:    若提供，则同时写入该文件（追加模式 utf-8）
        fmt:         日志格式
        datefmt:     时间格式
        use_color:   控制台是否上色；None 时自动检测 isatty
        propagate:   是否向上传播到 root logger（默认 False，避免重复打印）

    Returns:
        配置好的 :class:`logging.Logger` 实例。
    """
    logger = logging.getLogger(name)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = propagate

    # 检查是否已挂载等价 handler，避免重复
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_console:
        ch = logging.StreamHandler(stream=sys.stderr)
        if use_color is None:
            use_color = bool(getattr(sys.stderr, "isatty", lambda: False)())
        ch.setFormatter(ColorFormatter(fmt, datefmt, use_color=use_color))
        ch.setLevel(level)
        logger.addHandler(ch)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # 避免对同一文件挂载多个 FileHandler
        already = any(
            isinstance(h, logging.FileHandler)
            and Path(h.baseFilename).resolve() == log_file.resolve()
            for h in logger.handlers
        )
        if not already:
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setFormatter(logging.Formatter(fmt, datefmt))  # 文件不上色
            fh.setLevel(level)
            logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# 3. log_array_stats
# ---------------------------------------------------------------------------


def log_array_stats(
    logger: logging.Logger,
    name: str,
    arr: np.ndarray,
    *,
    level: int = logging.DEBUG,
) -> None:
    """记录 numpy 数组的关键诊断统计（shape / dtype / 范数 / 极值 / NaN 数）。

    主要用于调试 IMF-BLS 内部的 R / V / W 等矩阵。
    """
    if not isinstance(arr, np.ndarray):
        logger.log(level, "%s: not ndarray (type=%s)", name, type(arr).__name__)
        return
    if arr.size == 0:
        logger.log(level, "%s: shape=%s dtype=%s [empty]", name, arr.shape, arr.dtype)
        return

    n_nan = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
    n_inf = int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
    msg = (
        f"{name}: shape={tuple(arr.shape)} dtype={arr.dtype} "
        f"min={float(np.nanmin(arr)):.4e} max={float(np.nanmax(arr)):.4e} "
        f"mean={float(np.nanmean(arr)):.4e} ‖·‖_F={float(np.linalg.norm(arr)):.4e}"
    )
    if n_nan or n_inf:
        msg += f"  ⚠ nan={n_nan} inf={n_inf}"
    logger.log(level, msg)


# ---------------------------------------------------------------------------
# 4. ExperimentRecorder
# ---------------------------------------------------------------------------


@dataclass
class ExperimentRecorder:
    """结构化实验记录器：把每一步指标写入 JSONL + CSV，便于事后分析。

    典型用法::

        rec = ExperimentRecorder(out_dir="results/equal_scale/digits",
                                 experiment="equal_scale@digits")
        rec.log_config({"n_batches": 5, "lambda": 1e-3})
        for step, batch in enumerate(batches):
            ...
            rec.log_step(method="IMF-BLS", step=step,
                         accuracy=acc, train_time=t)
        rec.save_summary()

    输出文件::

        out_dir/metrics.jsonl   一行一条记录（结构化、易读）
        out_dir/metrics.csv     扁平 CSV（适合 Excel / pandas）
        out_dir/config.json     log_config 写入的元数据
        out_dir/summary.json    save_summary 自动聚合的 mean/std/last
        out_dir/run.log         （可选）若传 attach_logger 则同步落盘
    """

    out_dir: Union[str, Path]
    experiment: str = "experiment"
    overwrite: bool = False  # True 时初始化清空旧记录
    _records: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _csv_keys: List[str] = field(default_factory=list, init=False)
    _start_time: float = field(default_factory=time.time, init=False)

    # ---- 路径属性 ----

    @property
    def jsonl_path(self) -> Path:
        return Path(self.out_dir) / "metrics.jsonl"

    @property
    def csv_path(self) -> Path:
        return Path(self.out_dir) / "metrics.csv"

    @property
    def config_path(self) -> Path:
        return Path(self.out_dir) / "config.json"

    @property
    def summary_path(self) -> Path:
        return Path(self.out_dir) / "summary.json"

    @property
    def log_path(self) -> Path:
        return Path(self.out_dir) / "run.log"

    # ---- 初始化 ----

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.overwrite:
            for p in (self.jsonl_path, self.csv_path, self.summary_path):
                if p.exists():
                    p.unlink()
        else:
            # 恢复之前的记录
            if self.jsonl_path.exists():
                self._records = self._load_jsonl(self.jsonl_path)
                if self._records:
                    self._csv_keys = sorted({k for r in self._records for k in r})

    # ---- 元数据 ----

    def log_config(self, config: Mapping[str, Any]) -> None:
        """把实验配置写入 ``config.json``。"""
        payload = {
            "experiment": self.experiment,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": _to_json_safe(dict(config)),
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # ---- 单步记录 ----

    def log_step(self, **fields: Any) -> Dict[str, Any]:
        """追加一条记录，返回写入后的字典（含自动添加的 timestamp / elapsed）。

        建议至少包含：``method``、``step``、以及若干指标键。
        所有 numpy 标量会被转换为 Python 原生类型。
        """
        record: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": round(time.time() - self._start_time, 6),
        }
        record.update(_to_json_safe(fields))
        self._records.append(record)

        # 增量写 JSONL
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 增量写 CSV：若发现新键则重写表头
        new_keys = sorted({k for k in record} | set(self._csv_keys))
        if new_keys != self._csv_keys:
            self._csv_keys = new_keys
            self._rewrite_csv()
        else:
            with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_keys, extrasaction="ignore")
                writer.writerow(record)

        return record

    # ---- 查询/聚合 ----

    @property
    def records(self) -> List[Dict[str, Any]]:
        """返回所有已记录的副本。"""
        return list(self._records)

    def filter(self, **conditions: Any) -> List[Dict[str, Any]]:
        """按字段值过滤记录，例如 ``filter(method="IMF-BLS")``。"""
        return [
            r for r in self._records
            if all(r.get(k) == v for k, v in conditions.items())
        ]

    def summarize(self, group_by: str = "method") -> Dict[str, Dict[str, Any]]:
        """按 ``group_by`` 字段分组，对所有数值字段计算 mean/std/min/max/last。"""
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in self._records:
            key = str(r.get(group_by, "<none>"))
            groups.setdefault(key, []).append(r)

        summary: Dict[str, Dict[str, Any]] = {}
        for key, recs in groups.items():
            num_keys = _collect_numeric_keys(recs)
            stats: Dict[str, Any] = {"count": len(recs)}
            for nk in num_keys:
                values = [float(r[nk]) for r in recs if isinstance(r.get(nk), (int, float))]
                if not values:
                    continue
                arr = np.asarray(values, dtype=float)
                stats[nk] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=0)),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "last": float(arr[-1]),
                }
            summary[key] = stats
        return summary

    def save_summary(self, group_by: str = "method") -> Dict[str, Dict[str, Any]]:
        """生成 summary 并写入 ``summary.json``。"""
        summary = self.summarize(group_by=group_by)
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {"experiment": self.experiment, "group_by": group_by, "summary": summary},
                f, ensure_ascii=False, indent=2,
            )
        return summary

    # ---- 与 logger 整合 ----

    def attach_logger(
        self,
        name: Optional[str] = None,
        *,
        level: Union[int, str] = logging.INFO,
    ) -> logging.Logger:
        """返回一个同时输出到 ``run.log`` 的 logger。"""
        logger_name = name or f"imf_bls.{self.experiment}"
        return get_logger(logger_name, level=level, log_file=self.log_path)

    @contextmanager
    def stage(
        self,
        method: str,
        step: int,
        logger: Optional[logging.Logger] = None,
    ) -> Iterator[Dict[str, Any]]:
        """方便地"包一段代码 → 记录耗时与指标"。

        在 ``with`` 块内向 yield 出来的 dict 写入指标键，退出时自动:
            * 加入 ``method`` / ``step`` / ``train_time``
            * 调用 ``log_step``
            * 若提供 logger 则同步打印一行人类可读摘要
        """
        bag: Dict[str, Any] = {}
        t0 = time.perf_counter()
        try:
            yield bag
        finally:
            elapsed = time.perf_counter() - t0
            bag.setdefault("method", method)
            bag.setdefault("step", step)
            bag["train_time"] = round(elapsed, 6)
            self.log_step(**bag)
            if logger is not None:
                pretty = " | ".join(
                    f"{k}={_fmt_value(v)}" for k, v in bag.items()
                    if k not in ("method", "step")
                )
                logger.info("[%s] step=%d %s", method, step, pretty)

    # ---- 内部 ----

    def _rewrite_csv(self) -> None:
        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_keys, extrasaction="ignore")
            writer.writeheader()
            for r in self._records:
                writer.writerow(r)

    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out


# ---------------------------------------------------------------------------
# 工具：JSON 友好转换 / 类型推断
# ---------------------------------------------------------------------------


def _to_json_safe(obj: Any) -> Any:
    """递归把对象转成 JSON 友好形式（numpy → python 原生）。"""
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)  # 兜底


def _collect_numeric_keys(records: Sequence[Mapping[str, Any]]) -> List[str]:
    keys: set = set()
    for r in records:
        for k, v in r.items():
            if isinstance(v, bool):
                continue  # 布尔不视为数值
            if isinstance(v, (int, float)):
                keys.add(k)
    keys.discard("step")  # step 通常不需要 mean/std
    return sorted(keys)


def _fmt_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1e4 else f"{v:.4e}"
    return str(v)
