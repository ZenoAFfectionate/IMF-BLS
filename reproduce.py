# -*- coding: utf-8 -*-
"""复现论文实验的统一入口（论文 Section 4）。

复现的论文表格
==============

* **Table 5** (分类，equal-scale 数据流): LED / CIFAR10 / CIFAR100 / Waveform /
  Letter / Pendigits / Shuttle 上的最终准确率与总训练时间。
  → ``reproduce.py table5``
* **Table 6** (回归): Abalone / Bodyfat / Energy Eff. / Weather Izmir /
  Appliances Energy 上的 RMSE 与训练时间（每个数据集运行 ``--repeats`` 次取均值）。
  → ``reproduce.py table6``
* **Table 7** (分类，uncertain-scale 数据流): MNIST / Fashion-MNIST / EMNIST /
  NORB 在 5/10/15/20/25 个不均匀块上的 mean ± std。
  → ``reproduce.py table7``

输出
====

每个实验都会输出到 ``results/reproduce/<table_id>/<dataset>/``：

    metrics.json                - 详细指标（per-step / per-method / std）
    report.md                   - 该数据集的精美 markdown 报告
    runs/run.log                - 完整日志
    runs/steps.jsonl            - 结构化步骤记录

汇总后会自动产出 ``results/reproduce/<table_id>/summary.md``，
包含与论文同样格式的多数据集对比表。

使用
====

::

    # 单数据集
    python reproduce.py table5 --dataset pendigits
    python reproduce.py table6 --dataset abalone --repeats 5
    python reproduce.py table7 --dataset mnist --mnist_path data/mnist

    # 一键全量
    python reproduce.py all --skip mnist,emnist,norb,cifar10,cifar100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.baselines import (  # noqa: E402
    ApproximationMethodBLS,
    IncrementalBLS,
    RIBLS,
    TiBLS,
)
from src.bls_base import BLSConfig, NonIncrementalBLS  # noqa: E402
from src.imf_bls import IMFBLS  # noqa: E402
from utils.data import (  # noqa: E402
    load_classification_dataset,
    load_regression_dataset,
    one_hot_encode,
    split_into_batches,
    split_random_batches,
)
from utils.feature_layer import standardize_minmax  # noqa: E402
from utils.logger import ExperimentRecorder, get_logger  # noqa: E402
from utils.metrics import classification_accuracy, regression_rmse  # noqa: E402
from utils.paper_presets import get_preset, list_presets  # noqa: E402
from utils.timing import Timer  # noqa: E402


_GLOBAL_LOGGER = get_logger("reproduce", level="INFO")


# ===========================================================================
# 共用工具
# ===========================================================================


def _make_config(preset, seed: Optional[int] = None) -> BLSConfig:
    kwargs = dict(preset.bls_kwargs)
    if seed is not None:
        kwargs["seed"] = int(seed)
    return BLSConfig(**kwargs)


def _prepare_classification(name: str, mnist_path: Optional[str],
                            subsample_train: Optional[int] = None,
                            subsample_test: Optional[int] = None):
    """加载分类数据集并返回 ``(X_tr, Y_tr, X_te, Y_te, n_classes, preset)``。

    Args:
        subsample_train / subsample_test: 截取前 N 个训练 / 测试样本（用于快速验证）。
    """
    preset = get_preset(name)
    if name == "synthetic_classification":
        from utils.data import make_synthetic_classification
        X_tr, y_tr, X_te, y_te = make_synthetic_classification(
            n_train=preset.n_train or 1000,
            n_test=preset.n_test or 200,
            seed=preset.bls_kwargs.get("seed", 0),
        )
    elif name in {"mnist", "fashion_mnist", "emnist"}:
        X_tr, y_tr, X_te, y_te = load_classification_dataset(name, path=mnist_path)
    else:
        X_tr, y_tr, X_te, y_te = load_classification_dataset(name)

    if subsample_train and len(X_tr) > subsample_train:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X_tr), size=subsample_train, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
    if subsample_test and len(X_te) > subsample_test:
        rng = np.random.default_rng(1)
        idx = rng.choice(len(X_te), size=subsample_test, replace=False)
        X_te, y_te = X_te[idx], y_te[idx]

    # 归一化
    X_tr, X_te, _ = standardize_minmax(X_tr, X_te)
    n_classes = int(max(int(y_tr.max()), int(y_te.max())) + 1)
    Y_tr = one_hot_encode(y_tr, num_classes=n_classes)
    Y_te = one_hot_encode(y_te, num_classes=n_classes)
    return X_tr, Y_tr, X_te, Y_te, n_classes, preset


def _prepare_regression(name: str):
    preset = get_preset(name)
    if name == "synthetic_regression":
        from utils.data import make_synthetic_regression
        X_tr, y_tr, X_te, y_te = make_synthetic_regression(
            n_train=preset.n_train or 600,
            n_test=preset.n_test or 200,
            seed=preset.bls_kwargs.get("seed", 0),
        )
    else:
        X_tr, y_tr, X_te, y_te = load_regression_dataset(name)
    X_tr, X_te, _ = standardize_minmax(X_tr, X_te)
    return X_tr, y_tr.reshape(-1, 1), X_te, y_te.reshape(-1, 1), preset


def _eval_classification(model, X, Y) -> float:
    return classification_accuracy(Y, model.predict(X))


def _eval_regression(model, X, y) -> float:
    return regression_rmse(y, model.predict(X))


def _run_method(
    cls,
    cfg: BLSConfig,
    batches: List[Tuple[np.ndarray, np.ndarray]],
    X_te: np.ndarray,
    Y_te: np.ndarray,
    eval_fn,
    require_equal_batch: bool = False,
) -> Dict[str, Any]:
    """跑一个方法的完整流式过程，返回每步指标与时间。

    Returns:
        ``{"metric_per_step": [...], "time_per_step": [...]}``
    """
    if require_equal_batch:
        common = min(len(b[0]) for b in batches)
        batches = [(X[:common], Y[:common]) for X, Y in batches]

    metrics_per_step: List[float] = []
    times_per_step: List[float] = []

    model = cls(config=cfg)
    with Timer() as t:
        model.fit_initial(*batches[0])
    metrics_per_step.append(eval_fn(model, X_te, Y_te))
    times_per_step.append(t.elapsed)

    for X_b, Y_b in batches[1:]:
        with Timer() as t:
            model.add_data(X_b, Y_b)
        metrics_per_step.append(eval_fn(model, X_te, Y_te))
        times_per_step.append(t.elapsed)

    return {
        "metric_per_step": metrics_per_step,
        "time_per_step": times_per_step,
        "final_metric": metrics_per_step[-1],
        "total_time": float(sum(times_per_step)),
    }


# ===========================================================================
# Table 5：分类，等量数据流
# ===========================================================================


def reproduce_table5(
    dataset: str,
    output_dir: Path,
    seed: int = 0,
    mnist_path: Optional[str] = None,
    subsample_train: Optional[int] = None,
    subsample_test: Optional[int] = None,
) -> Dict[str, Any]:
    """复现论文 Table 5（分类等量数据流）。"""
    target = output_dir / "table5" / dataset
    rec = ExperimentRecorder(out_dir=target,
                             experiment=f"table5@{dataset}",
                             overwrite=True)
    log = rec.attach_logger(level="INFO")
    log.info("=" * 65)
    log.info("Reproducing paper Table 5 | dataset=%s", dataset)
    log.info("=" * 65)

    X_tr, Y_tr, X_te, Y_te, n_classes, preset = _prepare_classification(
        dataset, mnist_path, subsample_train, subsample_test
    )
    log.info("Train=%d, Test=%d, classes=%d, batches=%d",
             len(X_tr), len(X_te), n_classes, preset.equal_scale_n_batches)

    rec.log_config({
        "table": "5",
        "dataset": dataset,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_classes": n_classes,
        "n_batches": preset.equal_scale_n_batches,
        "seed": seed,
        "bls": preset.bls_kwargs,
    })

    batches = split_into_batches(X_tr, Y_tr,
                                 n_batches=preset.equal_scale_n_batches,
                                 shuffle=True, seed=seed)

    methods = {
        "Non-Incremental BLS":   (NonIncrementalBLS,        False, "joint"),
        "Incremental BLS":       (IncrementalBLS,           False, "stream"),
        "TiBLS":                 (TiBLS,                    True,  "stream"),
        "Approximation Method":  (ApproximationMethodBLS,   False, "stream"),
        "RI-BLS":                (RIBLS,                    False, "stream"),
        "IMF-BLS (Ours)":       (IMFBLS,                   False, "stream"),
    }

    results: Dict[str, Any] = {}
    for name, (cls, require_equal, mode) in methods.items():
        try:
            cfg = _make_config(preset, seed=seed)
            if mode == "joint":
                # 联合训练：一次性 fit
                with Timer() as t:
                    model = cls(config=cfg).fit_initial(X_tr, Y_tr)
                acc = _eval_classification(model, X_te, Y_te)
                info = {
                    "metric_per_step": [acc],
                    "time_per_step":   [t.elapsed],
                    "final_metric":    acc,
                    "total_time":      float(t.elapsed),
                }
            else:
                info = _run_method(cls, cfg, batches, X_te, Y_te,
                                   _eval_classification,
                                   require_equal_batch=require_equal)
            info["metric_name"] = "accuracy"
            results[name] = info
            log.info("  %-22s | final acc = %.4f | total = %.4fs",
                     name, info["final_metric"], info["total_time"])
            rec.log_step(method=name, final_metric=info["final_metric"],
                         total_time=info["total_time"])
        except Exception as e:
            log.warning("  %s 失败: %s", name, e)
            results[name] = {"error": str(e)}

    # 落盘 + 报告
    rec.save_summary(group_by="method")
    _save_metrics(target, results)
    _write_table5_report(target, dataset, results, preset, n_classes,
                         len(X_tr), len(X_te))
    return results


# ===========================================================================
# Table 6：回归
# ===========================================================================


def reproduce_table6(
    dataset: str,
    output_dir: Path,
    repeats: int = 5,
    seed: int = 0,
) -> Dict[str, Any]:
    """复现论文 Table 6（回归任务）。

    每个数据集运行 ``repeats`` 次（不同 seed），输出 train_time / test_rmse / std。
    """
    target = output_dir / "table6" / dataset
    rec = ExperimentRecorder(out_dir=target,
                             experiment=f"table6@{dataset}",
                             overwrite=True)
    log = rec.attach_logger(level="INFO")
    log.info("=" * 65)
    log.info("Reproducing paper Table 6 | dataset=%s | repeats=%d",
             dataset, repeats)
    log.info("=" * 65)

    X_tr, Y_tr, X_te, Y_te, preset = _prepare_regression(dataset)
    log.info("Train=%d, Test=%d, attrs=%d, target shape=%s",
             len(X_tr), len(X_te), X_tr.shape[1], Y_tr.shape)

    rec.log_config({
        "table": "6",
        "dataset": dataset,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "attributes": int(X_tr.shape[1]),
        "repeats": repeats,
        "seed": seed,
        "bls": preset.bls_kwargs,
    })

    methods = {
        "Non-Incremental BLS":   (NonIncrementalBLS,        False, "joint"),
        "Incremental BLS":       (IncrementalBLS,           False, "stream"),
        "Approximation Method":  (ApproximationMethodBLS,   False, "stream"),
        "TiBLS":                 (TiBLS,                    True,  "stream"),
        "RI-BLS":                (RIBLS,                    False, "stream"),
        "IMF-BLS (Ours)":       (IMFBLS,                   False, "stream"),
    }

    summary: Dict[str, Any] = {}
    for name, (cls, require_equal, mode) in methods.items():
        rmses = []
        times = []
        for r in range(repeats):
            try:
                cfg = _make_config(preset, seed=seed + r * 1000)
                if mode == "joint":
                    with Timer() as t:
                        model = cls(config=cfg).fit_initial(X_tr, Y_tr)
                    rmse = _eval_regression(model, X_te, Y_te)
                    elapsed = t.elapsed
                else:
                    batches = split_into_batches(
                        X_tr, Y_tr,
                        n_batches=preset.equal_scale_n_batches,
                        shuffle=True, seed=seed + r * 1000,
                    )
                    if require_equal:
                        common = min(len(b[0]) for b in batches)
                        batches = [(X[:common], Y[:common]) for X, Y in batches]

                    model = cls(config=cfg)
                    with Timer() as t:
                        model.fit_initial(*batches[0])
                        for X_b, Y_b in batches[1:]:
                            model.add_data(X_b, Y_b)
                    rmse = _eval_regression(model, X_te, Y_te)
                    elapsed = t.elapsed
                rmses.append(rmse)
                times.append(elapsed)
                rec.log_step(method=name, repeat=r, rmse=rmse, time=elapsed)
            except Exception as e:
                log.warning("  %s repeat=%d 失败: %s", name, r, e)

        if rmses:
            summary[name] = {
                "rmse_mean":  float(np.mean(rmses)),
                "rmse_std":   float(np.std(rmses)),
                "time_mean":  float(np.mean(times)),
                "time_std":   float(np.std(times)),
                "rmses":      rmses,
                "times":      times,
            }
            log.info("  %-22s | RMSE = %.4f ± %.4f | time = %.4f ± %.4fs",
                     name, summary[name]["rmse_mean"], summary[name]["rmse_std"],
                     summary[name]["time_mean"], summary[name]["time_std"])
        else:
            summary[name] = {"error": "all runs failed"}

    rec.save_summary(group_by="method")
    _save_metrics(target, summary)
    _write_table6_report(target, dataset, summary, preset,
                         len(X_tr), len(X_te), X_tr.shape[1])
    return summary


# ===========================================================================
# Table 7：分类，uncertain-scale 数据流
# ===========================================================================


def reproduce_table7(
    dataset: str,
    output_dir: Path,
    seed: int = 0,
    mnist_path: Optional[str] = None,
    n_batches_choices: Optional[List[int]] = None,
    n_repeats: Optional[int] = None,
    subsample_train: Optional[int] = None,
    subsample_test: Optional[int] = None,
) -> Dict[str, Any]:
    """复现论文 Table 7（分类不定数据流）。"""
    target = output_dir / "table7" / dataset
    rec = ExperimentRecorder(out_dir=target,
                             experiment=f"table7@{dataset}",
                             overwrite=True)
    log = rec.attach_logger(level="INFO")
    log.info("=" * 65)
    log.info("Reproducing paper Table 7 | dataset=%s", dataset)
    log.info("=" * 65)

    X_tr, Y_tr, X_te, Y_te, n_classes, preset = _prepare_classification(
        dataset, mnist_path, subsample_train, subsample_test
    )
    n_batches_choices = n_batches_choices or preset.uncertain_scale_n_batches
    n_repeats = n_repeats or preset.uncertain_scale_repeats

    log.info("Train=%d, Test=%d, n_batches=%s, repeats=%d",
             len(X_tr), len(X_te), n_batches_choices, n_repeats)

    rec.log_config({
        "table": "7",
        "dataset": dataset,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_classes": n_classes,
        "n_batches_choices": list(n_batches_choices),
        "n_repeats": n_repeats,
        "seed": seed,
        "bls": preset.bls_kwargs,
    })

    methods = {
        "Incremental BLS":       IncrementalBLS,
        "Approximation Method":  ApproximationMethodBLS,
        "RI-BLS":                RIBLS,
        "IMF-BLS (Ours)":       IMFBLS,
    }
    # 也跑一次联合训练作为参考准确率（表中常给）
    cfg_joint = _make_config(preset, seed=seed)
    with Timer() as t:
        joint = NonIncrementalBLS(config=cfg_joint).fit_initial(X_tr, Y_tr)
    joint_acc = _eval_classification(joint, X_te, Y_te)
    log.info("Reference: Non-Incremental BLS acc = %.4f (time = %.4fs)",
             joint_acc, t.elapsed)

    results: Dict[str, Any] = {
        "_non_incremental_reference": {
            "accuracy": float(joint_acc),
            "time":     float(t.elapsed),
        }
    }

    for n_b in n_batches_choices:
        log.info("  --- n_batches = %d ---", n_b)
        per_method = {m: {"accs": [], "times": []} for m in methods}
        for r in range(n_repeats):
            run_seed = seed + r * 1000
            batches = split_random_batches(X_tr, Y_tr, n_batches=n_b,
                                            seed=run_seed)
            for name, cls in methods.items():
                try:
                    cfg = _make_config(preset, seed=run_seed)
                    model = cls(config=cfg)
                    with Timer() as t:
                        model.fit_initial(*batches[0])
                        for X_b, Y_b in batches[1:]:
                            model.add_data(X_b, Y_b)
                    acc = _eval_classification(model, X_te, Y_te)
                    per_method[name]["accs"].append(acc)
                    per_method[name]["times"].append(t.elapsed)
                    rec.log_step(method=name, n_batches=n_b, repeat=r,
                                 acc=acc, time=t.elapsed)
                except Exception as e:
                    log.warning("    %s repeat=%d 失败: %s", name, r, e)

        slot: Dict[str, Any] = {}
        for m, vals in per_method.items():
            if vals["accs"]:
                slot[m] = {
                    "acc_mean":  float(np.mean(vals["accs"])),
                    "acc_std":   float(np.std(vals["accs"])),
                    "time_mean": float(np.mean(vals["times"])),
                    "time_std":  float(np.std(vals["times"])),
                }
                log.info("    %-22s | acc = %.4f ± %.4f | time = %.4f ± %.4fs",
                         m,
                         slot[m]["acc_mean"], slot[m]["acc_std"],
                         slot[m]["time_mean"], slot[m]["time_std"])
            else:
                slot[m] = {"error": "all runs failed"}
        results[f"n_batches={n_b}"] = slot

    # Overall：所有 batch 配置的均值
    overall = {}
    for m in methods:
        accs, times = [], []
        for k, v in results.items():
            if not k.startswith("n_batches=") or m not in v or "error" in v[m]:
                continue
            accs.append(v[m]["acc_mean"])
            times.append(v[m]["time_mean"])
        if accs:
            overall[m] = {
                "acc_mean":  float(np.mean(accs)),
                "acc_std":   float(np.std(accs)),
                "time_mean": float(np.mean(times)),
                "time_std":  float(np.std(times)),
            }
    results["Overall"] = overall

    rec.save_summary(group_by="method")
    _save_metrics(target, results)
    _write_table7_report(target, dataset, results, preset,
                         len(X_tr), len(X_te), n_classes,
                         n_batches_choices, n_repeats, joint_acc)
    return results


# ===========================================================================
# 报告生成器：精美 Markdown 表格
# ===========================================================================


def _save_metrics(target: Path, results: Dict[str, Any]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with open(target / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}"


def _fmt_time(x: float) -> str:
    return f"{x:.4f}"


def _write_table5_report(target: Path, dataset: str,
                         results: Dict[str, Any], preset,
                         n_classes: int, n_train: int, n_test: int) -> None:
    lines = [
        f"# Table 5 复现报告 — `{dataset}`",
        "",
        "**Setting**: Equal-scale data streams (论文 Section 4.1).",
        "",
        f"- 数据集大小: `train={n_train}` / `test={n_test}` / `classes={n_classes}`",
        f"- 增量 batch 数: **{preset.equal_scale_n_batches}**",
        f"- BLS 配置: `N1={preset.bls_kwargs['n_mapping_per_window']}, "
        f"N2={preset.bls_kwargs['n_mapping_windows']}, "
        f"N3={preset.bls_kwargs['n_enhancement']}, "
        f"λ={preset.bls_kwargs['reg_lambda']}`",
        "",
        "## Final accuracy & total training time",
        "",
        "| Method | Accuracy (%) | Total Time (s) |",
        "|---|---:|---:|",
    ]
    # 排序：Non-Incremental 优先；IMF-BLS 最后高亮
    order = [
        "Non-Incremental BLS",
        "Incremental BLS",
        "TiBLS",
        "Approximation Method",
        "RI-BLS",
        "IMF-BLS (Ours)",
    ]
    for name in order:
        if name not in results:
            continue
        info = results[name]
        if "error" in info:
            lines.append(f"| {name} | _failed_ | _failed_ |")
            continue
        acc_pct = _fmt_pct(info["final_metric"])
        tt = _fmt_time(info["total_time"])
        bold = "**" if name == "IMF-BLS (Ours)" else ""
        lines.append(f"| {bold}{name}{bold} | {bold}{acc_pct}{bold} | {bold}{tt}{bold} |")

    # 增量曲线
    lines += [
        "",
        "## Per-step accuracy (incremental learning curve)",
        "",
        "| Step | " + " | ".join(n for n in order if n in results and "metric_per_step" in results.get(n, {})) + " |",
        "|---|" + "|".join(["---:" for n in order if n in results and "metric_per_step" in results.get(n, {})]) + "|",
    ]
    avail = [n for n in order if n in results and "metric_per_step" in results.get(n, {})]
    if avail:
        max_steps = max(len(results[n]["metric_per_step"]) for n in avail)
        for s in range(max_steps):
            row = [f"| {s + 1}"]
            for n in avail:
                seq = results[n]["metric_per_step"]
                row.append(_fmt_pct(seq[s]) if s < len(seq) else "—")
            lines.append(" | ".join(row) + " |")

    # 去除中间多余空行（保证 markdown 表格连续）
    final = []
    prev_blank = False
    for line in lines:
        is_blank = (line.strip() == "")
        if is_blank and prev_blank:
            continue
        final.append(line)
        prev_blank = is_blank

    with open(target / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(final))


def _write_table6_report(target: Path, dataset: str,
                         summary: Dict[str, Any], preset,
                         n_train: int, n_test: int, attrs: int) -> None:
    lines = [
        f"# Table 6 复现报告 — `{dataset}` (Regression)",
        "",
        f"- 数据集: `train={n_train}` / `test={n_test}` / `attributes={attrs}`",
        f"- BLS 配置: `N1={preset.bls_kwargs['n_mapping_per_window']}, "
        f"N2={preset.bls_kwargs['n_mapping_windows']}, "
        f"N3={preset.bls_kwargs['n_enhancement']}, "
        f"λ={preset.bls_kwargs['reg_lambda']}`",
        "",
        "## Train Time / RMSE / STD（多次运行均值 ± std）",
        "",
        "| Method | Train Time (s) | Test RMSE | Test STD |",
        "|---|---:|---:|---:|",
    ]
    order = [
        "Non-Incremental BLS",
        "Incremental BLS",
        "Approximation Method",
        "TiBLS",
        "RI-BLS",
        "IMF-BLS (Ours)",
    ]
    for name in order:
        if name not in summary:
            continue
        info = summary[name]
        if "error" in info:
            lines.append(f"| {name} | _failed_ | _failed_ | _failed_ |")
            continue
        bold = "**" if name == "IMF-BLS (Ours)" else ""
        lines.append(
            f"| {bold}{name}{bold} | "
            f"{bold}{info['time_mean']:.4f}{bold} | "
            f"{bold}{info['rmse_mean']:.4f}{bold} | "
            f"{bold}{info['rmse_std']:.4f}{bold} |"
        )

    with open(target / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_table7_report(target: Path, dataset: str,
                         results: Dict[str, Any], preset,
                         n_train: int, n_test: int, n_classes: int,
                         n_batches_choices: List[int], n_repeats: int,
                         joint_acc: float) -> None:
    lines = [
        f"# Table 7 复现报告 — `{dataset}` (Uncertain-Scale)",
        "",
        f"- 数据集: `train={n_train}` / `test={n_test}` / `classes={n_classes}`",
        f"- Batch 配置: `{n_batches_choices}` × `{n_repeats}` repeats",
        f"- BLS 配置: `N1={preset.bls_kwargs['n_mapping_per_window']}, "
        f"N2={preset.bls_kwargs['n_mapping_windows']}, "
        f"N3={preset.bls_kwargs['n_enhancement']}, "
        f"λ={preset.bls_kwargs['reg_lambda']}`",
        f"- 联合训练 (Non-Incremental BLS) 参考准确率: **{joint_acc * 100:.2f}%**",
        "",
        "## Accuracy (%) ± std / Time (s) ± std",
        "",
    ]
    methods = ["Incremental BLS", "Approximation Method", "RI-BLS", "IMF-BLS (Ours)"]
    header = "| Batches | " + " | ".join(
        f"{m} Acc | {m} Time" for m in methods
    ) + " |"
    sep = "|---|" + "|".join(["---:|---:"] * len(methods)) + "|"
    lines.append(header)
    lines.append(sep)

    for n_b in n_batches_choices:
        key = f"n_batches={n_b}"
        if key not in results:
            continue
        slot = results[key]
        cells = [f"| {n_b}"]
        for m in methods:
            v = slot.get(m, {})
            if "error" in v:
                cells.append("_fail_")
                cells.append("_fail_")
            else:
                bold = "**" if m == "IMF-BLS (Ours)" else ""
                cells.append(f"{bold}{v['acc_mean'] * 100:.2f}±{v['acc_std'] * 100:.2f}{bold}")
                cells.append(f"{bold}{v['time_mean']:.4f}±{v['time_std']:.4f}{bold}")
        lines.append(" | ".join(cells) + " |")

    if "Overall" in results and results["Overall"]:
        cells = ["| Overall"]
        for m in methods:
            v = results["Overall"].get(m, {})
            if not v:
                cells.append("—")
                cells.append("—")
            else:
                bold = "**" if m == "IMF-BLS (Ours)" else ""
                cells.append(f"{bold}{v['acc_mean'] * 100:.2f}±{v['acc_std'] * 100:.2f}{bold}")
                cells.append(f"{bold}{v['time_mean']:.4f}±{v['time_std']:.4f}{bold}")
        lines.append(" | ".join(cells) + " |")

    with open(target / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ===========================================================================
# 跨数据集汇总
# ===========================================================================


def write_summary_md(table_id: str, output_dir: Path,
                     datasets: List[str]) -> None:
    """跨数据集汇总到 ``results/reproduce/<table_id>/summary.md``。"""
    base = output_dir / table_id
    base.mkdir(parents=True, exist_ok=True)
    rows: List[str] = []
    if table_id == "table5":
        rows.append("# Paper Table 5 — Final accuracy & training time")
        rows.append("")
        rows.append("| Dataset | "
                    "Incremental BLS Acc | Time | "
                    "TiBLS Acc | Time | "
                    "Approx. Acc | Time | "
                    "RI-BLS Acc | Time | "
                    "**IMF-BLS Acc** | **Time** |")
        rows.append("|---|" + "|".join(["---:"] * 10) + "|")
        for ds in datasets:
            jp = base / ds / "metrics.json"
            if not jp.exists():
                continue
            with open(jp) as f:
                m = json.load(f)
            cells = [f"| {ds}"]
            for name in ["Incremental BLS", "TiBLS", "Approximation Method",
                         "RI-BLS", "IMF-BLS (Ours)"]:
                if name not in m or "error" in m[name]:
                    cells.append("—")
                    cells.append("—")
                    continue
                bold = "**" if name == "IMF-BLS (Ours)" else ""
                cells.append(f"{bold}{m[name]['final_metric'] * 100:.2f}{bold}")
                cells.append(f"{bold}{m[name]['total_time']:.4f}{bold}")
            rows.append(" | ".join(cells) + " |")

    elif table_id == "table6":
        rows.append("# Paper Table 6 — Regression results")
        rows.append("")
        rows.append("| Dataset | Method | Train Time (s) | Test RMSE | Test STD |")
        rows.append("|---|---|---:|---:|---:|")
        for ds in datasets:
            jp = base / ds / "metrics.json"
            if not jp.exists():
                continue
            with open(jp) as f:
                m = json.load(f)
            for name in ["Non-Incremental BLS", "Incremental BLS",
                         "Approximation Method", "TiBLS", "RI-BLS",
                         "IMF-BLS (Ours)"]:
                if name not in m or "error" in m[name]:
                    continue
                bold = "**" if name == "IMF-BLS (Ours)" else ""
                rows.append(
                    f"| {ds} | {bold}{name}{bold} | "
                    f"{bold}{m[name]['time_mean']:.4f}{bold} | "
                    f"{bold}{m[name]['rmse_mean']:.4f}{bold} | "
                    f"{bold}{m[name]['rmse_std']:.4f}{bold} |"
                )

    elif table_id == "table7":
        rows.append("# Paper Table 7 — Uncertain-scale data streams")
        rows.append("")
        rows.append("| Dataset | Method | Acc (%) | Time (s) |")
        rows.append("|---|---|---:|---:|")
        for ds in datasets:
            jp = base / ds / "metrics.json"
            if not jp.exists():
                continue
            with open(jp) as f:
                m = json.load(f)
            ov = m.get("Overall", {})
            for name in ["Incremental BLS", "Approximation Method",
                         "RI-BLS", "IMF-BLS (Ours)"]:
                if name not in ov:
                    continue
                bold = "**" if name == "IMF-BLS (Ours)" else ""
                rows.append(
                    f"| {ds} | {bold}{name}{bold} | "
                    f"{bold}{ov[name]['acc_mean'] * 100:.2f}±"
                    f"{ov[name]['acc_std'] * 100:.2f}{bold} | "
                    f"{bold}{ov[name]['time_mean']:.4f}±"
                    f"{ov[name]['time_std']:.4f}{bold} |"
                )

    # 写入前去除连续空行（防止 markdown 表格中间被打断）
    final = []
    prev_blank = False
    for line in rows:
        is_blank = (line.strip() == "")
        if is_blank and prev_blank:
            continue
        final.append(line)
        prev_blank = is_blank

    with open(base / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(final))
    _GLOBAL_LOGGER.info("汇总报告 → %s", base / "summary.md")


# ===========================================================================
# CLI
# ===========================================================================


# 默认每张表运行的数据集（从论文 Table 5/6/7 中选取，可通过 --skip 跳过）
DEFAULT_TABLE5_DATASETS = [
    "pendigits", "letter", "shuttle", "waveform", "led",
]
DEFAULT_TABLE6_DATASETS = [
    "abalone", "bodyfat", "energy_efficiency",
    "weather_izmir", "appliances_energy",
]
DEFAULT_TABLE7_DATASETS = [
    "mnist", "fashion_mnist",
]


def _parse_dataset_list(spec: Optional[str], default: List[str]) -> List[str]:
    if spec is None:
        return list(default)
    parts = [s.strip() for s in spec.split(",") if s.strip()]
    return parts  # 空字符串显式禁用 → 返回空列表


def main() -> None:
    parser = argparse.ArgumentParser(
        description="复现 IMF-BLS 论文实验（Table 5 / 6 / 7）",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # table5
    p5 = sub.add_parser("table5", help="复现论文 Table 5（分类，等量数据流）")
    p5.add_argument("--dataset", type=str, required=True,
                    help=f"支持: {DEFAULT_TABLE5_DATASETS} 或 {list_presets()}")
    p5.add_argument("--seed", type=int, default=0)
    p5.add_argument("--mnist_path", type=str, default=None)
    p5.add_argument("--output_dir", type=str, default=None)
    p5.add_argument("--subsample_train", type=int, default=None,
                    help="可选：训练集随机下采样到 N 个样本（用于本地快速验证）")
    p5.add_argument("--subsample_test", type=int, default=None,
                    help="可选：测试集下采样到 N 个样本")

    # table6
    p6 = sub.add_parser("table6", help="复现论文 Table 6（回归）")
    p6.add_argument("--dataset", type=str, required=True,
                    help=f"支持: {DEFAULT_TABLE6_DATASETS}")
    p6.add_argument("--repeats", type=int, default=5)
    p6.add_argument("--seed", type=int, default=0)
    p6.add_argument("--output_dir", type=str, default=None)

    # table7
    p7 = sub.add_parser("table7", help="复现论文 Table 7（分类，不定数据流）")
    p7.add_argument("--dataset", type=str, required=True)
    p7.add_argument("--mnist_path", type=str, default=None)
    p7.add_argument("--seed", type=int, default=0)
    p7.add_argument("--n_batches", type=str, default=None,
                    help="逗号分隔，例如 5,10,15,20,25")
    p7.add_argument("--n_repeats", type=int, default=None)
    p7.add_argument("--output_dir", type=str, default=None)
    p7.add_argument("--subsample_train", type=int, default=None)
    p7.add_argument("--subsample_test", type=int, default=None)

    # all
    pa = sub.add_parser("all", help="一次跑全 Table 5 / 6 / 7")
    pa.add_argument("--datasets_table5", type=str, default=None)
    pa.add_argument("--datasets_table6", type=str, default=None)
    pa.add_argument("--datasets_table7", type=str, default=None)
    pa.add_argument("--skip", type=str, default="",
                    help="逗号分隔，要跳过的数据集名（如 mnist,emnist,norb）")
    pa.add_argument("--repeats", type=int, default=3,
                    help="Table 6 重复次数（论文为 5 次，但 5 次较慢）")
    pa.add_argument("--mnist_path", type=str, default=None)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()
    output_dir = Path(args.output_dir or os.path.join(ROOT, "results", "reproduce"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "table5":
        reproduce_table5(args.dataset, output_dir,
                         seed=args.seed, mnist_path=args.mnist_path,
                         subsample_train=args.subsample_train,
                         subsample_test=args.subsample_test)
        write_summary_md("table5", output_dir, [args.dataset])
    elif args.cmd == "table6":
        reproduce_table6(args.dataset, output_dir,
                         repeats=args.repeats, seed=args.seed)
        write_summary_md("table6", output_dir, [args.dataset])
    elif args.cmd == "table7":
        n_b = _parse_dataset_list(args.n_batches, [])
        n_b_int = [int(x) for x in n_b] if n_b else None
        reproduce_table7(args.dataset, output_dir,
                         seed=args.seed, mnist_path=args.mnist_path,
                         n_batches_choices=n_b_int,
                         n_repeats=args.n_repeats,
                         subsample_train=args.subsample_train,
                         subsample_test=args.subsample_test)
        write_summary_md("table7", output_dir, [args.dataset])
    elif args.cmd == "all":
        skip = {s.strip() for s in args.skip.split(",") if s.strip()}
        ds5 = [d for d in _parse_dataset_list(args.datasets_table5, DEFAULT_TABLE5_DATASETS)
               if d not in skip]
        ds6 = [d for d in _parse_dataset_list(args.datasets_table6, DEFAULT_TABLE6_DATASETS)
               if d not in skip]
        ds7 = [d for d in _parse_dataset_list(args.datasets_table7, DEFAULT_TABLE7_DATASETS)
               if d not in skip]

        for ds in ds5:
            try:
                reproduce_table5(ds, output_dir, seed=args.seed,
                                 mnist_path=args.mnist_path)
            except Exception as e:
                _GLOBAL_LOGGER.warning("table5/%s 失败: %s", ds, e)
        if ds5:
            write_summary_md("table5", output_dir, ds5)

        for ds in ds6:
            try:
                reproduce_table6(ds, output_dir,
                                 repeats=args.repeats, seed=args.seed)
            except Exception as e:
                _GLOBAL_LOGGER.warning("table6/%s 失败: %s", ds, e)
        if ds6:
            write_summary_md("table6", output_dir, ds6)

        for ds in ds7:
            try:
                reproduce_table7(ds, output_dir, seed=args.seed,
                                 mnist_path=args.mnist_path)
            except Exception as e:
                _GLOBAL_LOGGER.warning("table7/%s 失败: %s", ds, e)
        if ds7:
            write_summary_md("table7", output_dir, ds7)

        _GLOBAL_LOGGER.info("All done. 报告位置: %s", output_dir)


if __name__ == "__main__":
    main()
