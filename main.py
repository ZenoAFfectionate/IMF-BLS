# -*- coding: utf-8 -*-
"""IMF-BLS 实验入口（复现论文 Section 4 的三大场景）。

::

    Scenario 1 (Section 4.1)  equal_scale       — 等量数据流增量
    Scenario 2 (Section 4.2)  uncertain_scale   — 不定 scale 数据流增量
    Scenario 3 (Section 4.3)  data_and_nodes    — 数据 + 节点同时增量

对比方法 (论文 Section 4)::

    NonIncrementalBLS (joint, 上界)
    IncrementalBLS    (Greville pinv)
    RIBLS             (memory matrix U/V)
    TiBLS             (Sherman-Morrison-Woodbury, 仅 equal_scale)
    ApproximationMethodBLS (ridge 平均)
    IMFBLS           (Ours)

输出::

    results/<scenario>/<dataset>/metrics.json   — 全部数值结果
    results/<scenario>/<dataset>/accuracy.png   — 准确率/RMSE 曲线
    results/<scenario>/<dataset>/time.png       — 训练耗时柱状图

CLI 示例::

    # 1. 跑全部论文场景 + 全部内置数据集
    python main.py --scenario all --dataset all

    # 2. 仅在 digits 上跑 equal_scale
    python main.py --scenario equal_scale --dataset digits

    # 3. MNIST 实验（需 IDX 文件）
    python main.py --scenario equal_scale --dataset mnist --mnist_path ./data/mnist
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
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
from utils.metrics import classification_accuracy, regression_rmse  # noqa: E402
from utils.timing import Timer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("imf-bls")


# ============================================================================
# 数据集预设
# ============================================================================


@dataclass
class DatasetPreset:
    name: str
    task: str  # "classification" | "regression"
    loader_kwargs: Dict[str, Any]
    bls_kwargs: Dict[str, Any]
    needs_path: bool = False


CLASSIFICATION_PRESETS: Dict[str, DatasetPreset] = {
    "synthetic": DatasetPreset(
        name="synthetic", task="classification",
        loader_kwargs=dict(n_train=2000, n_test=500, n_features=20, n_classes=5),
        bls_kwargs=dict(
            n_mapping_per_window=10, n_mapping_windows=10,
            n_enhancement=600, reg_lambda=1e-6,
        ),
    ),
    "digits": DatasetPreset(
        name="digits", task="classification", loader_kwargs={},
        bls_kwargs=dict(
            n_mapping_per_window=10, n_mapping_windows=10,
            n_enhancement=1000, reg_lambda=1e-6,
        ),
    ),
    "iris": DatasetPreset(
        name="iris", task="classification", loader_kwargs={},
        bls_kwargs=dict(
            n_mapping_per_window=5, n_mapping_windows=5,
            n_enhancement=100, reg_lambda=1e-6,
        ),
    ),
    "mnist": DatasetPreset(
        name="mnist", task="classification", loader_kwargs={},
        bls_kwargs=dict(
            n_mapping_per_window=10, n_mapping_windows=10,
            n_enhancement=5000, reg_lambda=1e-6,
        ),
        needs_path=True,
    ),
    "fashion_mnist": DatasetPreset(
        name="fashion_mnist", task="classification", loader_kwargs={},
        bls_kwargs=dict(
            n_mapping_per_window=10, n_mapping_windows=10,
            n_enhancement=5000, reg_lambda=1e-6,
        ),
        needs_path=True,
    ),
}

REGRESSION_PRESETS: Dict[str, DatasetPreset] = {
    "synthetic_reg": DatasetPreset(
        name="synthetic_reg", task="regression",
        loader_kwargs=dict(n_train=2000, n_test=500, n_features=12),
        bls_kwargs=dict(
            n_mapping_per_window=8, n_mapping_windows=6,
            n_enhancement=300, reg_lambda=1e-6,
        ),
    ),
    "california": DatasetPreset(
        name="california", task="regression", loader_kwargs={},
        bls_kwargs=dict(
            n_mapping_per_window=8, n_mapping_windows=8,
            n_enhancement=600, reg_lambda=1e-6,
        ),
    ),
}


# ============================================================================
# 数据加载与配置工厂
# ============================================================================


def _prepare_dataset(name: str, mnist_path: Optional[str] = None):
    """返回 (X_tr, Y_tr, X_te, Y_te, task, preset)。"""
    if name in CLASSIFICATION_PRESETS:
        preset = CLASSIFICATION_PRESETS[name]
        if preset.needs_path:
            if not mnist_path or not os.path.isdir(mnist_path):
                raise FileNotFoundError(f"加载 {name} 需要 --mnist_path")
            X_tr, y_tr, X_te, y_te = load_classification_dataset(name=name, path=mnist_path)
        else:
            X_tr, y_tr, X_te, y_te = load_classification_dataset(
                name=name, **preset.loader_kwargs
            )
        num_classes = int(max(y_tr.max(), y_te.max()) + 1)
        Y_tr = one_hot_encode(y_tr, num_classes=num_classes)
        Y_te = one_hot_encode(y_te, num_classes=num_classes)
        X_tr_s, X_te_s, _ = standardize_minmax(X_tr, X_te)
        return X_tr_s, Y_tr, X_te_s, Y_te, "classification", preset

    if name in REGRESSION_PRESETS:
        preset = REGRESSION_PRESETS[name]
        loader_name = preset.name.replace("_reg", "")
        X_tr, y_tr, X_te, y_te = load_regression_dataset(name=loader_name, **preset.loader_kwargs)
        Y_tr = y_tr.reshape(-1, 1)
        Y_te = y_te.reshape(-1, 1)
        X_tr_s, X_te_s, _ = standardize_minmax(X_tr, X_te)
        return X_tr_s, Y_tr, X_te_s, Y_te, "regression", preset

    raise ValueError(f"未知数据集: {name}")


def _make_config(preset: DatasetPreset, seed: int = 0) -> BLSConfig:
    return BLSConfig(seed=seed, **preset.bls_kwargs)


def _eval(model, X_te: np.ndarray, Y_te: np.ndarray, task: str) -> float:
    if task == "classification":
        return classification_accuracy(Y_te, model.predict(X_te))
    return regression_rmse(Y_te, model.predict(X_te))


# ============================================================================
# Scenario 1: equal-scale data streams (Section 4.1)
# ============================================================================


def run_equal_scale(dataset: str, n_batches: int, seed: int,
                    mnist_path: Optional[str], output_dir: Path) -> Dict[str, Any]:
    logger.info("=" * 65)
    logger.info("Scenario 1 | dataset=%s | n_batches=%d", dataset, n_batches)
    logger.info("=" * 65)

    X_tr, Y_tr, X_te, Y_te, task, preset = _prepare_dataset(dataset, mnist_path)
    metric = "accuracy" if task == "classification" else "rmse"
    logger.info("Train=%d, Test=%d, target_dim=%d, task=%s",
                len(X_tr), len(X_te), Y_tr.shape[1], task)

    batches = split_into_batches(X_tr, Y_tr, n_batches=n_batches, shuffle=True, seed=seed)

    methods = {
        "Non-Incremental BLS":   NonIncrementalBLS,
        "Incremental BLS":       IncrementalBLS,
        "Approximation Method":  ApproximationMethodBLS,
        "RI-BLS":                RIBLS,
        "TI-BLS":                TiBLS,
        "IMF-BLS (Ours)":       IMFBLS,
    }

    results: Dict[str, Any] = {}
    for name, cls in methods.items():
        try:
            model = cls(config=_make_config(preset, seed=seed))
            metrics_per_step: List[float] = []
            times_per_step: List[float] = []

            if name == "Non-Incremental BLS":
                with Timer() as t:
                    model.fit_initial(X_tr, Y_tr)
                metrics_per_step.append(_eval(model, X_te, Y_te, task))
                times_per_step.append(t.elapsed)
            elif name == "TI-BLS":
                # TiBLS 严格要求等量 batch
                common = min(len(b[0]) for b in batches)
                bs = [(X[:common], Y[:common]) for X, Y in batches]
                with Timer() as t:
                    model.fit_initial(*bs[0])
                metrics_per_step.append(_eval(model, X_te, Y_te, task))
                times_per_step.append(t.elapsed)
                for X_b, Y_b in bs[1:]:
                    with Timer() as t:
                        model.add_data(X_b, Y_b)
                    metrics_per_step.append(_eval(model, X_te, Y_te, task))
                    times_per_step.append(t.elapsed)
            else:
                with Timer() as t:
                    model.fit_initial(*batches[0])
                metrics_per_step.append(_eval(model, X_te, Y_te, task))
                times_per_step.append(t.elapsed)
                for X_b, Y_b in batches[1:]:
                    with Timer() as t:
                        model.add_data(X_b, Y_b)
                    metrics_per_step.append(_eval(model, X_te, Y_te, task))
                    times_per_step.append(t.elapsed)

            results[name] = {
                "per_step_metric": metrics_per_step,
                "per_step_time":   times_per_step,
                "final_metric":    metrics_per_step[-1],
                "total_time":      float(sum(times_per_step)),
                "metric_name":     metric,
            }
            logger.info("  %-22s | final %s = %.4f | total = %.4fs",
                        name, metric, metrics_per_step[-1], sum(times_per_step))
        except Exception as e:
            logger.warning("  %s 失败: %s", name, e)
            results[name] = {"error": str(e)}

    _save_and_plot(output_dir, "equal_scale", dataset, results, metric)
    _print_summary(dataset, "equal_scale", results, metric)
    return results


# ============================================================================
# Scenario 2: uncertain-scale data streams (Section 4.2)
# ============================================================================


def run_uncertain_scale(dataset: str, n_batches_choices: List[int], n_repeats: int,
                        seed: int, mnist_path: Optional[str], output_dir: Path) -> Dict[str, Any]:
    logger.info("=" * 65)
    logger.info("Scenario 2 | dataset=%s | n_batches=%s | repeats=%d",
                dataset, n_batches_choices, n_repeats)
    logger.info("=" * 65)

    X_tr, Y_tr, X_te, Y_te, task, preset = _prepare_dataset(dataset, mnist_path)
    metric = "accuracy" if task == "classification" else "rmse"

    methods = {
        "Incremental BLS":       IncrementalBLS,
        "Approximation Method":  ApproximationMethodBLS,
        "RI-BLS":                RIBLS,
        "IMF-BLS (Ours)":       IMFBLS,
    }

    results: Dict[str, Any] = {}
    for n_b in n_batches_choices:
        logger.info("  --- n_batches = %d ---", n_b)
        per_method: Dict[str, Dict[str, List[float]]] = {
            m: {"metrics": [], "times": []} for m in methods
        }
        for r in range(n_repeats):
            run_seed = seed + r * 1000
            batches = split_random_batches(X_tr, Y_tr, n_batches=n_b, seed=run_seed)
            for name, cls in methods.items():
                try:
                    model = cls(config=_make_config(preset, seed=run_seed))
                    with Timer() as t:
                        model.fit_initial(*batches[0])
                        for X_b, Y_b in batches[1:]:
                            model.add_data(X_b, Y_b)
                    per_method[name]["metrics"].append(_eval(model, X_te, Y_te, task))
                    per_method[name]["times"].append(t.elapsed)
                except Exception as e:
                    logger.warning("    %s repeat=%d 失败: %s", name, r, e)

        summary: Dict[str, Any] = {}
        for m, vals in per_method.items():
            if vals["metrics"]:
                summary[m] = {
                    "metric_mean": float(np.mean(vals["metrics"])),
                    "metric_std":  float(np.std(vals["metrics"])),
                    "time_mean":   float(np.mean(vals["times"])),
                    "time_std":    float(np.std(vals["times"])),
                }
                logger.info("    %-22s | %s = %.4f ± %.4f | time = %.4f ± %.4f s",
                            m, metric,
                            summary[m]["metric_mean"], summary[m]["metric_std"],
                            summary[m]["time_mean"],   summary[m]["time_std"])
            else:
                summary[m] = {"error": "all runs failed"}
        results[f"n_batches={n_b}"] = summary

    _save_results(output_dir, "uncertain_scale", dataset, results)
    return results


# ============================================================================
# Scenario 3: concurrent data + node increments (Section 4.3)
# ============================================================================


def run_data_and_nodes(dataset: str, n_data_batches: int, n_node_steps: int,
                       node_step: int, seed: int, mnist_path: Optional[str],
                       output_dir: Path) -> Dict[str, Any]:
    logger.info("=" * 65)
    logger.info("Scenario 3 | dataset=%s | data_batches=%d | node_steps=%d × %d",
                dataset, n_data_batches, n_node_steps, node_step)
    logger.info("=" * 65)

    X_tr, Y_tr, X_te, Y_te, task, preset = _prepare_dataset(dataset, mnist_path)
    metric = "accuracy" if task == "classification" else "rmse"

    batches = split_into_batches(X_tr, Y_tr, n_batches=n_data_batches, shuffle=True, seed=seed)

    methods = {
        "Incremental BLS": IncrementalBLS,
        "IMF-BLS (Ours)": IMFBLS,
    }

    results: Dict[str, Any] = {}
    for name, cls in methods.items():
        try:
            model = cls(config=_make_config(preset, seed=seed))
            timeline: List[Dict[str, Any]] = []
            cum_time = 0.0
            X_seen, Y_seen = batches[0]

            with Timer() as t:
                model.fit_initial(*batches[0])
            cum_time += t.elapsed
            timeline.append({"step": "init", "metric": _eval(model, X_te, Y_te, task),
                             "cum_time": cum_time, "p": _model_width(model)})

            node_added = 0
            for k in range(1, n_data_batches):
                X_b, Y_b = batches[k]
                with Timer() as t:
                    model.add_data(X_b, Y_b)
                cum_time += t.elapsed
                X_seen = np.vstack([X_seen, X_b])
                Y_seen = np.vstack([Y_seen, Y_b])
                timeline.append({"step": f"data+{k}", "metric": _eval(model, X_te, Y_te, task),
                                 "cum_time": cum_time, "p": _model_width(model)})

                if node_added < n_node_steps:
                    with Timer() as t:
                        model.add_nodes(X_seen, Y_seen, n_new=node_step)
                    cum_time += t.elapsed
                    timeline.append({"step": f"node+{k}", "metric": _eval(model, X_te, Y_te, task),
                                     "cum_time": cum_time, "p": _model_width(model)})
                    node_added += 1

            results[name] = {
                "timeline":     timeline,
                "final_metric": timeline[-1]["metric"],
                "total_time":   cum_time,
                "metric_name":  metric,
            }
            logger.info("  %-22s | final %s = %.4f | time = %.4fs | p_final = %d",
                        name, metric, timeline[-1]["metric"], cum_time, timeline[-1]["p"])
        except Exception as e:
            logger.warning("  %s 失败: %s", name, e)
            results[name] = {"error": str(e)}

    _save_results(output_dir, "data_and_nodes", dataset, results)
    return results


def _model_width(model) -> int:
    """获取模型当前广义特征维度（用于 timeline 记录）。"""
    if hasattr(model, "width") and model.width:
        return int(model.width)
    return int(model.feature_layer.feature_dim)


# ============================================================================
# 输出工具
# ============================================================================


def _save_results(out_dir: Path, scenario: str, dataset: str,
                  results: Dict[str, Any], log: Optional[Any] = None) -> None:
    target = out_dir / scenario / dataset
    target.mkdir(parents=True, exist_ok=True)
    with open(target / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    (log or logger).info("结果已保存: %s", target / "metrics.json")


def _save_and_plot(out_dir: Path, scenario: str, dataset: str,
                   results: Dict[str, Any], metric: str,
                   log: Optional[Any] = None) -> None:
    _save_results(out_dir, scenario, dataset, results, log=log)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    target = out_dir / scenario / dataset

    # 准确率/RMSE 曲线
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, info in results.items():
        if "error" in info or "per_step_metric" not in info:
            continue
        ax.plot(info["per_step_metric"], marker="o", label=m)
    ax.set_xlabel("Incremental step")
    ax.set_ylabel(metric)
    ax.set_title(f"{scenario} on {dataset}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(target / "accuracy.png", dpi=120)
    plt.close(fig)

    # 时间柱状图
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names, times = [], []
    for m, info in results.items():
        if "error" in info:
            continue
        names.append(m)
        times.append(info["total_time"])
    ax.bar(names, times)
    ax.set_ylabel("Total time (s)")
    ax.set_title(f"Total training time ({scenario}, {dataset})")
    plt.xticks(rotation=20, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(target / "time.png", dpi=120)
    plt.close(fig)
    (log or logger).info("曲线已保存: %s", target)


def _print_summary(dataset: str, scenario: str, results: Dict[str, Any], metric: str) -> None:
    print()
    print(f"  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │ {scenario.upper():<15s} | {dataset:<43s} │")
    print(f"  ├──────────────────────────┬──────────────────┬─────────────┤")
    print(f"  │ {'Method':<24s} │ {'Final ' + metric:<16s} │ {'Time (s)':<11s} │")
    print(f"  ├──────────────────────────┼──────────────────┼─────────────┤")
    for m, info in results.items():
        if "error" in info:
            print(f"  │ {m:<24s} │ {'-':<16s} │ {info['error'][:11]:<11s} │")
        else:
            print(f"  │ {m:<24s} │ {info['final_metric']:<16.4f} │ {info['total_time']:<11.4f} │")
    print(f"  └──────────────────────────┴──────────────────┴─────────────┘")


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IMF-BLS 实验入口")
    p.add_argument("--scenario",
                   choices=["equal_scale", "uncertain_scale", "data_and_nodes", "all"],
                   default="equal_scale")
    p.add_argument("--dataset", default="synthetic",
                   help="synthetic / digits / iris / mnist / fashion_mnist / "
                        "synthetic_reg / california / all_classification / all_regression / all")
    p.add_argument("--n_batches", type=int, default=5)
    p.add_argument("--uncertain_n_batches", type=int, nargs="+", default=[5, 10, 15])
    p.add_argument("--uncertain_repeats", type=int, default=3)
    p.add_argument("--n_node_steps", type=int, default=2,
                   help="Scenario 3 节点增量次数")
    p.add_argument("--node_step", type=int, default=50,
                   help="Scenario 3 每次新增 enhancement 节点数")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mnist_path", default=None,
                   help="MNIST/Fashion-MNIST IDX 数据目录")
    p.add_argument("--output_dir", default=os.path.join(ROOT, "results"))
    return p.parse_args()


def _resolve_datasets(name: str, scenario: str) -> List[str]:
    if name == "all":
        if scenario == "data_and_nodes":
            return ["synthetic"] + (["digits"] if _has_sklearn() else [])
        return list(CLASSIFICATION_PRESETS.keys()) + list(REGRESSION_PRESETS.keys())
    if name == "all_classification":
        return list(CLASSIFICATION_PRESETS.keys())
    if name == "all_regression":
        return list(REGRESSION_PRESETS.keys())
    return [name]


def _has_sklearn() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = (["equal_scale", "uncertain_scale", "data_and_nodes"]
                 if args.scenario == "all" else [args.scenario])

    for scenario in scenarios:
        for ds in _resolve_datasets(args.dataset, scenario):
            preset = CLASSIFICATION_PRESETS.get(ds) or REGRESSION_PRESETS.get(ds)
            if preset is None:
                logger.warning("跳过未知数据集: %s", ds)
                continue
            if preset.needs_path and not args.mnist_path:
                logger.info("跳过 %s（未提供 --mnist_path）", ds)
                continue

            try:
                if scenario == "equal_scale":
                    run_equal_scale(ds, args.n_batches, args.seed,
                                    args.mnist_path, out_dir)
                elif scenario == "uncertain_scale":
                    run_uncertain_scale(ds, args.uncertain_n_batches,
                                        args.uncertain_repeats,
                                        args.seed, args.mnist_path, out_dir)
                else:  # data_and_nodes
                    run_data_and_nodes(ds, args.n_batches, args.n_node_steps,
                                       args.node_step, args.seed,
                                       args.mnist_path, out_dir)
            except Exception as e:
                logger.error("%s on %s 失败: %s", scenario, ds, e, exc_info=True)


if __name__ == "__main__":
    main()
