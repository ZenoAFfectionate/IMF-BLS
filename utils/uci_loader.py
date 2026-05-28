# -*- coding: utf-8 -*-
"""UCI / 公开数据集加载器（论文 Table 3、Table 4）。

提供论文实验中使用的 UCI 数据集的统一加载接口。所有数据集会自动下载
到 ``data/uci/<dataset>/`` 目录下并缓存。

支持的数据集
============

分类 (论文 Table 3)::

    pendigits   8000 train / 2992 test  / 16 dim / 10 cls
    letter      16000 train / 4000 test / 16 dim / 26 cls
    shuttle     43500 train / 14500 test / 9 dim / 7 cls
    waveform    4200 train / 800 test  / 40 dim / 3 cls
    led         160000 train / 40000 test / 24 dim / 10 cls (合成生成)

回归 (论文 Table 4)::

    abalone           2784 train / 1393 test / 8 dim
    bodyfat           168 train  / 84 test   / 14 dim
    weather_izmir     974 train  / 487 test  / 9 dim
    energy_efficiency 614 train  / 154 test  / 8 dim
    appliances_energy 15788 train / 3947 test / 6 dim

设计原则
========

* **自动缓存**: 文件优先从 ``DATA_DIR`` 读取，缺失时自动下载。
* **可重现切分**: 所有数据集使用固定 ``random_state=42`` 切分。
* **返回统一接口**: ``(X_train, y_train, X_test, y_test)``。
"""

from __future__ import annotations

import gzip
import os
import urllib.request
import zipfile
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 缓存目录与下载工具
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "uci"
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _download(url: str, target: str, force: bool = False) -> str:
    """下载 ``url`` 到 ``target``（已存在则跳过）。

    macOS 的 Python 默认 SSL 证书可能缺失，遇到证书错误时回退到不验证证书的请求。
    """
    if os.path.isfile(target) and not force:
        return target
    _ensure_dir(os.path.dirname(target))
    print(f"[uci_loader] 下载 {url} -> {target}")
    try:
        urllib.request.urlretrieve(url, target)
    except urllib.error.URLError as e:
        msg = str(e).lower()
        if "certificate" in msg or "ssl" in msg:
            # macOS 常见问题：fallback 到 unverified context
            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(url, context=ctx) as resp:
                with open(target, "wb") as f:
                    f.write(resp.read())
        else:
            raise
    return target


def _maybe_gunzip(src: str, dst: str) -> str:
    """若 ``src`` 是 .gz 文件则解压到 ``dst``，否则直接拷贝路径。"""
    if not src.endswith(".gz"):
        return src
    if os.path.isfile(dst):
        return dst
    with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(fin.read())
    return dst


def _split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float | int = 0.33,
    seed: int = 42,
):
    """简单 hold-out 切分（无 sklearn 依赖）。"""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = np.arange(n)
    rng.shuffle(idx)
    if isinstance(test_size, float):
        n_test = int(round(n * test_size))
    else:
        n_test = int(test_size)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


# ---------------------------------------------------------------------------
# 分类数据集
# ---------------------------------------------------------------------------


def load_pendigits(data_dir: Optional[str] = None):
    """Pen-Based Handwritten Digits (UCI)，8000 + 2992，16 维特征，10 类。"""
    base = os.path.join(data_dir or DEFAULT_DATA_DIR, "pendigits")
    _ensure_dir(base)
    train_url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "pendigits/pendigits.tra"
    )
    test_url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "pendigits/pendigits.tes"
    )
    f_tr = _download(train_url, os.path.join(base, "pendigits.tra"))
    f_te = _download(test_url, os.path.join(base, "pendigits.tes"))
    tr = np.loadtxt(f_tr, delimiter=",")
    te = np.loadtxt(f_te, delimiter=",")
    return (
        tr[:, :-1].astype(np.float64),
        tr[:, -1].astype(int),
        te[:, :-1].astype(np.float64),
        te[:, -1].astype(int),
    )


def load_letter(data_dir: Optional[str] = None):
    """Letter Recognition (UCI)，前 16000 训练 / 后 4000 测试，16 维 / 26 类。"""
    base = os.path.join(data_dir or DEFAULT_DATA_DIR, "letter")
    _ensure_dir(base)
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "letter-recognition/letter-recognition.data"
    )
    f = _download(url, os.path.join(base, "letter-recognition.data"))
    raw = np.loadtxt(f, delimiter=",", dtype=str)
    # 第 0 列是 A-Z 标签
    y = np.array([ord(c) - ord("A") for c in raw[:, 0]], dtype=int)
    X = raw[:, 1:].astype(np.float64)
    return X[:16000], y[:16000], X[16000:], y[16000:]


def load_shuttle(data_dir: Optional[str] = None):
    """Statlog Shuttle (UCI)，43500 + 14500，9 维 / 7 类。"""
    base = os.path.join(data_dir or DEFAULT_DATA_DIR, "shuttle")
    _ensure_dir(base)
    url_tr = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "statlog/shuttle/shuttle.trn.Z"
    )
    url_te = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "statlog/shuttle/shuttle.tst"
    )
    # .trn.Z 是 LZW 压缩；改用替代的 LIBSVM 镜像（更可靠）
    libsvm_tr = (
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/"
        "shuttle.scale"
    )
    libsvm_te = (
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/"
        "shuttle.scale.t"
    )
    try:
        f_tr = _download(libsvm_tr, os.path.join(base, "shuttle.scale"))
        f_te = _download(libsvm_te, os.path.join(base, "shuttle.scale.t"))
        X_tr, y_tr = _read_libsvm(f_tr, n_features=9)
        X_te, y_te = _read_libsvm(f_te, n_features=9)
        # libsvm 标签 1..7 → 0..6
        return X_tr, (y_tr - 1).astype(int), X_te, (y_te - 1).astype(int)
    except Exception as e:
        raise RuntimeError(
            f"无法下载 shuttle 数据集。请手动从 UCI 或 LIBSVM 下载到 {base}/。\n"
            f"原始错误: {e}"
        )


def load_waveform(data_dir: Optional[str] = None, seed: int = 42):
    """Waveform v1。优先使用 sklearn make_classification 风格生成；
    若已在本地放置 KEEL 数据则使用真实数据。

    论文：5000 个样本切 4200/800，40 维 / 3 类。
    """
    base = os.path.join(data_dir or DEFAULT_DATA_DIR, "waveform")
    _ensure_dir(base)
    target = os.path.join(base, "waveform.data")

    if os.path.isfile(target):
        # 用户手动放置的 KEEL .dat
        rows = []
        with open(target, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("@"):
                    continue
                rows.append([float(v) for v in line.replace(",", " ").split()])
        arr = np.array(rows, dtype=np.float64)
        X = arr[:, :-1]
        y = arr[:, -1].astype(int)
        return _split_train_test(X, y, test_size=800, seed=seed)

    # 兜底：使用 Breiman 1984 原始 waveform 生成程序
    # （3 个 21 维基础函数线性组合 + 19 维高斯噪声 = 40 维）
    rng = np.random.default_rng(seed)
    n_total = 5000

    h = np.zeros((3, 21), dtype=np.float64)
    # 三角形基函数
    for x in range(21):
        h[0, x] = max(6.0 - abs(x - 7), 0.0)
        h[1, x] = max(6.0 - abs(x - 11), 0.0)
        h[2, x] = max(6.0 - abs(x - 15), 0.0)

    # 类 0: 0.6 * h0 + 0.4 * h1 + 噪声
    # 类 1: 0.6 * h0 + 0.4 * h2 + 噪声
    # 类 2: 0.6 * h1 + 0.4 * h2 + 噪声
    pairs = [(0, 1), (0, 2), (1, 2)]
    y = rng.integers(0, 3, size=n_total)
    u = rng.uniform(0.0, 1.0, size=n_total)
    X_basis = np.zeros((n_total, 21), dtype=np.float64)
    for i in range(n_total):
        a, b = pairs[y[i]]
        X_basis[i] = u[i] * h[a] + (1.0 - u[i]) * h[b]
    # + 噪声 N(0, 1)
    X_basis = X_basis + rng.standard_normal((n_total, 21))
    # 19 维独立高斯噪声特征
    X_noise = rng.standard_normal((n_total, 19))
    X = np.concatenate([X_basis, X_noise], axis=1)
    return _split_train_test(X, y, test_size=800, seed=seed)


def load_led(data_dir: Optional[str] = None, n_samples: int = 50000,
             seed: int = 42):
    """LED Display Domain (合成生成)。

    论文 Table 3 用 160000+40000，但本地建议用 50000 即可（避免内存压力）。
    每个样本是 7-bit LED 显示 + 17 个无关 bit = 24 维；每位 10% 翻转噪声。
    """
    rng = np.random.default_rng(seed)
    digit_patterns = np.array([
        [1, 1, 1, 0, 1, 1, 1],  # 0
        [0, 0, 1, 0, 0, 1, 0],  # 1
        [1, 0, 1, 1, 1, 0, 1],  # 2
        [1, 0, 1, 1, 0, 1, 1],  # 3
        [0, 1, 1, 1, 0, 1, 0],  # 4
        [1, 1, 0, 1, 0, 1, 1],  # 5
        [1, 1, 0, 1, 1, 1, 1],  # 6
        [1, 0, 1, 0, 0, 1, 0],  # 7
        [1, 1, 1, 1, 1, 1, 1],  # 8
        [1, 1, 1, 1, 0, 1, 1],  # 9
    ], dtype=np.float64)

    y = rng.integers(0, 10, size=n_samples)
    X_led = digit_patterns[y].copy()
    # 10% bit flip 噪声
    flip_mask = rng.random(X_led.shape) < 0.1
    X_led = np.where(flip_mask, 1.0 - X_led, X_led)
    # 17 个无关随机 bit
    X_irrelevant = rng.integers(0, 2, size=(n_samples, 17)).astype(np.float64)
    X = np.concatenate([X_led, X_irrelevant], axis=1)
    # 论文：160000 / 40000
    n_test = int(n_samples * 0.2)
    return _split_train_test(X, y, test_size=n_test, seed=seed)


# ---------------------------------------------------------------------------
# 回归数据集
# ---------------------------------------------------------------------------


def load_abalone(data_dir: Optional[str] = None, seed: int = 42):
    """Abalone (UCI)。预测年龄环数。第 0 列性别 → one-hot 转 3 个 bit。"""
    base = os.path.join(data_dir or DEFAULT_DATA_DIR, "abalone")
    _ensure_dir(base)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    f = _download(url, os.path.join(base, "abalone.data"))

    sex_map = {"M": [1, 0, 0], "F": [0, 1, 0], "I": [0, 0, 1]}
    rows_X = []
    rows_y = []
    with open(f, "r") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            sex_oh = sex_map[parts[0]]
            feats = [float(x) for x in parts[1:-1]]
            target = float(parts[-1])
            rows_X.append(sex_oh + feats)
            rows_y.append(target)
    X = np.array(rows_X, dtype=np.float64)
    y = np.array(rows_y, dtype=np.float64)
    return _split_train_test(X, y, test_size=1393, seed=seed)


def load_bodyfat(data_dir: Optional[str] = None, seed: int = 42):
    """Bodyfat (LIBSVM 镜像)。252 样本，14 特征。"""
    base = os.path.join(data_dir or DEFAULT_DATA_DIR, "bodyfat")
    _ensure_dir(base)
    url = (
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/"
        "bodyfat"
    )
    f = _download(url, os.path.join(base, "bodyfat"))
    X, y = _read_libsvm(f, n_features=14)
    return _split_train_test(X, y, test_size=84, seed=seed)


def load_energy_efficiency(data_dir: Optional[str] = None, seed: int = 42):
    """Energy Efficiency (UCI)。预测制冷负荷 Y2。"""
    base = os.path.join(data_dir or DEFAULT_DATA_DIR, "energy_efficiency")
    _ensure_dir(base)
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/"
        "ENB2012_data.xlsx"
    )
    target = os.path.join(base, "ENB2012_data.xlsx")
    if not os.path.isfile(target):
        try:
            _download(url, target)
        except Exception:
            raise RuntimeError(
                f"无法下载 ENB2012_data.xlsx。请手动放置到 {target}"
            )

    try:
        import openpyxl
    except ImportError as e:
        raise ImportError(
            "加载 energy_efficiency 需要 openpyxl: pip install openpyxl"
        ) from e

    wb = openpyxl.load_workbook(target, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    # 第 0 行是表头，最后两列 Y1/Y2 是目标
    data = []
    for row in rows[1:]:
        if row[0] is None:
            continue
        data.append(row[:10])  # 8 features + Y1 + Y2
    arr = np.array(data, dtype=np.float64)
    X = arr[:, :8]
    y = arr[:, 9]  # 用 Y2（cooling load）
    return _split_train_test(X, y, test_size=154, seed=seed)


def load_appliances_energy(data_dir: Optional[str] = None, seed: int = 42):
    """Appliances Energy (UCI)。19735 个样本，预测家电能耗。"""
    base = os.path.join(data_dir or DEFAULT_DATA_DIR, "appliances_energy")
    _ensure_dir(base)
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/"
        "energydata_complete.csv"
    )
    f = _download(url, os.path.join(base, "energydata_complete.csv"))

    # 使用 csv 模块正确处理带引号字段
    import csv

    rows = []
    targets = []
    with open(f, "r", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)  # noqa: F841
        for parts in reader:
            if len(parts) < 25:
                continue
            try:
                target = float(parts[1])
                # 选 6 个气候相关特征：T1, RH_1, T_out, Press_mm_hg, RH_out, Windspeed
                feats = [
                    float(parts[3]),   # T1
                    float(parts[4]),   # RH_1
                    float(parts[21]),  # T_out
                    float(parts[22]),  # Press_mm_hg
                    float(parts[23]),  # RH_out
                    float(parts[24]),  # Windspeed
                ]
            except (ValueError, IndexError):
                continue
            rows.append(feats)
            targets.append(target)
    if not rows:
        raise RuntimeError(f"appliances_energy 解析失败：{f} 没有可用行")
    X = np.array(rows, dtype=np.float64)
    y = np.array(targets, dtype=np.float64)
    # 论文 15788 / 3947
    return _split_train_test(X, y, test_size=3947, seed=seed)


def load_weather_izmir(data_dir: Optional[str] = None, seed: int = 42):
    """Weather Izmir (KEEL)。1461 样本，9 维气象特征。

    由于原始 KEEL 镜像不稳定，可手动放置 ``izmir.dat`` 到 ``data/uci/weather_izmir/``。
    """
    base = os.path.join(data_dir or DEFAULT_DATA_DIR, "weather_izmir")
    _ensure_dir(base)
    target = os.path.join(base, "izmir.dat")
    if not os.path.isfile(target):
        # 尝试 KEEL 主源（有时不可达）
        url = (
            "https://sci2s.ugr.es/keel/dataset/data/regression/izmir.zip"
        )
        try:
            zip_path = _download(url, os.path.join(base, "izmir.zip"))
            with zipfile.ZipFile(zip_path) as zf:
                for name in zf.namelist():
                    if name.endswith(".dat"):
                        with zf.open(name) as src, open(target, "wb") as dst:
                            dst.write(src.read())
                        break
        except Exception:
            raise RuntimeError(
                f"无法下载 weather_izmir。请手动放置到 {target}（KEEL .dat 格式）"
            )

    rows = []
    with open(target, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("@"):
                continue
            rows.append([float(v) for v in line.replace(",", " ").split()])
    arr = np.array(rows, dtype=np.float64)
    X = arr[:, :-1]
    y = arr[:, -1]
    return _split_train_test(X, y, test_size=487, seed=seed)


# ---------------------------------------------------------------------------
# LIBSVM 格式解析
# ---------------------------------------------------------------------------


def _read_libsvm(path: str, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
    """读取 LIBSVM 稀疏文本格式：``label idx:val idx:val ...``。

    空文件返回 shape ``(0, n_features)`` 的 X 与 ``(0,)`` 的 y。
    """
    labels = []
    rows = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            labels.append(float(parts[0]))
            row = np.zeros(n_features, dtype=np.float64)
            for token in parts[1:]:
                idx_s, val_s = token.split(":")
                row[int(idx_s) - 1] = float(val_s)
            rows.append(row)
    if not rows:
        return (
            np.empty((0, n_features), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    X = np.array(rows, dtype=np.float64)
    y = np.array(labels, dtype=np.float64)
    return X, y


# ---------------------------------------------------------------------------
# 分类 / 回归统一入口
# ---------------------------------------------------------------------------


_CLASSIFICATION_LOADERS = {
    "pendigits": load_pendigits,
    "letter": load_letter,
    "shuttle": load_shuttle,
    "waveform": load_waveform,
    "led": load_led,
}


_REGRESSION_LOADERS = {
    "abalone": load_abalone,
    "bodyfat": load_bodyfat,
    "energy_efficiency": load_energy_efficiency,
    "appliances_energy": load_appliances_energy,
    "weather_izmir": load_weather_izmir,
}


def load_uci_classification(name: str, data_dir: Optional[str] = None):
    name = name.lower()
    if name not in _CLASSIFICATION_LOADERS:
        raise ValueError(
            f"未知 UCI 分类数据集 {name}；支持: {sorted(_CLASSIFICATION_LOADERS)}"
        )
    return _CLASSIFICATION_LOADERS[name](data_dir=data_dir)


def load_uci_regression(name: str, data_dir: Optional[str] = None):
    name = name.lower()
    if name not in _REGRESSION_LOADERS:
        raise ValueError(
            f"未知 UCI 回归数据集 {name}；支持: {sorted(_REGRESSION_LOADERS)}"
        )
    return _REGRESSION_LOADERS[name](data_dir=data_dir)


__all__ = [
    "DEFAULT_DATA_DIR",
    "load_pendigits",
    "load_letter",
    "load_shuttle",
    "load_waveform",
    "load_led",
    "load_abalone",
    "load_bodyfat",
    "load_energy_efficiency",
    "load_appliances_energy",
    "load_weather_izmir",
    "load_uci_classification",
    "load_uci_regression",
]
