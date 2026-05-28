#!/usr/bin/env bash
# ============================================================================
# IMF-BLS 论文实验复现脚本：一键运行全部 Table 5 / 6 / 7
# ----------------------------------------------------------------------------
# 默认跳过 MNIST / Fashion-MNIST 等需要手动下载 IDX 文件的数据集。
# 若已准备好 MNIST 数据，请使用 ``--with-mnist`` 选项并指定 MNIST_PATH。
#
# 用法::
#
#     bash scripts/run_all.sh                       # 跑核心 UCI 数据集
#     bash scripts/run_all.sh --with-mnist          # 同时跑 Table 7 MNIST
#     MNIST_PATH=data/mnist bash scripts/run_all.sh --with-mnist
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."   # 项目根目录

# 默认变量（可覆盖）
PYTHON="${PYTHON:-python3}"
REPEATS="${REPEATS:-3}"
MNIST_PATH="${MNIST_PATH:-data/mnist}"
SEED="${SEED:-0}"

WITH_MNIST=0
for arg in "$@"; do
    case "$arg" in
        --with-mnist) WITH_MNIST=1 ;;
        --help|-h)
            sed -n '2,12p' "$0"
            exit 0
            ;;
    esac
done

mkdir -p results/reproduce

echo "=================================================================="
echo "IMF-BLS Paper Reproduction Suite"
echo "=================================================================="
echo "  Python   : $($PYTHON --version 2>&1)"
echo "  Repeats  : $REPEATS  (Table 6 多次平均)"
echo "  Seed     : $SEED"
echo "  WITH_MNIST: $WITH_MNIST"
echo "=================================================================="

START_TS=$(date +%s)

# ------------------------------------------------------------------
# Table 5: 分类 / 等量数据流
# ------------------------------------------------------------------
echo ""
echo ">>> Reproducing Paper Table 5 (classification, equal-scale)"
for ds in pendigits letter shuttle waveform led; do
    echo "--- Dataset: $ds ---"
    $PYTHON reproduce.py table5 --dataset "$ds" --seed "$SEED" \
        || echo "[WARN] table5/$ds 失败，已跳过"
done

# ------------------------------------------------------------------
# Table 6: 回归
# ------------------------------------------------------------------
echo ""
echo ">>> Reproducing Paper Table 6 (regression)"
for ds in abalone bodyfat energy_efficiency weather_izmir appliances_energy; do
    echo "--- Dataset: $ds ---"
    $PYTHON reproduce.py table6 --dataset "$ds" --repeats "$REPEATS" \
        --seed "$SEED" \
        || echo "[WARN] table6/$ds 失败，已跳过"
done

# ------------------------------------------------------------------
# Table 7: 分类 / 不定数据流（默认仅在 --with-mnist 启用）
# ------------------------------------------------------------------
if [[ "$WITH_MNIST" == "1" ]]; then
    echo ""
    echo ">>> Reproducing Paper Table 7 (classification, uncertain-scale)"
    if [[ -d "$MNIST_PATH" ]]; then
        $PYTHON reproduce.py table7 --dataset mnist \
            --mnist_path "$MNIST_PATH" --seed "$SEED" \
            || echo "[WARN] table7/mnist 失败"
    else
        echo "[INFO] MNIST_PATH=$MNIST_PATH 不存在，跳过 mnist"
    fi

    if [[ -d "data/fashion_mnist" ]]; then
        $PYTHON reproduce.py table7 --dataset fashion_mnist \
            --mnist_path data/fashion_mnist --seed "$SEED" \
            || echo "[WARN] table7/fashion_mnist 失败"
    else
        echo "[INFO] data/fashion_mnist 不存在，跳过 fashion_mnist"
    fi
fi

# ------------------------------------------------------------------
# 汇总报告
# ------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "Generating cross-dataset summary reports"
echo "=================================================================="

# table5 / table6 / table7 的 summary.md 在每次运行时已生成；
# 这里再产出一份 README 索引。
cat > results/reproduce/README.md <<'MD'
# IMF-BLS 论文复现总览

| 论文表 | 报告 |
|---|---|
| Table 5 (分类 / 等量数据流) | [table5/summary.md](table5/summary.md) |
| Table 6 (回归)              | [table6/summary.md](table6/summary.md) |
| Table 7 (分类 / 不定数据流) | [table7/summary.md](table7/summary.md) |

每个数据集的详细报告位于 `<table_id>/<dataset>/report.md`，
包含 BLS 配置、数据规模、增量曲线、指标 ± std 等完整信息。
MD

END_TS=$(date +%s)
echo ""
echo "=================================================================="
echo "All experiments finished in $((END_TS - START_TS)) s"
echo "Reports at: results/reproduce/"
echo "  - results/reproduce/README.md             (索引)"
echo "  - results/reproduce/table5/summary.md     (Table 5 汇总)"
echo "  - results/reproduce/table6/summary.md     (Table 6 汇总)"
echo "  - results/reproduce/table7/summary.md     (Table 7 汇总)"
echo "=================================================================="
