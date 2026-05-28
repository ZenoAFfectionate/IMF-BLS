#!/usr/bin/env bash
# 复现论文 Table 7（分类，uncertain-scale 数据流）。
# 默认仅跑 mnist + fashion_mnist；需要 IDX 文件位于 data/<name>/。
#
# 用法::
#
#     MNIST_PATH=data/mnist bash scripts/table7.sh                   # 默认
#     bash scripts/table7.sh mnist                                   # 仅 mnist
#     bash scripts/table7.sh fashion_mnist                           # 仅 fashion
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
SEED="${SEED:-0}"

if [[ $# -gt 0 ]]; then
    datasets=("$@")
else
    datasets=(mnist fashion_mnist)
fi

OK_LIST=()
for ds in "${datasets[@]}"; do
    case "$ds" in
        mnist)
            DPATH="${MNIST_PATH:-data/mnist}"
            ;;
        fashion_mnist)
            DPATH="${FASHION_MNIST_PATH:-data/fashion_mnist}"
            ;;
        *)
            DPATH=""
            ;;
    esac

    if [[ -n "$DPATH" && ! -d "$DPATH" ]]; then
        echo "[INFO] $DPATH 不存在，跳过 $ds"
        continue
    fi

    echo "--- Table 7 / $ds (data dir = $DPATH) ---"
    if [[ -n "$DPATH" ]]; then
        $PYTHON reproduce.py table7 --dataset "$ds" \
            --mnist_path "$DPATH" --seed "$SEED" \
            && OK_LIST+=("$ds") \
            || echo "[WARN] $ds 失败"
    else
        $PYTHON reproduce.py table7 --dataset "$ds" --seed "$SEED" \
            && OK_LIST+=("$ds") \
            || echo "[WARN] $ds 失败"
    fi
done

# 多数据集时生成跨数据集 summary
if [[ ${#OK_LIST[@]} -gt 0 ]]; then
    OK_CSV=$(IFS=,; echo "${OK_LIST[*]}")
    $PYTHON -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
from reproduce import write_summary_md
write_summary_md('table7', Path('results/reproduce'), '$OK_CSV'.split(','))"
fi

echo "Done. → results/reproduce/table7/summary.md"
