#!/usr/bin/env bash
# 复现论文 Table 5（分类，equal-scale 数据流）。
#
# 用法::
#
#     bash scripts/table5.sh                         # 跑全部默认数据集
#     bash scripts/table5.sh pendigits letter        # 仅跑指定数据集
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
SEED="${SEED:-0}"

if [[ $# -gt 0 ]]; then
    DATASETS_CSV=$(IFS=,; echo "$*")
else
    DATASETS_CSV="pendigits,letter,shuttle,waveform,led"
fi

echo ">>> Table 5 reproduction (datasets: ${DATASETS_CSV//,/ })"
$PYTHON reproduce.py all \
    --datasets_table5 "$DATASETS_CSV" \
    --datasets_table6 "" \
    --datasets_table7 "" \
    --seed "$SEED"

echo "Done. → results/reproduce/table5/summary.md"
