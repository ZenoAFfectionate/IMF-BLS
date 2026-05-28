#!/usr/bin/env bash
# 复现论文 Table 6（回归任务）。
#
# 用法::
#
#     bash scripts/table6.sh                         # 跑全部默认数据集
#     bash scripts/table6.sh abalone bodyfat         # 仅跑指定数据集
#     REPEATS=5 bash scripts/table6.sh               # 自定义重复次数
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
SEED="${SEED:-0}"
REPEATS="${REPEATS:-3}"

if [[ $# -gt 0 ]]; then
    DATASETS_CSV=$(IFS=,; echo "$*")
else
    DATASETS_CSV="abalone,bodyfat,energy_efficiency,weather_izmir,appliances_energy"
fi

echo ">>> Table 6 reproduction (datasets: ${DATASETS_CSV//,/ }, repeats=$REPEATS)"
$PYTHON reproduce.py all \
    --datasets_table5 "" \
    --datasets_table6 "$DATASETS_CSV" \
    --datasets_table7 "" \
    --repeats "$REPEATS" \
    --seed "$SEED"

echo "Done. → results/reproduce/table6/summary.md"
