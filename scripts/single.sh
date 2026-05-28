#!/usr/bin/env bash
# 单数据集快捷复现脚本。
#
# 用法::
#
#     bash scripts/single.sh <table> <dataset> [extra_args ...]
#
# 示例::
#
#     bash scripts/single.sh table5 pendigits
#     bash scripts/single.sh table6 abalone --repeats 5
#     bash scripts/single.sh table7 mnist --mnist_path data/mnist
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"

if [[ $# -lt 2 ]]; then
    echo "用法: bash scripts/single.sh <table5|table6|table7> <dataset> [extra_args ...]"
    exit 1
fi

TABLE="$1"
DATASET="$2"
shift 2

$PYTHON reproduce.py "$TABLE" --dataset "$DATASET" "$@"

echo "Done. → results/reproduce/$TABLE/$DATASET/report.md"
