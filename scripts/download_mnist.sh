#!/usr/bin/env bash
# 下载论文使用的 MNIST 与 Fashion-MNIST 数据集到 data/mnist/ 与 data/fashion_mnist/。
#
# 用法::
#
#     bash scripts/download_mnist.sh
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data/mnist data/fashion_mnist

MNIST_BASE="https://storage.googleapis.com/cvdf-datasets/mnist"
FMNIST_BASE="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"

download() {
    local url="$1"
    local dst="$2"
    if [[ -f "$dst" ]]; then
        echo "[SKIP] 已存在: $dst"
        return
    fi
    echo "[DL]  $url -> $dst"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$url" -o "$dst"
    else
        wget -q "$url" -O "$dst"
    fi
}

# MNIST
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
         t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
    download "$MNIST_BASE/$f" "data/mnist/$f"
done

# Fashion-MNIST
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
         t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
    download "$FMNIST_BASE/$f" "data/fashion_mnist/$f"
done

echo ""
echo "Done."
echo "  data/mnist/         : $(ls -1 data/mnist/ | wc -l | tr -d ' ') files"
echo "  data/fashion_mnist/ : $(ls -1 data/fashion_mnist/ | wc -l | tr -d ' ') files"
