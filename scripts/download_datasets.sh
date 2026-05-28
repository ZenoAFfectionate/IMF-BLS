#!/usr/bin/env bash
# ============================================================================
#  download_datasets.sh
#  ----------------------------------------------------------------------------
#  下载 IMF-BLS 论文复现所需的全部 UCI / LIBSVM 数据集到 data/uci/。
#
#  Tables / 数据集对应关系（论文 Table 5 分类、Table 6 回归）：
#    Table 5 (Classification)
#      - letter        : UCI Letter Recognition
#      - pendigits     : UCI Pen-Based Digit Recognition
#      - shuttle       : LIBSVM shuttle.scale (+ .t)
#      - waveform      : UCI Waveform Database Generator (Version 1)
#      - led           : 由 sklearn / 内置 generator 生成，无需下载
#
#    Table 6 (Regression)
#      - abalone           : UCI Abalone
#      - bodyfat           : LIBSVM bodyfat
#      - appliances_energy : UCI Appliances Energy Prediction
#      - energy_efficiency : UCI Energy Efficiency (ENB2012)
#
#  用法：
#      bash scripts/download_datasets.sh           # 全量下载
#      bash scripts/download_datasets.sh letter    # 仅下载某一个数据集
#      bash scripts/download_datasets.sh -f        # 强制重新下载（覆盖）
#
#  依赖：curl 或 wget；可选 unzip（处理 waveform.zip 时使用）。
# ============================================================================

set -euo pipefail

# ---------- 0. 解析参数 ------------------------------------------------------
FORCE=0
TARGET=""
for arg in "$@"; do
  case "$arg" in
    -f|--force) FORCE=1 ;;
    -h|--help)
      sed -n '1,30p' "$0"
      exit 0
      ;;
    *) TARGET="$arg" ;;
  esac
done

# ---------- 1. 路径与工具检测 -------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data/uci"
mkdir -p "${DATA_DIR}"

if command -v curl >/dev/null 2>&1; then
  DL() { curl -L --fail --retry 3 --connect-timeout 30 -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
  DL() { wget -q --tries=3 --timeout=30 -O "$2" "$1"; }
else
  echo "[ERROR] 需要 curl 或 wget，请先安装。" >&2
  exit 1
fi

# ---------- 2. 通用下载函数 ---------------------------------------------------
# fetch <dataset_name> <subdir> <filename> <url>
fetch() {
  local name="$1" sub="$2" fname="$3" url="$4"
  if [[ -n "${TARGET}" && "${TARGET}" != "${name}" ]]; then
    return 0
  fi
  local out_dir="${DATA_DIR}/${sub}"
  local out_file="${out_dir}/${fname}"
  mkdir -p "${out_dir}"
  if [[ -s "${out_file}" && "${FORCE}" -eq 0 ]]; then
    printf "[skip]  %-22s %s (existing %s)\n" \
      "${name}" "${out_file#${REPO_ROOT}/}" "$(du -h "${out_file}" | cut -f1)"
    return 0
  fi
  printf "[fetch] %-22s <- %s\n" "${name}" "${url}"
  if DL "${url}" "${out_file}"; then
    echo "        ok    -> ${out_file#${REPO_ROOT}/} ($(du -h "${out_file}" | cut -f1))"
  else
    echo "        FAILED ${url}" >&2
    rm -f "${out_file}"
    return 1
  fi
}

# ---------- 3. 数据集清单 -----------------------------------------------------
# 注：UCI 在 2023 年迁移到了新域名，旧 archive.ics.uci.edu URL 仍可访问，
#     但作为冗余，部分数据集同时支持 LIBSVM 镜像。

UCI_BASE="https://archive.ics.uci.edu/ml/machine-learning-databases"
LIBSVM_BASE="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"

echo "================================================================"
echo " IMF-BLS dataset downloader"
echo " repo root : ${REPO_ROOT}"
echo " data dir  : ${DATA_DIR}"
[[ -n "${TARGET}" ]] && echo " target    : ${TARGET}"
[[ "${FORCE}" -eq 1 ]] && echo " mode      : FORCE (覆盖已有文件)"
echo "================================================================"

# ---- Classification (Table 5) ----
fetch abalone            abalone            abalone.data                "${UCI_BASE}/abalone/abalone.data"
fetch letter             letter             letter-recognition.data     "${UCI_BASE}/letter-recognition/letter-recognition.data"
fetch pendigits          pendigits          pendigits.tra               "${UCI_BASE}/pendigits/pendigits.tra"
fetch pendigits          pendigits          pendigits.tes               "${UCI_BASE}/pendigits/pendigits.tes"
fetch shuttle            shuttle            shuttle.scale               "${LIBSVM_BASE}/multiclass/shuttle.scale"
fetch shuttle            shuttle            shuttle.scale.t             "${LIBSVM_BASE}/multiclass/shuttle.scale.t"
fetch waveform           waveform           waveform.data.Z             "${UCI_BASE}/waveform/waveform.data.Z"

# ---- Regression (Table 6) ----
fetch bodyfat            bodyfat            bodyfat                     "${LIBSVM_BASE}/regression/bodyfat"
fetch appliances_energy  appliances_energy  energydata_complete.csv     "${UCI_BASE}/00374/energydata_complete.csv"
fetch energy_efficiency  energy_efficiency  ENB2012_data.xlsx           "${UCI_BASE}/00242/ENB2012_data.xlsx"

# ---------- 4. 后处理 --------------------------------------------------------
# waveform.data.Z 是旧式 compress(1) 格式，解压后保存为 waveform.zip 兼容 main.py。
WAVE_Z="${DATA_DIR}/waveform/waveform.data.Z"
WAVE_RAW="${DATA_DIR}/waveform/waveform.data"
WAVE_ZIP="${DATA_DIR}/waveform/waveform.zip"
if [[ -s "${WAVE_Z}" && ( ! -s "${WAVE_ZIP}" || "${FORCE}" -eq 1 ) ]]; then
  if command -v uncompress >/dev/null 2>&1; then
    uncompress -kf "${WAVE_Z}"
  elif command -v gunzip >/dev/null 2>&1; then
    gunzip -kf "${WAVE_Z}" 2>/dev/null || true
  fi
  if [[ -s "${WAVE_RAW}" ]]; then
    (cd "${DATA_DIR}/waveform" && zip -q -j waveform.zip waveform.data) \
      || cp "${WAVE_RAW}" "${WAVE_ZIP}"
    echo "[post]  waveform               -> waveform.zip"
  fi
fi

echo "================================================================"
echo " Done. 检查 ${DATA_DIR}:"
du -sh "${DATA_DIR}"/* 2>/dev/null | sed 's|^|   |'
echo "================================================================"
echo "下一步："
echo "   python reproduce.py --table 5      # 复现 Table 5 (分类)"
echo "   python reproduce.py --table 6      # 复现 Table 6 (回归)"
