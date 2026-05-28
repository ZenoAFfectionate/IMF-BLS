# `data/` — 数据集目录

> 仓库 **不入库** 任何数据集文件。请在仓库根目录执行：
>
> ```bash
> bash scripts/download_datasets.sh
> ```
>
> 即可自动下载论文 Section 4 / Table 5 / Table 6 所需的全部 UCI / LIBSVM 数据集。
>
> MNIST 请使用 `bash scripts/download_mnist.sh` 单独下载到 `data/mnist/`。

## 预期结构

```
data/
├── mnist/                                     (可选，MNIST 实验)
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz
│   └── t10k-labels-idx1-ubyte.gz
└── uci/
    ├── abalone/abalone.data
    ├── appliances_energy/energydata_complete.csv
    ├── bodyfat/bodyfat
    ├── energy_efficiency/ENB2012_data.xlsx
    ├── letter/letter-recognition.data
    ├── pendigits/{pendigits.tra, pendigits.tes}
    ├── shuttle/{shuttle.scale, shuttle.scale.t}
    └── waveform/waveform.zip
```

数据版权归原作者；本仓库不再分发。
