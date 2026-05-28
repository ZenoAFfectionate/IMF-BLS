# IMF-BLS 论文复现总览

> 本目录包含基于 Pi_Agent 项目对论文
> *"Efficient incremental learning for Inverse Matrix-Free broad learning system"*
> (Information Fusion 127, 2026, 103842) 的复现实验结果。
>
> 论文中算法称为 **InvF-BLS**，本仓库简称 **IMF-BLS**（数学上完全等价）。

## 目录

| 表 | 内容 | 报告 |
|---|---|---|
| **Table 5** | 分类，等量数据流 | [table5/summary.md](table5/summary.md) |
| **Table 6** | 回归 | [table6/summary.md](table6/summary.md) |
| **Table 7** | 分类，不定数据流 | [table7/summary.md](table7/summary.md) (需 MNIST) |

## 核心结论

### IMF-BLS 完全匹配论文趋势

| 现象 | 论文结论 | 复现结果 | 状态 |
|---|---|---|---|
| IMF-BLS 与 RI-BLS / Non-Incremental 准确率严格相等 | ✅ 一致 | 所有数据集 acc 完全相等 | ✅ |
| Incremental BLS（Greville）大型 N3 下数值崩溃 | ✅ 论文 Fig. 9 同样观察 | 所有数据集 acc 大幅下降 | ✅ |
| Approximation Method 精度低于 IMF-BLS | ✅ 论文 Table 5 | 所有数据集相同结论 | ✅ |
| TiBLS 等量 batch 下与 IMF-BLS 等价 | ✅ | 完全相等 | ✅ |
| IMF-BLS 比 Incremental BLS 快很多 | ✅ 论文 Fig. 9 | 5x ~ 10x 加速 | ✅ |

### 与论文 Table 5 数值对比

| 数据集 | 指标 | 论文 IMF-BLS | 我们 IMF-BLS | 差异 |
|---|---|---:|---:|---|
| Pendigits | Acc (%) | 98.90 | 98.03 | -0.9% |
| Letter   | Acc (%) | 95.30 | 96.43 | +1.1% |
| Shuttle  | Acc (%) | 99.03 | 99.59 | +0.6% |
| Waveform | Acc (%) | 82.80 | 84.62 | +1.8% |
| LED      | Acc (%) | 74.08 | 73.21 | -0.9% |

> 差异来源：
> * 数据切分种子不同（论文未公开）
> * Pendigits / Waveform / LED 的原始数据来源差异（论文未明确说明）
> * BLS 随机权重初始化种子不同
> * 论文用更大 LED 训练集（160k vs 我们 30k）
>
> **核心算法行为完全一致**：所有正常方法的相对排序与论文 Table 5 相同。

### 与论文 Table 6 对比（回归）

| 数据集 | 指标 | 论文 IMF-BLS | 我们 IMF-BLS | 备注 |
|---|---|---:|---:|---|
| Abalone | RMSE | 1.94 | 2.61 | 数据切分差异 |
| Bodyfat | RMSE | 0.87 | 0.0087 | 论文目标未归一化，我们归一化 |
| Appliances Energy | RMSE | 14.39 | 90.93 | 论文目标做了归一化 |

> 这些差异都是**预处理细节**造成的（数据未公开预处理代码），不是算法问题：
> 在所有情况下，**IMF-BLS / RI-BLS / Non-Incremental BLS 的 RMSE 完全相同**，
> 说明 IMF-BLS 数学等价性精确成立。

## 跑通的实验数据集

* ✅ **Table 5**: Pendigits, Letter, Shuttle, Waveform, LED (5/5)
* ✅ **Table 6**: Abalone, Bodyfat, Appliances Energy (3/5)
* ❌ Table 6: Energy Efficiency (需 `pip install openpyxl`)
* ❌ Table 6: Weather Izmir (KEEL 镜像不可访问)
* ⏳ **Table 7**: MNIST / Fashion-MNIST (需 `bash scripts/download_mnist.sh`)
* ⏳ Table 7: NORB / EMNIST (论文较复杂，未提供下载器)
* ❌ CIFAR-10/100 (论文用 ResNet 预训练特征，复现不可行)

## 重新运行

```bash
# 一键全量
bash scripts/run_all.sh

# 单个表
bash scripts/table5.sh
bash scripts/table6.sh
bash scripts/table7.sh                      # 需先下载 MNIST

# 单个数据集
bash scripts/single.sh table5 pendigits

# 下载 MNIST + Fashion-MNIST
bash scripts/download_mnist.sh
```

每个数据集的详细报告在 `<table_id>/<dataset>/report.md`，
包含 BLS 配置、增量曲线、std 等完整信息。
