# Table 5 复现报告 — `pendigits`

**Setting**: Equal-scale data streams (论文 Section 4.1).

- 数据集大小: `train=7494` / `test=3498` / `classes=10`
- 增量 batch 数: **5**
- BLS 配置: `N1=10, N2=10, N3=1000, λ=1e-06`

## Final accuracy & total training time

| Method | Accuracy (%) | Total Time (s) |
|---|---:|---:|
| Non-Incremental BLS | 98.03 | 0.0730 |
| Incremental BLS | 11.32 | 0.7156 |
| TiBLS | 98.03 | 0.3788 |
| Approximation Method | 97.08 | 0.1681 |
| RI-BLS | 98.03 | 0.0925 |
| **IMF-BLS (Ours)** | **98.03** | **0.5583** |

## Per-step accuracy (incremental learning curve)

| Step | Non-Incremental BLS | Incremental BLS | TiBLS | Approximation Method | RI-BLS | IMF-BLS (Ours) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 98.03 | 94.57 | 94.57 | 94.57 | 94.57 | 94.57 |
| 2 | — | 28.36 | 97.54 | 96.60 | 97.51 | 97.51 |
| 3 | — | 15.67 | 97.88 | 97.06 | 97.88 | 97.88 |
| 4 | — | 13.06 | 98.03 | 97.08 | 98.03 | 98.03 |
| 5 | — | 11.32 | 98.03 | 97.08 | 98.03 | 98.03 |