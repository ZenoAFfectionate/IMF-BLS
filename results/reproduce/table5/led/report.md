# Table 5 复现报告 — `led`

**Setting**: Equal-scale data streams (论文 Section 4.1).

- 数据集大小: `train=30000` / `test=8000` / `classes=10`
- 增量 batch 数: **5**
- BLS 配置: `N1=10, N2=10, N3=5000, λ=1e-06`

## Final accuracy & total training time

| Method | Accuracy (%) | Total Time (s) |
|---|---:|---:|
| Non-Incremental BLS | 73.21 | 3.0413 |
| Incremental BLS | 10.55 | 43.2923 |
| TiBLS | 73.21 | 23.2630 |
| Approximation Method | 63.46 | 9.9421 |
| RI-BLS | 73.21 | 5.3871 |
| **IMF-BLS (Ours)** | **73.21** | **29.7953** |

## Per-step accuracy (incremental learning curve)

| Step | Non-Incremental BLS | Incremental BLS | TiBLS | Approximation Method | RI-BLS | IMF-BLS (Ours) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 73.21 | 39.38 | 39.38 | 39.38 | 39.38 | 39.38 |
| 2 | — | 9.93 | 68.20 | 49.61 | 68.20 | 68.20 |
| 3 | — | 9.95 | 71.55 | 55.76 | 71.55 | 71.55 |
| 4 | — | 10.67 | 72.70 | 60.09 | 72.70 | 72.70 |
| 5 | — | 10.55 | 73.21 | 63.46 | 73.21 | 73.21 |