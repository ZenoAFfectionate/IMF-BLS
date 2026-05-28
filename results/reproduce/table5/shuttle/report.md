# Table 5 复现报告 — `shuttle`

**Setting**: Equal-scale data streams (论文 Section 4.1).

- 数据集大小: `train=43500` / `test=14500` / `classes=7`
- 增量 batch 数: **5**
- BLS 配置: `N1=10, N2=10, N3=3000, λ=1e-06`

## Final accuracy & total training time

| Method | Accuracy (%) | Total Time (s) |
|---|---:|---:|
| Non-Incremental BLS | 99.59 | 1.9259 |
| Incremental BLS | 77.12 | 67.5659 |
| TiBLS | 99.59 | 37.5663 |
| Approximation Method | 99.53 | 5.0914 |
| RI-BLS | 99.59 | 2.3781 |
| **IMF-BLS (Ours)** | **99.59** | **14.6306** |

## Per-step accuracy (incremental learning curve)

| Step | Non-Incremental BLS | Incremental BLS | TiBLS | Approximation Method | RI-BLS | IMF-BLS (Ours) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 99.59 | 99.47 | 99.47 | 99.47 | 99.47 | 99.47 |
| 2 | — | 0.30 | 99.56 | 99.52 | 99.56 | 99.56 |
| 3 | — | 0.28 | 99.57 | 99.54 | 99.57 | 99.57 |
| 4 | — | 0.33 | 99.57 | 99.52 | 99.56 | 99.56 |
| 5 | — | 77.12 | 99.59 | 99.53 | 99.59 | 99.59 |