# Table 5 复现报告 — `letter`

**Setting**: Equal-scale data streams (论文 Section 4.1).

- 数据集大小: `train=16000` / `test=4000` / `classes=26`
- 增量 batch 数: **5**
- BLS 配置: `N1=10, N2=10, N3=3000, λ=1e-06`

## Final accuracy & total training time

| Method | Accuracy (%) | Total Time (s) |
|---|---:|---:|
| Non-Incremental BLS | 96.43 | 0.7262 |
| Incremental BLS | 5.10 | 8.1030 |
| TiBLS | 96.43 | 4.5875 |
| Approximation Method | 93.27 | 2.5404 |
| RI-BLS | 96.43 | 1.4493 |
| **IMF-BLS (Ours)** | **96.43** | **7.3268** |

## Per-step accuracy (incremental learning curve)

| Step | Non-Incremental BLS | Incremental BLS | TiBLS | Approximation Method | RI-BLS | IMF-BLS (Ours) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 96.43 | 83.30 | 83.30 | 83.30 | 83.30 | 83.30 |
| 2 | — | 37.70 | 92.60 | 89.22 | 92.60 | 92.60 |
| 3 | — | 8.92 | 95.17 | 91.33 | 95.17 | 95.17 |
| 4 | — | 6.10 | 96.03 | 92.75 | 96.03 | 96.03 |
| 5 | — | 5.10 | 96.43 | 93.27 | 96.43 | 96.43 |