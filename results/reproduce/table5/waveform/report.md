# Table 5 复现报告 — `waveform`

**Setting**: Equal-scale data streams (论文 Section 4.1).

- 数据集大小: `train=4200` / `test=800` / `classes=3`
- 增量 batch 数: **5**
- BLS 配置: `N1=10, N2=10, N3=600, λ=1e-06`

## Final accuracy & total training time

| Method | Accuracy (%) | Total Time (s) |
|---|---:|---:|
| Non-Incremental BLS | 84.62 | 0.0233 |
| Incremental BLS | 29.38 | 0.1624 |
| TiBLS | 84.62 | 0.0892 |
| Approximation Method | 75.25 | 0.0527 |
| RI-BLS | 84.62 | 0.0322 |
| **IMF-BLS (Ours)** | **84.62** | **0.1584** |

## Per-step accuracy (incremental learning curve)

| Step | Non-Incremental BLS | Incremental BLS | TiBLS | Approximation Method | RI-BLS | IMF-BLS (Ours) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 84.62 | 59.75 | 59.75 | 59.75 | 59.75 | 59.75 |
| 2 | — | 34.62 | 78.12 | 64.88 | 78.12 | 78.12 |
| 3 | — | 31.25 | 82.75 | 70.88 | 82.75 | 82.75 |
| 4 | — | 29.62 | 82.88 | 74.62 | 82.88 | 82.88 |
| 5 | — | 29.38 | 84.62 | 75.25 | 84.62 | 84.62 |