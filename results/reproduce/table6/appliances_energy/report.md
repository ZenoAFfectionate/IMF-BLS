# Table 6 复现报告 — `appliances_energy` (Regression)

- 数据集: `train=15788` / `test=3947` / `attributes=6`
- BLS 配置: `N1=10, N2=10, N3=600, λ=1e-06`

## Train Time / RMSE / STD（多次运行均值 ± std）

| Method | Train Time (s) | Test RMSE | Test STD |
|---|---:|---:|---:|
| Non-Incremental BLS | 0.0727 | 90.9347 | 0.0487 |
| Incremental BLS | 2.7494 | 835.7369 | 80.2754 |
| Approximation Method | 0.1547 | 92.4274 | 0.0613 |
| TiBLS | 1.7538 | 90.9329 | 0.0491 |
| RI-BLS | 0.0773 | 90.9347 | 0.0487 |
| **IMF-BLS (Ours)** | **0.4233** | **90.9347** | **0.0487** |