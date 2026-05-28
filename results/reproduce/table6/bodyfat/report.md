# Table 6 复现报告 — `bodyfat` (Regression)

- 数据集: `train=168` / `test=84` / `attributes=14`
- BLS 配置: `N1=10, N2=10, N3=200, λ=1e-06`

## Train Time / RMSE / STD（多次运行均值 ± std）

| Method | Train Time (s) | Test RMSE | Test STD |
|---|---:|---:|---:|
| Non-Incremental BLS | 0.0015 | 0.0087 | 0.0012 |
| Incremental BLS | 0.0016 | 0.0087 | 0.0012 |
| Approximation Method | 0.0009 | 0.0065 | 0.0003 |
| TiBLS | 0.0021 | 0.0086 | 0.0010 |
| RI-BLS | 0.0025 | 0.0087 | 0.0012 |
| **IMF-BLS (Ours)** | **0.0129** | **0.0087** | **0.0012** |