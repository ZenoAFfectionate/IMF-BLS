# Table 6 复现报告 — `abalone` (Regression)

- 数据集: `train=2784` / `test=1393` / `attributes=10`
- BLS 配置: `N1=10, N2=10, N3=600, λ=1e-06`

## Train Time / RMSE / STD（多次运行均值 ± std）

| Method | Train Time (s) | Test RMSE | Test STD |
|---|---:|---:|---:|
| Non-Incremental BLS | 0.0177 | 2.6088 | 0.1872 |
| Incremental BLS | 0.0770 | 12.3740 | 2.5915 |
| Approximation Method | 0.0375 | 3.0181 | 0.1293 |
| TiBLS | 0.0478 | 2.6059 | 0.1933 |
| RI-BLS | 0.0252 | 2.6088 | 0.1872 |
| **IMF-BLS (Ours)** | **0.1292** | **2.6088** | **0.1872** |