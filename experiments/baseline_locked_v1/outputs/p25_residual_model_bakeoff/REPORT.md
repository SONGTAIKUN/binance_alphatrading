# P25 Residual Model Bake-off 报告

## 实验口径

六套 residualizer 统一使用 720h 滚动窗口、240 条最小观测、ex-ante labels、同一套 P20/P21 family library、strict-neutral allocator 和仅拼接 test 的 nested OOS 流程。P20 balanced 是查看结果前固定的主比较协议，其他协议只用于敏感性检查。

## 核心结果

- 固定 balanced 协议下，8bps OOS 收益最高：`crypto_ff_beta_shrink`（14.10%）。
- factor condition number 中位数最低：`legacy_orthogonalized`（1.07）。
- beta 每小时变动最低：`legacy_ridge`（0.001004）。
- 2024Q2-Q3 的 8bps 收益最好：`legacy_orthogonalized`（1.08%）。

## 主比较结果

| 模型 | 条件数中位数 | beta 变动 | residual vol CV | selected IC | active folds | OOS gross | OOS @8bps | Sharpe @6bps | maxDD @6bps | 2024Q2-Q3 @8bps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_full | 155.36 | 0.002649 | 0.396 | -0.0210 | 3.1% | 11.94% | 4.48% | 0.512 | -11.39% | -4.54% |
| legacy_no_sector | 20.62 | 0.001852 | 0.402 | -0.0191 | 6.2% | 16.97% | 9.16% | 0.791 | -12.14% | -4.54% |
| legacy_orthogonalized | 1.07 | 0.002046 | 0.399 | 0.0616 | 21.9% | 18.46% | 10.42% | 0.636 | -7.25% | 1.08% |
| legacy_ridge | 155.36 | 0.001004 | 0.397 | -0.0338 | 3.1% | 12.97% | 5.41% | 0.589 | -10.60% | -4.54% |
| crypto_ff | 22.29 | 0.002465 | 0.377 | 0.0493 | 6.2% | 17.40% | 9.54% | 0.891 | -8.79% | -1.62% |
| crypto_ff_beta_shrink | 22.29 | 0.001783 | 0.366 | 0.0426 | 9.4% | 22.35% | 14.10% | 1.169 | -8.79% | -1.85% |

## 协议敏感性：8bps OOS return

| 模型 | Balanced | Practical | Loose |
|---|---:|---:|---:|
| legacy_no_sector | 9.16% | 12.96% | 13.01% |
| legacy_orthogonalized | 10.42% | 2.97% | 0.42% |
| crypto_ff_beta_shrink | 14.10% | 11.50% | 8.79% |

## Review 结论

- `legacy_orthogonalized` 是结构诊断赢家：condition number 从 legacy 的 155.36 降到 1.07，并在 fixed balanced 下把 2024Q2-Q3 的 8bps 收益从 P15c 的 -4.54% 改善到 +1.08%。但它对 P20 协议较敏感，放宽到 practical/loose 后全样本表现明显退化。
- `crypto_ff_beta_shrink` 是 balanced 协议下的收益赢家，beta 和 residual volatility 也更稳定；但只激活 3/32 folds，且 2024Q2-Q3 仍为 -1.85%，证据宽度不足，不能仅凭 +14.10% 升级 baseline。
- `legacy_no_sector` 是最值得继续验证的简化候选。它消除了 BNB 的近奇异 sector 结构，condition number 降到约 20.62；在 practical/loose 下 8bps OOS 分别为 +12.96%/+13.01%，且 2024Q2-Q3 分别为 +6.15%/+10.19%。不过 balanced 结果较弱，下一步必须预注册协议后再做独立验证。
- 单一 `-resid_z` 的 mean-reversion IC 在六套模型里都接近 0。真正产生区分度的是 residual quality、beta stability 和 slow residual family，经 P20/P21 选择后再接 allocator。
- `legacy_ridge` 证明 estimator shrinkage 确实能稳定 beta，但并未修复 factor 共线性，也没有转化成更好的 OOS alpha；ridge 不能替代 factor design。

## 解读约束

- Ridge 可以稳定 beta，但不会降低原始 factor correlation；factor design 和 estimator design 必须分开评价。
- condition number 低不是充分条件，仍需同时改善 selected-label IC 和 portfolio OOS。
- active-fold rate 低意味着大部分时间仍是 P15c fallback，因此组合提升的证据覆盖面有限。
- 2024Q2-Q3 是事先固定的压力窗口，只用于评估，没有参与参数选择。

## 主要文件

- `bakeoff_scorecard.csv`：固定主协议的总比较。
- `factor_collinearity.csv`：逐币/全局 factor correlation 与 condition number。
- `residual_diagnostics.csv`：beta、residual volatility 与横截面 dispersion 稳定性。
- `simple_label_ic.csv`、`family_fold_metrics.csv`：label IC 与 P20/P21 诊断。
- `portfolio_summary.csv`、`cost_sensitivity.csv`、`fold_stability.csv`：nested OOS 组合诊断。
- `regime_2024q2_q3_summary.csv`：固定压力窗口结果。
