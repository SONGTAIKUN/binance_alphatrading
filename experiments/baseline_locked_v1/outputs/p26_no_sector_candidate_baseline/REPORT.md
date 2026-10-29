# P26 No-sector Residual v2 潜在新 Baseline

## 已正式固化的结构

- Alpha residual model：`alpha_residual_v2_no_sector_ols720`，仅使用 BTC、ETH、ex-self market 三个因子。
- Portfolio risk model：保留已验证的 `portfolio_risk_v1_p15c_namespaced`，字段迁入独立 `risk_beta_*` 命名空间。
- allocator 兼容字段只在入口由 `risk_beta_* -> beta_*` 显式映射，不再让 alpha beta 和 risk beta 静默混用。
- 固定 48/72/96/168h terminal residual labels 均使用 t 时刻 beta；波动率归一化只使用过去 336h。
- 横截面标签使用 [-1, 1] centered rank；标准选择标签为 `label_slow_low_turnover`。
- path-aware labels 与 adverse excursion 字段作为独立标签和诊断保留。
- 所有 residual/label 审计均为 build-fail gate。

## 自动审计

- 通过 12/12 项硬门槛，其中 Alpha beta 与 Portfolio risk beta 分别做 past-only 抽样复核。
- Alpha residual model 对 majors 与其他币统一取消 sector factor，因此 BNB 的结构性奇异问题已从 residual/label 生成层移除。

## Nested OOS 结果

- P26-core 不启用新 alpha family：全部 fold 保留 P15c 信号、eligibility 与已验证的 portfolio risk model；结构升级只作用于新的 residual/features/labels 研究入口。
- 默认 6bps：return 9.42%，gross 15.08%，Sharpe 0.800，maxDD -8.89%。
- 8bps：return 7.59%，Sharpe 0.656，maxDD -9.37%。
- 2024Q2-Q3 @8bps：return -4.54%，maxDD -6.86%。
- positive fold rate 62.50%，worst-5 folds 平均 -2.14%。
- 对照 P15c 默认 6bps return 9.42%，Sharpe 0.800。

## 被拒绝的 slow-label overlay

P20 practical 用 `label_slow_low_turnover` 激活 8/32 folds 后，6bps return 3.76%，gross 9.31%，Sharpe 0.277。该 overlay 明显劣于 P15c/P26-core，已作为失败实验隔离在 `nested_alpha_overlay_run/`，不会进入 baseline。

## 被拒绝的 risk-v2 替换

把 Portfolio risk model 也替换成 no-sector beta 并重投影全部 P15c 权重后，6bps return 5.44%，gross 10.88%，Sharpe 0.557。它低于 P15c，因此只保留在 `nested_risk_v2_reprojection_run/` 作为进一步研究备选。

## 定位与限制

P26 是后续探索应使用的潜在新 baseline，不覆盖正式 P15c。no-sector 与 P20 practical 是在查看 P25 后确定的，因此这条曲线不能被解释为未经观察的纯 OOS 证据；它的价值是把推荐结构固化成可复现、可审计、可继续做 prospective 验证的研究起点。

## 复现

```bash
python scripts/run_p26_no_sector_candidate_baseline.py
```
