# 研究链路与文件说明

这份 clean repo 现在包含两类入口：

1. 当前最新 baseline 的 replay：直接复现 P16/P15c 的 walk-forward OOS 曲线。
2. 从原始数据重新构建研究面板：生成 residual panel、慢速 residual labels、P2 手工慢 alpha grid、Binance derivatives 特征面板。

## 目录结构

- `configs/baseline.yaml`：20 个币的 universe、sector 分组、基础 residual/stat-arb 参数。
- `experiments/baseline_locked_v1/configs/baseline_locked_v1.yaml`：当前 locked baseline replay 使用的固定配置。
- `scripts/run_latest_baseline.py`：最新 P16/P15c baseline replay 入口。
- `scripts/run_baseline.py`：从 15m OHLCV kline 生成 `asset_residual_panel.csv`，并跑基础 walk-forward baseline。
- `scripts/build_p1_slow_residual_labels.py`：生成 48h/72h/96h/168h ex-ante residual labels、composite slow labels、path-aware labels。
- `scripts/run_p2_manual_slow_alpha_grid.py`：统一评估 OHLCV-only slow residual alpha families，输出 IC、PnL、turnover、long/short/asset attribution。
- `scripts/download_binance_derivatives.py`：从 Binance public data 下载 funding、index price、mark price、premium index、metrics。
- `scripts/build_binance_derivatives_panel.py`：把 Binance 原始 derivatives zip 转成 hourly derivatives panel。
- `scripts/run_p6b_derivatives_feature_expansion.py`：把 derivatives feature 接入 P6b diagnostics，输出 retained feature 列表。

## 从原始 OHLCV 重新生成 residual panel

默认配置假设 15m kline CSV 放在：

```text
data/klines/<SYMBOL>/15m.csv
```

每个 CSV 至少需要这些列：

```text
timestamp, open, high, low, close, volume, quote_volume, trades, taker_buy_base, taker_buy_quote
```

运行：

```bash
python scripts/run_baseline.py \
  --config configs/baseline.yaml \
  --output-dir experiments/baseline_locked_v1/inputs
```

这个命令会生成：

```text
experiments/baseline_locked_v1/inputs/asset_residual_panel.csv
```

同时会输出一份基础 baseline 的 folds、positions、equity 和 metrics。注意：当前 GitHub clean 包默认 replay 使用的是更小的 `p16_replay_panel_slim.csv.gz` 分片；`asset_residual_panel.csv` 是用于重建完整研究链路的原始 residual 面板。

## 生成 P1 慢速 residual labels

```bash
python scripts/build_p1_slow_residual_labels.py
```

输出：

```text
experiments/baseline_locked_v1/outputs/p1_slow_labels/slow_residual_labels.csv.gz
```

这里的 future residual return 用的是 t 时刻已知 beta，不使用未来 beta；path-aware label 会惩罚持有期内 adverse excursion。

## 跑 P2 手工慢 alpha grid

```bash
python scripts/run_p2_manual_slow_alpha_grid.py
```

输出包括：

```text
experiments/baseline_locked_v1/outputs/p2_manual_slow_alpha_grid/candidate_summary.csv
experiments/baseline_locked_v1/outputs/p2_manual_slow_alpha_grid/ic_summary.csv
experiments/baseline_locked_v1/outputs/p2_manual_slow_alpha_grid/manual_slow_feature_panel.csv.gz
```

这个阶段统一评估 residual momentum、high-momentum、OU-gated mean reversion、liquidity/volatility/beta quality 等 OHLCV-only slow features。

## 下载并构建 Binance derivatives 数据

先下载原始 zip：

```bash
python scripts/download_binance_derivatives.py \
  --config configs/baseline.yaml \
  --panel experiments/baseline_locked_v1/inputs/asset_residual_panel.csv \
  --workers 32
```

再构建 hourly panel：

```bash
python scripts/build_binance_derivatives_panel.py \
  --config configs/baseline.yaml \
  --panel experiments/baseline_locked_v1/inputs/asset_residual_panel.csv
```

输出：

```text
experiments/baseline_locked_v1/inputs/binance_derivatives/derivatives_hourly_panel.csv.gz
```

所有 derivatives 特征都按 ex-ante 方式构造；funding、metrics 等数据只做有限 forward fill，避免把缺失 archive 段误当作长期已知信息。

## 跑 P6b derivatives feature diagnostics

```bash
python scripts/run_p6b_derivatives_feature_expansion.py
```

输出：

```text
experiments/baseline_locked_v1/outputs/p6b_derivatives_feature_expansion/retained_features.csv
experiments/baseline_locked_v1/outputs/p6b_derivatives_feature_expansion/candidate_diagnostics.csv
```

这一步只做 diagnostics 和 retained feature selection，不直接把 derivatives 当成无约束 alpha 使用。P9/P16 里真正使用的 derivatives 增量特征需要继续通过 walk-forward/nested validation 固定到每个 test fold。

## 最新 baseline replay

如果只需要复现当前最新结果，不需要重新下载数据：

```bash
python scripts/run_latest_baseline.py
```

脚本会自动把：

```text
experiments/baseline_locked_v1/inputs/p16_replay_panel_slim.csv.gz.parts/part-*
```

合并成：

```text
experiments/baseline_locked_v1/inputs/p16_replay_panel_slim.csv.gz
```

并输出 replay 结果到：

```text
experiments/baseline_locked_v1/outputs/latest_baseline_replay/
```
