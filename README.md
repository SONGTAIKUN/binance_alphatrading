# Binance Alpha Trading Clean Baseline

这是当前最新一版多币种 crypto residual/stat-arb baseline 的干净可运行包。

本仓库只保留 P16/P15c 这一版 baseline 复现所需的文件：

- `scripts/run_latest_baseline.py`：最新 baseline replay 入口。
- `src/crypto_stat_arb/`：组合、特征、配置、walk-forward 基础库。
- `experiments/baseline_locked_v1/inputs/`：固定 walk-forward folds、replay 必需列清单、slim 面板分片。
- `experiments/baseline_locked_v1/outputs/p9_clean_baseline_package/selected_incremental_by_fold.csv`：P9 每个 fold 由 train/validation 选出的 AlphaGPT 增量公式和参数。
- `experiments/baseline_locked_v1/outputs/p16_clean_candidate_package/`：P16/P15c 最新候选 baseline 的完整结果包。
- `experiments/baseline_locked_v1/outputs/p3d_p4_gated_rerank_alphagpt/candidate_formulas.csv` 与 `outputs/p6b_derivatives_feature_expansion/retained_features.csv`：replay 固定公式文本和特征映射。

不包含历史实验输出、原始 Binance 下载缓存、archive 目录或 P1-P17 各轮中间产物。

## 运行方式

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_latest_baseline.py
```

首次运行时，脚本会自动把：

```text
experiments/baseline_locked_v1/inputs/p16_replay_panel_slim.csv.gz.parts/part-*
```

合并成：

```text
experiments/baseline_locked_v1/inputs/p16_replay_panel_slim.csv.gz
```

该合并后的 gzip 文件已被 `.gitignore` 忽略，避免把 140MB 单文件提交到 GitHub。

## 当前基线结果

默认成本假设为单边 `6bps`，即 `4bps fee + 2bps slippage`。

P16/P15c clean baseline 的关键结果：

- net total return: `16.62%`
- gross before cost: `21.06%`
- Sharpe: `2.427`
- max drawdown: `-1.79%`
- mean turnover: `0.3624`
- 20bps net return: `6.89%`
- 30bps net return: `0.43%`

详细报告在：

```text
experiments/baseline_locked_v1/outputs/p16_clean_candidate_package/REPORT.md
experiments/baseline_locked_v1/outputs/p16_clean_candidate_package/COLLABORATOR_BASELINE_SUMMARY.md
```

## 方法简述

策略先在 residual 层做横截面相对价值排序，再用 AlphaGPT 搜到的慢速 residual/derivatives 条件因子做 fold-local 增量 rerank。P9 固定每个 fold 的 AlphaGPT 信号选择；P16/P15c 在 P9 的可交易 long/short pool 上加入 extreme latent cluster cap 和 turnover-aware guard。

回测 replay 使用固定 walk-forward 选择：

1. 每个 fold 的公式和交易参数已经在该 fold 的 train/validation 中选定。
2. `run_latest_baseline.py` 只把这些固定选择应用到对应 test window。
3. 组合构建保持 intercept/BTC/ETH/market beta 严格中性，并保留 post-neutralization symbol cap。

因此这个包用于复现当前最新 baseline，不用于重新执行全部历史研究搜索。
