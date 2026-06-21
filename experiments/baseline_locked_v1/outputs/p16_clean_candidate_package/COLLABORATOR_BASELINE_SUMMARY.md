# 加密货币 Residual Relative-Value Baseline 总结

## 一句话概括

当前最新版研究基准是 `P16 / P15c extreme-cluster turnover-aware candidate`。它不是单币价格预测模型，也不是传统的两币 spread mean reversion，而是一个多币种横截面 residual relative-value / 统计套利系统：

1. 先把每个币的收益拆成 BTC、ETH、市场因子等共同风险暴露，以及剩下的个体 residual。
2. 用一套慢速、低换手、可解释的 residual / derivatives 因子库，对每个币做横截面相对排序。
3. AlphaGPT 在这些已审核的因子 primitive 上组合公式，寻找增量 rerank 信号。
4. 交易系统把排序信号转成 long top / short bottom 的市场中性组合，再通过严格 beta 中性优化器、单币权重上限、cluster cap 和 turnover guard 执行交易。

需要特别说明：P16 目前是 **research baseline candidate**。正式 baseline 仍保留为 P9，还没有被覆盖。P16 是在 P9 的信号层基础上增加了一个更稳健的组合构建 / 风控 overlay：当组合过度集中在某个 rolling residual cluster 中时，才触发温和的 cluster gross cap；如果这个调整会带来过多换手，则回退到 P9 sizing。

## 当前 Baseline 身份

- 名称：`p16_p15c_extreme_cluster_turnover_candidate`
- 父版本：`p9_clean_drop_k01_cap010`
- 数据频率：1 小时 K 线
- 交易 horizon：72 小时
- 调仓间隔：72 小时
- 执行延迟：1 根 bar
- 组合结构：做多 4 个币，做空 4 个币
- 默认成本假设：单边 6bps，其中 4bps fee + 2bps slippage
- 中性约束：intercept、BTC beta、ETH beta、market beta
- 单币最大权重：12.5%
- 当前状态：research baseline candidate，已进入下一轮 review

核心 OOS 结果：

| 版本 | 6bps net | 扣费前 gross | Sharpe | 最大回撤 | 20bps net | 30bps net |
|---|---:|---:|---:|---:|---:|---:|
| P9 official reference | 15.54% | 19.92% | 2.211 | -1.79% | 5.92% | -0.46% |
| P15b cluster cap reference | 16.02% | 20.44% | 2.315 | -1.80% | 6.31% | -0.13% |
| P16 / P15c candidate | 16.62% | 21.06% | 2.427 | -1.79% | 6.89% | 0.43% |

P16 相比 P9：

- 净收益提高约 +1.08%。
- 扣费前 gross 提高约 +1.14%。
- Sharpe 提高约 +0.216。
- 30bps 成本下从负收益改善为小幅正收益。
- 平均换手基本不变。
- rolling cluster HHI 降低约 5.87%。
- max cluster gross share 降低约 4.18%。

## 策略到底在预测什么

这个系统不直接预测原始价格涨跌。它预测的是 residual relative value。

可以把每个币的收益拆成：

```text
return_i
= beta_btc_i * BTC
+ beta_eth_i * ETH
+ beta_market_i * crypto_market
+ idiosyncratic residual_i
```

策略真正关心的是：

```text
在剥离 BTC / ETH / 市场共同风险之后，
哪些币的 residual 相对更有吸引力？
哪些币的 residual 相对更差？
这个 residual edge 是否足够稳定，能覆盖手续费、滑点、换手和组合风险？
```

所以它更接近现代多资产 stat-arb ranker，而不是经典 pair trading 中的单一 spread 策略。它交易的是一个市场中性的 residual portfolio。

## 因子库结构

当前因子库被分成几层。这个分层很重要，因为并不是所有有用变量都应该直接变成 long/short alpha。有些变量适合做排序因子，有些变量更适合做 risk state、gate 或 sizing conditioner。

### 1. 核心 Residual / OHLCV 因子

这一层只依赖 OHLCV 和 residualized return，是慢速 alpha 的基础。

Residual momentum / continuation：

- `res_mom_72_12`
- `res_mom_168_24`
- `res_mom_336_24`
- residual high momentum
- residual moving-average slope
- residual trend quality

这里的关键设计是：计算 momentum 时跳过最近 6 到 24 小时，避免把 microstructure reversal、liquidation wick 和 bar close noise 混进慢速信号里。

Residual mean reversion：

- residual z-score versus slow EWMA anchor
- OU speed
- half-life confidence
- beta stability
- stationarity filter

这类因子不会无条件启用。只有当 residual 行为看起来具备一定均值回复结构时，mean reversion 因子才更可信。

Liquidity / capacity：

- dollar volume trend
- low capacity penalty
- liquidity quality
- volume share trend
- turnover pressure

这些因子用于避免信号只在成本过高、容量不足或成交质量差的币上有效。

Volatility / beta quality：

- realized vol
- idiosyncratic vol
- vol-of-vol
- beta instability
- residual purity

这些既可以作为 alpha conditioner，也可以作为 risk-quality control。

当前保留下来的 OHLCV-only 慢因子包括：

| 因子 | 经济含义 |
|---|---|
| `C05_low_capacity_penalty` | residual momentum，同时惩罚低容量和 breakout risk |
| `C02_dollar_volume_mom` | 有 dollar volume trend 支撑的 residual momentum |
| `D04_low_vov_momentum` | residual momentum，但偏好低 vol-of-vol |
| `E03_basket_diversifier_mom` | 带 residual basket diversification 的 momentum |

### 2. Derivatives / Crowding 因子

后续实验加入了 funding、basis、premium、OI 和 trader positioning 相关数据。这些变量有用，但不能无约束地直接当 alpha 用，因为 raw crowding signal 很容易过拟合或在 squeeze regime 中失效。

Funding / carry：

- `H01_funding_contra_168`
- `H05_carry_unwind_contra`
- `H03_slow_mom_funding_penalty`

Basis / premium：

- `I01_premium_contra_168`
- `I05_mom_basis_quality`

Trader crowding / positioning：

- `J04_top_trader_contra`
- taker imbalance
- account long-short ratio
- top trader long-short ratio

Derivatives composite：

- `K02_uncrowded_slow_mom`
- `K05_deriv_p3d_diversifier`

一个重要经验是：`K01_deriv_crowding_contra` 在单独诊断里看起来很强，但作为直接 incremental alpha 比较脆弱。因此 P9 明确移除了 K01 驱动的增量公式；如果某个 fold 选到了 K01 相关增量，就回退到更保守的 P3d fold selection。

这体现了当前系统的原则：derivatives / crowding 数据很有价值，但多数时候更适合作为 conditioner、gate 或 risk-state variable，而不是完全自由的方向性排序信号。

### 3. Risk-State / 诊断变量

这一层不一定直接交易，更多用于诊断或风控模块：

- market vol regime
- residual dispersion
- cross-asset correlation
- funding / premium stress
- OI growth / unwind
- squeeze hazard
- liquidation hazard
- rolling residual cluster concentration
- symbol dependency
- turnover / cost sensitivity

当前 P16 已经把 rolling residual cluster concentration 用进 allocator。其他 risk-state 变量目前主要保存在诊断库里，等待后续更严格的风控模块验证。

## AlphaGPT 如何使用因子库

AlphaGPT 在这里不是自由发挥的黑盒预测器，而是一个受约束的公式搜索器。

### 输入 primitive

AlphaGPT 只能使用已经通过筛选的慢速 primitive，包括：

- residual momentum
- residual mean reversion quality
- liquidity / capacity quality
- vol / beta quality
- funding / carry
- basis / premium
- crowding / positioning
- derivatives composite features

每个 primitive 都必须是 ex-ante 的，也就是在时间 `t` 只使用 `t` 及之前已经可见的数据计算，不能包含未来收益、未来 beta 或未来 label。

### 允许的 operator

搜索空间刻意保持简单、慢速、可解释：

```text
ADD
SUB
MUL
NEG
EMA_SLOW
CS_RANK
CS_ZSCORE
CLIP
```

常用 lookback 是 168h 和 336h。我们避免短周期 lookback、1h delta、裸短周期反转、raw future return 和任何未来数据。

### 公式例子

以下是曾经在不同 fold 中被选中过的公式例子：

```text
EMA_SLOW(P6B(J04_top_trader_contra), 168)
```

含义：对 top trader long-short crowding 的反向信号做慢速 EMA。

```text
MUL(
  0.50 * EMA_SLOW(P6B(I01_premium_contra_168), 168),
  EMA_SLOW(P6B(K05_deriv_p3d_diversifier), 168)
)
```

含义：把 premium contra 信号与 derivatives diversifier 信号相乘，只在两者同时支持时增强。

```text
SUB(
  EMA_SLOW(P6B(D04_low_vov_momentum), 168),
  0.70 * P6B(E03_basket_diversifier_mom)
)
```

含义：偏好低 vol-of-vol 的 residual momentum，同时扣掉部分 basket diversifier momentum。

```text
EMA_SLOW(P6B(K02_uncrowded_slow_mom), 336)
```

含义：寻找不拥挤的慢速 residual momentum，并进一步做慢速平滑。

重点是：AlphaGPT 不发明新的市场数据，也不直接拿未来收益训练黑盒模型。它是在已审核的因子库上，用简单 operator 组合候选公式。

## AlphaGPT 如何评价候选公式

每个 walk-forward fold 独立处理：

1. 只使用该 fold 之前的 train / validation 数据。
2. 从 approved primitive library 生成候选公式。
3. 在 validation window 上评估每个候选公式。
4. 通过 hard accept 规则筛选：
   - validation net return 不能明显变差，最好改善；
   - Sharpe 不能明显变差，最好改善；
   - 20bps / 30bps 成本鲁棒性必须过关；
   - turnover 不能明显上升；
   - short-side contribution 不能过度集中；
   - beta / market exposure 必须保持严格中性；
   - 复杂公式会被惩罚。
5. 每个 fold 只选该 fold validation 中最优的公式。
6. 选出的公式只应用到下一段 test window。

如果某个 fold 没有任何候选公式通过 hard accept，就回退到已有 baseline，不强行使用 AlphaGPT 增量。

## AlphaGPT 是否决定多空方向

不直接决定。

这是当前系统最重要的安全设计之一：

1. P4 / P3d base score 先决定 long / short eligible pool。
2. AlphaGPT candidate 只在已有 long / short pool 内做 rerank。
3. AlphaGPT 不能把一个 P4 不想 short 的币强行推入 short book。
4. candidate weight 有上限。
5. P9 中 candidate rerank weight 被限制在 0.10。
6. 脆弱的 K01 crowding candidate 被剔除。

这样可以避免某个过拟合的 derivatives 公式把整个组合推向错误的 crowded short 或 squeeze-prone trade。

## 交易系统如何使用信号

交易系统把 factor score 转成严格约束的 residual relative-value portfolio。

### Step 1：计算 ex-ante 特征

每个 rebalance timestamp，系统计算：

- OHLCV residual features；
- slow residual momentum / mean reversion quality；
- liquidity / capacity features；
- funding / basis / premium / crowding features；
- beta / residual risk features；
- rolling cluster risk diagnostics。

所有 rolling 特征只使用当前 rebalance timestamp 之前的数据。

### Step 2：生成横截面 score

score 是分层构造的：

```text
base_score = P4 / P3d residual relative-value score
candidate_score = AlphaGPT formula score
final_score = base_score + capped candidate rerank adjustment
```

final score 的解释是横截面的：

- score 越高，越适合进入 long candidate；
- score 越低，越适合进入 short candidate；
- 它不是绝对价格预测，而是相对强弱 / 相对价值排序。

### Step 3：选择 long / short pool

组合构建采用 top / bottom 结构：

- 做多 top 4；
- 做空 bottom 4；
- 用 entry / exit hysteresis 降低反复进出；
- 用 minimum holding state 降低短周期 churn；
- 用 P4 / P3d side gating 限制 AlphaGPT 的方向影响。

### Step 4：严格中性组合优化

初始 long / short 权重会经过 constrained optimizer。

优化器约束包括：

- dollar neutral，也就是 intercept neutral；
- BTC beta neutral；
- ETH beta neutral；
- broad market beta neutral；
- 单币权重上限约 12.5%；
- long 仍然是 long，short 仍然是 short；
- gross exposure 约束；
- 如果优化不可行则回退。

因此，实际 realized gross exposure 往往低于名义 gross。系统优先保证中性、约束和可执行性，而不是强行打满杠杆。

### Step 5：P15c rolling cluster overlay

P16 在 P9 基础上增加了 P15c allocator overlay。

每个 rebalance timestamp，系统使用当前时间之前的 residual returns 拟合 rolling latent residual clusters，然后检查组合是否过度集中在某一个 residual cluster。

P15c 参数网格：

- rolling lookback：336h / 720h / 1440h；
- cluster 数：3 / 4；
- trigger threshold：75% / 80% / 85% max cluster gross share；
- cluster cap：60% / 65% / 70%；
- max additional turnover：0% / 2%。

每个 fold 中，cluster 参数只在 validation 上选择，然后只应用到下一段 test window。

P16 实际行为：

- cluster overlay 在 172 个 rebalance periods 中触发 37 次；
- 真正应用 20 次；
- 因 turnover guard 跳过 12 次；
- 平均 gross exposure retention 约 99.83%。

所以 P15c 不是全局 risk-off filter。它只在 cluster concentration 极端、且额外换手可接受时介入。

### Step 6：成本与执行

回测中包含：

- 1 bar execution delay；
- 默认单边 6bps 成本；
- 20bps 和 30bps 成本敏感性；
- turnover accounting；
- gross before cost 和 net after cost；
- fold-level attribution；
- symbol-level attribution。

P16 在 30bps 成本下仍然略微为正，这是相比早期 baseline 的重要进展。

## Walk-Forward 与防未来函数

整个流程是 nested walk-forward：

1. 每个 fold 都有 train / validation 和未来 test。
2. AlphaGPT 公式选择只使用 train / validation。
3. P15c allocator 参数选择只使用 train / validation。
4. test window 在参数选完之前不参与任何选择。
5. rolling cluster model 只使用 `timestamp < current_timestamp` 的 residual returns。
6. forward returns / labels 只用于回测评分和 IC 诊断，不用于实时交易特征。

因此，信号选择和交易参数选择都对最终 stitched test PnL 做到了 OOS。

## 最近实验带来的经验

近期做过的实验里，不是所有 idea 都提升了 baseline：

- raw crowding alpha 直接交易很脆弱，所以 P9 移除了 K01。
- broad regime gate 容易把有效 exposure 一起砍掉。
- squeeze / liquidation hazard 特征作为诊断有价值，但暂时还不是更好的直接 alpha。
- P17 symbol reliability shrinkage 没有改善 P16：收益下降，symbol dependency 也没有降低。
- P15c cluster-aware allocator 是目前最有效的增量：收益、成本鲁棒性和 cluster risk 都更好。

当前最重要的结论是：

```text
信号层应保持慢速、residual、受约束；
derivatives / crowding 特征更适合做增量 rerank 或 risk-state conditioner；
近期真正有效的提升主要来自 allocator / portfolio construction 层。
```

## 已知限制

这仍然是 research baseline，不是生产交易系统。

当前限制：

- universe 只有 20 个高流动性 crypto symbols；
- 执行模型是固定 bps fee / slippage，不是 order-book simulation；
- funding cashflow 还没有完整作为真实 carry PnL 建模；
- 没有完整模拟动态保证金、强平和 borrow/margin 限制；
- 交易所异常、下架、数据 stale 的工程处理还不完整；
- P16 已经通过 nested OOS review，但还需要 live paper trading。

## 给合作者的简短版本

这是一套多币种 crypto residual relative-value 策略。我们先剥离每个币对 BTC、ETH 和市场的共同风险暴露，然后在 residual 层构建慢速、低换手、可解释的因子库，包括 residual momentum、mean reversion quality、liquidity、volatility、funding、premium 和 crowding。AlphaGPT 不是黑盒预测器，而是在这套因子库上用简单 operator 组合公式，并且每个公式必须通过 fold-local validation 才能在下一段 test 中使用。AlphaGPT 只负责在已有 long / short pool 内 rerank，不直接决定多空方向。交易系统再把 score 转成 long top / short bottom 组合，通过严格 beta-neutral optimizer、symbol cap 和 rolling residual-cluster cap 执行。当前 P16 / P15c candidate 相比 P9 有更好的净收益、Sharpe、成本鲁棒性和 cluster-risk 控制，并且整个流程是 nested walk-forward、防未来函数的。

## 参考文件

- P16 报告：`experiments/baseline_locked_v1/outputs/p16_clean_candidate_package/REPORT.md`
- P16 manifest：`experiments/baseline_locked_v1/outputs/p16_clean_candidate_package/candidate_manifest.json`
- P16 PnL 曲线：`experiments/baseline_locked_v1/outputs/p16_clean_candidate_package/clean_candidate_pnl_curve.png`
- P9 selected formulas：`experiments/baseline_locked_v1/outputs/p9_clean_baseline_package/selected_incremental_by_fold.csv`
- P6b retained factor library：`experiments/baseline_locked_v1/outputs/p6b_derivatives_feature_expansion/retained_features.csv`
- P15c selected cluster specs：`experiments/baseline_locked_v1/outputs/p16_clean_candidate_package/fixed_selected_extreme_cluster_by_fold.csv`
