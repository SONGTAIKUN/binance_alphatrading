# 研究主线与版本关系

## 从配对机会转向资产级 residual

早期研究围绕 APT/FIL 等 pair 的 log-spread 与 z-score，用符号因子作为交易 gate / ranker，随后扩展到 190 个配对机会。2026-06-18 的归档说明记录了研究单位从 `timestamp × pair` 转为 `timestamp × symbol` 的切换。

主仓库展示资产级 residual 路线；早期配对源码、结果、q10.7 补充包和原始行情保存在完整档案中。

## 代表性主线：P4 → P3d → P9 → P15c/P16

1. **数据与 residual**：Binance 15m 行情转为完整 1h bars，利用历史窗口滚动估计共同因子，构造 residual 历史、横截面特征和延迟标签。
2. **P4 基础组合**：在明确的训练 / 验证边界内选择慢速相对价值组合参数。
3. **P3d 符号增量**：从固定 primitives 和操作符构造表达式，在对应历史窗口评价并选取增量 rerank。
4. **P9 衍生品增量**：检验 funding、OI、premium 等衍生品 primitives 对已有组合 pool 的边际贡献。
5. **P15c/P16 组合与回放**：引入极端 cluster cap 和 turnover guard，保存每折选择与可复查的结果包；P16 入口回放固定的 P15c 选择。

慢标签包含 48/72/96/168h 等 horizon；代表性组合以 72h 调仓为主。固定选择 replay 与完整嵌套重选是不同实验，不应把二者的结果直接混为一张最优绩效表。

## 后续研究到 P29

- P18–P22：因子衰减、因子选择与组合层目标错配、选择协议及新的经济 family。
- P23–P26：crypto FF residual、模型比较、no-sector residual 与标签审计，拆分 alpha residual 与 portfolio risk 的接口。
- P27–P29：no-sector 原生 remine、稳定 family ensemble、pool 内 tail-aware selection。

这些实验的结论包括无增量、selector fallback 和暂不升级正式基线。保留它们，是为了让后续研究能够接续已有证据。

## 2026-09-08 审查后的优先事项

当前最先需要复核的是共同评价引擎：持仓连续性、缺少候选时的账户动作、fold 边界清仓或继承规则、实际数量对应的换手与成本、小时盯市和 funding 现金流。随后在统一口径下比较简单信号、P4、P3d、P9 和 P15c。

审查还区分了特征时点因果性、单次实验的 fold 内选择边界，以及研究者反复查看同一段历史造成的选择偏差。`walk-forward` 名称本身不能替代后两者的证据。

完整论证、源码定位和既有输出核对见 [架构审查](../reviews/20260908_architecture_review/REVIEW_ZH.md)。
