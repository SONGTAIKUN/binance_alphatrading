# P27 no-sector 原生因子重挖报告

## 实验问题

本实验只回答一个问题：`no-sector residual` 本身缺乏 alpha，还是旧 A/B/C 因子语言与它不匹配。

- 因子层：从 no-sector residual 原生重建 48h 以上的 momentum、anchor reversion、OU stationarity、common-residual purity、beta stability、liquidity 与 slow quality gates。
- 搜索层：固定 240 个符号公式；每个 outer fold 只用 train label 筛选、validation 选择公式/平滑/执行参数，test 只回放。
- 交易层：P15c 的 portfolio risk beta、严格中性化、symbol cap、short confirmation 与 extreme-cluster allocator 保持不变。
- test 中的因子 IC 仅用于最终诊断，绝不参与该 fold 的公式选择。

## 总结结论

- 因子层：没有发现同时通过均值 IC、q25 fold IC、path IC、正 fold 比例和慢速自相关门槛的原生 primitive。
- 交易层：**不覆盖 P15c；保留为原生 re-mining 研究结果**。
- P27 相对 P15c 的 fold 胜率为 50.0%，相对 P26-full 为 50.0%。
- 公式 forced selection 比例 0.0%；共选择 20 个公式。
- 实际被选公式的 test slow Rank IC 均值 0.0267，q25 fold -0.0296，正 IC fold 比例 59.4%。
- cluster allocator fallback 比例 3.1%。

## 1. 因子层结果

```text
              primitive preferred_direction  oriented_mean_slow_rank_ic  oriented_q25_fold_ic  oriented_positive_fold_rate  oriented_path_rank_ic  mean_signal_turnover_72h  mean_signal_autocorr_72h  robust_pass
nf_d_low_downside_share             negated                    0.038155             -0.017961                      0.59375               0.028295                  0.434777                  0.485164        False
       nf_d_highloc_168             negated                    0.025785             -0.053180                      0.56250               0.016304                  0.516339                  0.361016        False
nf_d_common_adj_mom_336               as_is                    0.023852             -0.033565                      0.59375               0.027378                  0.329598                  0.697590        False
     nf_d_reversion_168               as_is                    0.021903             -0.045069                      0.53125               0.012286                  0.494955                  0.405988        False
     nf_d_reversion_336               as_is                    0.019922             -0.062808                      0.53125               0.010975                  0.414241                  0.553135        False
       nf_d_highloc_336             negated                    0.017983             -0.051893                      0.53125               0.009018                  0.403147                  0.582285        False
nf_d_volume_share_trend               as_is                    0.017893             -0.061349                      0.50000               0.020466                  0.281518                  0.752195        False
     nf_d_trend_eff_168             negated                    0.016202             -0.059907                      0.53125               0.008116                  0.476339                  0.426645        False
  nf_d_sign_persist_168             negated                    0.015870             -0.061576                      0.53125               0.007334                  0.437455                  0.515936        False
        nf_d_mom_168_24             negated                    0.015633             -0.058710                      0.53125               0.007881                  0.474330                  0.432129        False
     nf_d_reversion_504               as_is                    0.015570             -0.088714                      0.53125               0.005600                  0.377143                  0.615024        False
nf_d_common_adj_mom_168             negated                    0.014351             -0.051635                      0.53125               0.007122                  0.477500                  0.425658        False
```

## 2. 最终交易层结果

- P27 gross：12.77%；6bps net：7.04%；8bps net：5.20%。
- P27 Sharpe：0.549；MaxDD：-10.86%；positive fold rate：59.4%。
- P27 mean turnover：0.3409；估算单边 breakeven：13.86bps。
- P15c gross/net6/Sharpe：15.08% / 9.42% / 0.800。
- P26-full gross/net6/Sharpe：14.06% / 8.51% / 0.784。

### Matched summary
```text
       run  gross_total_return_before_cost  total_return   sharpe  max_drawdown  mean_turnover  positive_fold_rate  worst_5_fold_return_sum  median_fold_return
      P15c                        0.150801      0.094169 0.800291     -0.088949       0.347724             0.62500                -0.106890            0.002563
  P26_full                        0.140608      0.085110 0.784179     -0.091025       0.343747             0.56250                -0.074547            0.001701
P27_native                        0.127683      0.070408 0.549168     -0.108618       0.340906             0.59375                -0.113387            0.004037
```

### 成本敏感性
```text
experiment  tau_days  one_way_cost_bps  total_return  max_drawdown  mean_period_return  positive_period_rate  cost_drag_sum
      P15c       0.0               0.0      0.150801     -0.075171            0.000595              0.566116       0.000000
  P26_full       0.0               0.0      0.140608     -0.077199            0.000556              0.578512       0.000000
P27_native       0.0               0.0      0.127683     -0.097795            0.000488              0.521569       0.000000
      P15c       0.0               6.0      0.094169     -0.088949            0.000386              0.533058       0.050490
  P26_full       0.0               6.0      0.085110     -0.091025            0.000350              0.557851       0.049912
P27_native       0.0               6.0      0.070408     -0.108618            0.000283              0.482353       0.052159
      P15c       0.0               8.0      0.075914     -0.093736            0.000317              0.516529       0.067319
  P26_full       0.0               8.0      0.067215     -0.095588            0.000281              0.541322       0.066549
P27_native       0.0               8.0      0.051967     -0.112197            0.000215              0.478431       0.069545
      P15c       0.0              15.0      0.014377     -0.110298            0.000073              0.458678       0.126224
  P26_full       0.0              15.0      0.006862     -0.111383            0.000040              0.508264       0.124780
P27_native       0.0              15.0     -0.010122     -0.124613           -0.000024              0.454902       0.130396
      P15c       0.0              20.0     -0.027422     -0.121946           -0.000101              0.442149       0.168298
  P26_full       0.0              20.0     -0.034157     -0.124517           -0.000132              0.491736       0.166373
P27_native       0.0              20.0     -0.052224     -0.137265           -0.000194              0.443137       0.173862
      P15c       0.0              30.0     -0.105948     -0.144793           -0.000448              0.392562       0.252448
  P26_full       0.0              30.0     -0.111272     -0.153908           -0.000475              0.433884       0.249560
P27_native       0.0              30.0     -0.131157     -0.163147           -0.000535              0.400000       0.260793
```

### 季度 regime
```text
       run regime  one_way_cost_bps  periods  total_return
      P15c 2023Q4               0.0       21      0.044985
      P15c 2024Q1               0.0       23     -0.028130
      P15c 2024Q2               0.0       25     -0.016364
      P15c 2024Q3               0.0       22     -0.015288
      P15c 2024Q4               0.0       22      0.057146
      P15c 2025Q1               0.0       23      0.016619
      P15c 2025Q2               0.0       24      0.056829
      P15c 2025Q3               0.0       20     -0.014343
      P15c 2025Q4               0.0       22      0.034454
      P15c 2026Q1               0.0       22      0.000120
      P15c 2026Q2               0.0       18      0.010070
      P15c 2023Q4               8.0       21      0.038560
      P15c 2024Q1               8.0       23     -0.032946
      P15c 2024Q2               8.0       25     -0.023592
      P15c 2024Q3               8.0       22     -0.022368
      P15c 2024Q4               8.0       22      0.049705
      P15c 2025Q1               8.0       23      0.008762
      P15c 2025Q2               8.0       24      0.048456
      P15c 2025Q3               8.0       20     -0.019015
      P15c 2025Q4               8.0       22      0.028793
      P15c 2026Q1               8.0       22     -0.004680
      P15c 2026Q2               8.0       18      0.006306
  P26_full 2023Q4               0.0       22     -0.023687
  P26_full 2024Q1               0.0       27      0.004567
  P26_full 2024Q2               0.0       24     -0.023602
  P26_full 2024Q3               0.0       24     -0.011792
  P26_full 2024Q4               0.0       24      0.044091
  P26_full 2025Q1               0.0       20      0.046382
  P26_full 2025Q2               0.0       25      0.032176
  P26_full 2025Q3               0.0       22      0.003545
  P26_full 2025Q4               0.0       22      0.047017
  P26_full 2026Q1               0.0       15      0.002609
  P26_full 2026Q2               0.0       17      0.014585
  P26_full 2023Q4               8.0       22     -0.030709
  P26_full 2024Q1               8.0       27     -0.002371
  P26_full 2024Q2               8.0       24     -0.029113
  P26_full 2024Q3               8.0       24     -0.018809
  P26_full 2024Q4               8.0       24      0.036427
  P26_full 2025Q1               8.0       20      0.040489
  P26_full 2025Q2               8.0       25      0.023832
  P26_full 2025Q3               8.0       22     -0.002002
  P26_full 2025Q4               8.0       22      0.040421
  P26_full 2026Q1               8.0       15      0.000070
  P26_full 2026Q2               8.0       17      0.010492
P27_native 2023Q4               0.0       18     -0.009045
P27_native 2024Q1               0.0       27      0.020996
P27_native 2024Q2               0.0       24     -0.049989
P27_native 2024Q3               0.0       25      0.005723
P27_native 2024Q4               0.0       21      0.024271
P27_native 2025Q1               0.0       23      0.078388
P27_native 2025Q2               0.0       26      0.013084
P27_native 2025Q3               0.0       22     -0.012994
P27_native 2025Q4               0.0       25      0.032033
P27_native 2026Q1               0.0       26     -0.000309
P27_native 2026Q2               0.0       18      0.023735
P27_native 2023Q4               8.0       18     -0.014705
P27_native 2024Q1               8.0       27      0.013991
P27_native 2024Q2               8.0       24     -0.055140
P27_native 2024Q3               8.0       25     -0.003422
P27_native 2024Q4               8.0       21      0.018470
P27_native 2025Q1               8.0       23      0.070783
P27_native 2025Q2               8.0       26      0.005764
P27_native 2025Q3               8.0       22     -0.019006
P27_native 2025Q4               8.0       25      0.026654
P27_native 2026Q1               8.0       26     -0.006931
P27_native 2026Q2               8.0       18      0.019312
```

## 严格性审计

- 32/32 个 fold 通过 label closure、selection-before-test 与 test boundary 检查。
- 原生特征只使用 t 时刻及以前数据；交易使用 t+1 执行口径。
- factor test IC 是事后评价表，不参与选择；公式与 operator 集合在全实验前固定。

```text
 fold                 train_end                test_start                  test_end   validation_decision_end           train_label_end              selected_extreme_spec    max_test_decision_time  labels_close_before_validation  formula_selection_before_test  cluster_selection_before_test  test_decisions_inside_fold  all_pass
    1 2023-10-15 08:00:00+00:00 2023-10-15 08:00:00+00:00 2023-11-14 08:00:00+00:00 2023-10-12 07:00:00+00:00 2023-07-10 07:00:00+00:00  p15c_lb336_k3_trig75_cap60_turn00 2023-11-08 08:00:00+00:00                            True                           True                           True                        True      True
    2 2023-11-14 08:00:00+00:00 2023-11-14 08:00:00+00:00 2023-12-14 08:00:00+00:00 2023-11-11 07:00:00+00:00 2023-08-09 07:00:00+00:00  p15c_lb336_k3_trig75_cap60_turn00 2023-12-08 08:00:00+00:00                            True                           True                           True                        True      True
    3 2023-12-14 08:00:00+00:00 2023-12-14 08:00:00+00:00 2024-01-13 08:00:00+00:00 2023-12-11 07:00:00+00:00 2023-09-08 07:00:00+00:00  p15c_lb720_k4_trig75_cap60_turn00 2024-01-07 08:00:00+00:00                            True                           True                           True                        True      True
    4 2024-01-13 08:00:00+00:00 2024-01-13 08:00:00+00:00 2024-02-12 08:00:00+00:00 2024-01-10 07:00:00+00:00 2023-10-08 07:00:00+00:00  p15c_lb720_k3_trig85_cap70_turn00 2024-02-06 08:00:00+00:00                            True                           True                           True                        True      True
    5 2024-02-12 08:00:00+00:00 2024-02-12 08:00:00+00:00 2024-03-13 08:00:00+00:00 2024-02-09 07:00:00+00:00 2023-11-07 07:00:00+00:00  p15c_lb720_k3_trig75_cap60_turn02 2024-03-07 08:00:00+00:00                            True                           True                           True                        True      True
    6 2024-03-13 08:00:00+00:00 2024-03-13 08:00:00+00:00 2024-04-12 08:00:00+00:00 2024-03-10 07:00:00+00:00 2023-12-07 07:00:00+00:00 p15c_lb1440_k3_trig80_cap65_turn00 2024-04-06 08:00:00+00:00                            True                           True                           True                        True      True
    7 2024-04-12 08:00:00+00:00 2024-04-12 08:00:00+00:00 2024-05-12 08:00:00+00:00 2024-04-09 07:00:00+00:00 2024-01-06 07:00:00+00:00 p15c_lb1440_k3_trig75_cap60_turn00 2024-05-06 08:00:00+00:00                            True                           True                           True                        True      True
    8 2024-05-12 08:00:00+00:00 2024-05-12 08:00:00+00:00 2024-06-11 08:00:00+00:00 2024-05-09 07:00:00+00:00 2024-02-05 07:00:00+00:00  p15c_lb336_k3_trig80_cap65_turn02 2024-06-05 08:00:00+00:00                            True                           True                           True                        True      True
    9 2024-06-11 08:00:00+00:00 2024-06-11 08:00:00+00:00 2024-07-11 08:00:00+00:00 2024-06-08 07:00:00+00:00 2024-03-06 07:00:00+00:00  p15c_lb720_k4_trig85_cap70_turn00 2024-07-05 08:00:00+00:00                            True                           True                           True                        True      True
   10 2024-07-11 08:00:00+00:00 2024-07-11 08:00:00+00:00 2024-08-10 08:00:00+00:00 2024-07-08 07:00:00+00:00 2024-04-05 07:00:00+00:00 p15c_lb1440_k4_trig75_cap60_turn00 2024-08-04 08:00:00+00:00                            True                           True                           True                        True      True
   11 2024-08-10 08:00:00+00:00 2024-08-10 08:00:00+00:00 2024-09-09 08:00:00+00:00 2024-08-07 07:00:00+00:00 2024-05-05 07:00:00+00:00  p15c_lb720_k3_trig80_cap65_turn02 2024-09-03 08:00:00+00:00                            True                           True                           True                        True      True
   12 2024-09-09 08:00:00+00:00 2024-09-09 08:00:00+00:00 2024-10-09 08:00:00+00:00 2024-09-06 07:00:00+00:00 2024-06-04 07:00:00+00:00 p15c_lb1440_k3_trig75_cap60_turn02 2024-10-03 08:00:00+00:00                            True                           True                           True                        True      True
   13 2024-10-09 08:00:00+00:00 2024-10-09 08:00:00+00:00 2024-11-08 08:00:00+00:00 2024-10-06 07:00:00+00:00 2024-07-04 07:00:00+00:00  p15c_lb336_k3_trig80_cap65_turn02 2024-11-02 08:00:00+00:00                            True                           True                           True                        True      True
   14 2024-11-08 08:00:00+00:00 2024-11-08 08:00:00+00:00 2024-12-08 08:00:00+00:00 2024-11-05 07:00:00+00:00 2024-08-03 07:00:00+00:00  p15c_lb720_k3_trig80_cap65_turn02 2024-11-29 08:00:00+00:00                            True                           True                           True                        True      True
   15 2024-12-08 08:00:00+00:00 2024-12-08 08:00:00+00:00 2025-01-07 08:00:00+00:00 2024-12-05 07:00:00+00:00 2024-09-02 07:00:00+00:00  p15c_lb336_k3_trig85_cap70_turn02 2025-01-01 08:00:00+00:00                            True                           True                           True                        True      True
   16 2025-01-07 08:00:00+00:00 2025-01-07 08:00:00+00:00 2025-02-06 08:00:00+00:00 2025-01-04 07:00:00+00:00 2024-10-02 07:00:00+00:00 p15c_lb1440_k3_trig80_cap65_turn02 2025-01-31 08:00:00+00:00                            True                           True                           True                        True      True
   17 2025-02-06 08:00:00+00:00 2025-02-06 08:00:00+00:00 2025-03-08 08:00:00+00:00 2025-02-03 07:00:00+00:00 2024-11-01 07:00:00+00:00  p15c_lb720_k4_trig75_cap60_turn02 2025-02-27 08:00:00+00:00                            True                           True                           True                        True      True
   18 2025-03-08 08:00:00+00:00 2025-03-08 08:00:00+00:00 2025-04-07 08:00:00+00:00 2025-03-05 07:00:00+00:00 2024-12-01 07:00:00+00:00 p15c_lb1440_k4_trig75_cap60_turn00 2025-04-01 08:00:00+00:00                            True                           True                           True                        True      True
   19 2025-04-07 08:00:00+00:00 2025-04-07 08:00:00+00:00 2025-05-07 08:00:00+00:00 2025-04-04 07:00:00+00:00 2024-12-31 07:00:00+00:00  p15c_lb336_k4_trig85_cap70_turn02 2025-05-01 08:00:00+00:00                            True                           True                           True                        True      True
   20 2025-05-07 08:00:00+00:00 2025-05-07 08:00:00+00:00 2025-06-06 08:00:00+00:00 2025-05-04 07:00:00+00:00 2025-01-30 07:00:00+00:00 p15c_lb1440_k3_trig85_cap70_turn02 2025-05-31 08:00:00+00:00                            True                           True                           True                        True      True
   21 2025-06-06 08:00:00+00:00 2025-06-06 08:00:00+00:00 2025-07-06 08:00:00+00:00 2025-06-03 07:00:00+00:00 2025-03-01 07:00:00+00:00  p15c_lb720_k4_trig75_cap60_turn00 2025-06-30 08:00:00+00:00                            True                           True                           True                        True      True
   22 2025-07-06 08:00:00+00:00 2025-07-06 08:00:00+00:00 2025-08-05 08:00:00+00:00 2025-07-03 07:00:00+00:00 2025-03-31 07:00:00+00:00 p15c_lb1440_k3_trig75_cap60_turn00 2025-07-30 08:00:00+00:00                            True                           True                           True                        True      True
   23 2025-08-05 08:00:00+00:00 2025-08-05 08:00:00+00:00 2025-09-04 08:00:00+00:00 2025-08-02 07:00:00+00:00 2025-04-30 07:00:00+00:00 p15c_lb1440_k3_trig75_cap60_turn02 2025-08-29 08:00:00+00:00                            True                           True                           True                        True      True
   24 2025-09-04 08:00:00+00:00 2025-09-04 08:00:00+00:00 2025-10-04 08:00:00+00:00 2025-09-01 07:00:00+00:00 2025-05-30 07:00:00+00:00  p15c_lb336_k3_trig85_cap70_turn02 2025-09-28 08:00:00+00:00                            True                           True                           True                        True      True
   25 2025-10-04 08:00:00+00:00 2025-10-04 08:00:00+00:00 2025-11-03 08:00:00+00:00 2025-10-01 07:00:00+00:00 2025-06-29 07:00:00+00:00  p15c_lb720_k3_trig75_cap60_turn02 2025-10-28 08:00:00+00:00                            True                           True                           True                        True      True
   26 2025-11-03 08:00:00+00:00 2025-11-03 08:00:00+00:00 2025-12-03 08:00:00+00:00 2025-10-31 07:00:00+00:00 2025-07-29 07:00:00+00:00 p15c_lb1440_k3_trig80_cap65_turn02 2025-11-27 08:00:00+00:00                            True                           True                           True                        True      True
   27 2025-12-03 08:00:00+00:00 2025-12-03 08:00:00+00:00 2026-01-02 08:00:00+00:00 2025-11-30 07:00:00+00:00 2025-08-28 07:00:00+00:00      __P9_NO_EXTREME_CLUSTER_CAP__ 2025-12-27 08:00:00+00:00                            True                           True                           True                        True      True
   28 2026-01-02 08:00:00+00:00 2026-01-02 08:00:00+00:00 2026-02-01 08:00:00+00:00 2025-12-30 07:00:00+00:00 2025-09-27 07:00:00+00:00 p15c_lb1440_k4_trig75_cap60_turn00 2026-01-26 08:00:00+00:00                            True                           True                           True                        True      True
   29 2026-02-01 08:00:00+00:00 2026-02-01 08:00:00+00:00 2026-03-03 08:00:00+00:00 2026-01-29 07:00:00+00:00 2025-10-27 07:00:00+00:00  p15c_lb336_k3_trig75_cap60_turn00 2026-02-25 08:00:00+00:00                            True                           True                           True                        True      True
   30 2026-03-03 08:00:00+00:00 2026-03-03 08:00:00+00:00 2026-04-02 08:00:00+00:00 2026-02-28 07:00:00+00:00 2025-11-26 07:00:00+00:00  p15c_lb336_k4_trig75_cap60_turn02 2026-03-27 08:00:00+00:00                            True                           True                           True                        True      True
   31 2026-04-02 08:00:00+00:00 2026-04-02 08:00:00+00:00 2026-05-02 08:00:00+00:00 2026-03-30 07:00:00+00:00 2025-12-26 07:00:00+00:00 p15c_lb1440_k3_trig75_cap60_turn02 2026-04-26 08:00:00+00:00                            True                           True                           True                        True      True
   32 2026-05-02 08:00:00+00:00 2026-05-02 08:00:00+00:00 2026-06-01 08:00:00+00:00 2026-04-29 07:00:00+00:00 2026-01-25 07:00:00+00:00  p15c_lb720_k3_trig75_cap60_turn02 2026-05-26 08:00:00+00:00                            True                           True                           True                        True      True
```

## 公式选择频次
```text
formula_name  formula                                                                           smoothing_alpha  execution_spec
NSF012        NEG(nf_d_low_downside_share)                                                      0.50             native_lf50_ema35    3
NSF187        ADD(nf_d_reversion_168,0.25*nf_d_volume_share_trend)                              0.50             native_lf50_ema30    1
NSF049        SUB(nf_d_reversion_336,0.50*nf_d_low_downside_share)                              0.50             native_lf45_ema30    1
NSF040        GATE(nf_d_reversion_168,nf_q_stationarity_168)                                    0.25             native_lf45_ema35    1
                                                                                                0.50             native_lf50_ema35    1
NSF054        ADD(nf_d_reversion_336,0.50*nf_d_sign_persist_336)                                0.50             native_lf50_ema30    1
NSF146        ADD(nf_d_reversion_504,0.75*nf_d_mom_504_48)                                      1.00             native_lf45_ema35    1
NSF133        GATE(nf_d_mom_504_48,nf_q_low_amihud)                                             0.50             native_lf55_ema35    1
NSF054        ADD(nf_d_reversion_336,0.50*nf_d_sign_persist_336)                                0.25             native_lf45_ema35    1
NSF170        ADD(nf_d_sign_persist_336,0.50*nf_d_reversion_336)                                1.00             native_lf45_ema30    1
NSF111        SUB(nf_d_mom_336_24,0.75*nf_d_highloc_168)                                        1.00             native_lf50_ema35    1
NSF053        ADD(nf_d_reversion_168,0.75*nf_d_sign_persist_336)                                1.00             native_lf45_ema35    1
NSF008        NEG(nf_d_highloc_336)                                                             0.50             native_lf50_ema35    1
NSF012        NEG(nf_d_low_downside_share)                                                      0.50             native_lf45_ema35    1
NSF187        ADD(nf_d_reversion_168,0.25*nf_d_volume_share_trend)                              1.00             native_lf55_ema35    1
NSF092        GATED_ADD(nf_d_reversion_168,0.75*nf_d_highloc_504,nf_q_low_amihud)               1.00             native_lf50_ema35    1
                                                                                                0.25             native_lf45_ema30    1
NSF006        NEG(nf_d_highloc_168)                                                             1.00             native_lf45_ema35    1
NSF040        GATE(nf_d_reversion_168,nf_q_stationarity_168)                                    0.50             native_lf55_ema35    1
NSF092        GATED_ADD(nf_d_reversion_168,0.75*nf_d_highloc_504,nf_q_low_amihud)               0.25             native_lf45_ema35    1
                                                                                                                 native_lf50_ema35    1
NSF095        GATE(nf_d_reversion_504,nf_q_purity_336)                                          1.00             native_lf45_ema35    1
NSF063        ADD(nf_d_mom_504_48,0.50*nf_d_common_adj_mom_168)                                 0.50             native_lf50_ema35    1
NSF173        GATED_ADD(nf_d_volume_share_trend,0.25*nf_d_trend_eff_336,nf_q_stationarity_168)  0.25             native_lf45_ema35    1
NSF063        ADD(nf_d_mom_504_48,0.50*nf_d_common_adj_mom_168)                                 0.25             native_lf45_ema30    1
NSF179        GATE(nf_d_volume_share_trend,nf_q_low_amihud)                                     1.00             native_lf50_ema35    1
NSF235        ADD(nf_d_highloc_504,0.50*nf_d_volume_share_trend)                                0.50             native_lf45_ema35    1
NSF151        GATED_ADD(nf_d_mom_72_12,0.25*nf_d_mom_72_12,nf_q_low_vov)                        1.00             native_lf45_ema35    1
NSF145        GATED_ADD(nf_d_trend_eff_168,0.25*nf_d_sign_persist_168,nf_q_low_vov)             0.25             native_lf55_ema30    1
                                                                                                0.50             native_lf45_ema35    1
```
