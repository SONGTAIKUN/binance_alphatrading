# P18c Fold-Local Factor Selector Portfolio

P18c uses P18b fold-local selected factors as the portfolio score, then runs the strict-neutral allocator on test windows only.

- P18c estimated breakeven one-way cost: 25.70bps
- P15c estimated breakeven one-way cost: 16.72bps

## Comparison

```text
stage  periods  total_return  gross_total_return_before_cost   sharpe  max_drawdown  mean_rank_ic  mean_top_bottom_label_spread  mean_period_return  mean_gross_period_return  positive_period_rate  positive_gross_period_rate  mean_turnover  total_turnover  cost_drag  mean_period_cost  target_apply_rate  mean_abs_beta_btc  mean_abs_beta_eth  mean_abs_beta_market  mean_symbol_concentration  fold_count  positive_fold_rate  gross_max_drawdown  mean_gross_exposure  mean_max_abs_weight  max_abs_weight  total_stopped_weight  stopped_period_rate  max_abs_beta_btc  max_abs_beta_eth  max_abs_beta_market  mean_optimizer_success  max_optimizer_success  mean_optimizer_exposure_error  max_optimizer_exposure_error  mean_optimizer_target_gross  max_optimizer_target_gross  mean_optimizer_actual_gross  max_optimizer_actual_gross  fil_total_contribution  mean_target_turnover  total_target_turnover  mean_turnover_saved  total_turnover_saved  mean_target_gross_exposure  total_target_gross_exposure  turnover_reduction  gross_exposure_retention  mean_short_guard_removed  total_return_15bps  sharpe_15bps  max_drawdown_15bps  total_return_20bps  sharpe_20bps  max_drawdown_20bps  total_return_30bps  sharpe_30bps  max_drawdown_30bps  q25_segment_sharpe  long_short_balance  short_abs_contribution_share  mean_max_cluster_gross_share  mean_cluster_hhi  mean_active_cluster_count  mean_abs_pc1_exposure  mean_abs_pc2_exposure  mean_abs_pc3_exposure  mean_baseline_max_cluster_gross_share  mean_baseline_cluster_hhi  mean_gross_exposure_retention  cluster_hhi_reduction_vs_baseline  max_cluster_share_reduction_vs_baseline  mean_extreme_cluster_triggered  mean_cluster_cap_applied  mean_turnover_guard_skipped  mean_cluster_cap_solved  mean_turnover_add_vs_p9
 p18c      288      0.157986                        0.211107 0.642422     -0.137604      0.059743                      0.000567            0.000555                  0.000711              0.583333                    0.583333       0.259466       74.726245   0.044836          0.000156                1.0       2.199562e-15       2.939474e-15          1.270090e-14                   0.036426          32             0.53125           -0.130650             0.421570             0.099721           0.125                   0.0                  0.0      9.521724e-14      2.679697e-13         8.820861e-13                1.000000                    1.0                   1.412144e-14                  8.820908e-13                     0.750305                    0.976326                      0.42157                    0.949729               -0.003266              0.259466              74.726245             0.000000              0.000000                    0.421570                   121.412025            0.000000                  1.000000                  0.000000            0.082616      0.371283           -0.147936            0.042872      0.221215           -0.153625           -0.032318     -0.077422           -0.181899                 NaN                 NaN                           NaN                           NaN               NaN                        NaN                    NaN                    NaN                    NaN                                    NaN                        NaN                            NaN                                NaN                                      NaN                             NaN                       NaN                          NaN                      NaN                      NaN
 p15c      242      0.094169                        0.150801 0.800291     -0.088949      0.044239                      0.004329            0.000386                  0.000595              0.533058                    0.566116       0.347724       84.149199   0.050490          0.000209                1.0       4.258947e-15       5.946227e-15          1.970631e-14                   0.018025          32             0.62500           -0.075171             0.261507             0.081178           0.125                   0.0                  0.0      1.408583e-13      3.878772e-13         6.504623e-13                0.971074                    1.0                   8.323454e-13                  1.161715e-10                          NaN                         NaN                          NaN                         NaN               -0.020710              0.349167              84.498322             0.001795              0.434415                    0.261991                    63.401822            0.004132                  0.998154                  2.190083            0.014377      0.151632           -0.110298           -0.027422     -0.208593           -0.121946           -0.105948     -0.927402           -0.144793           -1.774536            0.166302                      0.083151                      0.596761          0.479974                   3.285124                0.04727               0.049629               0.063716                               0.628888                   0.517956                       0.998065                            0.07333                                 0.051086                         0.22314                  0.136364                     0.057851                 0.194215                 0.000054
```

## P18c Cost Sensitivity

```text
stage  one_way_cost_bps  total_return  gross_total_return    sharpe  max_drawdown  gross_max_drawdown  mean_turnover  total_turnover  positive_period_rate
 p18c               0.0      0.211107            0.211107  0.823756     -0.130650            -0.13065       0.259466       74.726245              0.583333
 p18c               6.0      0.157986            0.211107  0.642422     -0.137604            -0.13065       0.259466       74.726245              0.583333
 p18c              10.0      0.123866            0.211107  0.521771     -0.142211            -0.13065       0.259466       74.726245              0.583333
 p18c              15.0      0.082616            0.211107  0.371283     -0.147936            -0.13065       0.259466       74.726245              0.572917
 p18c              20.0      0.042872            0.211107  0.221215     -0.153625            -0.13065       0.259466       74.726245              0.559028
 p18c              30.0     -0.032318            0.211107 -0.077422     -0.181899            -0.13065       0.259466       74.726245              0.531250
```

## Segment Summary

```text
stage                      segment  one_way_cost_bps  periods  total_return  gross_total_return    sharpe  max_drawdown  gross_max_drawdown  mean_turnover  total_turnover
 p18c                         full               6.0      288      0.157986            0.211107  0.642422     -0.137604           -0.130650       0.259466       74.726245
 p18c                         full              15.0      288      0.082616            0.211107  0.371283     -0.147936           -0.130650       0.259466       74.726245
 p18c                         full              20.0      288      0.042872            0.211107  0.221215     -0.153625           -0.130650       0.259466       74.726245
 p18c                         full              30.0      288     -0.032318            0.211107 -0.077422     -0.181899           -0.130650       0.259466       74.726245
 p18c     pre_original_2023_2024H1               6.0       79      0.013305            0.024839  0.220121     -0.079786           -0.074845       0.238402       18.833776
 p18c     pre_original_2023_2024H1              15.0       79     -0.003759            0.024839  0.024833     -0.088097           -0.074845       0.238402       18.833776
 p18c     pre_original_2023_2024H1              20.0       79     -0.013117            0.024839 -0.083217     -0.094171           -0.074845       0.238402       18.833776
 p18c     pre_original_2023_2024H1              30.0       79     -0.031575            0.024839 -0.298280     -0.106202           -0.074845       0.238402       18.833776
 p18c original_window_2024_07_plus               6.0      209      0.142782            0.181754  0.890433     -0.137604           -0.130650       0.267428       55.892470
 p18c original_window_2024_07_plus              15.0      209      0.086701            0.181754  0.570804     -0.147936           -0.130650       0.267428       55.892470
 p18c original_window_2024_07_plus              20.0      209      0.056733            0.181754  0.393853     -0.153625           -0.130650       0.267428       55.892470
 p18c original_window_2024_07_plus              30.0      209     -0.000767            0.181754  0.041780     -0.170185           -0.130650       0.267428       55.892470
 p15c                         full               6.0      242      0.094169            0.150801  0.800291     -0.088949           -0.075171       0.347724       84.149199
 p15c                         full              15.0      242      0.014377            0.150801  0.151632     -0.110298           -0.075171       0.347724       84.149199
 p15c                         full              20.0      242     -0.027422            0.150801 -0.208593     -0.121946           -0.075171       0.347724       84.149199
 p15c                         full              30.0      242     -0.105948            0.150801 -0.927402     -0.144793           -0.075171       0.347724       84.149199
 p15c     pre_original_2023_2024H1               6.0       69     -0.014802           -0.001030 -0.376182     -0.064236           -0.057252       0.335255       23.132577
 p15c     pre_original_2023_2024H1              15.0       69     -0.035111           -0.001030 -0.945969     -0.075553           -0.057252       0.335255       23.132577
 p15c     pre_original_2023_2024H1              20.0       69     -0.046215           -0.001030 -1.261885     -0.082068           -0.057252       0.335255       23.132577
 p15c     pre_original_2023_2024H1              30.0       69     -0.068048           -0.001030 -1.891629     -0.095661           -0.057252       0.335255       23.132577
 p15c original_window_2024_07_plus               6.0      173      0.110608            0.151988  1.343946     -0.031342           -0.026890       0.352697       61.016622
 p15c original_window_2024_07_plus              15.0      173      0.051288            0.151988  0.655324     -0.039773           -0.026890       0.352697       61.016622
 p15c original_window_2024_07_plus              20.0      173      0.019704            0.151988  0.272611     -0.044426           -0.026890       0.352697       61.016622
 p15c original_window_2024_07_plus              30.0      173     -0.040667            0.151988 -0.491422     -0.053669           -0.026890       0.352697       61.016622
```

## Worst Fold Deltas vs P15c

```text
 fold                test_start  total_return_p18c  gross_total_return_before_cost_p18c  sharpe_p18c  max_drawdown_p18c  mean_turnover_p18c  total_return_p15c  gross_total_return_before_cost_p15c  sharpe_p15c  max_drawdown_p15c  mean_turnover_p15c  delta_total_return  delta_gross_total_return_before_cost  delta_sharpe  delta_max_drawdown  delta_mean_turnover
   14 2024-11-08 08:00:00+00:00          -0.054104                            -0.052266    -4.419444          -0.058695            0.357071           0.039028                             0.040756     8.095658          -0.004163            0.347263           -0.093132                             -0.093022    -12.515102           -0.054532             0.009807
    3 2023-12-14 08:00:00+00:00          -0.030200                            -0.028676    -2.928943          -0.045351            0.289886           0.002857                             0.003861     1.143732          -0.005303            0.238340           -0.033057                             -0.032537     -4.072675           -0.040048             0.051547
   13 2024-10-09 08:00:00+00:00          -0.035273                            -0.032558    -4.154284          -0.051590            0.517823          -0.006395                            -0.004783    -6.733796          -0.004125            0.449622           -0.028878                             -0.027775      2.579512           -0.047464             0.068200
   20 2025-05-07 08:00:00+00:00          -0.002648                            -0.002288    -1.158597          -0.000805            0.066749           0.025696                             0.027690     9.801984          -0.001648            0.405971           -0.028344                             -0.029978    -10.960581            0.000844            -0.339223
   16 2025-01-07 08:00:00+00:00          -0.016103                            -0.013992    -1.340977          -0.044506            0.394540           0.010660                             0.011608     2.193428          -0.013224            0.313827           -0.026763                             -0.025599     -3.534405           -0.031283             0.080713
    6 2024-03-13 08:00:00+00:00          -0.013464                            -0.012648    -3.441610          -0.012767            0.152788           0.013061                             0.014545     4.761025          -0.004100            0.349533           -0.026524                             -0.027193     -8.202634           -0.008667            -0.196745
    2 2023-11-14 08:00:00+00:00          -0.001686                            -0.001007    -0.492083          -0.013625            0.125938           0.018538                             0.020518     4.971546          -0.001694            0.405561           -0.020224                             -0.021525     -5.463629           -0.011932            -0.279623
   19 2025-04-07 08:00:00+00:00          -0.022790                            -0.021862    -3.939283          -0.025756            0.175046          -0.004616                            -0.003114    -2.270879          -0.009425            0.314109           -0.018175                             -0.018748     -1.668403           -0.016331            -0.139063
   15 2024-12-08 08:00:00+00:00          -0.001470                             0.000963    -0.109029          -0.016759            0.450855           0.012579                             0.014705     1.636792          -0.011465            0.389666           -0.014049                             -0.013742     -1.745821           -0.005294             0.061189
   21 2025-06-06 08:00:00+00:00           0.015916                             0.016952     2.748921          -0.012077            0.188512           0.028580                             0.030835     8.847496          -0.002549            0.524071           -0.012664                             -0.013883     -6.098575           -0.009528            -0.335559
```

## Common Selected Factors

```text
raw_name
taker_buy_base                            9
volume                                    9
cs_rank_realized_vol_medium               8
realized_vol_long                         8
deriv_open_interest                       7
cs_breakout_risk_72                       7
idio_vol_medium                           7
cs_res_ma_slope_48_168                    6
deriv_crowding_pressure_72                6
realized_vol_medium                       6
cs_zscore_idio_vol_medium                 6
p6b_J04_top_trader_contra                 6
cs_vol_of_vol_res_168                     6
cs_volume_share_trend_7d_30d              6
p6_G04_p3d_diversifier_candidate          5
cs_rank_idio_vol_medium                   5
p6_E01_uncrowded_mom                      5
cs_res_trend_quality_168_24               4
p6_F02_uptrend_mom                        4
cs_res_mom_168_24                         4
p6_G01_slow_consensus                     4
idio_vol_long                             4
deriv_top_sum_long_short_ratio_ewm_24     3
deriv_top_sum_long_short_ratio            3
p6b_J03_account_longs_contra              3
deriv_account_long_short_ratio_ewm_168    3
p6_G02_consensus_low_disagreement         3
deriv_top_sum_long_short_ratio_ewm_72     3
deriv_account_long_short_ratio_ewm_336    3
p6b_J01_oi_growth_mom_quality             3
cs_res_high_mom_168                       3
deriv_log_oi_value_delta_168              3
alpha_A_res_trend_quality                 3
cs_zscore_beta_btc                        3
beta_btc                                  3
cs_rank_beta_btc                          3
cs_vol_of_vol_336                         3
p6b_J02_oi_growth_crowding_fade           2
beta_market                               2
cs_res_high_mom_336                       2
```

## Asset Contribution

```text
  symbol  gross_contribution  avg_abs_weight  max_abs_weight  observations  abs_contribution_share
 XRPUSDT           -0.066907        0.024368           0.125           258                0.081473
AVAXUSDT           -0.062348        0.025654           0.125           271                0.075921
NEARUSDT           -0.050604        0.015841           0.125           257                0.061620
 SOLUSDT           -0.040147        0.013512           0.125           255                0.048887
 DOTUSDT           -0.025715        0.020202           0.125           271                0.031314
 BTCUSDT           -0.022316        0.038262           0.125           236                0.027175
  OPUSDT           -0.011330        0.027595           0.125           228                0.013796
 UNIUSDT           -0.010520        0.017925           0.125           245                0.012810
 ADAUSDT           -0.009442        0.012719           0.125           237                0.011498
ATOMUSDT           -0.005637        0.027440           0.125           268                0.006864
 FILUSDT           -0.003266        0.011551           0.125           223                0.003976
 LTCUSDT            0.000156        0.019267           0.125           280                0.000190
 ARBUSDT            0.006090        0.012901           0.125           252                0.007416
LINKUSDT            0.014687        0.011307           0.125           270                0.017885
 BCHUSDT            0.047082        0.045235           0.125           232                0.057333
 ETCUSDT            0.048215        0.021998           0.125           269                0.058712
 ETHUSDT            0.055204        0.031872           0.125           221                0.067222
 BNBUSDT            0.085115        0.042758           0.125           249                0.103645
 APTUSDT            0.119680        0.018776           0.125           255                0.145735
DOGEUSDT            0.136756        0.044869           0.125           266                0.166528
```
