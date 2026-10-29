# P18d Guarded Factor Selector

- P18d accepted fold rate: 15.62%
- P18d breakeven one-way cost: 17.53bps
- P18c breakeven one-way cost: 25.70bps
- P15c breakeven one-way cost: 16.72bps

## Comparison

```text
stage  periods  total_return  gross_total_return_before_cost   sharpe  max_drawdown  mean_rank_ic  mean_top_bottom_label_spread  mean_period_return  mean_gross_period_return  positive_period_rate  positive_gross_period_rate  mean_turnover  total_turnover  cost_drag  mean_period_cost  target_apply_rate  mean_abs_beta_btc  mean_abs_beta_eth  mean_abs_beta_market  mean_symbol_concentration  fold_count  positive_fold_rate  gross_max_drawdown  mean_gross_exposure  mean_max_abs_weight  max_abs_weight  total_stopped_weight  stopped_period_rate  max_abs_beta_btc  max_abs_beta_eth  max_abs_beta_market  mean_optimizer_success  max_optimizer_success  mean_optimizer_exposure_error  max_optimizer_exposure_error  mean_optimizer_target_gross  max_optimizer_target_gross  mean_optimizer_actual_gross  max_optimizer_actual_gross  fil_total_contribution  mean_target_turnover  total_target_turnover  mean_turnover_saved  total_turnover_saved  mean_target_gross_exposure  total_target_gross_exposure  turnover_reduction  gross_exposure_retention  mean_short_guard_removed  total_return_20bps  sharpe_20bps  max_drawdown_20bps  total_return_30bps  sharpe_30bps  max_drawdown_30bps  q25_segment_sharpe  long_short_balance  short_abs_contribution_share  mean_max_cluster_gross_share  mean_cluster_hhi  mean_active_cluster_count  mean_abs_pc1_exposure  mean_abs_pc2_exposure  mean_abs_pc3_exposure  mean_baseline_max_cluster_gross_share  mean_baseline_cluster_hhi  mean_gross_exposure_retention  cluster_hhi_reduction_vs_baseline  max_cluster_share_reduction_vs_baseline  mean_extreme_cluster_triggered  mean_cluster_cap_applied  mean_turnover_guard_skipped  mean_cluster_cap_solved  mean_turnover_add_vs_p9
 p18d      245      0.100092                        0.156201 0.830080     -0.089929      0.034914                      0.002609            0.000404                  0.000607              0.534694                    0.555102       0.338584       82.953149   0.049772          0.000203                1.0       3.935048e-15       5.132498e-15          1.667136e-14                   0.020444          32             0.62500           -0.074212             0.284799             0.084413           0.125                   0.0                  0.0      1.408583e-13      3.878772e-13         6.504623e-13                0.975510                    1.0                   8.167373e-13                  1.161715e-10                     0.622625                    0.761688                     0.368997                    0.746567               -0.000131              0.339549              83.189449             0.001313              0.321593                    0.285159                    69.863948            0.002841                  0.998737                  1.767347                 NaN           NaN                 NaN                 NaN           NaN                 NaN                 NaN                 NaN                           NaN                           NaN               NaN                        NaN                    NaN                    NaN                    NaN                                    NaN                        NaN                            NaN                                NaN                                      NaN                             NaN                       NaN                          NaN                      NaN                      NaN
 p18c      288      0.157986                        0.211107 0.642422     -0.137604      0.059743                      0.000567            0.000555                  0.000711              0.583333                    0.583333       0.259466       74.726245   0.044836          0.000156                1.0       2.199562e-15       2.939474e-15          1.270090e-14                   0.036426          32             0.53125           -0.130650             0.421570             0.099721           0.125                   0.0                  0.0      9.521724e-14      2.679697e-13         8.820861e-13                1.000000                    1.0                   1.412144e-14                  8.820908e-13                     0.750305                    0.976326                     0.421570                    0.949729               -0.003266              0.259466              74.726245             0.000000              0.000000                    0.421570                   121.412025            0.000000                  1.000000                  0.000000                 NaN           NaN                 NaN                 NaN           NaN                 NaN                 NaN                 NaN                           NaN                           NaN               NaN                        NaN                    NaN                    NaN                    NaN                                    NaN                        NaN                            NaN                                NaN                                      NaN                             NaN                       NaN                          NaN                      NaN                      NaN
 p15c      242      0.094169                        0.150801 0.800291     -0.088949      0.044239                      0.004329            0.000386                  0.000595              0.533058                    0.566116       0.347724       84.149199   0.050490          0.000209                1.0       4.258947e-15       5.946227e-15          1.970631e-14                   0.018025          32             0.62500           -0.075171             0.261507             0.081178           0.125                   0.0                  0.0      1.408583e-13      3.878772e-13         6.504623e-13                0.971074                    1.0                   8.323454e-13                  1.161715e-10                          NaN                         NaN                          NaN                         NaN               -0.020710              0.349167              84.498322             0.001795              0.434415                    0.261991                    63.401822            0.004132                  0.998154                  2.190083           -0.027422     -0.208593           -0.121946           -0.105948     -0.927402           -0.144793           -1.774536            0.166302                      0.083151                      0.596761          0.479974                   3.285124                0.04727               0.049629               0.063716                               0.628888                   0.517956                       0.998065                            0.07333                                 0.051086                         0.22314                  0.136364                     0.057851                 0.194215                 0.000054
```

## Cost Sensitivity

```text
stage  one_way_cost_bps  total_return  gross_total_return    sharpe  max_drawdown  gross_max_drawdown  mean_turnover  total_turnover  positive_period_rate
 p18d               0.0      0.156201            0.156201  1.247278     -0.074212           -0.074212       0.338584       82.953149              0.555102
 p18d               6.0      0.100092            0.156201  0.830080     -0.089929           -0.074212       0.338584       82.953149              0.534694
 p18d              10.0      0.064198            0.156201  0.551691     -0.100261           -0.074212       0.338584       82.953149              0.514286
 p18d              15.0      0.020966            0.156201  0.203676     -0.113014           -0.074212       0.338584       82.953149              0.485714
 p18d              20.0     -0.020518            0.156201 -0.144100     -0.125588           -0.074212       0.338584       82.953149              0.469388
 p18d              30.0     -0.098524            0.156201 -0.837786     -0.152235           -0.074212       0.338584       82.953149              0.428571
 p18c               0.0      0.211107            0.211107  0.823756     -0.130650           -0.130650       0.259466       74.726245              0.583333
 p18c               6.0      0.157986            0.211107  0.642422     -0.137604           -0.130650       0.259466       74.726245              0.583333
 p18c              10.0      0.123866            0.211107  0.521771     -0.142211           -0.130650       0.259466       74.726245              0.583333
 p18c              15.0      0.082616            0.211107  0.371283     -0.147936           -0.130650       0.259466       74.726245              0.572917
 p18c              20.0      0.042872            0.211107  0.221215     -0.153625           -0.130650       0.259466       74.726245              0.559028
 p18c              30.0     -0.032318            0.211107 -0.077422     -0.181899           -0.130650       0.259466       74.726245              0.531250
 p15c               0.0      0.150801            0.150801  1.232243     -0.075171           -0.075171       0.347724       84.149199              0.566116
 p15c               6.0      0.094169            0.150801  0.800291     -0.088949           -0.075171       0.347724       84.149199              0.533058
 p15c              10.0      0.057963            0.150801  0.512032     -0.098499           -0.075171       0.347724       84.149199              0.504132
 p15c              15.0      0.014377            0.150801  0.151632     -0.110298           -0.075171       0.347724       84.149199              0.458678
 p15c              20.0     -0.027422            0.150801 -0.208593     -0.121946           -0.075171       0.347724       84.149199              0.442149
 p15c              30.0     -0.105948            0.150801 -0.927402     -0.144793           -0.075171       0.347724       84.149199              0.392562
```

## Segment Summary

```text
stage                      segment  one_way_cost_bps  periods  total_return  gross_total_return    sharpe  max_drawdown  gross_max_drawdown  mean_turnover  total_turnover
 p18d                         full               6.0      245      0.100092            0.156201  0.830080     -0.089929           -0.074212       0.338584       82.953149
 p18d                         full              15.0      245      0.020966            0.156201  0.203676     -0.113014           -0.074212       0.338584       82.953149
 p18d                         full              20.0      245     -0.020518            0.156201 -0.144100     -0.125588           -0.074212       0.338584       82.953149
 p18d                         full              30.0      245     -0.098524            0.156201 -0.837786     -0.152235           -0.074212       0.338584       82.953149
 p18d     pre_original_2023_2024H1               6.0       69     -0.002500            0.011121 -0.031460     -0.066883           -0.059208       0.327545       22.600621
 p18d     pre_original_2023_2024H1              15.0       69     -0.022593            0.011121 -0.561807     -0.078280           -0.059208       0.327545       22.600621
 p18d     pre_original_2023_2024H1              20.0       69     -0.033584            0.011121 -0.855359     -0.084554           -0.059208       0.327545       22.600621
 p18d     pre_original_2023_2024H1              30.0       69     -0.055203            0.011121 -1.439451     -0.096977           -0.059208       0.327545       22.600621
 p18d original_window_2024_07_plus               6.0      176      0.102848            0.143484  1.247444     -0.049270           -0.042158       0.342912       60.352528
 p18d original_window_2024_07_plus              15.0      176      0.044567            0.143484  0.571170     -0.059843           -0.042158       0.342912       60.352528
 p18d original_window_2024_07_plus              20.0      176      0.013520            0.143484  0.195054     -0.065668           -0.042158       0.342912       60.352528
 p18d original_window_2024_07_plus              30.0      176     -0.045852            0.143484 -0.556209     -0.079410           -0.042158       0.342912       60.352528
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

## Selected Guard Counts

```text
selected_guard_spec
__P15C_FALLBACK__                27
p18d_mom35_weak40_dis50_reg70     2
p18d_mom35_weak40_dis35_reg70     1
p18d_mom50_weak40_dis35_reg35     1
p18d_mom65_weak40_dis50_reg70     1
```

## Worst Fold Deltas vs P15c

```text
 fold                test_start           selected_guard_spec  fallback_to_p15c  total_return_p18d  gross_total_return_before_cost_p18d  sharpe_p18d  max_drawdown_p18d  mean_turnover_p18d  total_return_p15c  gross_total_return_before_cost_p15c  sharpe_p15c  max_drawdown_p15c  mean_turnover_p15c  delta_total_return  delta_gross_total_return_before_cost  delta_sharpe  delta_max_drawdown  delta_mean_turnover
 14.0 2024-11-08 08:00:00+00:00 p18d_mom35_weak40_dis35_reg70             False          -0.004553                            -0.003302    -1.184338          -0.014387            0.232526           0.039028                             0.040756     8.095658          -0.004163            0.347263           -0.043581                             -0.044058     -9.279996           -0.010224            -0.114738
 30.0 2026-03-03 08:00:00+00:00 p18d_mom50_weak40_dis35_reg35             False          -0.006847                            -0.004689    -2.162931          -0.006606            0.401318          -0.001243                             0.000254    -1.826603          -0.001526            0.311984           -0.005604                             -0.004943     -0.336329           -0.005081             0.089334
 28.0 2026-01-02 08:00:00+00:00 p18d_mom35_weak40_dis50_reg70             False           0.004691                             0.005104     3.904338          -0.000794            0.076095          -0.001867                            -0.000410    -1.757180          -0.002355            0.304018            0.006558                              0.005514      5.661518            0.001560            -0.227923
  9.0 2024-06-11 08:00:00+00:00 p18d_mom35_weak40_dis50_reg70             False           0.019052                             0.020373     4.799650          -0.005038            0.240220           0.001269                             0.002876     2.074107          -0.001456            0.297009            0.017783                              0.017497      2.725542           -0.003582            -0.056789
 32.0 2026-05-02 08:00:00+00:00 p18d_mom65_weak40_dis50_reg70             False           0.032503                             0.034258    10.380191          -0.000870            0.315566           0.002269                             0.003598     0.925308          -0.007317            0.245374            0.030235                              0.030660      9.454884            0.006446             0.070191
```

## Asset Contribution

```text
  symbol  gross_contribution  avg_abs_weight  max_abs_weight  observations  abs_contribution_share
 UNIUSDT           -0.026135        0.011790        0.125000           212                0.074318
 BNBUSDT           -0.020411        0.029705        0.125000           202                0.058040
 ETCUSDT           -0.018245        0.022713        0.125000           209                0.051881
 DOTUSDT           -0.016582        0.011679        0.125000           217                0.047153
 BTCUSDT           -0.016550        0.026386        0.125000           185                0.047062
 ADAUSDT           -0.003422        0.012739        0.125000           198                0.009732
 FILUSDT           -0.000131        0.011644        0.125000           197                0.000371
AVAXUSDT            0.001689        0.020641        0.125000           212                0.004803
 BCHUSDT            0.005563        0.022820        0.125000           205                0.015820
ATOMUSDT            0.006177        0.013123        0.111792           213                0.017564
 ETHUSDT            0.007285        0.021483        0.125000           203                0.020717
 SOLUSDT            0.010129        0.013537        0.125000           211                0.028802
 ARBUSDT            0.010740        0.010821        0.113180           215                0.030539
DOGEUSDT            0.020852        0.023820        0.125000           204                0.059296
 XRPUSDT            0.021139        0.014207        0.125000           215                0.060110
NEARUSDT            0.023833        0.011647        0.125000           215                0.067773
LINKUSDT            0.023899        0.013545        0.125000           207                0.067958
 APTUSDT            0.029732        0.014354        0.125000           215                0.084546
  OPUSDT            0.042848        0.011264        0.108571           214                0.121842
 LTCUSDT            0.046305        0.019837        0.125000           211                0.131673
```
