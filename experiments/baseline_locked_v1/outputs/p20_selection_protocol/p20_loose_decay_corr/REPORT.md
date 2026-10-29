# p20_loose_decay_corr

Loose diagnostic protocol to test whether any filtered P18d edge remains after homogeneity and decay checks.

- accepted fold rate after guard validation: 0.00%
- protocol pass fold rate before guard validation: 18.75%
- breakeven one-way cost: 16.72bps

## Comparison

```text
               stage  periods  total_return  gross_total_return_before_cost   sharpe  max_drawdown  mean_rank_ic  mean_top_bottom_label_spread  mean_period_return  mean_gross_period_return  positive_period_rate  positive_gross_period_rate  mean_turnover  total_turnover  cost_drag  mean_period_cost  target_apply_rate  mean_abs_beta_btc  mean_abs_beta_eth  mean_abs_beta_market  mean_symbol_concentration  fold_count  positive_fold_rate  gross_max_drawdown  mean_gross_exposure  mean_max_abs_weight  max_abs_weight  total_stopped_weight  stopped_period_rate  max_abs_beta_btc  max_abs_beta_eth  max_abs_beta_market  mean_optimizer_success  max_optimizer_success  mean_optimizer_exposure_error  max_optimizer_exposure_error  fil_total_contribution  mean_target_turnover  total_target_turnover  mean_turnover_saved  total_turnover_saved  mean_target_gross_exposure  total_target_gross_exposure  turnover_reduction  gross_exposure_retention  mean_short_guard_removed  mean_optimizer_target_gross  max_optimizer_target_gross  mean_optimizer_actual_gross  max_optimizer_actual_gross  total_return_20bps  sharpe_20bps  max_drawdown_20bps  total_return_30bps  sharpe_30bps  max_drawdown_30bps  q25_segment_sharpe  long_short_balance  short_abs_contribution_share  mean_max_cluster_gross_share  mean_cluster_hhi  mean_active_cluster_count  mean_abs_pc1_exposure  mean_abs_pc2_exposure  mean_abs_pc3_exposure  mean_baseline_max_cluster_gross_share  mean_baseline_cluster_hhi  mean_gross_exposure_retention  cluster_hhi_reduction_vs_baseline  max_cluster_share_reduction_vs_baseline  mean_extreme_cluster_triggered  mean_cluster_cap_applied  mean_turnover_guard_skipped  mean_cluster_cap_solved  mean_turnover_add_vs_p9
p20_loose_decay_corr      242      0.094169                        0.150801 0.800291     -0.088949      0.044239                      0.004329            0.000386                  0.000595              0.533058                    0.566116       0.347724       84.149199   0.050490          0.000209                1.0       4.258947e-15       5.946227e-15          1.970631e-14                   0.018025          32             0.62500           -0.075171             0.261507             0.081178           0.125                   0.0                  0.0      1.408583e-13      3.878772e-13         6.504623e-13                0.971074                    1.0                   8.323454e-13                  1.161715e-10               -0.020710              0.349167              84.498322             0.001795              0.434415                    0.261991                    63.401822            0.004132                  0.998154                  2.190083                          NaN                         NaN                          NaN                         NaN                 NaN           NaN                 NaN                 NaN           NaN                 NaN                 NaN                 NaN                           NaN                           NaN               NaN                        NaN                    NaN                    NaN                    NaN                                    NaN                        NaN                            NaN                                NaN                                      NaN                             NaN                       NaN                          NaN                      NaN                      NaN
                p18d      245      0.100092                        0.156201 0.830080     -0.089929      0.034914                      0.002609            0.000404                  0.000607              0.534694                    0.555102       0.338584       82.953149   0.049772          0.000203                1.0       3.935048e-15       5.132498e-15          1.667136e-14                   0.020444          32             0.62500           -0.074212             0.284799             0.084413           0.125                   0.0                  0.0      1.408583e-13      3.878772e-13         6.504623e-13                0.975510                    1.0                   8.167373e-13                  1.161715e-10               -0.000131              0.339549              83.189449             0.001313              0.321593                    0.285159                    69.863948            0.002841                  0.998737                  1.767347                     0.622625                    0.761688                     0.368997                    0.746567                 NaN           NaN                 NaN                 NaN           NaN                 NaN                 NaN                 NaN                           NaN                           NaN               NaN                        NaN                    NaN                    NaN                    NaN                                    NaN                        NaN                            NaN                                NaN                                      NaN                             NaN                       NaN                          NaN                      NaN                      NaN
                p18c      288      0.157986                        0.211107 0.642422     -0.137604      0.059743                      0.000567            0.000555                  0.000711              0.583333                    0.583333       0.259466       74.726245   0.044836          0.000156                1.0       2.199562e-15       2.939474e-15          1.270090e-14                   0.036426          32             0.53125           -0.130650             0.421570             0.099721           0.125                   0.0                  0.0      9.521724e-14      2.679697e-13         8.820861e-13                1.000000                    1.0                   1.412144e-14                  8.820908e-13               -0.003266              0.259466              74.726245             0.000000              0.000000                    0.421570                   121.412025            0.000000                  1.000000                  0.000000                     0.750305                    0.976326                     0.421570                    0.949729                 NaN           NaN                 NaN                 NaN           NaN                 NaN                 NaN                 NaN                           NaN                           NaN               NaN                        NaN                    NaN                    NaN                    NaN                                    NaN                        NaN                            NaN                                NaN                                      NaN                             NaN                       NaN                          NaN                      NaN                      NaN
                p15c      242      0.094169                        0.150801 0.800291     -0.088949      0.044239                      0.004329            0.000386                  0.000595              0.533058                    0.566116       0.347724       84.149199   0.050490          0.000209                1.0       4.258947e-15       5.946227e-15          1.970631e-14                   0.018025          32             0.62500           -0.075171             0.261507             0.081178           0.125                   0.0                  0.0      1.408583e-13      3.878772e-13         6.504623e-13                0.971074                    1.0                   8.323454e-13                  1.161715e-10               -0.020710              0.349167              84.498322             0.001795              0.434415                    0.261991                    63.401822            0.004132                  0.998154                  2.190083                          NaN                         NaN                          NaN                         NaN           -0.027422     -0.208593           -0.121946           -0.105948     -0.927402           -0.144793           -1.774536            0.166302                      0.083151                      0.596761          0.479974                   3.285124                0.04727               0.049629               0.063716                               0.628888                   0.517956                       0.998065                            0.07333                                 0.051086                         0.22314                  0.136364                     0.057851                 0.194215                 0.000054
```

## Cost Sensitivity

```text
               stage  one_way_cost_bps  total_return  gross_total_return    sharpe  max_drawdown  gross_max_drawdown  mean_turnover  total_turnover  positive_period_rate
p20_loose_decay_corr               6.0      0.094169            0.150801  0.800291     -0.088949           -0.075171       0.347724       84.149199              0.533058
p20_loose_decay_corr              15.0      0.014377            0.150801  0.151632     -0.110298           -0.075171       0.347724       84.149199              0.458678
p20_loose_decay_corr              20.0     -0.027422            0.150801 -0.208593     -0.121946           -0.075171       0.347724       84.149199              0.442149
p20_loose_decay_corr              30.0     -0.105948            0.150801 -0.927402     -0.144793           -0.075171       0.347724       84.149199              0.392562
                p18d               6.0      0.100092            0.156201  0.830080     -0.089929           -0.074212       0.338584       82.953149              0.534694
                p18d              15.0      0.020966            0.156201  0.203676     -0.113014           -0.074212       0.338584       82.953149              0.485714
                p18d              20.0     -0.020518            0.156201 -0.144100     -0.125588           -0.074212       0.338584       82.953149              0.469388
                p18d              30.0     -0.098524            0.156201 -0.837786     -0.152235           -0.074212       0.338584       82.953149              0.428571
                p18c               6.0      0.157986            0.211107  0.642422     -0.137604           -0.130650       0.259466       74.726245              0.583333
                p18c              15.0      0.082616            0.211107  0.371283     -0.147936           -0.130650       0.259466       74.726245              0.572917
                p18c              20.0      0.042872            0.211107  0.221215     -0.153625           -0.130650       0.259466       74.726245              0.559028
                p18c              30.0     -0.032318            0.211107 -0.077422     -0.181899           -0.130650       0.259466       74.726245              0.531250
                p15c               6.0      0.094169            0.150801  0.800291     -0.088949           -0.075171       0.347724       84.149199              0.533058
                p15c              15.0      0.014377            0.150801  0.151632     -0.110298           -0.075171       0.347724       84.149199              0.458678
                p15c              20.0     -0.027422            0.150801 -0.208593     -0.121946           -0.075171       0.347724       84.149199              0.442149
                p15c              30.0     -0.105948            0.150801 -0.927402     -0.144793           -0.075171       0.347724       84.149199              0.392562
```

## Segment Summary

```text
               stage                      segment  one_way_cost_bps  periods  total_return  gross_total_return    sharpe  max_drawdown  gross_max_drawdown  mean_turnover  total_turnover
p20_loose_decay_corr                         full               6.0      242      0.094169            0.150801  0.800291     -0.088949           -0.075171       0.347724       84.149199
p20_loose_decay_corr                         full              15.0      242      0.014377            0.150801  0.151632     -0.110298           -0.075171       0.347724       84.149199
p20_loose_decay_corr                         full              20.0      242     -0.027422            0.150801 -0.208593     -0.121946           -0.075171       0.347724       84.149199
p20_loose_decay_corr                         full              30.0      242     -0.105948            0.150801 -0.927402     -0.144793           -0.075171       0.347724       84.149199
p20_loose_decay_corr     pre_original_2023_2024H1               6.0       69     -0.014802           -0.001030 -0.376182     -0.064236           -0.057252       0.335255       23.132577
p20_loose_decay_corr     pre_original_2023_2024H1              15.0       69     -0.035111           -0.001030 -0.945969     -0.075553           -0.057252       0.335255       23.132577
p20_loose_decay_corr     pre_original_2023_2024H1              20.0       69     -0.046215           -0.001030 -1.261885     -0.082068           -0.057252       0.335255       23.132577
p20_loose_decay_corr     pre_original_2023_2024H1              30.0       69     -0.068048           -0.001030 -1.891629     -0.095661           -0.057252       0.335255       23.132577
p20_loose_decay_corr original_window_2024_07_plus               6.0      173      0.110608            0.151988  1.343946     -0.031342           -0.026890       0.352697       61.016622
p20_loose_decay_corr original_window_2024_07_plus              15.0      173      0.051288            0.151988  0.655324     -0.039773           -0.026890       0.352697       61.016622
p20_loose_decay_corr original_window_2024_07_plus              20.0      173      0.019704            0.151988  0.272611     -0.044426           -0.026890       0.352697       61.016622
p20_loose_decay_corr original_window_2024_07_plus              30.0      173     -0.040667            0.151988 -0.491422     -0.053669           -0.026890       0.352697       61.016622
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

## Fallback Reasons

```text
fallback_reason
too_few_surviving_factors    26
guard_validation_failed       6
```

## Worst Fold Deltas vs P15c

```text
 fold                test_start selected_guard_spec  fallback_to_p15c  total_return_p20_loose_decay_corr  gross_total_return_before_cost_p20_loose_decay_corr  sharpe_p20_loose_decay_corr  max_drawdown_p20_loose_decay_corr  mean_turnover_p20_loose_decay_corr  total_return_p15c  gross_total_return_before_cost_p15c  sharpe_p15c  max_drawdown_p15c  mean_turnover_p15c  delta_total_return  delta_gross_total_return_before_cost  delta_sharpe  delta_max_drawdown  delta_mean_turnover
    1 2023-10-15 08:00:00+00:00   __P15C_FALLBACK__              True                           0.023579                                             0.025929                     7.599070                          -0.002528                            0.425775           0.023579                             0.025929     7.599070          -0.002528            0.425775                 0.0                                   0.0           0.0                 0.0                  0.0
   30 2026-03-03 08:00:00+00:00   __P15C_FALLBACK__              True                          -0.001243                                             0.000254                    -1.826603                          -0.001526                            0.311984          -0.001243                             0.000254    -1.826603          -0.001526            0.311984                 0.0                                   0.0           0.0                 0.0                  0.0
   29 2026-02-01 08:00:00+00:00   __P15C_FALLBACK__              True                          -0.000375                                             0.000276                    -0.679260                          -0.001738                            0.181055          -0.000375                             0.000276    -0.679260          -0.001738            0.181055                 0.0                                   0.0           0.0                 0.0                  0.0
   28 2026-01-02 08:00:00+00:00   __P15C_FALLBACK__              True                          -0.001867                                            -0.000410                    -1.757180                          -0.002355                            0.304018          -0.001867                            -0.000410    -1.757180          -0.002355            0.304018                 0.0                                   0.0           0.0                 0.0                  0.0
   27 2025-12-03 08:00:00+00:00   __P15C_FALLBACK__              True                           0.009683                                             0.011150                     2.535720                          -0.008090                            0.346416           0.009683                             0.011150     2.535720          -0.008090            0.346416                 0.0                                   0.0           0.0                 0.0                  0.0
   26 2025-11-03 08:00:00+00:00   __P15C_FALLBACK__              True                           0.015646                                             0.017413                     5.309208                          -0.001309                            0.362774           0.015646                             0.017413     5.309208          -0.001309            0.362774                 0.0                                   0.0           0.0                 0.0                  0.0
   25 2025-10-04 08:00:00+00:00   __P15C_FALLBACK__              True                           0.004608                                             0.005538                     3.583117                          -0.002786                            0.220396           0.004608                             0.005538     3.583117          -0.002786            0.220396                 0.0                                   0.0           0.0                 0.0                  0.0
   24 2025-09-04 08:00:00+00:00   __P15C_FALLBACK__              True                          -0.005671                                            -0.004169                    -1.238327                          -0.020505                            0.359102          -0.005671                            -0.004169    -1.238327          -0.020505            0.359102                 0.0                                   0.0           0.0                 0.0                  0.0
   23 2025-08-05 08:00:00+00:00   __P15C_FALLBACK__              True                           0.004091                                             0.004736                     3.723597                          -0.002173                            0.153005           0.004091                             0.004736     3.723597          -0.002173            0.153005                 0.0                                   0.0           0.0                 0.0                  0.0
   22 2025-07-06 08:00:00+00:00   __P15C_FALLBACK__              True                          -0.016271                                            -0.014882                    -8.231773                          -0.015294                            0.390650          -0.016271                            -0.014882    -8.231773          -0.015294            0.390650                 0.0                                   0.0           0.0                 0.0                  0.0
   21 2025-06-06 08:00:00+00:00   __P15C_FALLBACK__              True                           0.028580                                             0.030835                     8.847496                          -0.002549                            0.524071           0.028580                             0.030835     8.847496          -0.002549            0.524071                 0.0                                   0.0           0.0                 0.0                  0.0
   20 2025-05-07 08:00:00+00:00   __P15C_FALLBACK__              True                           0.025696                                             0.027690                     9.801984                          -0.001648                            0.405971           0.025696                             0.027690     9.801984          -0.001648            0.405971                 0.0                                   0.0           0.0                 0.0                  0.0
```

## Factor Protocol Audit

```text
 fold  protocol_pass           fallback_reason  quality_pass_count  corr_reject_count  surviving_factor_count  validation_high_risk_rate  recent_high_risk_rate  pre_mean_abs_factor_corr  post_mean_abs_factor_corr
    1          False too_few_surviving_factors                   3                  1                       2                   0.110153               0.163205                  0.323176                   0.087039
    2          False too_few_surviving_factors                   5                  3                       2                   0.250479               0.706231                  0.587981                   0.003255
    3           True                                             6                  2                       4                   0.376916               0.486647                  0.402494                   0.365943
    4          False too_few_surviving_factors                   5                  3                       2                   0.574713               0.712166                  0.833795                   0.739114
    5          False too_few_surviving_factors                   6                  3                       3                   0.521552               0.097923                  0.605802                   0.414073
    6          False too_few_surviving_factors                   5                  3                       2                   0.490900               0.976261                  0.824666                   0.727017
    7           True                                             8                  4                       4                   0.462165               0.249258                  0.549204                   0.445677
    8           True                                             7                  3                       4                   0.469828               0.201780                  0.592469                   0.522113
    9          False too_few_surviving_factors                   1                  0                       1                   0.322318               0.124629                  0.512108                   0.000000
   10          False too_few_surviving_factors                   5                  4                       1                   0.213123               0.219585                  0.878672                   0.000000
   11          False too_few_surviving_factors                   4                  3                       1                   0.211207               0.525223                  0.429904                   0.000000
   12           True                                             8                  4                       4                   0.216475               0.115727                  0.749874                   0.678762
   13           True                                             8                  3                       5                   0.221743               0.222552                  0.521027                   0.403545
   14           True                                             6                  2                       4                   0.122605               0.320475                  0.463756                   0.260099
   15          False too_few_surviving_factors                   6                  3                       3                   0.414751               1.000000                  0.548367                   0.770496
   16          False too_few_surviving_factors                   6                  3                       3                   0.547893               0.109792                  0.352715                   0.308041
   17          False too_few_surviving_factors                   0                  0                       0                   0.575192               0.302671                  0.393404                        NaN
   18          False too_few_surviving_factors                   4                  2                       2                   0.311782               0.471810                  0.653882                   0.071435
   19          False too_few_surviving_factors                   6                  3                       3                   0.247126               0.169139                  0.493828                   0.042003
   20          False too_few_surviving_factors                   6                  3                       3                   0.186782               0.026706                  0.253663                   0.096594
   21          False too_few_surviving_factors                   5                  3                       2                   0.167146               0.109792                  0.344432                   0.192725
   22          False too_few_surviving_factors                   6                  4                       2                   0.140805               0.109792                  0.786163                   0.623604
   23          False too_few_surviving_factors                   0                  0                       0                   0.162835               0.160237                  0.818982                        NaN
   24          False too_few_surviving_factors                   0                  0                       0                   0.153257               0.124629                  0.401889                        NaN
   25          False too_few_surviving_factors                   5                  2                       3                   0.125479               0.181009                  0.370774                   0.099594
   26          False too_few_surviving_factors                   8                  5                       3                   0.174330               0.068249                  0.410027                   0.262274
   27          False too_few_surviving_factors                   1                  0                       1                   0.388889               0.587537                  0.463534                   0.000000
   28          False too_few_surviving_factors                   0                  0                       0                   0.398467               0.000000                  0.431844                        NaN
   29          False too_few_surviving_factors                   3                  0                       3                   0.333333               0.278932                  0.354316                   0.631063
   30          False too_few_surviving_factors                   4                  1                       3                   0.188218               0.157270                  0.738507                   0.596156
   31          False too_few_surviving_factors                   0                  0                       0                   0.186303               0.065282                  0.220336                        NaN
   32          False too_few_surviving_factors                   0                  0                       0                   0.133142               0.000000                  0.310547                        NaN
```

## Asset Contribution

```text
  symbol  gross_contribution  avg_abs_weight  max_abs_weight  observations  abs_contribution_share
 BNBUSDT           -0.032330        0.026796        0.125000           198                0.076850
 UNIUSDT           -0.029043        0.010731        0.125000           206                0.069038
 DOTUSDT           -0.024249        0.013167        0.125000           215                0.057641
 BCHUSDT           -0.020934        0.016197        0.125000           215                0.049760
 FILUSDT           -0.020710        0.009673        0.125000           208                0.049229
  OPUSDT           -0.008316        0.009381        0.125000           215                0.019768
 BTCUSDT           -0.002797        0.028238        0.125000           178                0.006648
DOGEUSDT            0.001544        0.014298        0.125000           200                0.003669
ATOMUSDT            0.008596        0.011840        0.124715           205                0.020434
 SOLUSDT            0.008756        0.015945        0.125000           204                0.020814
 ARBUSDT            0.008873        0.010735        0.125000           205                0.021092
 ETHUSDT            0.014396        0.020802        0.125000           188                0.034221
 ADAUSDT            0.020902        0.013643        0.125000           206                0.049686
LINKUSDT            0.022192        0.014200        0.125000           198                0.052751
 ETCUSDT            0.025970        0.017948        0.125000           198                0.061733
 LTCUSDT            0.027359        0.020128        0.125000           205                0.065035
 XRPUSDT            0.030346        0.015558        0.125000           209                0.072135
 APTUSDT            0.031455        0.014339        0.125000           212                0.074770
AVAXUSDT            0.034826        0.018288        0.125000           208                0.082784
NEARUSDT            0.047093        0.011041        0.125000           213                0.111942
```
