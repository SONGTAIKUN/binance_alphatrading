# Long-History Nested OOS Pipeline

This package reruns fold-local parameter selection on the long 20-symbol common Binance history.

Scope:

- P4 committee parameters are selected inside each fold's training window.
- P3d slow AlphaGPT incremental formulas and weights are selected on fold-local validation windows.
- P9 derivatives incremental formulas and weights are selected on fold-local validation windows.
- P15c allocator/cluster-cap parameters are selected on fold-local validation windows.
- Final PnL is stitched from test windows only.
- The P6/P6b primitive library is treated as a predeclared candidate universe; no full-sample retained-feature filter is used for this run.

Inputs:

- asset panel: `experiments/baseline_locked_v1/inputs/long_history/baseline_panel/asset_residual_panel.csv`
- manual slow features: `experiments/baseline_locked_v1/outputs/long_history_nested_oos/inputs/manual_slow_feature_panel_long.csv.gz`
- P6/P6b features: `experiments/baseline_locked_v1/outputs/long_history_nested_oos/inputs/p6b_feature_panel_long.csv.gz`
- folds: `experiments/baseline_locked_v1/outputs/long_history_nested_oos/inputs/long_formula_selections.csv`

## Stage Summary

```text
stage  periods  total_return  gross_total_return_before_cost   sharpe  max_drawdown  mean_rank_ic  mean_top_bottom_label_spread  mean_period_return  mean_gross_period_return  positive_period_rate  positive_gross_period_rate  mean_turnover  total_turnover  cost_drag  mean_period_cost  target_apply_rate  mean_abs_beta_btc  mean_abs_beta_eth  mean_abs_beta_market  mean_symbol_concentration  fold_count  positive_fold_rate  gross_max_drawdown  mean_gross_exposure  mean_max_abs_weight  max_abs_weight  total_stopped_weight  stopped_period_rate  max_abs_beta_btc  max_abs_beta_eth  max_abs_beta_market  mean_optimizer_success  max_optimizer_success  mean_optimizer_exposure_error  max_optimizer_exposure_error  mean_optimizer_target_gross  max_optimizer_target_gross  mean_optimizer_actual_gross  max_optimizer_actual_gross  fil_total_contribution  mean_target_turnover  total_target_turnover  mean_turnover_saved  total_turnover_saved  mean_target_gross_exposure  total_target_gross_exposure  turnover_reduction  gross_exposure_retention  mean_short_guard_removed
   p4      245      0.075859                        0.135011 0.604857     -0.056572      0.039731                      0.000838            0.000315                  0.000534              0.522449                    0.546939       0.364272       89.246544   0.053548          0.000219                1.0       3.502362e-15       4.418059e-15          2.287766e-14                   0.019789          32             0.62500           -0.053663             0.272637             0.086770           0.125                   0.0                  0.0      1.496221e-13      1.471093e-13         8.197223e-13                1.000000                    1.0                   2.370353e-14                  8.197269e-13                     0.570296                    0.860877                     0.272637                    0.753508                0.025770              0.364272              89.246544             0.000000              0.000000                    0.272637                    66.796052            0.000000                  1.000000                  2.228571
  p3d      244      0.032135                        0.086729 0.290037     -0.111271      0.043612                      0.001859            0.000145                  0.000356              0.504098                    0.536885       0.352134       85.920750   0.051552          0.000211                1.0       3.410053e-15       6.353740e-15          2.178861e-14                   0.019384          32             0.53125           -0.096708             0.267040             0.084239           0.125                   0.0                  0.0      8.450250e-14      2.954997e-13         8.153658e-13                1.000000                    1.0                   2.294022e-14                  8.153692e-13                     0.569449                    0.860877                     0.267040                    0.753508               -0.023082              0.352134              85.920750             0.000000              0.000000                    0.267040                    65.157753            0.000000                  1.000000                  2.200820
   p9      242      0.084681                        0.140470 0.714731     -0.089025      0.044239                      0.003318            0.000351                  0.000558              0.533058                    0.561983       0.345579       83.629998   0.050178          0.000207                1.0       4.667127e-15       6.608185e-15          2.647323e-14                   0.018701          32             0.59375           -0.074777             0.261991             0.083475           0.125                   0.0                  0.0      1.408583e-13      3.878772e-13         8.674797e-13                1.000000                    1.0                   2.859505e-14                  8.674783e-13                     0.571912                    0.907115                     0.261991                    0.832037               -0.019703              0.345579              83.629998             0.000000              0.000000                    0.261991                    63.401822            0.000000                  1.000000                  2.190083
 p15c      242      0.094169                        0.150801 0.800291     -0.088949      0.044239                      0.004329            0.000386                  0.000595              0.533058                    0.566116       0.347724       84.149199   0.050490          0.000209                1.0       4.258947e-15       5.946227e-15          1.970631e-14                   0.018025          32             0.62500           -0.075171             0.261507             0.081178           0.125                   0.0                  0.0      1.408583e-13      3.878772e-13         6.504623e-13                0.971074                    1.0                   8.323454e-13                  1.161715e-10                          NaN                         NaN                          NaN                         NaN               -0.020710              0.349167              84.498322             0.001795              0.434415                    0.261991                    63.401822            0.004132                  0.998154                  2.190083
```

## P15c Cost Sensitivity

```text
stage  one_way_cost_bps  periods  total_return  gross_total_return    sharpe  max_drawdown  gross_max_drawdown  mean_turnover  total_turnover  positive_period_rate
 p15c               0.0    242.0      0.150801            0.150801  1.232243     -0.075171           -0.075171       0.347724       84.149199              0.566116
 p15c               6.0    242.0      0.094169            0.150801  0.800291     -0.088949           -0.075171       0.347724       84.149199              0.533058
 p15c              15.0    242.0      0.014377            0.150801  0.151632     -0.110298           -0.075171       0.347724       84.149199              0.458678
 p15c              20.0    242.0     -0.027422            0.150801 -0.208593     -0.121946           -0.075171       0.347724       84.149199              0.442149
 p15c              30.0    242.0     -0.105948            0.150801 -0.927402     -0.144793           -0.075171       0.347724       84.149199              0.392562
```

## P15c Segment Summary

```text
                     segment  one_way_cost_bps  periods  total_return  gross_total_return    sharpe  max_drawdown  gross_max_drawdown  mean_turnover  total_turnover  positive_period_rate
                        full               6.0    242.0      0.094169            0.150801  0.800291     -0.088949           -0.075171       0.347724       84.149199              0.533058
                        full              15.0    242.0      0.014377            0.150801  0.151632     -0.110298           -0.075171       0.347724       84.149199              0.458678
                        full              20.0    242.0     -0.027422            0.150801 -0.208593     -0.121946           -0.075171       0.347724       84.149199              0.442149
                        full              30.0    242.0     -0.105948            0.150801 -0.927402     -0.144793           -0.075171       0.347724       84.149199              0.392562
    pre_original_2023_2024H1               6.0     71.0     -0.014599           -0.000412 -0.365249     -0.064236           -0.057252       0.335482       23.819226              0.535211
    pre_original_2023_2024H1              15.0     71.0     -0.035508           -0.000412 -0.943620     -0.075578           -0.057252       0.335482       23.819226              0.478873
    pre_original_2023_2024H1              20.0     71.0     -0.046935           -0.000412 -1.264298     -0.082761           -0.057252       0.335482       23.819226              0.422535
    pre_original_2023_2024H1              30.0     71.0     -0.069391           -0.000412 -1.903519     -0.096964           -0.057252       0.335482       23.819226              0.366197
original_window_2024_07_plus               6.0    171.0      0.110378            0.151276  1.349263     -0.031342           -0.026890       0.352807       60.329974              0.532164
original_window_2024_07_plus              15.0    171.0      0.051721            0.151276  0.664373     -0.039773           -0.026890       0.352807       60.329974              0.450292
original_window_2024_07_plus              20.0    171.0      0.020474            0.151276  0.283733     -0.044426           -0.026890       0.352807       60.329974              0.450292
original_window_2024_07_plus              30.0    171.0     -0.039283            0.151276 -0.476175     -0.053669           -0.026890       0.352807       60.329974              0.403509
```

## Selection Counts

### p4

```text
selected_spec
a00_b10_c00_lf45_ema35           3
a02_b04_c04_lf50_ema30           3
a00_b04_c06_lf45_c09b11_ema35    2
a04_b00_c06_lf55_comp2_ema35     2
a00_b10_c00_lf55_comp2_ema35     2
a06_b04_c00_lf55_ema30           2
a06_b04_c00_lf45_ema30           2
a02_b04_c04_lf55_ema30           2
a06_b00_c04_lf45_ema30           1
a00_b00_c10_lf55_ema30           1
a00_b06_c04_lf45_ema30           1
a02_b02_c06_lf55_c09b11_ema35    1
a00_b10_c00_lf50_ema30           1
a00_b10_c00_lf55_c09b11_ema35    1
a00_b04_c06_lf45_comp2_ema35     1
a00_b02_c08_lf55_comp2_ema35     1
a02_b06_c02_lf55_comp2_ema35     1
a04_b00_c06_lf50_ema30           1
a00_b04_c06_lf55_c09b11_ema35    1
a00_b04_c06_lf50_c09b11_ema35    1
a02_b02_c06_lf50_c09b11_ema35    1
a08_b02_c00_lf45_ema35           1
```

### p3d

```text
formula_name
__P4_ONLY__    4
agpt_0255      2
agpt_0362      2
agpt_0320      2
agpt_0113      2
agpt_0311      1
agpt_0126      1
agpt_0037      1
agpt_0336      1
agpt_0070      1
agpt_0182      1
agpt_0199      1
agpt_0300      1
agpt_0217      1
agpt_0274      1
agpt_0165      1
agpt_0013      1
agpt_0068      1
agpt_0322      1
agpt_0177      1
agpt_0091      1
agpt_0136      1
agpt_0250      1
agpt_0340      1
agpt_0001      1
```

### p9

```text
deriv_formula_name
__P4_ONLY__     3
p7_agpt_0049    3
p7_agpt_0079    2
p7_agpt_0078    2
p7_agpt_0058    2
p7_agpt_0066    1
p7_agpt_0062    1
p7_agpt_0111    1
p7_agpt_0096    1
p7_agpt_0075    1
p7_agpt_0129    1
p7_agpt_0055    1
p7_agpt_0059    1
p7_agpt_0029    1
p7_agpt_0109    1
p7_agpt_0156    1
p7_agpt_0046    1
p7_agpt_0016    1
p7_agpt_0021    1
p7_agpt_0091    1
p7_agpt_0026    1
p7_agpt_0077    1
p7_agpt_0052    1
p7_agpt_0070    1
p7_agpt_0115    1
```

### p15c

```text
selected_extreme_spec
p15c_lb1440_k3_trig75_cap60_turn02    6
p15c_lb720_k3_trig75_cap60_turn02     3
p15c_lb336_k3_trig75_cap60_turn02     3
p15c_lb1440_k3_trig85_cap70_turn02    2
p15c_lb1440_k4_trig75_cap60_turn00    2
p15c_lb720_k3_trig75_cap60_turn00     2
p15c_lb336_k3_trig80_cap65_turn00     2
p15c_lb336_k3_trig75_cap60_turn00     1
p15c_lb720_k4_trig80_cap65_turn02     1
p15c_lb720_k4_trig75_cap60_turn00     1
p15c_lb720_k3_trig85_cap70_turn02     1
p15c_lb720_k3_trig80_cap65_turn02     1
p15c_lb720_k4_trig75_cap60_turn02     1
p15c_lb1440_k4_trig75_cap60_turn02    1
p15c_lb336_k4_trig85_cap70_turn00     1
p15c_lb1440_k3_trig80_cap65_turn02    1
p15c_lb336_k4_trig80_cap65_turn00     1
p15c_lb336_k4_trig75_cap60_turn02     1
p15c_lb1440_k4_trig85_cap70_turn02    1
```

## P15c Asset Contribution

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
