# P19 Fold14 Mismatch Audit

## Scope

- fold: 14
- train_end: 2024-11-08 08:00:00+00:00
- validation window: 2024-08-10 08:00:00+00:00 to 2024-11-05 07:00:00+00:00
- test window: 2024-11-08 08:00:00+00:00 to 2024-12-05 07:00:00+00:00
- selected P18d guard: `p18d_mom35_weak40_dis35_reg70`

## P19a Config Replay

```text
        stage window                        config  validation_selection_score  test_selection_score_vs_p15c_ref  hard_accept_on_validation  selected_by_p18d  periods  total_return  gross_total_return_before_cost    sharpe  max_drawdown  gross_max_drawdown  mean_turnover  total_turnover  cost_drag  positive_period_rate  positive_gross_period_rate  mean_rank_ic  mean_top_bottom_label_spread  total_return_20bps  sharpe_20bps  max_drawdown_20bps  total_return_30bps  sharpe_30bps  max_drawdown_30bps  mean_high_risk_regime  mean_regime_gross_scale  test_minus_validation_total_return
p15c_selected   test                  __P15C_REF__                         NaN                               NaN                      False              True        8      0.039028                        0.040756  8.095658     -0.004163           -0.003989       0.347263        2.778107   0.001667              0.875000                    0.875000      0.218421                      0.038596            0.035006      7.135676            -0.00457            0.032141      6.470159           -0.004860                    NaN                      NaN                            0.021895
     p18c_raw   test                      p18c_raw                         NaN                               NaN                      False             False        9     -0.054104                       -0.052266 -4.419444     -0.058695           -0.057962       0.357071        3.213637   0.001928              0.222222                    0.222222     -0.058145                     -0.014817                 NaN           NaN                 NaN                 NaN           NaN                 NaN                    NaN                      NaN                           -0.137092
p18d_selected   test p18d_mom35_weak40_dis35_reg70                         NaN                               NaN                      False             False        9     -0.004553                       -0.003302 -1.184338     -0.014387           -0.013953       0.232526        2.092731   0.001256              0.444444                    0.444444     -0.042774                     -0.008874           -0.007466     -1.972816            -0.01540           -0.009542     -2.540416           -0.016123                    1.0                      0.7                                 NaN
```

- guard configs beating P15c on fold14 test: 0/36

### Top Test Guard Configs

```text
                       config  total_return  gross_total_return_before_cost    sharpe  max_drawdown  mean_turnover  selected_by_p18d
p18d_mom35_weak40_dis50_reg35     -0.001913                       -0.001246 -0.957960     -0.007620       0.123599             False
p18d_mom35_weak60_dis50_reg35     -0.001913                       -0.001246 -0.957960     -0.007620       0.123599             False
p18d_mom35_weak40_dis35_reg35     -0.002255                       -0.001628 -1.184338     -0.007211       0.116263             False
p18d_mom35_weak60_dis35_reg35     -0.002255                       -0.001628 -1.184338     -0.007211       0.116263             False
p18d_mom35_weak40_dis50_reg50     -0.002747                       -0.001796 -0.957960     -0.010873       0.176570             False
p18d_mom35_weak60_dis50_reg50     -0.002747                       -0.001796 -0.957960     -0.010873       0.176570             False
p18d_mom35_weak60_dis35_reg50     -0.003234                       -0.002340 -1.184338     -0.010291       0.166090             False
p18d_mom35_weak40_dis35_reg50     -0.003234                       -0.002340 -1.184338     -0.010291       0.166090             False
p18d_mom35_weak40_dis50_reg70     -0.003874                       -0.002543 -0.957960     -0.015201       0.247197             False
p18d_mom35_weak60_dis50_reg70     -0.003874                       -0.002543 -0.957960     -0.015201       0.247197             False
```

### Worst Test Guard Configs

```text
                       config  total_return  gross_total_return_before_cost    sharpe  max_drawdown  mean_turnover  selected_by_p18d
p18d_mom65_weak60_dis50_reg70     -0.024497                       -0.023424 -6.054402     -0.021956       0.203191             False
p18d_mom65_weak40_dis50_reg70     -0.024497                       -0.023424 -6.054402     -0.021956       0.203191             False
p18d_mom65_weak60_dis35_reg70     -0.022416                       -0.021398 -5.737505     -0.020451       0.192537             False
p18d_mom65_weak40_dis35_reg70     -0.022416                       -0.021398 -5.737505     -0.020451       0.192537             False
p18d_mom65_weak40_dis50_reg50     -0.017530                       -0.016759 -6.054402     -0.015704       0.145137             False
p18d_mom65_weak60_dis50_reg50     -0.017530                       -0.016759 -6.054402     -0.015704       0.145137             False
p18d_mom50_weak60_dis35_reg70     -0.016773                       -0.015619 -5.993883     -0.014553       0.216947             False
p18d_mom50_weak40_dis35_reg70     -0.016773                       -0.015619 -5.993883     -0.014553       0.216947             False
p18d_mom50_weak60_dis50_reg70     -0.016421                       -0.015200 -5.839472     -0.014979       0.229451             False
p18d_mom50_weak40_dis50_reg70     -0.016421                       -0.015200 -5.839472     -0.014979       0.229451             False
```

## P19b Largest Regime Shifts

```text
                                     feature  validation_mean  test_mean  test_minus_validation  test_percentile_full_sample  validation_std  test_std  validation_nonnull  test_nonnull
                              p18d_high_risk         0.122664   0.981453               0.858789                     0.723035        0.328051  0.134919                2087           647
deriv_top_sum_long_short_ratio_ewm_72_median         0.437801   0.914459               0.476658                     0.864902        0.142597  0.106072                2087           647
deriv_account_long_short_ratio_ewm_72_median         0.821772   1.014847               0.193076                     0.756767        0.069437  0.157796                2087           647
         deriv_abs_funding_stress_168_median         0.715599   0.896307               0.180709                     0.783074        0.371725  0.655798                2087           647
         deriv_log_oi_value_delta_168_median         0.017568   0.194501               0.176933                     0.964686        0.552372  0.088996                2087           647
           deriv_crowding_pressure_72_median         0.316476   0.466170               0.149693                     0.863549        0.034508  0.074808                2087           647
               deriv_oi_growth_72_336_median         0.007231   0.127733               0.120503                     0.976794        0.060434  0.029331                2087           647
                                beta_eth_abs         0.149422   0.260552               0.111131                     0.863691        0.010558  0.061276                2087           647
              deriv_taker_pressure_24_median         0.106499   0.025279              -0.081220                     0.079284        0.031429  0.022890                2087           647
  deriv_taker_long_short_ratio_ewm_24_median         0.106499   0.025279              -0.081220                     0.079284        0.031429  0.022890                2087           647
                              market_ret_72h         0.001393   0.069885               0.068492                     0.903201        0.049090  0.062533                2087           647
                                beta_btc_abs         0.163465   0.228538               0.065074                     0.903521        0.017746  0.026424                2087           647
           deriv_log_oi_value_delta_168_disp         0.123817   0.177550               0.053734                     0.820202        0.051603  0.042023                2087           647
           deriv_abs_basis_stress_168_median         0.748532   0.800257               0.051725                     0.650646        0.279286  0.358745                2087           647
           deriv_abs_funding_stress_168_disp         0.681449   0.639578              -0.041871                     0.633372        0.473981  0.501445                2087           647
  deriv_account_long_short_ratio_ewm_72_disp         0.320570   0.361795               0.041225                     0.652515        0.070968  0.033537                2087           647
                             beta_market_abs         0.679336   0.643984              -0.035352                     0.272226        0.046603  0.029612                2087           647
            deriv_log_oi_value_delta_24_disp         0.049381   0.081010               0.031628                     0.871469        0.047538  0.025935                2087           647
                 deriv_oi_growth_72_336_disp         0.054957   0.084929               0.029972                     0.824790        0.015503  0.017798                2087           647
             deriv_abs_basis_stress_168_disp         0.570477   0.546149              -0.024328                     0.505927        0.136801  0.168794                2087           647
```

## P19c Factor IC Decay

```text
                        raw_name source  selection_score  weight_scale  is_momentum_like  is_weak_or_decaying  validation_ic_mean_recomputed  test_ic_mean  test_minus_validation_ic  validation_top_bottom_spread  test_top_bottom_spread  spread_inverted
           p6_G01_slow_consensus    p6b        13.142641          0.40             False                 True                       0.158379     -0.031126                 -0.189505                      0.260687               -0.083910             True
p6_G04_p3d_diversifier_candidate    p6b         9.594142          0.40             False                 True                       0.136334     -0.041005                 -0.177339                      0.235707               -0.128699             True
        p6_C02_dollar_volume_mom    p6b         8.518437          0.14              True                 True                       0.124741     -0.042251                 -0.166992                      0.217713               -0.067624             True
       p6_C01_liq_persistent_mom    p6b        10.726181          0.14              True                 True                       0.142307     -0.017417                 -0.159724                      0.224626               -0.062906             True
 p6b_J02_oi_growth_crowding_fade    p6b        12.845234          0.40             False                 True                       0.132300     -0.025476                 -0.157777                      0.191663               -0.031441             True
   p6b_J01_oi_growth_mom_quality    p6b         8.669910          0.14              True                 True                       0.127969     -0.022991                 -0.150960                      0.233652               -0.048076             True
             cs_res_high_mom_336   slow        10.199396          0.14              True                 True                       0.126035     -0.006117                 -0.132152                      0.213200                0.022737            False
             cs_breakout_risk_72    p6b         8.864592          0.40             False                 True                       0.093387      0.052757                 -0.040629                      0.156104                0.052754            False
```

## P19d Symbol Delta P18d minus P15c

```text
  symbol  delta_contribution  delta_weight_contribution  p18d_contribution  p15c_contribution  avg_abs_delta_weight  max_abs_delta_weight  observations
 XRPUSDT           -0.025303                  -0.025303          -0.009029           0.016274              0.018787              0.054264             8
NEARUSDT           -0.023729                  -0.023729          -0.007028           0.016701              0.022961              0.075227             8
 ETCUSDT           -0.016742                  -0.016742           0.000087           0.016829              0.015545              0.092284             9
DOGEUSDT           -0.014319                  -0.014319           0.001199           0.015517              0.009176              0.021243             9
 DOTUSDT           -0.008968                  -0.008968          -0.011456          -0.002488              0.021600              0.055919             9
 BTCUSDT           -0.007265                  -0.007265          -0.000452           0.006813              0.053858              0.126648             7
 ADAUSDT           -0.005463                  -0.005463          -0.008496          -0.003033              0.010538              0.030151             7
AVAXUSDT           -0.005334                  -0.005334          -0.000751           0.004583              0.028018              0.117803             9
 ETHUSDT           -0.004855                  -0.004855          -0.005050          -0.000195              0.027261              0.077825             9
LINKUSDT           -0.001361                  -0.001361           0.001663           0.003025              0.011947              0.061247             9
 APTUSDT           -0.000729                  -0.000729          -0.000139           0.000591              0.017957              0.106370             9
 FILUSDT            0.000193                   0.000193          -0.000754          -0.000947              0.003144              0.007218             9
 UNIUSDT            0.000340                   0.000340          -0.001351          -0.001691              0.011581              0.036344             9
 SOLUSDT            0.002170                   0.002170          -0.000458          -0.002628              0.009910              0.023592             9
 ARBUSDT            0.002586                   0.002586          -0.000050          -0.002636              0.005785              0.011315             9
  OPUSDT            0.007906                   0.007906          -0.002142          -0.010048              0.034568              0.125000             7
 BNBUSDT            0.008038                   0.008038           0.007562          -0.000476              0.024241              0.079186             9
ATOMUSDT            0.008628                   0.008628           0.008843           0.000215              0.008878              0.062172             9
 BCHUSDT            0.019488                   0.019488           0.016442          -0.003046              0.025858              0.091714             8
 LTCUSDT            0.021295                   0.021295           0.008149          -0.013146              0.032844              0.112016             9
```

### Category Delta

```text
    diff_category  delta_contribution  delta_weight_contribution  abs_delta_weight  rows
resized_same_side           -0.022297                  -0.022297          0.606635    43
        p15c_only           -0.011977                  -0.011977          0.527525    25
        p18d_only           -0.005601                  -0.005601          0.702129    46
        sign_flip           -0.003551                  -0.003551          1.448284    57
```

### Timestamp Delta

```text
                timestamp  delta_contribution  p18d_contribution  p15c_contribution  gross_delta_turnover_proxy
2024-11-08 08:00:00+00:00           -0.010412           0.000352           0.010764                    0.501240
2024-11-11 08:00:00+00:00           -0.014287           0.004157           0.018444                    0.388310
2024-11-14 08:00:00+00:00           -0.003885          -0.003407           0.000478                    0.529295
2024-11-17 08:00:00+00:00           -0.001402           0.000339           0.001741                    0.202270
2024-11-20 08:00:00+00:00           -0.006923          -0.000572           0.006351                    0.355175
2024-11-23 08:00:00+00:00            0.009939           0.009939           0.000000                    0.280052
2024-11-26 08:00:00+00:00           -0.009138          -0.004342           0.004796                    0.362358
2024-11-29 08:00:00+00:00           -0.005963          -0.004332           0.001631                    0.312272
2024-12-02 08:00:00+00:00           -0.001355          -0.005344          -0.003989                    0.353599
```

## P19e Allocator Projection Summary

```text
    window  rows  mean_raw_gross  mean_final_gross  projection_delta_contribution  raw_contribution  final_contribution  mean_abs_weight_change  max_abs_weight_change
validation   580            29.0         11.267720                      -0.067885          0.140606            0.072721                0.032644                  0.125
      test   180             9.0          1.564507                       0.312590         -0.315800           -0.003210                0.044309                  0.125
```

## Initial Read

- selected P18d test delta vs P15c: -4.3581%.
- guard grid test configs beating P15c: 0/36.
- selected factor spread inversions in test: 6/8.
- worst symbol delta: XRPUSDT -2.5303%.
- worst position-diff category: resized_same_side -2.2297%.
- allocator projection delta contribution on test: 31.2590%.
