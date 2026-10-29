# P16 Clean Candidate Package

## Decision

- Candidate: `p16_p15c_extreme_cluster_turnover_candidate`.
- Status: `candidate_promoted_for_next_review`.
- Official baseline remains `p9_clean_drop_k01_cap010` until explicit promotion.
- P16 freezes the P15c nested OOS result and fold-local selected extreme-cluster specs.

## Manifest
```json
{
  "name": "p16_p15c_extreme_cluster_turnover_candidate",
  "status": "candidate_promoted_for_next_review",
  "parent": "p9_clean_baseline_package",
  "source_stage": "p15c_extreme_cluster_turnover",
  "official_baseline_unchanged": "p9_clean_drop_k01_cap010",
  "candidate_metrics": {
    "total_return_6bps": 0.16623337593677867,
    "gross_return_before_cost": 0.2106183896572873,
    "sharpe_6bps": 2.426798400308199,
    "max_drawdown": -0.017877398356266117,
    "mean_turnover": 0.36240003426895684,
    "return_30bps": 0.0042773406871929875,
    "cluster_hhi_reduction_vs_p9": 0.05872151629820632,
    "max_cluster_share_reduction_vs_p9": 0.04175608060997671,
    "trigger_rate": 0.21511627906976744,
    "apply_rate": 0.11627906976744186
  },
  "delta_vs_p9": {
    "total_return": 0.010837948546240872,
    "gross_return_before_cost": 0.011382280341293516,
    "sharpe": 0.21563533906910948,
    "return_30bps": 0.008895229594446108
  },
  "delta_vs_p15b": {
    "total_return": 0.006055031179138526,
    "gross_return_before_cost": 0.006174557577156126,
    "sharpe": 0.1120686009220937,
    "return_30bps": 0.005582342916423899
  }
}
```

## Variant Summary
```text
 periods  total_return  gross_total_return_before_cost   sharpe  max_drawdown  mean_rank_ic  mean_top_bottom_label_spread  mean_period_return  mean_gross_period_return  positive_period_rate  positive_gross_period_rate  mean_turnover  total_turnover  cost_drag  mean_period_cost  target_apply_rate  mean_abs_beta_btc  mean_abs_beta_eth  mean_abs_beta_market  mean_symbol_concentration  fold_count  positive_fold_rate  gross_max_drawdown  mean_gross_exposure  mean_max_abs_weight  max_abs_weight  total_stopped_weight  stopped_period_rate  max_abs_beta_btc  max_abs_beta_eth  max_abs_beta_market  mean_optimizer_success  max_optimizer_success  mean_optimizer_exposure_error  max_optimizer_exposure_error  mean_optimizer_target_gross  max_optimizer_target_gross  mean_optimizer_actual_gross  max_optimizer_actual_gross  fil_total_contribution  mean_target_turnover  total_target_turnover  mean_turnover_saved  total_turnover_saved  mean_target_gross_exposure  total_target_gross_exposure  turnover_reduction  gross_exposure_retention  mean_short_guard_removed  total_return_20bps  sharpe_20bps  max_drawdown_20bps  total_return_30bps  sharpe_30bps  max_drawdown_30bps  q25_segment_sharpe  long_short_balance  short_abs_contribution_share  mean_max_cluster_gross_share  mean_cluster_hhi  mean_active_cluster_count  mean_abs_pc1_exposure  mean_abs_pc2_exposure  mean_abs_pc3_exposure  mean_baseline_max_cluster_gross_share  mean_baseline_cluster_hhi  mean_gross_exposure_retention  cluster_hhi_reduction_vs_baseline  max_cluster_share_reduction_vs_baseline                    variant  delta_vs_p9_total_return  delta_vs_p9_gross_total_return_before_cost  delta_vs_p9_sharpe  delta_vs_p9_max_drawdown  worst_5_fold_return_sum  median_fold_return  q25_fold_return  mean_cluster_cap_success  mean_cluster_cap_active
     172      0.166233                        0.210618 2.426798     -0.017877      0.027561                      0.004183            0.000903                  0.001120              0.569767                    0.598837       0.362400       62.332806   0.037400          0.000217                1.0       3.235685e-15       3.528800e-15          1.069056e-14                   0.018626          23            0.695652           -0.016528             0.269956             0.082173           0.125                   0.0                  0.0      1.219486e-13      1.205966e-13         3.247671e-13                0.970930                    1.0                   6.541230e-13                  3.316301e-11                     0.593070                    0.850596                     0.277005                    0.534351                0.031500              0.363106              62.454278             0.000872              0.150014                    0.270424                    46.512894            0.001945                  0.998269                  2.168605            0.068851      1.069046           -0.023922            0.004277      0.089618           -0.053281           -0.303840            0.867467                      0.433733                      0.612881          0.507237                   3.220930               0.047570               0.049646               0.056255                               0.639588                   0.538881                       0.998296                           0.058722                                 0.041756         p16_p15c_candidate                  0.010838                                    0.011382            0.215635             -1.123550e-10                -0.032251            0.005002        -0.000726                       NaN                      NaN
     172      0.160178                        0.204444 2.314730     -0.017990      0.027561                      0.003712            0.000873                  0.001091              0.558140                    0.593023       0.363275       62.483243   0.037490          0.000218                1.0       2.021704e-15       1.376326e-15          7.062173e-15                   0.018481          23            0.695652           -0.016528             0.267452             0.081767           0.125                   0.0                  0.0      1.038934e-13      8.151639e-14         3.247671e-13                0.959302                    1.0                   9.004152e-13                  5.650973e-11                     0.588114                    0.850596                     0.272220                    0.769368                0.033734              0.364431              62.682115             0.002823              0.485641                    0.270424                    46.512894            0.003173                  0.989012                  2.168605            0.063073      0.970619           -0.031516           -0.001305      0.002499           -0.063086           -0.704078            0.888903                      0.444452                      0.582999          0.487202                   3.197674               0.048214               0.047888               0.059539                               0.632200                   0.526217                       0.989709                           0.074142                                 0.077825 p15b_cluster_cap_reference                  0.004783                                    0.005208            0.103567             -1.128175e-04                -0.033210            0.004551        -0.001597                  0.959302                 0.348837
     172      0.155395                        0.199236 2.211163     -0.017877      0.027561                      0.003196            0.000849                  0.001066              0.552326                    0.581395       0.361314       62.146037   0.037288          0.000217                1.0       3.385727e-15       2.321514e-15          1.036098e-14                   0.019352          23            0.695652           -0.016528             0.270424             0.084603           0.125                   0.0                  0.0      1.950627e-13      7.492347e-14         4.433329e-13                1.000000                    1.0                   1.158352e-14                  4.433398e-13                     0.580226                    0.850596                     0.270424                    0.769368                0.031314              0.361314              62.146037             0.000000              0.000000                    0.270424                    46.512899            0.000000                  1.000000                  2.168605            0.059190      0.898562           -0.022960           -0.004618     -0.047381           -0.054684           -1.088102            0.917962                      0.458981                           NaN               NaN                        NaN                    NaN                    NaN                    NaN                                    NaN                        NaN                            NaN                                NaN                                      NaN               p9_reference                  0.000000                                    0.000000            0.000000              0.000000e+00                -0.039310            0.005933        -0.001773                       NaN                      NaN
```

## Reality Summary
```text
                                 check         value  pass
              net_delta_vs_p9_positive  1.083795e-02  True
            net_delta_vs_p15b_positive  6.055031e-03  True
           sharpe_delta_vs_p9_positive  2.156353e-01  True
         sharpe_delta_vs_p15b_positive  1.120686e-01  True
    return_20bps_positive_and_above_p9  9.661281e-03  True
    return_30bps_positive_and_above_p9  8.895230e-03  True
    turnover_not_higher_than_102pct_p9  1.003005e+00  True
        maxdd_not_worse_than_105pct_p9 -1.123550e-10  True
      cluster_hhi_reduction_vs_p9_3pct  5.872152e-02  True
max_cluster_share_reduction_vs_p9_3pct  4.175608e-02  True
   extreme_overlay_apply_rate_le_20pct  1.162791e-01  True
 extreme_overlay_trigger_rate_le_35pct  2.151163e-01  True
     worst5_fold_sum_not_worse_than_p9  7.058671e-03  True
                   bootstrap_vs_p9_q05 -2.742129e-04  True
                 bootstrap_vs_p15b_q05 -4.303532e-04  True
        positive_delta_fold_rate_vs_p9  3.913043e-01  True
      positive_delta_fold_rate_vs_p15b  3.913043e-01  True
             start_fold_positive_vs_p9  4.782609e-01  True
         symbol_dropout_positive_vs_p9  9.500000e-01  True
           strict_mean_abs_market_beta  1.069056e-14  True
            strict_max_abs_market_beta  3.247671e-13  True
```

## Cost Sensitivity
```text
                experiment  tau_days  one_way_cost_bps  total_return  max_drawdown  mean_period_return  positive_period_rate  cost_drag_sum                    variant  p9_total_return  p9_max_drawdown  p15b_total_return  p15b_max_drawdown  delta_vs_p9_total_return  delta_vs_p15b_total_return  delta_vs_p9_max_drawdown  delta_vs_p15b_max_drawdown
        p16_p15c_candidate       0.0               0.0      0.210618     -0.016528        1.120305e-03              0.598837       0.000000         p16_p15c_candidate         0.199236        -0.016528           0.204444          -0.016528                  0.011382                    0.006175             -8.809298e-11                3.330669e-16
        p16_p15c_candidate       0.0               4.0      0.180846     -0.017428        9.753451e-04              0.587209       0.024933         p16_p15c_candidate         0.169829        -0.017428           0.174751          -0.017428                  0.011017                    0.006095             -1.042735e-10               -4.440892e-16
        p16_p15c_candidate       0.0               6.0      0.166233     -0.017877        9.028651e-04              0.569767       0.037400         p16_p15c_candidate         0.155395        -0.017877           0.160178          -0.017990                  0.010838                    0.006055             -1.123550e-10                1.128174e-04
        p16_p15c_candidate       0.0               8.0      0.151800     -0.018327        8.303851e-04              0.546512       0.049866         p16_p15c_candidate         0.141138        -0.018327           0.145785          -0.019148                  0.010662                    0.006015             -1.204300e-10                8.212746e-04
        p16_p15c_candidate       0.0              10.0      0.137545     -0.018776        7.579051e-04              0.517442       0.062333         p16_p15c_candidate         0.127056        -0.018776           0.131569          -0.020305                  0.010489                    0.005976             -1.285004e-10                1.528620e-03
        p16_p15c_candidate       0.0              12.0      0.123464     -0.019225        6.854251e-04              0.511628       0.074799         p16_p15c_candidate         0.113146        -0.019225           0.117528          -0.021460                  0.010318                    0.005936             -1.365642e-10                2.234855e-03
        p16_p15c_candidate       0.0              15.0      0.102667     -0.019898        5.767051e-04              0.500000       0.093499         p16_p15c_candidate         0.092600        -0.019898           0.096791          -0.023829                  0.010067                    0.005877             -1.486495e-10                3.930562e-03
        p16_p15c_candidate       0.0              20.0      0.068851     -0.023922        3.955050e-04              0.494186       0.124666         p16_p15c_candidate         0.059190        -0.022960           0.063073          -0.031516                  0.009661                    0.005778             -9.614188e-04                7.593977e-03
        p16_p15c_candidate       0.0              30.0      0.004277     -0.053281        3.310501e-05              0.459302       0.186998         p16_p15c_candidate        -0.004618        -0.054684          -0.001305          -0.063086                  0.008895                    0.005582              1.402570e-03                9.804764e-03
p15b_cluster_cap_reference       0.0               0.0      0.204444     -0.016528        1.090761e-03              0.593023       0.000000 p15b_cluster_cap_reference         0.199236        -0.016528           0.204444          -0.016528                  0.005208                    0.000000             -8.809331e-11                0.000000e+00
p15b_cluster_cap_reference       0.0               4.0      0.174751     -0.017428        9.454510e-04              0.569767       0.024993 p15b_cluster_cap_reference         0.169829        -0.017428           0.174751          -0.017428                  0.004922                    0.000000             -1.042730e-10                0.000000e+00
p15b_cluster_cap_reference       0.0               6.0      0.160178     -0.017990        8.727960e-04              0.558140       0.037490 p15b_cluster_cap_reference         0.155395        -0.017877           0.160178          -0.017990                  0.004783                    0.000000             -1.128175e-04                0.000000e+00
p15b_cluster_cap_reference       0.0               8.0      0.145785     -0.019148        8.001411e-04              0.540698       0.049987 p15b_cluster_cap_reference         0.141138        -0.018327           0.145785          -0.019148                  0.004647                    0.000000             -8.212747e-04                0.000000e+00
p15b_cluster_cap_reference       0.0              10.0      0.131569     -0.020305        7.274862e-04              0.517442       0.062483 p15b_cluster_cap_reference         0.127056        -0.018776           0.131569          -0.020305                  0.004513                    0.000000             -1.528620e-03                0.000000e+00
p15b_cluster_cap_reference       0.0              12.0      0.117528     -0.021460        6.548312e-04              0.511628       0.074980 p15b_cluster_cap_reference         0.113146        -0.019225           0.117528          -0.021460                  0.004382                    0.000000             -2.234855e-03                0.000000e+00
p15b_cluster_cap_reference       0.0              15.0      0.096791     -0.023829        5.458488e-04              0.500000       0.093725 p15b_cluster_cap_reference         0.092600        -0.019898           0.096791          -0.023829                  0.004190                    0.000000             -3.930562e-03                0.000000e+00
p15b_cluster_cap_reference       0.0              20.0      0.063073     -0.031516        3.642115e-04              0.482558       0.124966 p15b_cluster_cap_reference         0.059190        -0.022960           0.063073          -0.031516                  0.003883                    0.000000             -8.555395e-03                0.000000e+00
p15b_cluster_cap_reference       0.0              30.0     -0.001305     -0.063086        9.368279e-07              0.430233       0.187450 p15b_cluster_cap_reference        -0.004618        -0.054684          -0.001305          -0.063086                  0.003313                    0.000000             -8.402194e-03                0.000000e+00
     p9_official_reference       0.0               0.0      0.199236     -0.016528        1.065868e-03              0.581395       0.000000      p9_official_reference         0.199236        -0.016528           0.204444          -0.016528                  0.000000                   -0.005208              0.000000e+00                8.809331e-11
     p9_official_reference       0.0               4.0      0.169829     -0.017428        9.213424e-04              0.569767       0.024858      p9_official_reference         0.169829        -0.017428           0.174751          -0.017428                  0.000000                   -0.004922              0.000000e+00                1.042730e-10
     p9_official_reference       0.0               6.0      0.155395     -0.017877        8.490796e-04              0.552326       0.037288      p9_official_reference         0.155395        -0.017877           0.160178          -0.017990                  0.000000                   -0.004783              0.000000e+00                1.128175e-04
     p9_official_reference       0.0               8.0      0.141138     -0.018327        7.768168e-04              0.534884       0.049717      p9_official_reference         0.141138        -0.018327           0.145785          -0.019148                  0.000000                   -0.004647              0.000000e+00                8.212747e-04
     p9_official_reference       0.0              10.0      0.127056     -0.018776        7.045539e-04              0.517442       0.062146      p9_official_reference         0.127056        -0.018776           0.131569          -0.020305                  0.000000                   -0.004513              0.000000e+00                1.528620e-03
     p9_official_reference       0.0              12.0      0.113146     -0.019225        6.322911e-04              0.500000       0.074575      p9_official_reference         0.113146        -0.019225           0.117528          -0.021460                  0.000000                   -0.004382              0.000000e+00                2.234855e-03
     p9_official_reference       0.0              15.0      0.092600     -0.019898        5.238968e-04              0.488372       0.093219      p9_official_reference         0.092600        -0.019898           0.096791          -0.023829                  0.000000                   -0.004190              0.000000e+00                3.930562e-03
     p9_official_reference       0.0              20.0      0.059190     -0.022960        3.432398e-04              0.482558       0.124292      p9_official_reference         0.059190        -0.022960           0.063073          -0.031516                  0.000000                   -0.003883              0.000000e+00                8.555395e-03
     p9_official_reference       0.0              30.0     -0.004618     -0.054684       -1.807441e-05              0.447674       0.186438      p9_official_reference        -0.004618        -0.054684          -0.001305          -0.063086                  0.000000                   -0.003313              0.000000e+00                8.402194e-03
```

## Cluster Delta Vs P9
```text
                 metric  p15b_mean  p9_mean     delta  reduction_pct  periods
            cluster_hhi   0.507237 0.538881 -0.031644       0.058722      172
max_cluster_gross_share   0.612881 0.639588 -0.026707       0.041756      172
       abs_pc1_exposure   0.047570 0.044970  0.002600      -0.057817      172
       abs_pc2_exposure   0.049646 0.047508  0.002137      -0.044990      172
         gross_exposure   0.269956 0.270424 -0.000468       0.001731      172
```

## Cluster Delta Vs P15b
```text
                 metric  p16_p15c_candidate_mean  p15b_cluster_cap_reference_mean     delta  reduction_pct  periods
            cluster_hhi                 0.507237                         0.487202  0.020035      -0.041123      172
max_cluster_gross_share                 0.612881                         0.582999  0.029882      -0.051256      172
       abs_pc1_exposure                 0.047570                         0.048214 -0.000644       0.013351      172
       abs_pc2_exposure                 0.049646                         0.047888  0.001757      -0.036695      172
         gross_exposure                 0.269956                         0.267452  0.002503      -0.009360      172
```

## Activation Summary
```text
                        metric      value
                       periods 172.000000
             triggered_periods  37.000000
               applied_periods  20.000000
turnover_guard_skipped_periods  12.000000
                  trigger_rate   0.215116
                    apply_rate   0.116279
      turnover_guard_skip_rate   0.069767
       mean_turnover_add_vs_p9   0.001152
 mean_gross_exposure_retention   0.998296
```

## Frozen Fold-Local Specs
```text
 fold              selected_extreme_spec  fallback_to_p9_sizing  lookback_hours  cluster_k  trigger_share  cluster_cap  max_turnover_add  validation_selection_score  validation_total_return  validation_sharpe  validation_cluster_hhi_reduction_vs_p9
    1 p15c_lb1440_k3_trig75_cap60_turn02                  False            1440          3           0.75         0.60              0.02                    1.336023                 0.027589           3.192100                                0.066925
    2  p15c_lb336_k3_trig75_cap60_turn00                  False             336          3           0.75         0.60              0.00                    2.926396                 0.080977           5.108784                                0.080896
    3  p15c_lb336_k4_trig75_cap60_turn02                  False             336          4           0.75         0.60              0.02                    0.379117                 0.030073           3.361989                                0.028814
    4  p15c_lb336_k3_trig80_cap65_turn02                  False             336          3           0.80         0.65              0.02                    1.491950                 0.036953           3.878243                                0.081248
    5  p15c_lb720_k3_trig80_cap65_turn02                  False             720          3           0.80         0.65              0.02                    2.287766                 0.054515           2.633766                                0.089844
    6  p15c_lb720_k3_trig80_cap65_turn02                  False             720          3           0.80         0.65              0.02                    7.944816                 0.165584           6.216041                                0.069860
    7  p15c_lb720_k4_trig75_cap60_turn02                  False             720          4           0.75         0.60              0.02                    1.655648                 0.062523           6.569560                                0.073841
    8  p15c_lb720_k3_trig80_cap65_turn00                  False             720          3           0.80         0.65              0.00                    4.810954                 0.090507           6.886970                                0.068102
    9 p15c_lb1440_k4_trig75_cap60_turn00                  False            1440          4           0.75         0.60              0.00                    1.552553                 0.085251           4.859826                                0.052895
   10 p15c_lb1440_k3_trig75_cap60_turn00                  False            1440          3           0.75         0.60              0.00                    2.708358                 0.082749           5.125124                                0.053382
   11  p15c_lb336_k3_trig75_cap60_turn00                  False             336          3           0.75         0.60              0.00                    1.201014                 0.029295           5.131167                                0.048572
   12      __P9_NO_EXTREME_CLUSTER_CAP__                   True             336          3            NaN          NaN               NaN                    0.074052                 0.074052          10.493060                                0.000000
   13 p15c_lb1440_k4_trig80_cap65_turn02                  False            1440          4           0.80         0.65              0.02                    1.210685                 0.054745           9.182414                                0.105253
   14 p15c_lb1440_k3_trig75_cap60_turn02                  False            1440          3           0.75         0.60              0.02                    1.976640                 0.041166           5.874940                                0.092454
   15  p15c_lb720_k3_trig85_cap70_turn00                  False             720          3           0.85         0.70              0.00                    0.853290                 0.040844           6.793451                                0.069999
   16      __P9_NO_EXTREME_CLUSTER_CAP__                   True             336          3            NaN          NaN               NaN                    0.040603                 0.040603           3.843028                                0.000000
   17  p15c_lb336_k4_trig75_cap60_turn02                  False             336          4           0.75         0.60              0.02                    4.364372                 0.038365           9.266159                                0.097896
   18 p15c_lb1440_k3_trig80_cap65_turn02                  False            1440          3           0.80         0.65              0.02                    1.214407                 0.123896           3.229282                                0.090384
   19  p15c_lb336_k3_trig85_cap70_turn02                  False             336          3           0.85         0.70              0.02                    3.864577                 0.135893           4.426325                                0.086540
   20 p15c_lb1440_k4_trig85_cap70_turn00                  False            1440          4           0.85         0.70              0.00                    0.686400                 0.021104           6.046278                                0.046169
   21 p15c_lb1440_k4_trig75_cap60_turn00                  False            1440          4           0.75         0.60              0.00                   -0.024065                 0.040010           3.788553                                0.017791
   22  p15c_lb336_k3_trig85_cap70_turn02                  False             336          3           0.85         0.70              0.02                    2.441106                 0.059145           4.783492                                0.055261
   23  p15c_lb336_k3_trig85_cap70_turn00                  False             336          3           0.85         0.70              0.00                    1.726624                 0.075191           4.956916                                0.053523
```

## Fold Delta Vs P9
```text
 fold  p16_p15c_candidate_total_return  p16_p15c_candidate_gross_total_return_before_cost  p16_p15c_candidate_sharpe  p16_p15c_candidate_max_drawdown  p16_p15c_candidate_mean_turnover p16_p15c_candidate_formula_name  p9_total_return  p9_gross_total_return_before_cost  p9_sharpe  p9_max_drawdown  p9_mean_turnover p9_formula_name  delta_total_return  delta_gross_total_return_before_cost  delta_sharpe  delta_max_drawdown  delta_mean_turnover
    1                        -0.003161                                          -0.002101                  -3.106298                        -0.003115                          0.221464                     __P4_ONLY__        -0.002631                          -0.001569  -2.453167        -0.003115          0.221753     __P4_ONLY__       -5.301933e-04                         -5.320634e-04 -6.531314e-01       -6.095117e-09        -2.887058e-04
    2                        -0.004009                                          -0.002431                  -1.500142                        -0.006164                          0.292907                    p7_agpt_0008        -0.009313                          -0.007747  -3.496208        -0.008687          0.291896    p7_agpt_0008        5.304112e-03                          5.316491e-03  1.996065e+00        2.522467e-03         1.010840e-03
    3                         0.004516                                           0.006202                   2.341837                        -0.007213                          0.349771                    p7_agpt_0083         0.005933                           0.007625   2.949018        -0.007213          0.350305    p7_agpt_0083       -1.417636e-03                         -1.422335e-03 -6.071811e-01        3.961379e-09        -5.336411e-04
    4                         0.005002                                           0.006810                   3.261364                        -0.001088                          0.428198                    p7_agpt_0150         0.003533                           0.005292   1.971834        -0.001088          0.417092    p7_agpt_0150        1.469818e-03                          1.518912e-03  1.289530e+00       -4.262765e-08         1.110613e-02
    5                         0.055703                                           0.057597                  19.144517                         0.000000                          0.375746                     __P4_ONLY__         0.048604                           0.050467  13.538245         0.000000          0.372069     __P4_ONLY__        7.099236e-03                          7.129130e-03  5.606273e+00        0.000000e+00         3.677396e-03
    6                         0.020308                                           0.022428                   3.642384                        -0.001304                          0.433759                     __P4_ONLY__         0.024371                           0.026508   3.929953        -0.001304          0.435662     __P4_ONLY__       -4.063112e-03                         -4.079707e-03 -2.875684e-01       -5.720346e-10        -1.903762e-03
    7                         0.004830                                           0.005795                   1.619187                        -0.010953                          0.267355                    p7_agpt_0047         0.004830                           0.005795   1.619179        -0.010953          0.267354    p7_agpt_0047        2.564625e-08                          2.629197e-08  7.853400e-06        3.039789e-09         1.738375e-07
    8                        -0.001095                                           0.000993                  -0.415247                        -0.008277                          0.497426                    p7_agpt_0102        -0.001095                           0.000993  -0.415246        -0.008277          0.497426    p7_agpt_0102       -2.985833e-09                         -3.177208e-09 -1.123741e-06       -3.333750e-09        -4.389805e-08
    9                        -0.000357                                           0.001172                  -0.192433                        -0.006254                          0.363916                     __P4_ONLY__        -0.003048                          -0.001518  -1.760959        -0.006755          0.365099     __P4_ONLY__        2.690953e-03                          2.689658e-03  1.568526e+00        5.013981e-04        -1.182749e-03
   10                         0.009076                                           0.010952                   4.821272                        -0.002849                          0.442914                    p7_agpt_0102         0.009076                           0.010952   4.821279        -0.002849          0.442914    p7_agpt_0102       -8.197772e-09                         -8.060637e-09 -6.929011e-06       -8.801584e-09         3.562711e-08
   11                         0.015307                                           0.017285                   6.959346                        -0.002078                          0.406230                    p7_agpt_0115         0.015307                           0.017285   6.959344        -0.002078          0.406230    p7_agpt_0115        4.053743e-09                          4.225002e-09  2.498220e-06       -1.611003e-13         3.373811e-08
   12                         0.010960                                           0.013395                   3.895788                        -0.007020                          0.573910                    p7_agpt_0006         0.010960                           0.013395   3.895789        -0.007020          0.573910    p7_agpt_0006       -1.062823e-08                         -1.042984e-08 -1.425376e-06        1.028692e-09         5.146289e-08
   13                        -0.017625                                          -0.016179                 -15.738613                        -0.017877                          0.407129                    p7_agpt_0019        -0.017625                          -0.016179 -15.738610        -0.017877          0.407129    p7_agpt_0019       -3.468856e-10                         -2.869331e-10 -2.344274e-06       -1.123548e-10         1.699109e-08
   14                         0.011119                                           0.012078                   4.539003                        -0.004069                          0.264027                     __P4_ONLY__         0.012433                           0.013367   4.691574        -0.004218          0.256811     __P4_ONLY__       -1.314385e-03                         -1.289008e-03 -1.525709e-01        1.486859e-04         7.216002e-03
   15                        -0.005005                                          -0.003228                  -1.773023                        -0.013485                          0.424857                    p7_agpt_0034        -0.006692                          -0.004939  -2.413229        -0.013630          0.419721    p7_agpt_0034        1.687082e-03                          1.711081e-03  6.402061e-01        1.448610e-04         5.135198e-03
   16                         0.001116                                           0.002177                   0.847381                        -0.004371                          0.196142                    p7_agpt_0021         0.001116                           0.002177   0.847379        -0.004371          0.196142    p7_agpt_0021        2.462582e-09                          2.075999e-09  2.264161e-06        3.217123e-09        -7.187225e-08
   17                         0.005879                                           0.007407                   4.391011                        -0.001765                          0.361939                    p7_agpt_0111         0.006433                           0.007956   4.703910        -0.001765          0.360352    p7_agpt_0111       -5.542126e-04                         -5.482579e-04 -3.128992e-01        9.137472e-10         1.586995e-03
   18                         0.012176                                           0.014150                   2.926607                        -0.005612                          0.361529                    p7_agpt_0129         0.013027                           0.014997   3.138723        -0.005612          0.360391    p7_agpt_0129       -8.509573e-04                         -8.462588e-04 -2.121158e-01        3.706363e-10         1.138545e-03
   19                         0.006823                                           0.009364                   1.917997                        -0.004229                          0.467916                     __P4_ONLY__         0.006823                           0.009364   1.918002        -0.004229          0.467915     __P4_ONLY__       -2.699435e-08                         -2.687044e-08 -5.047387e-06       -4.800299e-09         3.317734e-08
   20                        -0.002451                                          -0.001643                  -3.035300                        -0.001772                          0.224782                    p7_agpt_0111        -0.002451                          -0.001643  -3.035296        -0.001772          0.224782    p7_agpt_0111       -3.481772e-09                         -3.556447e-09 -3.998331e-06       -9.993500e-12        -2.002802e-08
   21                         0.001205                                           0.002689                   1.419808                        -0.002377                          0.352702                    p7_agpt_0111         0.001205                           0.002689   1.419774        -0.002377          0.352702    p7_agpt_0111        3.199973e-08                          3.019921e-08  3.424746e-05        2.995785e-10        -4.385384e-07
   22                         0.007420                                           0.009340                   2.417824                        -0.004907                          0.352779                     __P4_ONLY__         0.007420                           0.009340   2.417826        -0.004907          0.352779     __P4_ONLY__       -7.561436e-09                         -7.146945e-09 -2.244560e-06        1.140815e-09         7.854618e-08
   23                         0.018553                                           0.019708                  10.524021                         0.000000                          0.270685                     __P4_ONLY__         0.018553                           0.019708  10.524021         0.000000          0.270685     __P4_ONLY__       -4.153924e-09                         -4.205408e-09  5.082379e-07        0.000000e+00        -1.137691e-08
```

## Fold Delta Vs P15b
```text
 fold  p16_p15c_candidate_total_return  p16_p15c_candidate_gross_total_return_before_cost  p16_p15c_candidate_sharpe  p16_p15c_candidate_max_drawdown  p16_p15c_candidate_mean_turnover p16_p15c_candidate_formula_name  p15b_total_return  p15b_gross_total_return_before_cost  p15b_sharpe  p15b_max_drawdown  p15b_mean_turnover p15b_formula_name  delta_total_return  delta_gross_total_return_before_cost  delta_sharpe  delta_max_drawdown  delta_mean_turnover
    1                        -0.003161                                          -0.002101                  -3.106298                        -0.003115                          0.221464                     __P4_ONLY__          -0.002631                            -0.001569    -2.453170          -0.003115            0.221753       __P4_ONLY__           -0.000530                             -0.000532     -0.653128       -1.001803e-16            -0.000289
    2                        -0.004009                                          -0.002431                  -1.500142                        -0.006164                          0.292907                    p7_agpt_0008          -0.002971                            -0.001429    -1.211860          -0.005203            0.286000      p7_agpt_0008           -0.001038                             -0.001002     -0.288282       -9.611310e-04             0.006907
    3                         0.004516                                           0.006202                   2.341837                        -0.007213                          0.349771                    p7_agpt_0083           0.008898                             0.010720     6.289747          -0.003898            0.376216      p7_agpt_0083           -0.004383                             -0.004518     -3.947910       -3.314960e-03            -0.026445
    4                         0.005002                                           0.006810                   3.261364                        -0.001088                          0.428198                    p7_agpt_0150           0.002774                             0.004538     1.592155          -0.000960            0.418617      p7_agpt_0150            0.002228                              0.002272      1.669209       -1.274857e-04             0.009582
    5                         0.055703                                           0.057597                  19.144517                         0.000000                          0.375746                     __P4_ONLY__           0.053766                             0.055695    15.044859           0.000000            0.383286       __P4_ONLY__            0.001937                              0.001901      4.099658        0.000000e+00            -0.007540
    6                         0.020308                                           0.022428                   3.642384                        -0.001304                          0.433759                     __P4_ONLY__           0.024371                             0.026508     3.929944          -0.001304            0.435663       __P4_ONLY__           -0.004063                             -0.004080     -0.287560        0.000000e+00            -0.001904
    7                         0.004830                                           0.005795                   1.619187                        -0.010953                          0.267355                    p7_agpt_0047           0.004551                             0.005528     1.534761          -0.010959            0.270486      p7_agpt_0047            0.000278                              0.000267      0.084426        5.978240e-06            -0.003131
    8                        -0.001095                                           0.000993                  -0.415247                        -0.008277                          0.497426                    p7_agpt_0102          -0.000563                             0.001492    -0.196296          -0.008200            0.489454      p7_agpt_0102           -0.000532                             -0.000499     -0.218950       -7.664580e-05             0.007972
    9                        -0.000357                                           0.001172                  -0.192433                        -0.006254                          0.363916                     __P4_ONLY__          -0.003127                            -0.001652    -2.090717          -0.006735            0.351888       __P4_ONLY__            0.002770                              0.002824      1.898284        4.810277e-04             0.012029
   10                         0.009076                                           0.010952                   4.821272                        -0.002849                          0.442914                    p7_agpt_0102           0.009076                             0.010952     4.821272          -0.002849            0.442914      p7_agpt_0102            0.000000                              0.000000      0.000000        0.000000e+00             0.000000
   11                         0.015307                                           0.017285                   6.959346                        -0.002078                          0.406230                    p7_agpt_0115           0.015307                             0.017285     6.959346          -0.002078            0.406230      p7_agpt_0115            0.000000                              0.000000      0.000000        0.000000e+00             0.000000
   12                         0.010960                                           0.013395                   3.895788                        -0.007020                          0.573910                    p7_agpt_0006           0.010960                             0.013395     3.895788          -0.007020            0.573910      p7_agpt_0006            0.000000                              0.000000      0.000000        0.000000e+00             0.000000
   13                        -0.017625                                          -0.016179                 -15.738613                        -0.017877                          0.407129                    p7_agpt_0019          -0.017625                            -0.016179   -15.738613          -0.017877            0.407129      p7_agpt_0019            0.000000                              0.000000      0.000000        0.000000e+00             0.000000
   14                         0.011119                                           0.012078                   4.539003                        -0.004069                          0.264027                     __P4_ONLY__           0.008271                             0.009225     4.169022          -0.003963            0.263230       __P4_ONLY__            0.002848                              0.002853      0.369981       -1.054754e-04             0.000797
   15                        -0.005005                                          -0.003228                  -1.773023                        -0.013485                          0.424857                    p7_agpt_0034          -0.006692                            -0.004939    -2.413229          -0.013630            0.419721      p7_agpt_0034            0.001687                              0.001711      0.640206        1.448652e-04             0.005135
   16                         0.001116                                           0.002177                   0.847381                        -0.004371                          0.196142                    p7_agpt_0021           0.001116                             0.002177     0.847381          -0.004371            0.196142      p7_agpt_0021            0.000000                              0.000000      0.000000        0.000000e+00             0.000000
   17                         0.005879                                           0.007407                   4.391011                        -0.001765                          0.361939                    p7_agpt_0111           0.001664                             0.003187     1.988186          -0.002085            0.361904      p7_agpt_0111            0.004215                              0.004221      2.402825        3.203773e-04             0.000035
   18                         0.012176                                           0.014150                   2.926607                        -0.005612                          0.361529                    p7_agpt_0129           0.013027                             0.014997     3.138723          -0.005612            0.360391      p7_agpt_0129           -0.000851                             -0.000846     -0.212116        0.000000e+00             0.001138
   19                         0.006823                                           0.009364                   1.917997                        -0.004229                          0.467916                     __P4_ONLY__           0.006823                             0.009364     1.917997          -0.004229            0.467916       __P4_ONLY__            0.000000                              0.000000      0.000000        0.000000e+00             0.000000
   20                        -0.002451                                          -0.001643                  -3.035300                        -0.001772                          0.224782                    p7_agpt_0111          -0.002795                            -0.001968    -2.979043          -0.002080            0.230359      p7_agpt_0111            0.000345                              0.000325     -0.056256        3.077352e-04            -0.005578
   21                         0.001205                                           0.002689                   1.419808                        -0.002377                          0.352702                    p7_agpt_0111           0.000896                             0.002457     1.116090          -0.002668            0.371163      p7_agpt_0111            0.000309                              0.000232      0.303718        2.906884e-04            -0.018461
   22                         0.007420                                           0.009340                   2.417824                        -0.004907                          0.352779                     __P4_ONLY__           0.007420                             0.009340     2.417824          -0.004907            0.352779       __P4_ONLY__            0.000000                              0.000000      0.000000        0.000000e+00             0.000000
   23                         0.018553                                           0.019708                  10.524021                         0.000000                          0.270685                     __P4_ONLY__           0.018553                             0.019708    10.524021           0.000000            0.270685       __P4_ONLY__            0.000000                              0.000000      0.000000        0.000000e+00             0.000000
```

## Bootstrap
```text
                       metric  fold_count  observed_mean_delta  bootstrap_q05  bootstrap_q50  bootstrap_q95  positive_delta_fold_rate  sign_flip_p_value_mean_le_0
  p16_p15c_candidate_minus_p9          23             0.000414      -0.000274       0.000401       0.001205                  0.391304                       0.1935
p16_p15c_candidate_minus_p15b          23             0.000227      -0.000430       0.000230       0.000869                  0.391304                       0.2941
```

## Start Fold Sensitivity Vs P9
```text
 start_fold  p16_p15c_candidate       p9  delta_total_return
          1            0.166233 0.155395        1.083795e-02
          2            0.169932 0.158444        1.148847e-02
          3            0.174641 0.169334        5.307468e-03
          4            0.169361 0.162437        6.924113e-03
          5            0.163540 0.158345        5.195566e-03
          6            0.102147 0.104654       -2.506988e-03
          7            0.080210 0.078373        1.837252e-03
          8            0.075018 0.073190        1.828394e-03
          9            0.076197 0.074366        1.830401e-03
         10            0.076581 0.077651       -1.069890e-03
         11            0.066898 0.067958       -1.060258e-03
         12            0.050813 0.051857       -1.044277e-03
         13            0.039421 0.040454       -1.032946e-03
         14            0.058069 0.059121       -1.051477e-03
         15            0.046434 0.046114        3.199616e-04
         16            0.051698 0.053162       -1.464137e-03
         17            0.050526 0.051988       -1.462508e-03
         18            0.044386 0.045264       -8.780474e-04
         19            0.031823 0.031823       -1.024036e-08
         20            0.024830 0.024830        1.730618e-08
         21            0.027348 0.027348        2.093447e-08
         22            0.026111 0.026111       -1.188647e-08
         23            0.018553 0.018553       -4.153924e-09
```

## Start Fold Sensitivity Vs P15b
```text
 start_fold     p15b  p16_p15c_candidate  delta_total_return
          1 0.160178            0.166233            0.006055
          2 0.163239            0.169932            0.006693
          3 0.166706            0.174641            0.007936
          4 0.156416            0.169361            0.012945
          5 0.153216            0.163540            0.010324
          6 0.094375            0.102147            0.007772
          7 0.068339            0.080210            0.011872
          8 0.063498            0.075018            0.011520
          9 0.064098            0.076197            0.012099
         10 0.067436            0.076581            0.009146
         11 0.057835            0.066898            0.009063
         12 0.041886            0.050813            0.008927
         13 0.030591            0.039421            0.008830
         14 0.049081            0.058069            0.008988
         15 0.040476            0.046434            0.005958
         16 0.047486            0.051698            0.004212
         17 0.046318            0.050526            0.004208
         18 0.044580            0.044386           -0.000195
         19 0.031148            0.031823            0.000675
         20 0.024160            0.024830            0.000670
         21 0.027031            0.027348            0.000317
         22 0.026111            0.026111            0.000000
         23 0.018553            0.018553            0.000000
```

## Symbol Dropout Vs P9
```text
symbol_dropped  p16_p15c_candidate       p9  delta_total_return
      LINKUSDT            0.096629 0.086995            0.009634
       XRPUSDT            0.126071 0.110682            0.015389
       ARBUSDT            0.126241 0.114600            0.011641
       APTUSDT            0.129362 0.119059            0.010303
       FILUSDT            0.129965 0.119649            0.010316
       SOLUSDT            0.133592 0.121926            0.011666
      NEARUSDT            0.135240 0.123406            0.011834
       LTCUSDT            0.141482 0.140655            0.000827
       BTCUSDT            0.150228 0.140537            0.009691
       ETCUSDT            0.151016 0.143272            0.007744
       BNBUSDT            0.153245 0.150776            0.002469
       BCHUSDT            0.153693 0.142277            0.011416
       ETHUSDT            0.157780 0.156956            0.000824
       DOTUSDT            0.162874 0.163455           -0.000581
      DOGEUSDT            0.166094 0.158161            0.007933
       UNIUSDT            0.187006 0.178318            0.008688
      ATOMUSDT            0.195099 0.182709            0.012390
      AVAXUSDT            0.196193 0.163185            0.033007
        OPUSDT            0.204172 0.189774            0.014399
       ADAUSDT            0.207607 0.192264            0.015343
```

## Symbol Dropout Vs P15b
```text
symbol_dropped     p15b  p16_p15c_candidate  delta_total_return
      LINKUSDT 0.092132            0.096629            0.004497
       XRPUSDT 0.112199            0.126071            0.013872
       ARBUSDT 0.123838            0.126241            0.002403
       APTUSDT 0.124076            0.129362            0.005286
       FILUSDT 0.121610            0.129965            0.008356
       SOLUSDT 0.133617            0.133592           -0.000024
      NEARUSDT 0.125410            0.135240            0.009830
       LTCUSDT 0.157493            0.141482           -0.016011
       BTCUSDT 0.151796            0.150228           -0.001568
       ETCUSDT 0.149462            0.151016            0.001554
       BNBUSDT 0.148809            0.153245            0.004436
       BCHUSDT 0.154107            0.153693           -0.000415
       ETHUSDT 0.155018            0.157780            0.002762
       DOTUSDT 0.138994            0.162874            0.023880
      DOGEUSDT 0.174205            0.166094           -0.008112
       UNIUSDT 0.181963            0.187006            0.005043
      ATOMUSDT 0.194481            0.195099            0.000617
      AVAXUSDT 0.168755            0.196193            0.027438
        OPUSDT 0.192681            0.204172            0.011491
       ADAUSDT 0.188075            0.207607            0.019532
```

## Side Contribution
```text
 side  gross_contribution_sum  mean_contribution  positive_position_rate  mean_abs_weight  max_abs_weight  position_count  symbol_count  abs_contribution_share                    variant
 long                0.109115           0.000100                0.462454         0.021260           0.125            1092            20                0.566267         p16_p15c_candidate
short                0.083577           0.000046                0.513691         0.012714           0.125            1826            20                0.433733         p16_p15c_candidate
 long                0.104227           0.000099                0.460302         0.021740           0.125            1058            20                0.555548 p15b_cluster_cap_reference
short                0.083384           0.000047                0.515376         0.013098           0.125            1756            20                0.444452 p15b_cluster_cap_reference
 long                0.099185           0.000087                0.459790         0.020329           0.125            1144            20                0.541019      p9_official_reference
short                0.084145           0.000044                0.513795         0.012106           0.125            1921            20                0.458981      p9_official_reference
```

## Asset Contribution
```text
  symbol  gross_contribution_sum  long_contribution_sum  short_contribution_sum  mean_contribution  positive_position_rate  mean_abs_weight  max_abs_weight  position_count  abs_contribution_share                    variant
LINKUSDT                0.060793               0.049598                0.011195       4.536776e-04                0.514925         0.019363        0.125000             134                0.131208         p16_p15c_candidate
 XRPUSDT                0.034711               0.004230                0.030480       2.314047e-04                0.513333         0.019383        0.125000             150                0.074916         p16_p15c_candidate
 ARBUSDT                0.034686              -0.009556                0.044243       2.297094e-04                0.476821         0.017845        0.125000             151                0.074863         p16_p15c_candidate
 APTUSDT                0.032218              -0.007974                0.040192       2.065230e-04                0.474359         0.011195        0.125000             156                0.069535         p16_p15c_candidate
 FILUSDT                0.031500               0.016350                0.015150       2.172430e-04                0.475862         0.009685        0.110496             145                0.067986         p16_p15c_candidate
 SOLUSDT                0.028235               0.027709                0.000526       2.046019e-04                0.514493         0.016795        0.122560             138                0.060939         p16_p15c_candidate
NEARUSDT                0.026820              -0.005440                0.032260       1.697473e-04                0.518987         0.012522        0.125000             158                0.057885         p16_p15c_candidate
 LTCUSDT                0.021234               0.011856                0.009378       1.378844e-04                0.454545         0.017633        0.125000             154                0.045829         p16_p15c_candidate
 BTCUSDT                0.013765               0.017800               -0.004035       1.042827e-04                0.545455         0.026470        0.125000             132                0.029709         p16_p15c_candidate
 ETCUSDT                0.012857               0.024174               -0.011317       8.687010e-05                0.432432         0.012767        0.125000             148                0.027749         p16_p15c_candidate
 BNBUSDT                0.010827               0.020253               -0.009426       7.571261e-05                0.496503         0.026399        0.125000             143                0.023368         p16_p15c_candidate
 BCHUSDT                0.010644               0.018137               -0.007493       6.911681e-05                0.532468         0.013478        0.125000             154                0.022973         p16_p15c_candidate
 ETHUSDT                0.006986               0.020884               -0.013898       5.252905e-05                0.511278         0.025253        0.125000             133                0.015079         p16_p15c_candidate
 DOTUSDT                0.002735              -0.007085                0.009821       1.912914e-05                0.489510         0.014496        0.125000             143                0.005904         p16_p15c_candidate
DOGEUSDT               -0.000080               0.010551               -0.010631      -5.641259e-07                0.531915         0.015935        0.125000             141                0.000172         p16_p15c_candidate
 UNIUSDT               -0.017852              -0.000811               -0.017041      -1.182271e-04                0.509934         0.013338        0.125000             151                0.038530         p16_p15c_candidate
ATOMUSDT               -0.024709              -0.026625                0.001916      -1.727894e-04                0.468531         0.011317        0.125000             143                0.053329         p16_p15c_candidate
AVAXUSDT               -0.025407              -0.014079               -0.011328      -1.587937e-04                0.506250         0.017332        0.125000             160                0.054836         p16_p15c_candidate
  OPUSDT               -0.032067              -0.016956               -0.015111      -2.181439e-04                0.496599         0.007114        0.125000             147                0.069210         p16_p15c_candidate
 ADAUSDT               -0.035204              -0.023901               -0.011304      -2.569653e-04                0.430657         0.012478        0.115677             137                0.075981         p16_p15c_candidate
LINKUSDT                0.059594               0.048573                0.011021       4.655783e-04                0.515625         0.020883        0.125000             128                0.139619 p15b_cluster_cap_reference
 XRPUSDT                0.042138               0.013224                0.028913       2.946702e-04                0.524476         0.018781        0.125000             143                0.098722 p15b_cluster_cap_reference
 FILUSDT                0.033734               0.019532                0.014202       2.409580e-04                0.464286         0.009125        0.091368             140                0.079034 p15b_cluster_cap_reference
 ARBUSDT                0.031712              -0.009133                0.040845       2.217603e-04                0.482517         0.018675        0.125000             143                0.074296 p15b_cluster_cap_reference
 APTUSDT                0.031666              -0.009557                0.041223       2.056227e-04                0.474026         0.011221        0.125000             154                0.074188 p15b_cluster_cap_reference
NEARUSDT                0.030409              -0.000676                0.031086       2.027296e-04                0.513333         0.012659        0.125000             150                0.071244 p15b_cluster_cap_reference
 SOLUSDT                0.023092               0.030480               -0.007388       1.710521e-04                0.511111         0.016548        0.102447             135                0.054101 p15b_cluster_cap_reference
 DOTUSDT                0.018425              -0.010157                0.028582       1.306741e-04                0.489362         0.014958        0.125000             141                0.043167 p15b_cluster_cap_reference
 BNBUSDT                0.009448               0.021254               -0.011806       6.998738e-05                0.488889         0.027691        0.125000             135                0.022136 p15b_cluster_cap_reference
 ETCUSDT                0.008965               0.022340               -0.013374       6.269429e-05                0.419580         0.013888        0.125000             143                0.021004 p15b_cluster_cap_reference
 BTCUSDT                0.007132               0.009506               -0.002374       5.798031e-05                0.560976         0.028182        0.125000             123                0.016708 p15b_cluster_cap_reference
 BCHUSDT                0.005104               0.013617               -0.008513       3.271500e-05                0.532051         0.013985        0.125000             156                0.011957 p15b_cluster_cap_reference
 ETHUSDT                0.004070               0.021560               -0.017490       3.131064e-05                0.530769         0.025034        0.125000             130                0.009536 p15b_cluster_cap_reference
 LTCUSDT                0.001732               0.004197               -0.002465       1.194739e-05                0.441379         0.019628        0.125000             145                0.004059 p15b_cluster_cap_reference
AVAXUSDT               -0.007417              -0.010643                0.003225      -4.912234e-05                0.529801         0.018049        0.125000             151                0.017378 p15b_cluster_cap_reference
DOGEUSDT               -0.012243               0.000372               -0.012615      -8.936592e-05                0.503650         0.015020        0.125000             137                0.028684 p15b_cluster_cap_reference
 UNIUSDT               -0.018808              -0.001030               -0.017777      -1.315220e-04                0.524476         0.014827        0.125000             143                0.044063 p15b_cluster_cap_reference
 ADAUSDT               -0.024150              -0.016402               -0.007747      -1.872068e-04                0.426357         0.013786        0.122932             129                0.056579 p15b_cluster_cap_reference
  OPUSDT               -0.027615              -0.017290               -0.010325      -1.931108e-04                0.475524         0.007324        0.125000             143                0.064697 p15b_cluster_cap_reference
ATOMUSDT               -0.029378              -0.025540               -0.003838      -2.026065e-04                0.489655         0.010504        0.125000             145                0.068828 p15b_cluster_cap_reference
LINKUSDT                0.060216               0.049359                0.010857       4.210915e-04                0.510490         0.019081        0.125000             143                0.140406      p9_official_reference
 XRPUSDT                0.039244               0.008001                0.031243       2.564946e-04                0.509804         0.018362        0.125000             153                0.091505      p9_official_reference
 ARBUSDT                0.035746              -0.007629                0.043375       2.262424e-04                0.468354         0.017478        0.125000             158                0.083350      p9_official_reference
 APTUSDT                0.032081              -0.008253                0.040334       1.980286e-04                0.469136         0.010321        0.125000             162                0.074803      p9_official_reference
 FILUSDT                0.031314               0.017111                0.014203       2.033390e-04                0.467532         0.009205        0.110496             154                0.073016      p9_official_reference
 SOLUSDT                0.029283               0.029963               -0.000680       2.047740e-04                0.517483         0.015852        0.122560             143                0.068279      p9_official_reference
NEARUSDT                0.027915              -0.005817                0.033732       1.702120e-04                0.524390         0.011134        0.125000             164                0.065089      p9_official_reference
 BTCUSDT                0.012887               0.017620               -0.004733       9.545892e-05                0.555556         0.025407        0.125000             135                0.030049      p9_official_reference
 LTCUSDT                0.012352               0.005080                0.007272       7.719893e-05                0.456250         0.017106        0.125000             160                0.028801      p9_official_reference
 BCHUSDT                0.011240               0.016617               -0.005377       6.938222e-05                0.537037         0.012912        0.125000             162                0.026208      p9_official_reference
 ETCUSDT                0.010252               0.023846               -0.013594       6.488805e-05                0.430380         0.012081        0.125000             158                0.023905      p9_official_reference
 BNBUSDT                0.003570               0.014292               -0.010722       2.445522e-05                0.493151         0.025346        0.125000             146                0.008325      p9_official_reference
 ETHUSDT               -0.001731               0.020206               -0.021937      -1.227341e-05                0.517730         0.023070        0.125000             141                0.004035      p9_official_reference
DOGEUSDT               -0.002657               0.008828               -0.011485      -1.819652e-05                0.520548         0.015171        0.125000             146                0.006195      p9_official_reference
AVAXUSDT               -0.006743              -0.014804                0.008061      -4.037664e-05                0.520958         0.017210        0.125000             167                0.015722      p9_official_reference
 DOTUSDT               -0.007087              -0.005755               -0.001332      -4.632166e-05                0.483660         0.014594        0.125000             153                0.016525      p9_official_reference
 UNIUSDT               -0.019837              -0.002364               -0.017473      -1.255497e-04                0.506329         0.013117        0.125000             158                0.046254      p9_official_reference
ATOMUSDT               -0.023629              -0.026866                0.003237      -1.476785e-04                0.487500         0.010232        0.125000             160                0.055095      p9_official_reference
  OPUSDT               -0.029304              -0.017007               -0.012297      -1.820130e-04                0.478261         0.006979        0.125000             161                0.068329      p9_official_reference
 ADAUSDT               -0.031784              -0.023243               -0.008541      -2.254159e-04                0.425532         0.012446        0.122932             141                0.074110      p9_official_reference
```
