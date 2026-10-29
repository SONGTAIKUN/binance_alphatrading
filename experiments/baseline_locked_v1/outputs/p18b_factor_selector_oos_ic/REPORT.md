# P18b Factor Selector OOS IC

Each fold selects factors using only that fold's historical validation diagnostics, then evaluates the equal-weight oriented selected-factor score on the next test segment.

## OOS Summary

```text
 horizon  fold_count  mean_oos_ic  median_oos_ic  q25_fold_oos_ic  positive_fold_rate  mean_top_bottom_spread  mean_turnover_72h  mean_autocorr_72h  mean_selected_count
      48          32     0.058140       0.063600         0.019819             0.84375                0.002468           0.227540           0.765080                  8.0
      72          32     0.054846       0.051647        -0.001748             0.75000                0.003404           0.214485           0.772907                  8.0
      96          32     0.055506       0.045306        -0.013607             0.65625                0.004072           0.210936           0.769205                  8.0
     168          32     0.053134       0.059912        -0.051489             0.65625                0.005107           0.165206           0.842168                  8.0
```

## Common Selected Factors

```text
raw_name
volume                                    35
taker_buy_base                            33
deriv_open_interest                       32
cs_res_ma_slope_48_168                    29
realized_vol_long                         27
cs_breakout_risk_72                       25
deriv_crowding_pressure_72                25
cs_volume_share_trend_7d_30d              24
p6b_J04_top_trader_contra                 22
idio_vol_medium                           20
cs_vol_of_vol_res_168                     19
p6_E01_uncrowded_mom                      18
p6_G04_p3d_diversifier_candidate          18
cs_rank_realized_vol_medium               18
realized_vol_medium                       18
cs_zscore_idio_vol_medium                 18
cs_res_mom_168_24                         17
idio_vol_long                             17
cs_rank_idio_vol_medium                   17
deriv_top_sum_long_short_ratio_ewm_72     16
deriv_top_sum_long_short_ratio_ewm_24     16
deriv_top_sum_long_short_ratio            16
p6_G01_slow_consensus                     16
deriv_top_sum_long_short_ratio_ewm_168    15
deriv_account_long_short_ratio_ewm_336    14
cs_res_high_mom_336                       14
cs_rank_beta_btc                          13
p6_G02_consensus_low_disagreement         13
cs_res_trend_quality_168_24               13
beta_btc                                  13
p6b_J02_oi_growth_crowding_fade           12
cs_vol_of_vol_336                         12
deriv_top_sum_long_short_ratio_ewm_336    12
cs_zscore_beta_btc                        12
alpha_A_res_trend_quality                 12
p6b_J03_account_longs_contra              11
deriv_account_long_short_ratio_ewm_168    11
p6_F02_uptrend_mom                        11
cs_zscore_realized_vol_medium             11
deriv_log_oi_value_delta_168              11
```

## Weakest OOS Folds

```text
 fold  horizon                test_start                  test_end  selected_count  oos_ic_mean  oos_ic_q25_ts  positive_ts_rate  top_bottom_spread_mean  mean_turnover_72h  mean_autocorr_72h                                                                                                                                                                                                                                                                               selected_factors
   13      168 2024-10-09 08:00:00+00:00 2024-11-08 08:00:00+00:00               8    -0.160673      -0.299179          0.246879               -0.051128           0.324171           0.594987                                                                       cs_res_high_mom_336|p6_G01_slow_consensus|p6_C01_liq_persistent_mom|p6_G04_p3d_diversifier_candidate|p6_C02_dollar_volume_mom|deriv_log_oi_value_delta_168|p6b_J01_oi_growth_mom_quality|p6b_J02_oi_growth_crowding_fade
   30       96 2026-03-03 08:00:00+00:00 2026-04-02 08:00:00+00:00               8    -0.121439      -0.223091          0.248266               -0.012261           0.332160           0.670992                                                                             p6_G02_consensus_low_disagreement|p6_G04_p3d_diversifier_candidate|p6_D04_low_vov_momentum|cs_res_high_mom_336|alpha_A_res_trend_quality|cs_res_trend_quality_168_24|cs_res_ma_slope_48_168|p6_D01_compression_mom
   30       72 2026-03-03 08:00:00+00:00 2026-04-02 08:00:00+00:00               8    -0.106109      -0.208350          0.242718               -0.008707           0.355094           0.625288                                                                  p6_D03_range_compression_quality|p6_G04_p3d_diversifier_candidate|cs_res_ma_slope_48_168|cs_res_trend_quality_168_24|alpha_A_res_trend_quality|p6_D04_low_vov_momentum|p6_G02_consensus_low_disagreement|p6_E01_uncrowded_mom
   30      168 2026-03-03 08:00:00+00:00 2026-04-02 08:00:00+00:00               8    -0.102230      -0.203625          0.263523               -0.011323           0.288042           0.727145                                                                         cs_res_high_mom_336|p6b_K05_deriv_p3d_diversifier|p6_D04_low_vov_momentum|p6_G04_p3d_diversifier_candidate|p6_G02_consensus_low_disagreement|cs_res_high_mom_168|alpha_A_res_trend_quality|cs_res_trend_quality_168_24
   14      168 2024-11-08 08:00:00+00:00 2024-12-08 08:00:00+00:00               8    -0.095453      -0.313108          0.341193               -0.052664           0.156697           0.868221                                                              cs_res_high_mom_336|deriv_crowding_pressure_72|p6_G01_slow_consensus|deriv_account_long_short_ratio_ewm_336|p6b_J04_top_trader_contra|deriv_top_count_long_short_ratio_ewm_336|cs_res_high_mom_168|deriv_top_sum_long_short_ratio
   23      168 2025-08-05 08:00:00+00:00 2025-09-04 08:00:00+00:00               8    -0.078781      -0.187514          0.382802               -0.002080           0.081589           0.979003                                                                                    deriv_account_long_short_ratio_ewm_72|deriv_account_long_short_ratio_ewm_24|p6b_J03_account_longs_contra|deriv_account_long_short_ratio_ewm_168|taker_buy_base|volume|deriv_open_interest|realized_vol_long
   23       96 2025-08-05 08:00:00+00:00 2025-09-04 08:00:00+00:00               8    -0.074908      -0.200058          0.349515               -0.000292           0.104803           0.968224                                                    deriv_account_long_short_ratio_ewm_72|deriv_account_long_short_ratio_ewm_24|deriv_account_long_short_ratio|deriv_account_long_short_ratio_ewm_168|p6b_J03_account_longs_contra|volume|taker_buy_base|deriv_account_long_short_ratio_ewm_336
   30       48 2026-03-03 08:00:00+00:00 2026-04-02 08:00:00+00:00               8    -0.073407      -0.222598          0.345354               -0.004630           0.319763           0.677426                                                         p6_D03_range_compression_quality|p6_D04_low_vov_momentum|p6_G04_p3d_diversifier_candidate|cs_res_ma_slope_48_168|p6b_K05_deriv_p3d_diversifier|p6_G02_consensus_low_disagreement|cs_res_trend_quality_168_24|alpha_A_res_trend_quality
    1      168 2023-10-15 08:00:00+00:00 2023-11-14 08:00:00+00:00               8    -0.069761      -0.224579          0.395284               -0.004804           0.288270           0.582564                                                                                         cs_res_ma_slope_48_168|p6_F05_high_vol_defensive_quality|deriv_log_oi_value_delta_168|deriv_oi_growth_72_336|p6b_J02_oi_growth_crowding_fade|p6_E01_uncrowded_mom|cs_res_mom_168_24|p6_F02_uptrend_mom
   19      168 2025-04-07 08:00:00+00:00 2025-05-07 08:00:00+00:00               8    -0.069406      -0.166881          0.343967               -0.009994           0.098779           0.958834                               p6_F02_uptrend_mom|deriv_top_sum_long_short_ratio_ewm_168|deriv_top_sum_long_short_ratio_ewm_336|deriv_top_sum_long_short_ratio_ewm_72|p6b_J04_top_trader_contra|deriv_top_sum_long_short_ratio_ewm_24|deriv_crowding_pressure_72|deriv_top_sum_long_short_ratio
   28      168 2026-01-02 08:00:00+00:00 2026-02-01 08:00:00+00:00               8    -0.062360      -0.258053          0.380028               -0.010425           0.049469           0.988284                                                                                                                                                                      deriv_open_interest|cs_rank_beta_market|beta_market|cs_zscore_beta_market|volume|taker_buy_base|beta_btc|cs_rank_beta_btc
   23       72 2025-08-05 08:00:00+00:00 2025-09-04 08:00:00+00:00               8    -0.057523      -0.200360          0.374480                0.000625           0.106412           0.967531                                                    deriv_account_long_short_ratio_ewm_72|deriv_account_long_short_ratio_ewm_24|p6b_J03_account_longs_contra|deriv_account_long_short_ratio_ewm_168|deriv_account_long_short_ratio|deriv_account_long_short_ratio_ewm_336|taker_buy_base|volume
   18      168 2025-03-08 08:00:00+00:00 2025-04-07 08:00:00+00:00               8    -0.054528      -0.216748          0.346741                0.002456           0.106328           0.952493 cs_volume_share_trend_7d_30d|cs_volume_share_trend_7d_30d|deriv_top_count_long_short_ratio_ewm_336|deriv_top_count_long_short_ratio_ewm_72|deriv_account_long_short_ratio_ewm_336|p6b_J03_account_longs_contra|deriv_account_long_short_ratio_ewm_168|deriv_top_count_long_short_ratio_ewm_168
   32      168 2026-05-02 08:00:00+00:00 2026-06-01 08:00:00+00:00               8    -0.050476      -0.230216          0.420250               -0.016695           0.181936           0.798695                                         cs_volume_share_trend_7d_30d|cs_volume_share_trend_7d_30d|deriv_account_long_short_ratio_ewm_336|cs_vol_of_vol_res_168|cs_vol_of_vol_res_168|p6b_J03_account_longs_contra|deriv_account_long_short_ratio_ewm_168|deriv_account_long_short_ratio_ewm_72
   13       96 2024-10-09 08:00:00+00:00 2024-11-08 08:00:00+00:00               8    -0.048431      -0.182578          0.406380               -0.017393           0.356116           0.544479                                                                  p6_F01_normal_regime_mom|p6_G01_slow_consensus|p6b_J01_oi_growth_mom_quality|p6_C01_liq_persistent_mom|deriv_log_oi_value_delta_168|p6b_J02_oi_growth_crowding_fade|p6_G04_p3d_diversifier_candidate|p6_C02_dollar_volume_mom
   26       48 2025-11-03 08:00:00+00:00 2025-12-03 08:00:00+00:00               8    -0.046259      -0.218086          0.416089                0.004625           0.186113           0.817232                                                     cs_dollar_volume_trend_7d_30d|cs_volume_share_trend_7d_30d_p6|cs_volume_share_trend_7d_30d|cs_volume_share_trend_7d_30d|p6b_H01_funding_contra_168|p6b_H05_carry_unwind_contra|deriv_funding_rate_ewm_168|deriv_funding_annualized_ewm_168
   19       96 2025-04-07 08:00:00+00:00 2025-05-07 08:00:00+00:00               8    -0.041468      -0.199633          0.396671               -0.002911           0.115585           0.911600                                     p6_F01_normal_regime_mom|p6_F02_uptrend_mom|deriv_top_sum_long_short_ratio_ewm_336|deriv_top_sum_long_short_ratio_ewm_168|p6b_J04_top_trader_contra|deriv_crowding_pressure_72|deriv_top_sum_long_short_ratio_ewm_72|deriv_top_sum_long_short_ratio_ewm_24
    1       96 2023-10-15 08:00:00+00:00 2023-11-14 08:00:00+00:00               8    -0.039433      -0.169444          0.411928                0.001307           0.339728           0.509290                                                                                             cs_res_ma_slope_48_168|p6_F05_high_vol_defensive_quality|p6_E01_uncrowded_mom|cs_res_mom_168_24|p6_E02_purity_mom|deriv_oi_growth_72_336|p6b_J02_oi_growth_crowding_fade|alpha_A_res_trend_quality
    9      168 2024-06-11 08:00:00+00:00 2024-07-11 08:00:00+00:00               8    -0.036829      -0.152282          0.398058                0.000549           0.063727           0.984185                                                                           taker_buy_base|volume|deriv_open_interest|deriv_top_count_long_short_ratio_ewm_168|deriv_top_count_long_short_ratio_ewm_72|p6b_J04_top_trader_contra|deriv_taker_long_short_ratio_ewm_336|deriv_crowding_pressure_72
   23       48 2025-08-05 08:00:00+00:00 2025-09-04 08:00:00+00:00               8    -0.034730      -0.184025          0.416089               -0.000741           0.122436           0.957807                                  deriv_account_long_short_ratio_ewm_24|deriv_account_long_short_ratio_ewm_72|deriv_account_long_short_ratio|p6b_J03_account_longs_contra|deriv_account_long_short_ratio_ewm_168|deriv_account_long_short_ratio_ewm_336|deriv_top_count_long_short_ratio|volume
   26       72 2025-11-03 08:00:00+00:00 2025-12-03 08:00:00+00:00               8    -0.032378      -0.202597          0.425798                0.007118           0.188780           0.810689                                                     cs_dollar_volume_trend_7d_30d|cs_volume_share_trend_7d_30d|cs_volume_share_trend_7d_30d|cs_volume_share_trend_7d_30d_p6|deriv_funding_annualized_ewm_168|deriv_funding_rate_ewm_168|p6b_H05_carry_unwind_contra|p6b_H01_funding_contra_168
   26       96 2025-11-03 08:00:00+00:00 2025-12-03 08:00:00+00:00               8    -0.031125      -0.205030          0.400832                0.007515           0.188126           0.812768                                                     cs_dollar_volume_trend_7d_30d|cs_volume_share_trend_7d_30d_p6|cs_volume_share_trend_7d_30d|cs_volume_share_trend_7d_30d|deriv_funding_annualized_ewm_168|deriv_funding_rate_ewm_168|p6b_H01_funding_contra_168|p6b_H05_carry_unwind_contra
   19       72 2025-04-07 08:00:00+00:00 2025-05-07 08:00:00+00:00               8    -0.028191      -0.179205          0.407767                0.000125           0.119614           0.901506                                     p6_F01_normal_regime_mom|p6_F02_uptrend_mom|deriv_top_sum_long_short_ratio_ewm_336|deriv_top_sum_long_short_ratio_ewm_168|deriv_top_sum_long_short_ratio_ewm_72|deriv_top_sum_long_short_ratio_ewm_24|p6b_J04_top_trader_contra|deriv_crowding_pressure_72
   28       96 2026-01-02 08:00:00+00:00 2026-02-01 08:00:00+00:00               8    -0.019802      -0.169755          0.393897               -0.003808           0.050334           0.990258                                                                                                                                                   deriv_open_interest|volume|taker_buy_base|cs_rank_beta_market|beta_market|cs_zscore_beta_market|cs_vol_compression_168_504|cs_vol_of_vol_336
   25       96 2025-10-04 08:00:00+00:00 2025-11-03 08:00:00+00:00               8    -0.015579      -0.228376          0.542302               -0.003026           0.118459           0.909891                                                                              cs_zscore_beta_btc|cs_rank_beta_btc|beta_btc|deriv_top_sum_long_short_ratio|cs_volume_share_trend_7d_30d|cs_volume_share_trend_7d_30d|deriv_top_sum_long_short_ratio_ewm_24|deriv_top_sum_long_short_ratio_ewm_72
   13       72 2024-10-09 08:00:00+00:00 2024-11-08 08:00:00+00:00               8    -0.013048      -0.151503          0.471567               -0.008791           0.352934           0.553631                                                                  p6b_J01_oi_growth_mom_quality|p6_F01_normal_regime_mom|p6_C02_dollar_volume_mom|p6_G01_slow_consensus|p6b_J02_oi_growth_crowding_fade|p6_C01_liq_persistent_mom|p6_G04_p3d_diversifier_candidate|deriv_log_oi_value_delta_168
   14       96 2024-11-08 08:00:00+00:00 2024-12-08 08:00:00+00:00               8    -0.012950      -0.200038          0.539528               -0.017463           0.252818           0.669366                                                                  p6_G01_slow_consensus|cs_res_high_mom_336|p6b_J02_oi_growth_crowding_fade|p6_C01_liq_persistent_mom|p6_G04_p3d_diversifier_candidate|deriv_top_sum_long_short_ratio|cs_breakout_risk_72|deriv_top_sum_long_short_ratio_ewm_24
   28       72 2026-01-02 08:00:00+00:00 2026-02-01 08:00:00+00:00               8    -0.012854      -0.195279          0.428571               -0.000796           0.062471           0.986239                                                                                                                                      deriv_open_interest|taker_buy_base|volume|beta_market|cs_rank_beta_market|cs_zscore_beta_market|deriv_funding_rate_ewm_24|deriv_funding_annualized_ewm_24
   14       72 2024-11-08 08:00:00+00:00 2024-12-08 08:00:00+00:00               8    -0.012301      -0.127231          0.545076               -0.022543           0.299629           0.674619                                                                                p6_G01_slow_consensus|p6b_J02_oi_growth_crowding_fade|p6_C01_liq_persistent_mom|cs_res_high_mom_336|p6_G04_p3d_diversifier_candidate|cs_breakout_risk_72|p6b_J01_oi_growth_mom_quality|p6_C02_dollar_volume_mom
    1       72 2023-10-15 08:00:00+00:00 2023-11-14 08:00:00+00:00               8    -0.011809      -0.142359          0.513176                0.004810           0.314525           0.552697                                                                                                              cs_res_ma_slope_48_168|cs_vol_of_vol_res_168|cs_vol_of_vol_res_168|p6_F02_uptrend_mom|deriv_oi_growth_72_336|cs_res_mom_168_24|p6_E01_uncrowded_mom|p6_G03_liq_purity_compression
```
