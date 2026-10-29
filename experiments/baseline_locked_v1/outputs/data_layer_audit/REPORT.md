# Data Layer Audit

This audit checks raw 15m Binance klines, hourly derivatives, residual/replay panels, fold boundaries, and preprocessing choices.

## Topline Findings

```text
                                      check value status
               kline_missing_15m_bars_total     0   pass
           kline_duplicate_timestamps_total     0   pass
                  kline_bad_ohlc_rows_total     0   pass
            asset_missing_hourly_timestamps     0   pass
            asset_min_symbols_per_timestamp    20   pass
      asset_duplicate_timestamp_symbol_rows     0   pass
      derivatives_missing_hourly_timestamps     0   pass
      derivatives_min_symbols_per_timestamp    20   pass
derivatives_duplicate_timestamp_symbol_rows     0   pass
           replay_missing_hourly_timestamps     0   pass
           replay_min_symbols_per_timestamp    20   pass
     replay_duplicate_timestamp_symbol_rows     0   pass
                 fold_train_before_test_all  True   pass
                 fold_test_inside_panel_all  True   pass
               asset_minus_derivatives_keys     0   pass
               derivatives_minus_asset_keys     0   pass
                    replay_minus_asset_keys     0   pass
                    asset_minus_replay_keys     0   pass
          derivatives_freshness_field_count    11   pass
                 derivatives_max_stale_rate   0.0   pass
       robust_preprocessing_missing_columns     0   pass
              transform_review_action_count     6 review
```

## Kline Summary

```text
  symbol   rows                     start                       end  duplicate_timestamps  missing_15m_bars  extra_15m_bars  bad_ohlc_rows  nonpositive_volume_rows  nonpositive_quote_volume_rows  close_time_lag_ms_median  close_time_lag_ms_min  close_time_lag_ms_max  interval_min  interval_max  abs_log_ret_p99  abs_log_ret_max  zero_volume_rows
 ADAUSDT 115618 2023-03-01 00:00:00+00:00 2026-06-17 08:15:00+00:00                     0                 0               0              0                        0                              0                  899999.0               899999.0               899999.0          15.0          15.0         0.016253         0.344931                 0
 APTUSDT 115618 2023-03-01 00:00:00+00:00 2026-06-17 08:15:00+00:00                     0                 0               0              0                        0                              0                  899999.0               899999.0               899999.0          15.0          15.0         0.017925         0.292861                 0
 ARBUSDT 113446 2023-03-23 15:00:00+00:00 2026-06-17 08:15:00+00:00                     0                 0               0              0                        0                              0                  899999.0               899999.0               899999.0          15.0          15.0         0.017953         0.393587                 0
ATOMUSDT 115618 2023-03-01 00:00:00+00:00 2026-06-17 08:15:00+00:00                     0                 0               0              0                        0                              0                  899999.0               899999.0               899999.0          15.0          15.0         0.014337         0.294347                 0
AVAXUSDT 115618 2023-03-01 00:00:00+00:00 2026-06-17 08:15:00+00:00                     0                 0               0              0                        0                              0                  899999.0               899999.0               899999.0          15.0          15.0         0.016575         0.242840                 0
 BCHUSDT 115618 2023-03-01 00:00:00+00:00 2026-06-17 08:15:00+00:00                     0                 0               0              0                        0                              0                  899999.0               899999.0               899999.0          15.0          15.0         0.015745         0.176803                 0
 BNBUSDT 115618 2023-03-01 00:00:00+00:00 2026-06-17 08:15:00+00:00                     0                 0               0              0                        0                              0                  899999.0               899999.0               899999.0          15.0          15.0         0.010494         0.124028                 0
 BTCUSDT 115618 2023-03-01 00:00:00+00:00 2026-06-17 08:15:00+00:00                     0                 0               0              0                        0                              0                  899999.0               899999.0               899999.0          15.0          15.0         0.009192         0.108978                 5
DOGEUSDT 115618 2023-03-01 00:00:00+00:00 2026-06-17 08:15:00+00:00                     0                 0               0              0                        0                              0                  899999.0               899999.0               899999.0          15.0          15.0         0.017734         0.170838                 0
 DOTUSDT 115618 2023-03-01 00:00:00+00:00 2026-06-17 08:15:00+00:00                     0                 0               0              0                        0                              0                  899999.0               899999.0               899999.0          15.0          15.0         0.015195         0.485836                 0
 ETCUSDT 115618 2023-03-01 00:00:00+00:00 2026-06-17 08:15:00+00:00                     0                 0               0              0                        0                              0                  899999.0               899999.0               899999.0          15.0          15.0         0.014755         0.391435                 0
 ETHUSDT 115618 2023-03-01 00:00:00+00:00 2026-06-17 08:15:00+00:00                     0                 0               0              0                        0                              0                  899999.0               899999.0               899999.0          15.0          15.0         0.012007         0.135187                 0
 FILUSDT 115618 2023-03-01 00:00:00+00:00 2026-06-17 08:15:00+00:00                     0                 0               0              0                        0                              0                  899999.0               899999.0               899999.0          15.0          15.0         0.017999         0.326910                 0
LINKUSDT 115618 2023-03-01 00:00:00+00:00 2026-06-17 08:15:00+00:00                     0                 0               0              0                        0                              0                  899999.0               899999.0               899999.0          15.0          15.0         0.015835         0.321703                 0
 LTCUSDT 115618 2023-03-01 00:00:00+00:00 2026-06-17 08:15:00+00:00                     0                 0               0              0                        0                              0                  899999.0               899999.0               899999.0          15.0          15.0         0.014377         0.298607                 0
NEARUSDT 115618 2023-03-01 00:00:00+00:00 2026-06-17 08:15:00+00:00                     0                 0               0              0                        0                              0                  899999.0               899999.0               899999.0          15.0          15.0         0.019000         0.171429                 0
  OPUSDT 115618 2023-03-01 00:00:00+00:00 2026-06-17 08:15:00+00:00                     0                 0               0              0                        0                              0                  899999.0               899999.0               899999.0          15.0          15.0         0.018692         0.497630                 0
 SOLUSDT 115618 2023-03-01 00:00:00+00:00 2026-06-17 08:15:00+00:00                     0                 0               0              0                        0                              0                  899999.0               899999.0               899999.0          15.0          15.0         0.016402         0.164138                 0
 UNIUSDT 115618 2023-03-01 00:00:00+00:00 2026-06-17 08:15:00+00:00                     0                 0               0              0                        0                              0                  899999.0               899999.0               899999.0          15.0          15.0         0.017205         0.417503                 0
 XRPUSDT 115618 2023-03-01 00:00:00+00:00 2026-06-17 08:15:00+00:00                     0                 0               0              0                        0                              0                  899999.0               899999.0               899999.0          15.0          15.0         0.015918         0.277338                 0
```

## Kline Gaps

```text
No gaps.
```

## Panel Summaries

```text
            panel   rows  symbols  timestamps                     start                       end  expected_hourly_timestamps  missing_hourly_timestamps  extra_hourly_timestamps  expected_balanced_rows  missing_balanced_rows  duplicate_timestamp_symbol_rows  min_symbols_per_timestamp  q01_symbols_per_timestamp  median_symbols_per_timestamp
      asset_panel 567220       20       28361 2023-03-23 15:00:00+00:00 2026-06-17 07:00:00+00:00                       28361                          0                        0                  567220                      0                                0                         20                       20.0                          20.0
derivatives_panel 567220       20       28361 2023-03-23 15:00:00+00:00 2026-06-17 07:00:00+00:00                       28361                          0                        0                  567220                      0                                0                         20                       20.0                          20.0
     replay_panel 567220       20       28361 2023-03-23 15:00:00+00:00 2026-06-17 07:00:00+00:00                       28361                          0                        0                  567220                      0                                0                         20                       20.0                          20.0
```

## Source Alignment

```text
                      source   rows  unique_timestamp_symbol  symbols  timestamps                     start                       end
                       asset 567220                 567220.0     20.0     28361.0 2023-03-23 15:00:00+00:00 2026-06-17 07:00:00+00:00
                 derivatives 567220                 567220.0     20.0     28361.0 2023-03-23 15:00:00+00:00 2026-06-17 07:00:00+00:00
                      replay 567220                 567220.0     20.0     28361.0 2023-03-23 15:00:00+00:00 2026-06-17 07:00:00+00:00
asset_minus_derivatives_keys      0                      NaN      NaN         NaN                       NaT                       NaT
derivatives_minus_asset_keys      0                      NaN      NaN         NaN                       NaT                       NaT
     replay_minus_asset_keys      0                      NaN      NaN         NaN                       NaT                       NaT
     asset_minus_replay_keys      0                      NaN      NaN         NaN                       NaT                       NaT
```

## Preprocessing Transform Review

```text
              field_family                                                                                                    current_state                                                                                                                       recommended_input                                                  action
                     price                                       asset panel has log_close/log_ret/simple_ret; residual model uses log_ret.                                             log price for level features; log return for beta/residual; never raw price across symbols.                                                    keep
volume/quote_volume/trades                            asset panel has log_quote_volume, log-volume diffs, liquidity share, and liquidity_z.                                    log1p volume/quote_volume/trades, rolling share, rolling z, and log-diff for liquidity acceleration.                                                    done
       taker buy/sell flow                                    asset panel has normalized taker buy ratio, taker imbalance, EWMA, and delta.                     use imbalance = 2*taker_buy/volume - 1, then EWMA/delta/z; raw taker quantities should be log-scaled or normalized.                                                    done
             open_interest                                          derivatives panel has OI level, log OI value, EWM, delta, z, OI growth.                                     log OI value level for leverage stock; log-diff/delta for leverage flow; OI/volume for crowdedness. mostly ok; add OI/volume if not present in active panel
         long_short_ratios                             derivatives panel stores log(ratio), EWM, deltas, rolling z, and freshness metadata.                                                      keep log-ratio convention documented; raw ratio is skewed and asymmetric around 1.                                                    done
                   funding                                              funding level, annualized, EWM, delta, z, acceleration are present.                                       level is economically meaningful; use EWM/delta/z and carry-adjusted return. Do not only diff it.                                                    keep
             premium/basis                                                   premium and mark-index basis level, EWM, delta, z are present.                                          level for rich/cheap state, delta for worsening/normalization, z for cross-time comparability.                                                    keep
             beta/residual                                        beta, residual_log_ret, residual z/momentum/quality features are present. residual return should be modeled in log-return units; residual level anchors should use cumulative residual index/log residual equity.                       check residual index construction
  cross_sectional_features                                                        many features are already CS rank/zscore or EMA of ranks.                                          preferred for ranking alpha; keep raw transformed values only for diagnostics and calibration.                                                    keep
              missing_data warmup NaNs exist; derivatives raw fields are shifted by 1h and now include source age/stale/freshness metadata.                    never fill alpha features cross-sectionally; if forward-fill derivatives, add freshness age and cap max stale hours.                                                    done
```

## Derivatives Freshness

```text
                           field  age_notna_rate  age_p50_hours  age_p95_hours  age_max_hours  stale_rate  freshness_weight_p01  freshness_weight_p50
  deriv_account_long_short_ratio        0.999644            1.0            1.0            3.0         0.0                   1.0                   1.0
        deriv_funding_annualized        0.986178            5.0            8.0            9.0         0.0                   1.0                   1.0
              deriv_funding_rate        0.986178            5.0            8.0            9.0         0.0                   1.0                   1.0
              deriv_log_oi_value        0.999644            1.0            1.0            3.0         0.0                   1.0                   1.0
          deriv_mark_index_basis        0.986213            1.0            1.0            2.0         0.0                   1.0                   1.0
                  deriv_oi_value        0.999644            1.0            1.0            3.0         0.0                   1.0                   1.0
             deriv_open_interest        0.999644            1.0            1.0            3.0         0.0                   1.0                   1.0
             deriv_premium_index        0.985686            1.0            1.0            2.0         0.0                   1.0                   1.0
    deriv_taker_long_short_ratio        0.999644            1.0            1.0            3.0         0.0                   1.0                   1.0
deriv_top_count_long_short_ratio        0.999644            1.0            1.0            3.0         0.0                   1.0                   1.0
  deriv_top_sum_long_short_ratio        0.999644            1.0            1.0            3.0         0.0                   1.0                   1.0
```

## Robust Preprocessing

```text
                        column                            purpose  present  notna_rate  finite_rate       p01       p50       p99
                    log_volume                   volume log scale     True    1.000000     1.000000  8.038679 13.431144 19.774962
            log_volume_diff_24                volume acceleration     True    0.999154     0.999154 -1.892695 -0.022536  2.088801
      log_quote_volume_diff_24          quote-volume acceleration     True    0.999154     0.999154 -1.890733 -0.023741  2.093847
                  volume_share       cross-sectional volume share     True    1.000000     1.000000  0.000012  0.002424  0.779593
            quote_volume_share cross-sectional quote-volume share     True    1.000000     1.000000  0.000770  0.007519  0.540440
  quote_volume_share_change_24     relative liquidity share trend     True    0.999154     0.999154 -1.456366 -0.009885  1.609995
          taker_buy_base_ratio    normalized taker base buy share     True    1.000000     1.000000  0.372810  0.491109  0.609696
         taker_buy_quote_ratio   normalized taker quote buy share     True    1.000000     1.000000  0.372900  0.491172  0.609770
         taker_imbalance_quote         normalized taker imbalance     True    1.000000     1.000000 -0.254200 -0.017656  0.219540
  taker_imbalance_quote_ewm_24               slow taker imbalance     True    0.999753     0.999753 -0.074778 -0.017487  0.037264
taker_imbalance_quote_delta_24        taker pressure acceleration     True    0.998343     0.998343 -0.042544  0.000016  0.041992
            log_high_low_range       range-based volatility proxy     True    1.000000     1.000000  0.002124  0.009725  0.049212
          close_location_value                 bar close location     True    1.000000     1.000000 -0.975610  0.023256  0.992453
                     amihud_1h               liquidity cost proxy     True    0.999965     0.999965  0.000000  0.000439  0.007209
            log_ret_winsor_720        past-only winsorized return     True    0.999965     0.999965 -0.022896  0.000000  0.022583
          log_ret_robust_z_720       past-vol standardized return     True    0.991502     0.991502 -2.895375  0.000000  2.807439
       abs_log_ret_outlier_720              past-vol outlier flag     True    0.991502     0.991502  0.000000  0.000000  0.000000
```
