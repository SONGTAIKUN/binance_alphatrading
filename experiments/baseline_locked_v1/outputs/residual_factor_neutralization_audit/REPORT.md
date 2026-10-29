# Residual / Factor / Neutralization Audit

This audit numerically recomputes factor inputs, rolling betas, residual identities, forward residual labels, and realized portfolio exposures.

## Topline

```text
                                         check        value status
              factor_construction_max_abs_diff 1.504873e-16   pass
            factor_construction_mismatch_count 0.000000e+00   pass
           rolling_beta_past_only_max_abs_diff 1.715295e-13   pass
               rolling_beta_min_past_valid_obs 2.400000e+02   pass
                residual_identity_max_abs_diff 5.282233e-16   pass
              factor_model_max_abs_factor_corr 9.829262e-01 review
             factor_model_max_condition_number 2.542046e+17 review
 factor_model_median_residual_to_raw_vol_ratio 5.425396e-01   pass
                factor_model_min_approx_oos_r2 5.122651e-01   pass
             forward_return_label_max_abs_diff 3.652460e-15   pass
      residual_label_current_beta_max_abs_diff 5.353357e-15   pass
   long_history_p15c_nested_max_net_weight_abs 9.027067e-16   pass
  long_history_p15c_nested_max_abs_beta_market 6.502333e-13   pass
long_history_p15c_nested_max_missing_beta_rows 0.000000e+00   pass
        p16_clean_candidate_max_net_weight_abs 8.049117e-16   pass
       p16_clean_candidate_max_abs_beta_market 3.252659e-13   pass
     p16_clean_candidate_max_missing_beta_rows 0.000000e+00   pass
     latest_baseline_cost15_max_net_weight_abs 8.049117e-16   pass
    latest_baseline_cost15_max_abs_beta_market 3.252659e-13   pass
  latest_baseline_cost15_max_missing_beta_rows 0.000000e+00   pass
```

## Factor Construction

```text
            field  valid_compared  max_abs_diff  mismatch_count
   factor_btc_ret          567201  0.000000e+00               0
   factor_eth_ret          567201  0.000000e+00               0
factor_market_ret          567200  1.365011e-16               0
factor_sector_ret          567200  1.504873e-16               0
```

## Rolling Beta Past-Only

```text
 samples  symbols  min_past_valid_obs  max_past_max_abs_diff  p95_past_max_abs_diff  mean_past_max_abs_diff  median_current_inclusive_max_abs_diff  share_inclusive_diff_larger_than_past
     240       20                 240           1.715295e-13           9.246492e-14            2.703240e-14                               0.002669                                    1.0
```

## Residual Identity

```text
                     check  valid_rows  max_abs_diff  p99_abs_diff  mean_abs_diff
predicted_log_ret_identity      562400  5.282233e-16  1.578598e-16   4.543257e-17
 residual_log_ret_identity      562400  5.277896e-16  2.005506e-16   6.291976e-17
```

## Factor-Model Quality

```text
  symbol  valid_model_rate  residual_to_raw_vol_ratio  approx_out_of_sample_r2  max_abs_factor_corr  factor_design_condition_number
 ADAUSDT          0.991502                   0.504157                 0.745826             0.980632                    2.394157e+02
 APTUSDT          0.991502                   0.600846                 0.638984             0.982926                    2.624020e+02
 ARBUSDT          0.991502                   0.463432                 0.785230             0.981694                    2.532435e+02
ATOMUSDT          0.991502                   0.517425                 0.732271             0.982258                    2.532411e+02
AVAXUSDT          0.991502                   0.510491                 0.739399             0.982128                    2.575873e+02
 BCHUSDT          0.991502                   0.698380                 0.512265             0.942921                    7.222849e+01
 BNBUSDT          0.991502                   0.612415                 0.624947             0.966099                    2.542046e+17
 BTCUSDT          0.991502                   0.526792                 0.722490             0.937364                    4.727590e+01
DOGEUSDT          0.991502                   0.568841                 0.676419             0.928638                    6.097830e+01
 DOTUSDT          0.991502                   0.449271                 0.798156             0.981646                    2.462943e+02
 ETCUSDT          0.991502                   0.489293                 0.760593             0.926401                    5.921542e+01
 ETHUSDT          0.991502                   0.454127                 0.793768             0.906536                    3.314877e+01
 FILUSDT          0.991502                   0.556130                 0.690720             0.908314                    4.821032e+01
LINKUSDT          0.991502                   0.494791                 0.755182             0.906542                    5.377775e+01
 LTCUSDT          0.991502                   0.577670                 0.666298             0.936930                    6.523634e+01
NEARUSDT          0.991502                   0.585549                 0.657132             0.982791                    2.613663e+02
  OPUSDT          0.991502                   0.528950                 0.720212             0.982206                    2.555671e+02
 SOLUSDT          0.991502                   0.577005                 0.667066             0.981689                    2.616490e+02
 UNIUSDT          0.991502                   0.628318                 0.605216             0.929557                    6.560042e+01
 XRPUSDT          0.991502                   0.655086                 0.570862             0.936775                    6.615277e+01
```

## Forward Residual Labels

```text
 horizon  execution_delay_bars  valid_rows  fwd_max_abs_diff  label_max_abs_diff
       4                     1      562300      3.652027e-15        3.784299e-15
       8                     1      562220      3.652027e-15        3.891852e-15
      24                     1      561900      3.652460e-15        4.016752e-15
      48                     1      561420      3.652460e-15        4.364564e-15
      72                     1      560940      3.652460e-15        4.510281e-15
      96                     1      560460      3.649858e-15        4.801715e-15
     168                     1      559020      3.652027e-15        5.353357e-15
```

## Portfolio Neutralization

```text
                     run                                                                                                                                                                path  exists  periods  position_rows  max_missing_beta_rows  mean_net_weight_abs  max_net_weight_abs  mean_abs_beta_btc  max_abs_beta_btc  mean_abs_beta_eth  max_abs_beta_eth  mean_abs_beta_market  max_abs_beta_market  mean_abs_beta_sector  max_abs_beta_sector  mean_gross_exposure  max_gross_exposure  mean_long_short_gross_gap  max_long_short_gross_gap  max_abs_symbol_weight  optimizer_success_rate  max_optimizer_exposure_error
long_history_p15c_nested experiments/baseline_locked_v1/outputs/long_history_nested_oos/p15c_nested/nested_extreme_cluster_run    True      242           4086                      0         2.793045e-16        9.027067e-16       4.270252e-15      1.408195e-13       5.960971e-15      3.878807e-13          1.973878e-14         6.502333e-13              0.017515             0.096609             0.261507            0.790435               2.790754e-16              9.020562e-16                  0.125                0.971074                  1.161715e-10
     p16_clean_candidate               experiments/baseline_locked_v1/outputs/p16_clean_candidate_package/nested_candidate_run    True      172           2918                      0         2.658266e-16        8.049117e-16       3.456011e-15      1.223600e-13       3.644004e-15      1.207915e-13          1.123173e-14         3.252659e-13              0.018048             0.080599             0.269956            0.769368               2.640209e-16              8.049117e-16                  0.125                0.970930                  3.316301e-11
  latest_baseline_cost15                    experiments/baseline_locked_v1/outputs/latest_baseline_cost15/nested_candidate_run    True      172           2918                      0         2.675865e-16        8.049117e-16       3.453692e-15      1.223600e-13       3.616105e-15      1.207915e-13          1.118772e-14         3.252659e-13              0.018048             0.080599             0.269956            0.769368               2.657153e-16              8.049117e-16                  0.125                0.970930                  3.316301e-11
```

## Interpretation

- Factor returns are same-bar completed returns; signals are assumed to be formed after the bar close.
- Rolling betas must match OLS fitted on rows strictly before t.
- Residual labels may use future returns because they are labels, but the factor exposure used in the label must be beta_t, not future beta.
- Portfolio neutralization is checked on realized position weights joined to timestamp-t betas.
