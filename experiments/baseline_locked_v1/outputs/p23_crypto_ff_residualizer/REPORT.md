# P23 Crypto FF-Like Rolling Residualizer

This package upgrades the crypto FF-like idea into a rolling past-only residualizer. It is still a research diagnostic, not a full AlphaGPT/P15c rerun.

## Topline

```text
                                     check                value status
              factor_max_abs_pairwise_corr              0.85551   pass
                factor_top_correlated_pair VOL~BETA corr=0.8555   info
                   factor_condition_number            22.286941   pass
  median_residual_to_raw_vol_ratio_current             0.543096   info
median_residual_to_raw_vol_ratio_crypto_ff              0.48039   pass
       median_rolling_resid_vol_cv_current             0.396157   info
     median_rolling_resid_vol_cv_crypto_ff             0.377363   pass
             median_beta_1h_change_current             0.002841   info
           median_beta_1h_change_crypto_ff             0.002315   pass
       mean_ic_current_score_current_label              0.00187   info
                 mean_ic_ff_score_ff_label             0.001821 review
      allocator_gross_return_current_score            -0.208964   info
           allocator_gross_return_ff_score             -0.22047 review
```

## Factor Collinearity

```text
    factor      vif  max_abs_pairwise_corr  condition_number_corr  min_eigenvalue_corr  max_eigenvalue_corr  complete_rows
       MKT 2.167765               0.703329                    NaN                  NaN                  NaN            NaN
       LIQ 1.140781               0.300757                    NaN                  NaN                  NaN            NaN
       MOM 1.229845               0.332177                    NaN                  NaN                  NaN            NaN
       REV 1.005725               0.038666                    NaN                  NaN                  NaN            NaN
     CARRY 1.082667               0.241600                    NaN                  NaN                  NaN            NaN
  CROWDING 1.243904               0.332177                    NaN                  NaN                  NaN            NaN
       VOL 4.584705               0.855510                    NaN                  NaN                  NaN            NaN
      BETA 4.443823               0.855510                    NaN                  NaN                  NaN            NaN
__SYSTEM__      NaN               0.855510              22.286941               0.1258             2.803701        27627.0
```

```text
factor_1 factor_2     corr  abs_corr
     VOL     BETA 0.855510  0.855510
     MKT     BETA 0.703329  0.703329
     MKT      VOL 0.697050  0.697050
     MOM CROWDING 0.332177  0.332177
     LIQ     BETA 0.300757  0.300757
     MKT      LIQ 0.267660  0.267660
CROWDING      VOL 0.261570  0.261570
   CARRY CROWDING 0.241600  0.241600
     MOM      VOL 0.221946  0.221946
     LIQ      VOL 0.201413  0.201413
CROWDING     BETA 0.184012  0.184012
     MKT CROWDING 0.180751  0.180751
```

## Residual Quality

```text
    model  median_resid_to_raw  median_r2_like  median_30d_vol_cv
  current             0.543096        0.704855           0.396157
crypto_ff             0.480390        0.769199           0.377363
```

## Beta Stability

```text
    model  median_abs_1h_change  median_rolling_7d_std  median_rolling_30d_std
crypto_ff              0.002315               0.032149                0.083059
  current              0.002841               0.033032                0.080250
```

## Label IC

```text
                          test  observations  mean_ic  median_ic    q25_ic  positive_ic_rate    ic_ir
current_score_vs_current_label         27992 0.001870  -0.001504 -0.168421          0.496928 0.007628
     current_score_vs_ff_label         27241 0.002125   0.001504 -0.162406          0.501083 0.009041
     ff_score_vs_current_label         27332 0.002337   0.003008 -0.162406          0.502964 0.009973
          ff_score_vs_ff_label         27186 0.001821   0.000000 -0.181955          0.498786 0.007050
```

## Allocator Smoke Test

```text
                            run  periods  total_return  gross_total_return_before_cost    sharpe  max_drawdown  mean_rank_ic  mean_top_bottom_label_spread  mean_period_return  mean_gross_period_return  positive_period_rate  positive_gross_period_rate  mean_turnover  total_turnover  cost_drag  mean_period_cost  target_apply_rate  mean_abs_beta_btc  mean_abs_beta_eth  mean_abs_beta_market  mean_symbol_concentration
  current_neg_resid_z_allocator      288     -0.334027                       -0.208964 -0.732183     -0.335325     -0.007179                     -0.002275           -0.001234                 -0.000638              0.475694                    0.482639       0.994826      286.509762   0.171906          0.000597                1.0       3.210180e-16       2.135777e-16          1.410749e-15                   0.173450
crypto_ff_neg_resid_z_allocator      287     -0.343765                       -0.220470 -0.809445     -0.393108     -0.007634                     -0.000938           -0.001308                 -0.000709              0.491289                    0.508711       0.998745      286.639743   0.171984          0.000599                1.0       4.088553e-16       3.082987e-16          1.097663e-15                   0.179149
```

## Caveats

- MOM/REV characteristics are built from a rolling MKT-only stage-0 residual, not from the old BTC/ETH/sector residual.
- The 20-symbol universe is small, so factor-mimicking portfolios still have material self-inclusion noise.
- The allocator section is a constant-score smoke test using the existing strict-neutral allocator; it does not re-run AlphaGPT, P20 selection, or P15c cluster logic.
- A production replacement would need nested OOS factor/allocator selection and likely leave-one-symbol-out factor construction.
