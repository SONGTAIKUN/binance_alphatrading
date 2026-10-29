# Crypto Fama-French-Like Factor Diagnostics

This is not a strategy backtest. It builds factor-mimicking portfolio returns with t-1 characteristics and diagnoses collinearity, turnover, and simple explanatory power.

## Topline

```text
                                     check               value status
           crypto_ff_max_abs_pairwise_corr             0.69705   pass
             crypto_ff_top_correlated_pair MKT~VOL corr=0.6970   info
           crypto_ff_corr_condition_number           11.107426   pass
                         crypto_ff_max_vif            2.479153   pass
crypto_ff_median_residual_to_raw_vol_ratio            0.498973   pass
                       crypto_ff_median_r2            0.751019   pass
                   highest_turnover_factor                 REV   info
```

## Factor Correlation Matrix

```text
               MKT       LIQ       MOM       REV     CARRY  CROWDING       VOL      BETA
MKT       1.000000  0.267660  0.053248  0.028652  0.126138  0.180751  0.697050 -0.314709
LIQ       0.267660  1.000000 -0.050947 -0.029110  0.045084  0.095512  0.201413 -0.549444
MOM       0.053248 -0.050947  1.000000 -0.055157  0.064417  0.313478  0.222882 -0.092351
REV       0.028652 -0.029110 -0.055157  1.000000 -0.011903  0.017478  0.001465 -0.010898
CARRY     0.126138  0.045084  0.064417 -0.011903  1.000000  0.241600  0.171013 -0.224716
CROWDING  0.180751  0.095512  0.313478  0.017478  0.241600  1.000000  0.261570 -0.263112
VOL       0.697050  0.201413  0.222882  0.001465  0.171013  0.261570  1.000000 -0.467730
BETA     -0.314709 -0.549444 -0.092351 -0.010898 -0.224716 -0.263112 -0.467730  1.000000
```

## Top Pairwise Correlations

```text
factor_1 factor_2      corr  abs_corr
     MKT      VOL  0.697050  0.697050
     LIQ     BETA -0.549444  0.549444
     VOL     BETA -0.467730  0.467730
     MKT     BETA -0.314709  0.314709
     MOM CROWDING  0.313478  0.313478
     MKT      LIQ  0.267660  0.267660
CROWDING     BETA -0.263112  0.263112
CROWDING      VOL  0.261570  0.261570
   CARRY CROWDING  0.241600  0.241600
   CARRY     BETA -0.224716  0.224716
     MOM      VOL  0.222882  0.222882
     LIQ      VOL  0.201413  0.201413
```

## VIF / Condition

```text
    factor      vif  max_abs_pairwise_corr  condition_number_corr  min_eigenvalue_corr  max_eigenvalue_corr  complete_rows
       MKT 2.091791               0.697050                    NaN                  NaN                  NaN            NaN
       LIQ 1.545544               0.549444                    NaN                  NaN                  NaN            NaN
       MOM 1.178262               0.313478                    NaN                  NaN                  NaN            NaN
       REV 1.009115               0.055157                    NaN                  NaN                  NaN            NaN
     CARRY 1.107133               0.241600                    NaN                  NaN                  NaN            NaN
  CROWDING 1.240845               0.313478                    NaN                  NaN                  NaN            NaN
       VOL 2.479153               0.697050                    NaN                  NaN                  NaN            NaN
      BETA 1.907006               0.549444                    NaN                  NaN                  NaN            NaN
__SYSTEM__      NaN               0.697050              11.107426             0.226876             2.520005        27627.0
```

## Factor Return / Turnover Summary

```text
  factor  observations  mean_hourly_return  hourly_vol  annualized_vol  hourly_sharpe_like  autocorr_1h  mean_one_way_turnover  mean_valid_signal_count  mean_gross_weight
     MKT         28360           -0.000025    0.007492        0.701226           -0.003321    -0.004699               0.000018                19.999295           0.999965
     LIQ         28360           -0.000050    0.003686        0.344962           -0.013688     0.003985               0.267063                19.999295           1.999929
     MOM         28024            0.000013    0.003548        0.332058            0.003543     0.025914               0.134886                19.762350           1.976235
     REV         28064           -0.000065    0.003401        0.318277           -0.019004     0.011044               1.389144                19.790557           1.979056
   CARRY         27928           -0.000039    0.003500        0.327593           -0.011187    -0.003614               0.030403                19.676669           1.969465
CROWDING         27879           -0.000031    0.003367        0.315108           -0.009127     0.020517               0.167375                19.648778           1.966010
     VOL         28304           -0.000031    0.004848        0.453738           -0.006391    -0.002067               0.019704                19.959804           1.995980
    BETA         28119            0.000062    0.004062        0.380218            0.015298     0.024023               0.017647                19.829343           1.982934
```

## Portfolio Construction Diagnostics

```text
  factor  mean_valid_signal_count  mean_one_way_turnover  mean_gross_weight  mean_max_abs_weight
    BETA                19.829343               0.017647           1.982934             0.165245
   CARRY                19.676669               0.030403           1.969465             0.164395
CROWDING                19.648778               0.167375           1.966010             0.163995
     LIQ                19.999295               0.267063           1.999929             0.166661
     MKT                19.999295               0.000018           0.999965             0.049998
     MOM                19.762350               0.134886           1.976235             0.164686
     REV                19.790557               1.389144           1.979056             0.164921
     VOL                19.959804               0.019704           1.995980             0.166332
```

## Asset Regression Quality

```text
  symbol  observations  factor_count  design_condition_number  raw_ret_std  residual_ret_std  residual_to_raw_vol_ratio  full_sample_r2
 ADAUSDT         27627             8                11.107426     0.009407          0.004433                   0.471215        0.777957
 APTUSDT         27627             8                11.107426     0.010242          0.005634                   0.550067        0.697427
 ARBUSDT         27627             8                11.107426     0.010310          0.004328                   0.419830        0.823743
ATOMUSDT         27627             8                11.107426     0.008423          0.003879                   0.460588        0.787859
AVAXUSDT         27627             8                11.107426     0.009616          0.004567                   0.474900        0.774470
 BCHUSDT         27627             8                11.107426     0.008711          0.005561                   0.638422        0.592418
 BNBUSDT         27627             8                11.107426     0.005835          0.003311                   0.567356        0.678107
 BTCUSDT         27627             8                11.107426     0.004917          0.002440                   0.496269        0.753717
DOGEUSDT         27627             8                11.107426     0.009676          0.004949                   0.511427        0.738443
 DOTUSDT         27627             8                11.107426     0.008905          0.003722                   0.417980        0.825293
 ETCUSDT         27627             8                11.107426     0.008368          0.003810                   0.455276        0.792723
 ETHUSDT         27627             8                11.107426     0.006653          0.002953                   0.443964        0.802896
 FILUSDT         27627             8                11.107426     0.010202          0.005118                   0.501676        0.748321
LINKUSDT         27627             8                11.107426     0.009193          0.004369                   0.475298        0.774091
 LTCUSDT         27627             8                11.107426     0.008086          0.004514                   0.558260        0.688346
NEARUSDT         27627             8                11.107426     0.010930          0.005700                   0.521514        0.728023
  OPUSDT         27627             8                11.107426     0.010914          0.004924                   0.451199        0.796419
 SOLUSDT         27627             8                11.107426     0.009285          0.004680                   0.504031        0.745953
 UNIUSDT         27627             8                11.107426     0.010443          0.005982                   0.572765        0.671940
 XRPUSDT         27627             8                11.107426     0.008665          0.005262                   0.607279        0.631213
```

## Comparison To Current Residual Model

```text
                                model  symbols  median_residual_to_raw_vol_ratio  mean_residual_to_raw_vol_ratio   min_r2  median_r2  max_design_condition_number  median_design_condition_number                                                                                  note
           crypto_ff_like_full_sample       20                          0.498973                        0.504966 0.592418   0.751019                 1.110743e+01                       11.107426 Full-sample explanatory regression for diagnostics only; not a walk-forward backtest.
current_btc_eth_market_sector_rolling       20                          0.542540                        0.549948 0.512265   0.705466                 2.542046e+17                      155.822100    Rolling past-only current model quality from residual_factor_neutralization_audit.
```

## Caveats

- Universe has only 20 symbols, so long-short factor portfolios are much noisier than equity FF portfolios.
- MOM and REV are built from the existing residual model, so this is a prototype of the construction idea rather than a clean replacement residualizer.
- The asset regression is full-sample explanatory diagnostics only; it is not walk-forward and not an OOS performance claim.
