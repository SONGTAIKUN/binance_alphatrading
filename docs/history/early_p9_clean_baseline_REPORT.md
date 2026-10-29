# P9 Clean Baseline Package

## Design

- P9 is not a fresh formula search. It packages the already nested/OOS P7b selections with two global risk overlays.
- Overlay 1: any selected increment using `K01_deriv_crowding_contra` falls back to the P3d fold selection.
- Overlay 2: candidate rerank weights are capped at `0.10` after the P7b validation choice.
- This keeps the P7b per-fold train/validation selection record while removing the P8-identified fragile K01 and oversized candidate weights.

## Summary
```text
 periods  total_return  gross_total_return_before_cost   sharpe  max_drawdown  mean_rank_ic  mean_top_bottom_label_spread  mean_period_return  mean_gross_period_return  positive_period_rate  positive_gross_period_rate  mean_turnover  total_turnover  cost_drag  mean_period_cost  target_apply_rate  mean_abs_beta_btc  mean_abs_beta_eth  mean_abs_beta_market  mean_symbol_concentration  fold_count  positive_fold_rate  gross_max_drawdown  mean_gross_exposure  mean_max_abs_weight  max_abs_weight  total_stopped_weight  stopped_period_rate  max_abs_beta_btc  max_abs_beta_eth  max_abs_beta_market  mean_optimizer_success  max_optimizer_success  mean_optimizer_exposure_error  max_optimizer_exposure_error  mean_optimizer_target_gross  max_optimizer_target_gross  mean_optimizer_actual_gross  max_optimizer_actual_gross  fil_total_contribution  mean_target_turnover  total_target_turnover  mean_turnover_saved  total_turnover_saved  mean_target_gross_exposure  total_target_gross_exposure  turnover_reduction  gross_exposure_retention  mean_short_guard_removed                      run  horizon  net_corr_to_p3d  net_corr_to_p7b  net_corr_to_p4_nested
     172      0.155395                        0.199236 2.211163     -0.017877      0.027561                      0.003196            0.000849                  0.001066              0.552326                    0.581395       0.361314       62.146037   0.037288          0.000217                1.0       3.385727e-15       2.321514e-15          1.036098e-14                   0.019352          23            0.695652           -0.016528             0.270424             0.084603           0.125                   0.0                  0.0      1.950627e-13      7.492347e-14         4.433329e-13                     1.0                    1.0                   1.158352e-14                  4.433398e-13                     0.580226                    0.850596                     0.270424                    0.769368                0.031314              0.361314              62.146037                  0.0                   0.0                    0.270424                    46.512899                 0.0                       1.0                  2.168605 p9_clean_drop_k01_cap010       72         0.938793         0.963637               0.736082
```

## Comparison
```text
 periods  total_return  gross_total_return_before_cost   sharpe  max_drawdown  mean_rank_ic  mean_top_bottom_label_spread  mean_period_return  mean_gross_period_return  positive_period_rate  positive_gross_period_rate  mean_turnover  total_turnover  cost_drag  mean_period_cost  target_apply_rate  mean_abs_beta_btc  mean_abs_beta_eth  mean_abs_beta_market  mean_symbol_concentration  fold_count  positive_fold_rate  gross_max_drawdown  mean_gross_exposure  mean_max_abs_weight  max_abs_weight  total_stopped_weight  stopped_period_rate  max_abs_beta_btc  max_abs_beta_eth  max_abs_beta_market  mean_optimizer_success  max_optimizer_success  mean_optimizer_exposure_error  max_optimizer_exposure_error  mean_optimizer_target_gross  max_optimizer_target_gross  mean_optimizer_actual_gross  max_optimizer_actual_gross  fil_total_contribution  mean_target_turnover  total_target_turnover  mean_turnover_saved  total_turnover_saved  mean_target_gross_exposure  total_target_gross_exposure  turnover_reduction  gross_exposure_retention  mean_short_guard_removed                      run  horizon  net_corr_to_p3d  net_corr_to_p7b  net_corr_to_p4_nested  p3d_total_return  delta_vs_p3d_total_return  p3d_gross_total_return_before_cost  delta_vs_p3d_gross_total_return_before_cost  p3d_sharpe  delta_vs_p3d_sharpe  p3d_max_drawdown  delta_vs_p3d_max_drawdown  p3d_gross_max_drawdown  delta_vs_p3d_gross_max_drawdown  p3d_mean_turnover  delta_vs_p3d_mean_turnover  p3d_cost_drag  delta_vs_p3d_cost_drag  p3d_positive_fold_rate  delta_vs_p3d_positive_fold_rate  p7b_total_return  delta_vs_p7b_total_return  p7b_gross_total_return_before_cost  delta_vs_p7b_gross_total_return_before_cost  p7b_sharpe  delta_vs_p7b_sharpe  p7b_max_drawdown  delta_vs_p7b_max_drawdown  p7b_gross_max_drawdown  delta_vs_p7b_gross_max_drawdown  p7b_mean_turnover  delta_vs_p7b_mean_turnover  p7b_cost_drag  delta_vs_p7b_cost_drag  p7b_positive_fold_rate  delta_vs_p7b_positive_fold_rate
     172      0.155395                        0.199236 2.211163     -0.017877      0.027561                      0.003196            0.000849                  0.001066              0.552326                    0.581395       0.361314       62.146037   0.037288          0.000217                1.0       3.385727e-15       2.321514e-15          1.036098e-14                   0.019352          23            0.695652           -0.016528             0.270424             0.084603           0.125                   0.0                  0.0      1.950627e-13      7.492347e-14         4.433329e-13                     1.0                    1.0                   1.158352e-14                  4.433398e-13                     0.580226                    0.850596                     0.270424                    0.769368                0.031314              0.361314              62.146037                  0.0                   0.0                    0.270424                    46.512899                 0.0                       1.0                  2.168605 p9_clean_drop_k01_cap010       72         0.938793         0.963637               0.736082          0.126883                   0.028512                            0.169051                                     0.030185    1.804529             0.406634         -0.021446                   0.003569               -0.019268                         0.002739           0.354271                    0.007043       0.036773                0.000514                0.565217                         0.130435          0.139616                   0.015779                            0.182177                                     0.017059    1.998336             0.212827         -0.017877              -1.249001e-16               -0.016528                     6.245005e-17           0.355667                    0.005647       0.036705                0.000583                0.695652                              0.0
```

## Variant Summary
```text
 periods  total_return  gross_total_return_before_cost   sharpe  max_drawdown  mean_rank_ic  mean_top_bottom_label_spread  mean_period_return  mean_gross_period_return  positive_period_rate  positive_gross_period_rate  mean_turnover  total_turnover  cost_drag  mean_period_cost  target_apply_rate  mean_abs_beta_btc  mean_abs_beta_eth  mean_abs_beta_market  mean_symbol_concentration  fold_count  positive_fold_rate  gross_max_drawdown  mean_gross_exposure  mean_max_abs_weight  max_abs_weight  total_stopped_weight  stopped_period_rate  max_abs_beta_btc  max_abs_beta_eth  max_abs_beta_market  mean_optimizer_success  max_optimizer_success  mean_optimizer_exposure_error  max_optimizer_exposure_error  mean_optimizer_target_gross  max_optimizer_target_gross  mean_optimizer_actual_gross  max_optimizer_actual_gross  fil_total_contribution  mean_target_turnover  total_target_turnover  mean_turnover_saved  total_turnover_saved  mean_target_gross_exposure  total_target_gross_exposure  turnover_reduction  gross_exposure_retention  mean_short_guard_removed                  variant  delta_vs_p3d_total_return  delta_vs_p7b_total_return  delta_vs_p3d_sharpe  delta_vs_p7b_sharpe
     172      0.155395                        0.199236 2.211163     -0.017877      0.027561                      0.003196            0.000849                  0.001066              0.552326                    0.581395       0.361314       62.146037   0.037288          0.000217                1.0       3.385727e-15       2.321514e-15          1.036098e-14                   0.019352          23            0.695652           -0.016528             0.270424             0.084603           0.125                   0.0                  0.0      1.950627e-13      7.492347e-14         4.433329e-13                     1.0                    1.0                   1.158352e-14                  4.433398e-13                     0.580226                    0.850596                     0.270424                    0.769368                0.031314              0.361314              62.146037                  0.0                   0.0                    0.270424                    46.512899                 0.0                       1.0                  2.168605 p9_clean_drop_k01_cap010                   0.028512               1.577937e-02             0.406634             0.212827
     171      0.149644                        0.193077 2.140611     -0.017877      0.032166                      0.002899            0.000825                  0.001042              0.555556                    0.590643       0.361873       61.880302   0.037128          0.000217                1.0       3.788162e-15       2.363162e-15          1.298774e-14                   0.019645          23            0.695652           -0.016528             0.272764             0.085374           0.125                   0.0                  0.0      1.950627e-13      7.492347e-14         5.044192e-13                     1.0                    1.0                   1.421162e-14                  5.043743e-13                     0.583203                    0.850596                     0.272764                    0.769368                0.021755              0.361873              61.880302                  0.0                   0.0                    0.272764                    46.642594                 0.0                       1.0                  2.157895                 drop_k01                   0.022761               1.002808e-02             0.336082             0.142274
     173      0.144895                        0.188296 2.060483     -0.017877      0.030279                      0.002584            0.000791                  0.001007              0.549133                    0.583815       0.358856       62.082106   0.037249          0.000215                1.0       3.715591e-15       3.202361e-15          1.136920e-14                   0.019215          23            0.695652           -0.016528             0.269495             0.084299           0.125                   0.0                  0.0      1.950627e-13      2.382869e-13         4.433329e-13                     1.0                    1.0                   1.349863e-14                  4.433398e-13                     0.580706                    0.850596                     0.269495                    0.769368                0.031440              0.358856              62.082106                  0.0                   0.0                    0.269495                    46.622701                 0.0                       1.0                  2.173410                   cap010                   0.018011               5.278628e-03             0.255954             0.062147
     175      0.144354                        0.186988 2.036375     -0.017163      0.034298                      0.002122            0.000780                  0.000989              0.542857                    0.577143       0.348729       61.027536   0.036617          0.000209                1.0       5.936753e-15       4.657814e-15          1.968225e-14                   0.018282          23            0.652174           -0.016407             0.263998             0.082993           0.125                   0.0                  0.0      2.038474e-13      3.065950e-13         4.340139e-13                     1.0                    1.0                   2.077952e-14                  4.340348e-13                     0.580043                    0.850596                     0.263998                    0.575769                0.018941              0.348729              61.027536                  0.0                   0.0                    0.263998                    46.199584                 0.0                       1.0                  2.171429                 scale075                   0.017470               4.737751e-03             0.231847             0.038039
     172      0.139616                        0.182177 1.998336     -0.017877      0.034333                      0.002191            0.000769                  0.000983              0.563953                    0.598837       0.355667       61.174785   0.036705          0.000213                1.0       4.048147e-15       2.200988e-15          1.550859e-14                   0.019134          23            0.695652           -0.016528             0.269067             0.084945           0.125                   0.0                  0.0      1.950627e-13      7.492347e-14         5.044192e-13                     1.0                    1.0                   1.654308e-14                  5.043743e-13                     0.582931                    0.850596                     0.269067                    0.575769                0.022369              0.355667              61.174785                  0.0                   0.0                    0.269067                    46.279542                 0.0                       1.0                  2.162791          p7b_full_replay                   0.012733               8.326673e-17             0.193807             0.000000
```

## Selected By Fold
```text
 fold                       p4_spec p3d_formula_name deriv_formula_name  candidate_weight_long  candidate_weight_short  high_risk_candidate_cap  hard_accept  fallback_to_p3d  validation_total_return  validation_sharpe                                                                                                                                                                                                                                              deriv_formula
    1        a00_b08_c02_lf45_ema30        agpt_0035        __P4_ONLY__                   0.00                    0.00                     0.00        False             True                 0.041541           4.330422                                                                                                                                                                                        P3d fallback: P9 drops K01_deriv_crowding_contra selected increment
    2  a00_b06_c04_lf45_comp2_ema35        agpt_0017       p7_agpt_0008                   0.10                    0.10                     0.05         True            False                 0.074353           4.707471                                                                                                                                                                                                                   EMA_SLOW(P6B(J04_top_trader_contra),168)
    3  a00_b10_c00_lf55_comp2_ema35        agpt_0311       p7_agpt_0083                   0.05                    0.05                     0.03         True            False                 0.029688           3.324968                                                                                                                                                           MUL(0.50*EMA_SLOW(P6B(I01_premium_contra_168),168),EMA_SLOW(P6B(K05_deriv_p3d_diversifier),168))
    4  a00_b06_c04_lf50_comp2_ema35        agpt_0034       p7_agpt_0150                   0.10                    0.10                     0.10         True            False                 0.055467           6.050554 ADD(ADD(MUL(0.50*EMA_SLOW(P6B(C02_dollar_volume_mom),336),EMA_SLOW(P6B(I05_mom_basis_quality),168)),EMA_SLOW(P6B(H03_slow_mom_funding_penalty),336)),NEG(ADD(EMA_SLOW(P6B(H05_carry_unwind_contra),168),0.70*EMA_SLOW(P6B(H05_carry_unwind_contra),168))))
    5  a02_b02_c06_lf55_comp2_ema35        agpt_0282        __P4_ONLY__                   0.00                    0.00                     0.00        False             True                 0.049840           2.385923                                                                                                                                                                                        P3d fallback: P9 drops K01_deriv_crowding_contra selected increment
    6        a02_b08_c00_lf45_ema35        agpt_0264        __P4_ONLY__                   0.00                    0.00                     0.00        False             True                 0.172009           5.648066                                                                                                                                                                                        P3d fallback: P9 drops K01_deriv_crowding_contra selected increment
    7 a00_b10_c00_lf55_c09b11_ema35        agpt_0362       p7_agpt_0047                   0.10                    0.10                     0.05         True            False                 0.060365           6.116689                                                                                                                                                                                                        NEG(0.35*EMA_SLOW(P6B(H01_funding_contra_168),168))
    8 a02_b06_c02_lf50_c09b11_ema35        agpt_0145       p7_agpt_0102                   0.10                    0.10                     0.10         True            False                 0.085013           5.732193                                                                                                                                                                          SUB(EMA_SLOW(P6B(D04_low_vov_momentum),168),0.70*P6B(E03_basket_diversifier_mom))
    9        a04_b06_c00_lf55_ema30      __P4_ONLY__        __P4_ONLY__                   0.00                    0.00                     0.00        False             True                 0.081966           4.605260                                                                                                                                                                                  P3d fallback: no derivative increment passed validation hard-accept rules
   10        a04_b06_c00_lf55_ema30        agpt_0276       p7_agpt_0102                   0.10                    0.10                     0.10         True            False                 0.076940           4.671940                                                                                                                                                                          SUB(EMA_SLOW(P6B(D04_low_vov_momentum),168),0.70*P6B(E03_basket_diversifier_mom))
   11        a00_b08_c02_lf50_ema30        agpt_0273       p7_agpt_0115                   0.10                    0.10                     0.10         True            False                 0.049093           6.084012                                                                                                               ADD(1.20*EMA_SLOW(P6B(K05_deriv_p3d_diversifier),168),0.70*NEG(MUL(0.50*P6B(H05_carry_unwind_contra),0.50*P6B(E03_basket_diversifier_mom))))
   12        a00_b08_c02_lf55_ema30        agpt_0015       p7_agpt_0006                   0.05                    0.05                     0.03         True            False                 0.074052          10.493057                                                                                                                                                                                                               EMA_SLOW(P6B(K05_deriv_p3d_diversifier),336)
   13        a00_b08_c02_lf50_ema30        agpt_0361       p7_agpt_0019                   0.05                    0.05                     0.03         True            False                 0.057007           9.498275                                                                                                                                                                                                                                P6B(K02_uncrowded_slow_mom)
   14 a00_b00_c10_lf55_c09b11_ema35      __P4_ONLY__        __P4_ONLY__                   0.00                    0.00                     0.00        False             True                 0.038103           5.498119                                                                                                                                                                                  P3d fallback: no derivative increment passed validation hard-accept rules
   15 a00_b00_c10_lf45_c09b11_ema35        agpt_0030       p7_agpt_0034                   0.10                    0.10                     0.10         True            False                 0.053351           8.704972                                                                                                                                                                                                                                 P6B(C02_dollar_volume_mom)
   16        a00_b02_c08_lf50_ema35        agpt_0125       p7_agpt_0021                   0.10                    0.10                     0.10         True            False                 0.052295           5.539271                                                                                                                                                                                                                  EMA_SLOW(P6B(K02_uncrowded_slow_mom),336)
   17 a02_b08_c00_lf45_c09b11_ema35        agpt_0312       p7_agpt_0111                   0.10                    0.10                     0.10         True            False                 0.031120           7.690081               MUL(ADD(EMA_SLOW(P6B(H05_carry_unwind_contra),168),0.50*0.20*EMA_SLOW(P6B(E03_basket_diversifier_mom),336)),SUB(0.50*EMA_SLOW(P6B(C05_low_capacity_penalty),336),0.70*SUB(P6B(H05_carry_unwind_contra),0.35*P6B(C05_low_capacity_penalty))))
   18        a04_b06_c00_lf55_ema30        agpt_0270       p7_agpt_0129                   0.05                    0.05                     0.03         True            False                 0.121898           3.177613                                                                                                                                                                SUB(EMA_SLOW(P6B(J04_top_trader_contra),336),0.35*EMA_SLOW(P6B(J04_top_trader_contra),168))
   19  a06_b04_c00_lf50_comp2_ema35        agpt_0244        __P4_ONLY__                   0.00                    0.00                     0.00        False             True                 0.148785           4.732933                                                                                                                                                                                        P3d fallback: P9 drops K01_deriv_crowding_contra selected increment
   20  a02_b08_c00_lf55_comp2_ema35        agpt_0129       p7_agpt_0111                   0.10                    0.10                     0.05         True            False                 0.020148           5.938998               MUL(ADD(EMA_SLOW(P6B(H05_carry_unwind_contra),168),0.50*0.20*EMA_SLOW(P6B(E03_basket_diversifier_mom),336)),SUB(0.50*EMA_SLOW(P6B(C05_low_capacity_penalty),336),0.70*SUB(P6B(H05_carry_unwind_contra),0.35*P6B(C05_low_capacity_penalty))))
   21        a06_b00_c04_lf50_ema30        agpt_0213       p7_agpt_0111                   0.10                    0.10                     0.10         True            False                 0.046878           3.932451               MUL(ADD(EMA_SLOW(P6B(H05_carry_unwind_contra),168),0.50*0.20*EMA_SLOW(P6B(E03_basket_diversifier_mom),336)),SUB(0.50*EMA_SLOW(P6B(C05_low_capacity_penalty),336),0.70*SUB(P6B(H05_carry_unwind_contra),0.35*P6B(C05_low_capacity_penalty))))
   22        a06_b00_c04_lf50_ema30      __P4_ONLY__        __P4_ONLY__                   0.00                    0.00                     0.00        False             True                 0.053521           4.468906                                                                                                                                                                                  P3d fallback: no derivative increment passed validation hard-accept rules
   23        a06_b00_c04_lf50_ema35        agpt_0240        __P4_ONLY__                   0.00                    0.00                     0.00        False             True                 0.073855           4.708033                                                                                                                                                                                        P3d fallback: P9 drops K01_deriv_crowding_contra selected increment
```

## Primitive Usage
```text
                   primitive                family                        score_col  selected_fold_count  selected_weight_sum  p6b_primary_mean_ic  p6b_turnover_72h  p6b_mean_abs_p3d_corr
  E03_basket_diversifier_mom E_crowding_dispersion    p6_E03_basket_diversifier_mom                    6                 0.60             0.032075          0.435026               0.232386
     H05_carry_unwind_contra       H_funding_carry      p6b_H05_carry_unwind_contra                    5                 0.50             0.042941          0.271708               0.171296
    C05_low_capacity_penalty    C_liquidity_volume      p6_C05_low_capacity_penalty                    3                 0.30             0.029592          0.361894               0.240550
   K05_deriv_p3d_diversifier   K_derivatives_combo    p6b_K05_deriv_p3d_diversifier                    3                 0.20             0.045055          0.396231               0.236737
        D04_low_vov_momentum  D_volatility_quality          p6_D04_low_vov_momentum                    2                 0.20             0.031589          0.366858               0.242409
       C02_dollar_volume_mom    C_liquidity_volume         p6_C02_dollar_volume_mom                    2                 0.20             0.022540          0.365043               0.232274
       J04_top_trader_contra         J_oi_crowding        p6b_J04_top_trader_contra                    2                 0.15             0.048766          0.126201               0.169983
      K02_uncrowded_slow_mom   K_derivatives_combo       p6b_K02_uncrowded_slow_mom                    2                 0.15             0.034417          0.367594               0.237287
      H01_funding_contra_168       H_funding_carry       p6b_H01_funding_contra_168                    1                 0.10             0.042941          0.271708               0.171296
H03_slow_mom_funding_penalty       H_funding_carry p6b_H03_slow_mom_funding_penalty                    1                 0.10             0.037441          0.367823               0.230491
       I05_mom_basis_quality       I_basis_premium        p6b_I05_mom_basis_quality                    1                 0.10             0.028916          0.393657               0.229191
      I01_premium_contra_168       I_basis_premium       p6b_I01_premium_contra_168                    1                 0.05             0.021450          0.273454               0.198239
   K01_deriv_crowding_contra   K_derivatives_combo    p6b_K01_deriv_crowding_contra                    0                 0.00             0.045213          0.298552               0.173520
```

## Cost Sensitivity
```text
              experiment  tau_days  one_way_cost_bps  total_return  max_drawdown  mean_period_return  positive_period_rate  cost_drag_sum  p3d_total_return  p3d_max_drawdown  delta_vs_p3d_total_return  delta_vs_p3d_max_drawdown  p7b_total_return  p7b_max_drawdown  delta_vs_p7b_total_return  delta_vs_p7b_max_drawdown
p9_clean_drop_k01_cap010       0.0               0.0      0.199236     -0.016528            0.001066              0.581395       0.000000          0.169051         -0.019268                   0.030185                   0.002739          0.182177         -0.016528                   0.017059               6.245005e-17
p9_clean_drop_k01_cap010       0.0               6.0      0.155395     -0.017877            0.000849              0.552326       0.037288          0.126883         -0.021446                   0.028512                   0.003569          0.139616         -0.017877                   0.015779              -1.249001e-16
p9_clean_drop_k01_cap010       0.0              10.0      0.127056     -0.018776            0.000705              0.517442       0.062146          0.099614         -0.022896                   0.027442                   0.004120          0.112091         -0.019589                   0.014965               8.131726e-04
p9_clean_drop_k01_cap010       0.0              15.0      0.092600     -0.019898            0.000524              0.488372       0.093219          0.066447         -0.024706                   0.026153                   0.004807          0.078611         -0.022234                   0.013989               2.336029e-03
p9_clean_drop_k01_cap010       0.0              20.0      0.059190     -0.022960            0.000343              0.482558       0.124292          0.034274         -0.030128                   0.024916                   0.007168          0.046131         -0.025559                   0.013059               2.598685e-03
p9_clean_drop_k01_cap010       0.0              30.0     -0.004618     -0.054684           -0.000018              0.447674       0.186438         -0.027210         -0.066153                   0.022593                   0.011469         -0.015943         -0.056163                   0.011325               1.479293e-03
```

## Variant Cost Sensitivity
```text
              experiment  tau_days  one_way_cost_bps  total_return  max_drawdown  mean_period_return  positive_period_rate  cost_drag_sum                  variant
p9_clean_drop_k01_cap010       0.0               0.0      0.199236     -0.016528            0.001066              0.581395       0.000000 p9_clean_drop_k01_cap010
p9_clean_drop_k01_cap010       0.0               6.0      0.155395     -0.017877            0.000849              0.552326       0.037288 p9_clean_drop_k01_cap010
p9_clean_drop_k01_cap010       0.0              20.0      0.059190     -0.022960            0.000343              0.482558       0.124292 p9_clean_drop_k01_cap010
p9_clean_drop_k01_cap010       0.0              30.0     -0.004618     -0.054684           -0.000018              0.447674       0.186438 p9_clean_drop_k01_cap010
         p7b_full_replay       0.0               0.0      0.182177     -0.016528            0.000983              0.598837       0.000000          p7b_full_replay
         p7b_full_replay       0.0               6.0      0.139616     -0.017877            0.000769              0.563953       0.036705          p7b_full_replay
         p7b_full_replay       0.0              20.0      0.046131     -0.025559            0.000271              0.494186       0.122350          p7b_full_replay
         p7b_full_replay       0.0              30.0     -0.015943     -0.056163           -0.000084              0.459302       0.183524          p7b_full_replay
                  cap010       0.0               0.0      0.188296     -0.016528            0.001007              0.583815       0.000000                   cap010
                  cap010       0.0               6.0      0.144895     -0.017877            0.000791              0.549133       0.037249                   cap010
                  cap010       0.0              20.0      0.049648     -0.023778            0.000289              0.479769       0.124164                   cap010
                  cap010       0.0              30.0     -0.013528     -0.054684           -0.000070              0.445087       0.186246                   cap010
                drop_k01       0.0               0.0      0.193077     -0.016528            0.001042              0.590643       0.000000                 drop_k01
                drop_k01       0.0               6.0      0.149644     -0.017877            0.000825              0.555556       0.037128                 drop_k01
                drop_k01       0.0              20.0      0.054308     -0.025559            0.000318              0.491228       0.123761                 drop_k01
                drop_k01       0.0              30.0     -0.008943     -0.057958           -0.000044              0.461988       0.185641                 drop_k01
                scale075       0.0               0.0      0.186988     -0.016407            0.000989              0.577143       0.000000                 scale075
                scale075       0.0               6.0      0.144354     -0.017163            0.000780              0.542857       0.036617                 scale075
                scale075       0.0              20.0      0.050695     -0.023778            0.000291              0.485714       0.122055                 scale075
                scale075       0.0              30.0     -0.011506     -0.053902           -0.000057              0.457143       0.183083                 scale075
```

## Regime Summary
```text
                regime_type          regime  period_count  total_return  gross_total_return   sharpe  max_drawdown  cost_drag_sum  mean_turnover  mean_gross_exposure  positive_period_rate  mean_rank_ic
                crash_state          normal           125      0.133121            0.164495 2.554993     -0.021529       0.027348       0.364634             0.262186              0.592000      0.044230
                crash_state        crash_7d            47      0.019657            0.029834 1.208434     -0.021211       0.009940       0.352486             0.292333              0.446809     -0.016770
           liquidity_bucket        high_liq            63      0.067128            0.081988 2.156485     -0.017877       0.013848       0.366356             0.269435              0.523810      0.033177
           liquidity_bucket         mid_liq            53      0.049326            0.061826 3.074665     -0.006374       0.011857       0.372848             0.271125              0.566038      0.027627
           liquidity_bucket         low_liq            56      0.031820            0.043828 1.761841     -0.011366       0.011583       0.344726             0.270873              0.571429      0.021181
               market_trend          up_30d            74      0.119392            0.136769 3.472212     -0.020128       0.015435       0.347640             0.264806              0.648649      0.039991
               market_trend        down_30d            98      0.032163            0.054952 1.014552     -0.012213       0.021852       0.371640             0.274666              0.479592      0.018175
          market_vol_bucket        high_vol            57      0.057683            0.070485 2.107605     -0.026417       0.012050       0.352330             0.263113              0.508772      0.009828
          market_vol_bucket         mid_vol            54      0.057321            0.071028 3.005919     -0.008571       0.012897       0.398047             0.284322              0.537037      0.038938
          market_vol_bucket         low_vol            61      0.033162            0.045980 1.721029     -0.024367       0.012341       0.337191             0.264952              0.606557      0.034060
 residual_dispersion_bucket high_resid_disp            53      0.069849            0.082850 2.736157     -0.021727       0.012099       0.380470             0.284483              0.547170      0.008471
 residual_dispersion_bucket  mid_resid_disp            56      0.040912            0.052586 2.330136     -0.009143       0.011164       0.332259             0.228062              0.571429      0.050404
 residual_dispersion_bucket  low_resid_disp            63      0.037515            0.052154 1.653174     -0.021280       0.014025       0.371026             0.296251              0.539683      0.023317
volume_concentration_bucket       mid_vconc            56      0.112908            0.127370 5.010441     -0.007020       0.012939       0.385079             0.294096              0.625000      0.045761
volume_concentration_bucket       low_vconc            56      0.037371            0.050560 1.401261     -0.044987       0.012652       0.376544             0.275268              0.553571      0.022788
volume_concentration_bucket      high_vconc            60      0.000777            0.012552 0.068544     -0.015068       0.011697       0.324919             0.243809              0.483333      0.015029
```

## Side Contribution
```text
 side  gross_contribution_sum  mean_contribution  positive_position_rate  mean_abs_weight  max_abs_weight  position_count  symbol_count  abs_contribution_share
 long                0.099185           0.000087                0.459790         0.020329           0.125            1144            20                0.541019
short                0.084145           0.000044                0.513795         0.012106           0.125            1921            20                0.458981
```

## Asset Contribution
```text
  symbol  gross_contribution_sum  long_contribution_sum  short_contribution_sum  mean_contribution  positive_position_rate  mean_abs_weight  max_abs_weight  position_count  abs_contribution_share
LINKUSDT                0.060216               0.049359                0.010857           0.000421                0.510490         0.019081        0.125000             143                0.140406
 XRPUSDT                0.039244               0.008001                0.031243           0.000256                0.509804         0.018362        0.125000             153                0.091505
 ARBUSDT                0.035746              -0.007629                0.043375           0.000226                0.468354         0.017478        0.125000             158                0.083350
 APTUSDT                0.032081              -0.008253                0.040334           0.000198                0.469136         0.010321        0.125000             162                0.074803
 FILUSDT                0.031314               0.017111                0.014203           0.000203                0.467532         0.009205        0.110496             154                0.073016
 SOLUSDT                0.029283               0.029963               -0.000680           0.000205                0.517483         0.015852        0.122560             143                0.068279
NEARUSDT                0.027915              -0.005817                0.033732           0.000170                0.524390         0.011134        0.125000             164                0.065089
 BTCUSDT                0.012887               0.017620               -0.004733           0.000095                0.555556         0.025407        0.125000             135                0.030049
 LTCUSDT                0.012352               0.005080                0.007272           0.000077                0.456250         0.017106        0.125000             160                0.028801
 BCHUSDT                0.011240               0.016617               -0.005377           0.000069                0.537037         0.012912        0.125000             162                0.026208
 ETCUSDT                0.010252               0.023846               -0.013594           0.000065                0.430380         0.012081        0.125000             158                0.023905
 BNBUSDT                0.003570               0.014292               -0.010722           0.000024                0.493151         0.025346        0.125000             146                0.008325
 ETHUSDT               -0.001731               0.020206               -0.021937          -0.000012                0.517730         0.023070        0.125000             141                0.004035
DOGEUSDT               -0.002657               0.008828               -0.011485          -0.000018                0.520548         0.015171        0.125000             146                0.006195
AVAXUSDT               -0.006743              -0.014804                0.008061          -0.000040                0.520958         0.017210        0.125000             167                0.015722
 DOTUSDT               -0.007087              -0.005755               -0.001332          -0.000046                0.483660         0.014594        0.125000             153                0.016525
 UNIUSDT               -0.019837              -0.002364               -0.017473          -0.000126                0.506329         0.013117        0.125000             158                0.046254
ATOMUSDT               -0.023629              -0.026866                0.003237          -0.000148                0.487500         0.010232        0.125000             160                0.055095
  OPUSDT               -0.029304              -0.017007               -0.012297          -0.000182                0.478261         0.006979        0.125000             161                0.068329
 ADAUSDT               -0.031784              -0.023243               -0.008541          -0.000225                0.425532         0.012446        0.122932             141                0.074110
```

## Fold Delta Vs P3d
```text
 fold  p9_total_return  p9_gross_total_return_before_cost  p9_sharpe  p9_max_drawdown  p9_mean_turnover p9_formula_name  p3d_total_return  p3d_gross_total_return_before_cost  p3d_sharpe  p3d_max_drawdown  p3d_mean_turnover p3d_formula_name  delta_total_return  delta_gross_total_return_before_cost  delta_sharpe  delta_max_drawdown  delta_mean_turnover
    1        -0.002631                          -0.001569  -2.453167        -0.003115          0.221753     __P4_ONLY__         -0.002631                           -0.001569   -2.453167         -0.003115           0.221753        agpt_0035       -5.334275e-17                         -4.987330e-17  0.000000e+00       -1.994932e-17         0.000000e+00
    2        -0.009313                          -0.007747  -3.496208        -0.008687          0.291896    p7_agpt_0008         -0.009313                           -0.007747   -3.496208         -0.008687           0.291896        agpt_0017        4.122217e-11                          1.273297e-12  3.379497e-08        2.565827e-10        -7.401989e-09
    3         0.005933                           0.007625   2.949018        -0.007213          0.350305    p7_agpt_0083          0.005933                            0.007625    2.949018         -0.007213           0.350305        agpt_0311        1.232996e-09                          1.218886e-09 -1.161352e-07       -3.075478e-09        -3.177404e-09
    4         0.003533                           0.005292   1.971834        -0.001088          0.417092    p7_agpt_0150         -0.008203                           -0.006562   -7.598459         -0.003585           0.393046        agpt_0034        1.173526e-02                          1.185359e-02  9.570293e+00        2.497315e-03         2.404591e-02
    5         0.048604                           0.050467  13.538245         0.000000          0.372069     __P4_ONLY__          0.048604                            0.050467   13.538245          0.000000           0.372069        agpt_0282        3.469447e-17                          7.632783e-17  0.000000e+00        0.000000e+00         5.551115e-17
    6         0.024371                           0.026508   3.929953        -0.001304          0.435662     __P4_ONLY__          0.024371                            0.026508    3.929953         -0.001304           0.435662        agpt_0264        6.938894e-17                          2.428613e-17  0.000000e+00       -4.575333e-17         5.551115e-17
    7         0.004830                           0.005795   1.619179        -0.010953          0.267354    p7_agpt_0047          0.004830                            0.005795    1.619182         -0.010953           0.267354        agpt_0362       -1.014477e-08                         -1.041885e-08 -3.263461e-06       -3.082448e-09        -7.379157e-08
    8        -0.001095                           0.000993  -0.415246        -0.008277          0.497426    p7_agpt_0102         -0.003154                           -0.001222   -1.315837         -0.007868           0.461189        agpt_0145        2.058947e-03                          2.214765e-03  9.005917e-01       -4.088176e-04         3.623742e-02
    9        -0.003048                          -0.001518  -1.760959        -0.006755          0.365099     __P4_ONLY__         -0.003048                           -0.001518   -1.760959         -0.006755           0.365099      __P4_ONLY__       -6.765422e-17                         -2.320193e-17  0.000000e+00       -9.367507e-17         0.000000e+00
   10         0.009076                           0.010952   4.821279        -0.002849          0.442914    p7_agpt_0102          0.007729                            0.009561    4.781594         -0.002849           0.432827        agpt_0276        1.347187e-03                          1.391844e-03  3.968492e-02        1.981670e-09         1.008746e-02
   11         0.015307                           0.017285   6.959344        -0.002078          0.406230    p7_agpt_0115          0.015727                            0.017600    7.217711         -0.002078           0.384689        agpt_0273       -4.196348e-04                         -3.151445e-04 -2.583671e-01        4.343912e-13         2.154074e-02
   12         0.010960                           0.013395   3.895789        -0.007020          0.573910    p7_agpt_0006          0.007056                            0.009783    1.979272         -0.011707           0.563651        agpt_0015        3.903122e-03                          3.611847e-03  1.916517e+00        4.687554e-03         1.025938e-02
   13        -0.017625                          -0.016179 -15.738610        -0.017877          0.407129    p7_agpt_0019         -0.009660                           -0.008158   -3.869377         -0.011664           0.315360        agpt_0361       -7.964177e-03                         -8.021666e-03 -1.186923e+01       -6.213499e-03         9.176868e-02
   14         0.012433                           0.013367   4.691574        -0.004218          0.256811     __P4_ONLY__          0.012433                            0.013367    4.691574         -0.004218           0.256811      __P4_ONLY__        9.194034e-17                          9.367507e-17  0.000000e+00       -3.556183e-17         5.551115e-17
   15        -0.006692                          -0.004939  -2.413229        -0.013630          0.419721    p7_agpt_0034         -0.010083                           -0.008800   -4.083311         -0.016071           0.359019        agpt_0030        3.390526e-03                          3.861327e-03  1.670082e+00        2.441393e-03         6.070208e-02
   16         0.001116                           0.002177   0.847379        -0.004371          0.196142    p7_agpt_0021         -0.005462                           -0.004394   -3.887022         -0.007463           0.198621        agpt_0125        6.578147e-03                          6.570361e-03  4.734400e+00        3.091425e-03        -2.478649e-03
   17         0.006433                           0.007956   4.703910        -0.001765          0.360352    p7_agpt_0111          0.006433                            0.007956    4.703915         -0.001765           0.360352        agpt_0312       -4.362494e-09                         -4.471533e-09 -4.671048e-06       -5.114041e-09        -2.447508e-08
   18         0.013027                           0.014997   3.138723        -0.005612          0.360391    p7_agpt_0129          0.014265                            0.016071    3.475652         -0.004436           0.330121        agpt_0270       -1.238562e-03                         -1.074512e-03 -3.369295e-01       -1.175981e-03         3.027009e-02
   19         0.006823                           0.009364   1.918002        -0.004229          0.467915     __P4_ONLY__          0.006823                            0.009364    1.918002         -0.004229           0.467915        agpt_0244        9.801188e-17                          9.540979e-17  0.000000e+00       -5.117434e-17         5.551115e-17
   20        -0.002451                          -0.001643  -3.035296        -0.001772          0.224782    p7_agpt_0111         -0.004389                           -0.003646  -11.971419         -0.002402           0.248535        agpt_0129        1.938591e-03                          2.002851e-03  8.936123e+00        6.300661e-04        -2.375332e-02
   21         0.001205                           0.002689   1.419774        -0.002377          0.352702    p7_agpt_0111         -0.002501                           -0.000797   -1.729530         -0.006964           0.406395        agpt_0213        3.706589e-03                          3.485896e-03  3.149303e+00        4.586980e-03        -5.369252e-02
   22         0.007420                           0.009340   2.417826        -0.004907          0.352779     __P4_ONLY__          0.007420                            0.009340    2.417826         -0.004907           0.352779      __P4_ONLY__        6.071532e-17                          5.204170e-17  0.000000e+00       -1.734723e-17         0.000000e+00
   23         0.018553                           0.019708  10.524021         0.000000          0.270685     __P4_ONLY__          0.018553                            0.019708   10.524021          0.000000           0.270685        agpt_0240        8.326673e-17                          8.326673e-17  0.000000e+00        0.000000e+00         0.000000e+00
```

## Fold Delta Vs P7b
```text
 fold  p9_total_return  p9_gross_total_return_before_cost  p9_sharpe  p9_max_drawdown  p9_mean_turnover p9_formula_name  p7b_total_return  p7b_gross_total_return_before_cost  p7b_sharpe  p7b_max_drawdown  p7b_mean_turnover p7b_formula_name  delta_total_return  delta_gross_total_return_before_cost  delta_sharpe  delta_max_drawdown  delta_mean_turnover
    1        -0.002631                          -0.001569  -2.453167        -0.003115          0.221753     __P4_ONLY__         -0.004474                           -0.003439   -3.583308         -0.003814           0.216449     p7_agpt_0002        1.842880e-03                          1.869749e-03  1.130141e+00        6.986118e-04         5.304130e-03
    2        -0.009313                          -0.007747  -3.496208        -0.008687          0.291896    p7_agpt_0008         -0.009313                           -0.007747   -3.496208         -0.008687           0.291896     p7_agpt_0008       -2.949030e-17                         -3.469447e-18  0.000000e+00       -6.765422e-17         0.000000e+00
    3         0.005933                           0.007625   2.949018        -0.007213          0.350305    p7_agpt_0083          0.005933                            0.007625    2.949018         -0.007213           0.350305     p7_agpt_0083        2.949030e-17                          8.847090e-17  0.000000e+00       -1.734723e-18         0.000000e+00
    4         0.003533                           0.005292   1.971834        -0.001088          0.417092    p7_agpt_0150          0.002576                            0.004315    1.684147         -0.000019           0.482093     p7_agpt_0150        9.569973e-04                          9.761407e-04  2.876871e-01       -1.068482e-03        -6.500110e-02
    5         0.048604                           0.050467  13.538245         0.000000          0.372069     __P4_ONLY__          0.040997                            0.042838    8.762795         -0.000026           0.328339     p7_agpt_0003        7.606621e-03                          7.629881e-03  4.775449e+00        2.648114e-05         4.372962e-02
    6         0.024371                           0.026508   3.929953        -0.001304          0.435662     __P4_ONLY__          0.024371                            0.026508    3.929947         -0.001304           0.435663     p7_agpt_0101        3.876048e-08                          3.848354e-08  6.072898e-06       -4.117264e-10        -7.179468e-08
    7         0.004830                           0.005795   1.619179        -0.010953          0.267354    p7_agpt_0047          0.004830                            0.005795    1.619179         -0.010953           0.267354     p7_agpt_0047        9.540979e-18                          9.107298e-17  2.220446e-16       -8.847090e-17         0.000000e+00
    8        -0.001095                           0.000993  -0.415246        -0.008277          0.497426    p7_agpt_0102         -0.001095                            0.000993   -0.415246         -0.008277           0.497426     p7_agpt_0102        5.444910e-10                          5.683752e-10  2.273029e-07        0.000000e+00         5.471182e-09
    9        -0.003048                          -0.001518  -1.760959        -0.006755          0.365099     __P4_ONLY__         -0.003048                           -0.001518   -1.760959         -0.006755           0.365099      __P4_ONLY__       -6.765422e-17                         -2.320193e-17  0.000000e+00       -9.367507e-17         0.000000e+00
   10         0.009076                           0.010952   4.821279        -0.002849          0.442914    p7_agpt_0102          0.005464                            0.007271    3.743143         -0.002849           0.427772     p7_agpt_0102        3.612651e-03                          3.681891e-03  1.078136e+00       -5.637851e-17         1.514214e-02
   11         0.015307                           0.017285   6.959344        -0.002078          0.406230    p7_agpt_0115          0.015307                            0.017285    6.959346         -0.002078           0.406230     p7_agpt_0115       -4.395785e-09                         -4.604634e-09 -2.828887e-06       -3.906580e-10        -4.140086e-08
   12         0.010960                           0.013395   3.895789        -0.007020          0.573910    p7_agpt_0006          0.010960                            0.013395    3.895789         -0.007020           0.573910     p7_agpt_0006        4.163336e-17                          4.510281e-17  0.000000e+00       -5.811324e-17         0.000000e+00
   13        -0.017625                          -0.016179 -15.738610        -0.017877          0.407129    p7_agpt_0019         -0.017625                           -0.016179  -15.738610         -0.017877           0.407129     p7_agpt_0019       -8.673617e-17                         -1.734723e-17  0.000000e+00       -3.469447e-17         5.551115e-17
   14         0.012433                           0.013367   4.691574        -0.004218          0.256811     __P4_ONLY__          0.012433                            0.013367    4.691574         -0.004218           0.256811      __P4_ONLY__        9.194034e-17                          9.367507e-17  0.000000e+00       -3.556183e-17         5.551115e-17
   15        -0.006692                          -0.004939  -2.413229        -0.013630          0.419721    p7_agpt_0034         -0.007139                           -0.005465   -2.278485         -0.017163           0.400771     p7_agpt_0034        4.464457e-04                          5.261720e-04 -1.347434e-01        3.533417e-03         1.895020e-02
   16         0.001116                           0.002177   0.847379        -0.004371          0.196142    p7_agpt_0021          0.001116                            0.002177    0.847378         -0.004371           0.196142     p7_agpt_0021        1.076036e-09                          1.230288e-09  8.508620e-07       -3.635470e-10         2.839100e-08
   17         0.006433                           0.007956   4.703910        -0.001765          0.360352    p7_agpt_0111          0.006433                            0.007956    4.703910         -0.001765           0.360352     p7_agpt_0111       -6.061618e-13                          2.301940e-12 -5.886003e-10        8.391725e-17         6.884368e-10
   18         0.013027                           0.014997   3.138723        -0.005612          0.360391    p7_agpt_0129          0.013027                            0.014997    3.138723         -0.005612           0.360391     p7_agpt_0129        3.642919e-17                          3.816392e-17  0.000000e+00       -3.035766e-17         0.000000e+00
   19         0.006823                           0.009364   1.918002        -0.004229          0.467915     __P4_ONLY__          0.007195                            0.009350    2.069546         -0.002300           0.396628     p7_agpt_0158       -3.715355e-04                          1.413068e-05 -1.515441e-01       -1.929423e-03         7.128739e-02
   20        -0.002451                          -0.001643  -3.035296        -0.001772          0.224782    p7_agpt_0111         -0.002451                           -0.001643   -3.035296         -0.001772           0.224782     p7_agpt_0111       -4.466913e-17                         -4.445229e-17 -4.440892e-16        0.000000e+00         2.775558e-17
   21         0.001205                           0.002689   1.419774        -0.002377          0.352702    p7_agpt_0111          0.001205                            0.002689    1.419795         -0.002377           0.352702     p7_agpt_0111       -2.136114e-08                         -1.968156e-08 -2.140939e-05        8.091226e-10         4.065167e-07
   22         0.007420                           0.009340   2.417826        -0.004907          0.352779     __P4_ONLY__          0.007420                            0.009340    2.417826         -0.004907           0.352779      __P4_ONLY__        6.071532e-17                          5.204170e-17  0.000000e+00       -1.734723e-17         0.000000e+00
   23         0.018553                           0.019708  10.524021         0.000000          0.270685     __P4_ONLY__          0.018553                            0.019708   10.524022          0.000000           0.270685     p7_agpt_0045        1.006747e-08                          1.033937e-08 -1.467407e-06        0.000000e+00         6.175569e-08
```

## Bootstrap Checks
```text
      metric  fold_count  observed_mean_delta  bootstrap_q05  bootstrap_q50  bootstrap_q95  positive_delta_fold_rate  sign_flip_p_value_mean_le_0
p9_minus_p3d          23             0.001089      -0.000056       0.001054       0.002275                  0.695652                       0.0771
p9_minus_p7b          23             0.000613       0.000106       0.000568       0.001261                  0.652174                       0.0201
```

## Start Fold Sensitivity Vs P3d
```text
 start_fold      p3d       p9  delta_total_return
          1 0.126883 0.155395        2.851203e-02
          2 0.129856 0.158444        2.858725e-02
          3 0.140478 0.169334        2.885600e-02
          4 0.133751 0.162437        2.868579e-02
          5 0.143128 0.158345        1.521714e-02
          6 0.090142 0.104654        1.451181e-02
          7 0.064207 0.078373        1.416655e-02
          8 0.059092 0.073190        1.409847e-02
          9 0.062442 0.074366        1.192402e-02
         10 0.065691 0.077651        1.196047e-02
         11 0.057517 0.067958        1.044104e-02
         12 0.041143 0.051857        1.071393e-02
         13 0.033848 0.040454        6.606298e-03
         14 0.043933 0.059121        1.518805e-02
         15 0.031113 0.046114        1.500153e-02
         16 0.041615 0.053162        1.154718e-02
         17 0.047336 0.051988        4.652460e-03
         18 0.040641 0.045264        4.622725e-03
         19 0.026005 0.031823        5.817711e-03
         20 0.019052 0.024830        5.778285e-03
         21 0.023544 0.027348        3.803371e-03
         22 0.026111 0.026111       -4.440892e-16
         23 0.018553 0.018553        2.220446e-16
```

## Start Fold Sensitivity Vs P7b
```text
 start_fold      p7b       p9  delta_total_return
          1 0.139616 0.155395        1.577937e-02
          2 0.144738 0.158444        1.370581e-02
          3 0.155499 0.169334        1.383466e-02
          4 0.148684 0.162437        1.375306e-02
          5 0.145733 0.158345        1.261204e-02
          6 0.100611 0.104654        4.043580e-03
          7 0.074426 0.078373        3.947338e-03
          8 0.069262 0.073190        3.928365e-03
          9 0.070434 0.074366        3.932671e-03
         10 0.073707 0.077651        3.944695e-03
         11 0.067872 0.067958        8.606362e-05
         12 0.051772 0.051857        8.477063e-05
         13 0.040370 0.040454        8.385166e-05
         14 0.059035 0.059121        8.535602e-05
         15 0.046030 0.046114        8.430779e-05
         16 0.053551 0.053162       -3.886464e-04
         17 0.052377 0.051988       -3.882144e-04
         18 0.045650 0.045264       -3.857328e-04
         19 0.032203 0.031823       -3.807726e-04
         20 0.024830 0.024830       -1.173567e-08
         21 0.027348 0.027348       -1.176450e-08
         22 0.026111 0.026111        1.014217e-08
         23 0.018553 0.018553        1.006747e-08
```

## Symbol Dropout Vs P3d
```text
symbol_dropped      p3d       p9  delta_total_return
      LINKUSDT 0.096276 0.086995           -0.009281
       XRPUSDT 0.092915 0.110682            0.017766
       ARBUSDT 0.092090 0.114600            0.022510
       APTUSDT 0.102445 0.119059            0.016614
       FILUSDT 0.092456 0.119649            0.027194
       SOLUSDT 0.094041 0.121926            0.027886
      NEARUSDT 0.104730 0.123406            0.018675
       BTCUSDT 0.103520 0.140537            0.037017
       LTCUSDT 0.098065 0.140655            0.042590
       BCHUSDT 0.110067 0.142277            0.032210
       ETCUSDT 0.095307 0.143272            0.047964
       BNBUSDT 0.140569 0.150776            0.010207
       ETHUSDT 0.112458 0.156956            0.044498
      DOGEUSDT 0.132308 0.158161            0.025852
      AVAXUSDT 0.126562 0.163185            0.036624
       DOTUSDT 0.142919 0.163455            0.020536
       UNIUSDT 0.136264 0.178318            0.042054
      ATOMUSDT 0.168574 0.182709            0.014135
        OPUSDT 0.155166 0.189774            0.034608
       ADAUSDT 0.164834 0.192264            0.027430
```

## Symbol Dropout Vs P7b
```text
symbol_dropped      p7b       p9  delta_total_return
      LINKUSDT 0.080334 0.086995            0.006661
       XRPUSDT 0.097735 0.110682            0.012947
       ARBUSDT 0.101784 0.114600            0.012816
       APTUSDT 0.100572 0.119059            0.018487
       FILUSDT 0.114295 0.119649            0.005354
       SOLUSDT 0.118561 0.121926            0.003366
      NEARUSDT 0.105722 0.123406            0.017683
       BTCUSDT 0.125055 0.140537            0.015482
       LTCUSDT 0.125277 0.140655            0.015378
       BCHUSDT 0.141012 0.142277            0.001264
       ETCUSDT 0.114277 0.143272            0.028994
       BNBUSDT 0.130988 0.150776            0.019788
       ETHUSDT 0.133526 0.156956            0.023430
      DOGEUSDT 0.140004 0.158161            0.018156
      AVAXUSDT 0.153844 0.163185            0.009341
       DOTUSDT 0.142591 0.163455            0.020864
       UNIUSDT 0.165776 0.178318            0.012542
      ATOMUSDT 0.171605 0.182709            0.011104
        OPUSDT 0.177950 0.189774            0.011824
       ADAUSDT 0.160768 0.192264            0.031496
```

## Reality Summary
```text
                         check     value  pass
           p9_vs_p3d_net_delta  0.028512  True
        p9_vs_p3d_sharpe_delta  0.406634  True
           p9_vs_p7b_net_delta  0.015779  True
        p9_vs_p7b_sharpe_delta  0.212827  True
               p9_20bps_return  0.059190  True
               p9_30bps_return -0.004618  True
            p9_short_abs_share  0.458981  True
    bootstrap_delta_vs_p3d_q05 -0.000056  True
    bootstrap_delta_vs_p7b_q05  0.000106  True
    start_fold_positive_vs_p3d  0.956522  True
    start_fold_positive_vs_p7b  0.739130  True
symbol_dropout_positive_vs_p3d  0.950000  True
symbol_dropout_positive_vs_p7b  1.000000  True
```
