# P26 Residual / Label 自动审计

该审计在组合回放前执行；任一硬门槛失败时脚本立即终止，不生成候选 baseline。

```text
                        check        value    threshold direction status
         past_only_alpha_beta 1.329492e-14 1.000000e-10       max   pass
past_only_portfolio_risk_beta 1.327827e-13 1.000000e-10       max   pass
      alpha_residual_identity 2.853620e-16 1.000000e-12       max   pass
       terminal_label_formula 2.775558e-16 1.000000e-10       max   pass
      condition_number_median 2.062349e+01 1.000000e+02       max   pass
         bnb_condition_number 2.041357e+01 1.000000e+03       max   pass
  beta_mean_abs_hourly_change 2.319197e-03 3.000000e-03       max   pass
            residual_coverage 9.915024e-01 9.500000e-01       min   pass
               label_coverage 9.855435e-01 9.500000e-01       min   pass
           centered_rank_mean 4.614364e-17 1.000000e-10       max   pass
 slow_label_composite_formula 1.665335e-16 1.000000e-12       max   pass
   alpha_sector_factor_absent 1.000000e+00 1.000000e+00       min   pass
```

完整诊断见 `audit_diagnostics.csv`。
