# Long History Fixed P16 Replay

This run tests the locked P16/P15c baseline on the extended common 20-symbol history.

Important interpretation note:

- This is a fixed-policy long-history stress replay.
- Folds before the original locked selection window map to source fold 1, because those periods were not part of the original nested AlphaGPT/P15c parameter-selection process.
- Therefore this is useful for stability and regime stress, but it is not a fresh strict nested parameter-selection OOS result for the added 2023-2024H1 history.

Inputs:

- panel: `experiments/baseline_locked_v1/inputs/long_history/p16_long_replay_panel.csv.gz`
- first panel timestamp: `2023-03-23 15:00:00+00:00`
- last panel timestamp: `2026-06-17 07:00:00+00:00`
- symbols: `20`
- rows: `567220`

Stress folds:

- first test start: `2023-10-15 08:00:00+00:00`
- last test end: `2026-06-01 08:00:00+00:00`
- folds: `32`
- pre-original folds using mapped source fold 1: `9`

Default 6bps metrics:

- total_return: 5.0135%
- gross_total_return_before_cost: 10.8278%
- sharpe: 0.419
- max_drawdown: -9.7130%
- mean_turnover: 0.3808

Cost sensitivity:

- 0bps: return 10.83%, Sharpe 0.842, maxDD -7.68%
- 6bps: return 5.01%, Sharpe 0.419, maxDD -9.71%
- 10bps: return 1.31%, Sharpe 0.135, maxDD -11.04%
- 15bps: return -3.14%, Sharpe -0.220, maxDD -12.68%
- 20bps: return -7.40%, Sharpe -0.576, maxDD -14.28%
- 30bps: return -15.36%, Sharpe -1.289, maxDD -17.56%

15bps segment summary:

- full_long_history: net -3.14%, gross 10.83%, Sharpe -0.220, maxDD -12.68%, mean_turnover 0.381
- pre_original_2023_2024H1: net -6.46%, gross -2.75%, Sharpe -1.379, maxDD -7.78%, mean_turnover 0.365
- original_window_2024_07_plus: net 3.55%, gross 13.96%, Sharpe 0.480, maxDD -5.31%, mean_turnover 0.387

Fold stability at 15bps:

- positive fold rate: 46.88%
- worst 15bps fold: fold 2, 2023-11-14 08:00:00+00:00 to 2023-12-14 08:00:00+00:00, return -3.58%, maxDD -4.23%
