# Model Final Report

## Overall Metrics

| model        | rows_evaluated | mae    | rmse   | mse      | mean_signed_error | within_2_5_lb_accuracy | within_5_lb_accuracy | within_10_lb_accuracy |
| ------------ | -------------- | ------ | ------ | -------- | ----------------- | ---------------------- | -------------------- | --------------------- |
| Baseline     | 420.0          | 21.938 | 38.849 | 1509.263 | 2.098             | 0.2381                 | 0.3738               | 0.5024                |
| XGBoost      | 420.0          | 21.979 | 35.188 | 1238.165 | -5.999            | 0.0738                 | 0.1929               | 0.3714                |
| XGBoost Plus | 420.0          | 23.201 | 36.725 | 1348.691 | -5.283            | 0.0738                 | 0.169                | 0.3548                |

## Notes

- MAE, RMSE, MSE, and mean signed error are measured in pounds.
- Within-threshold accuracy columns report the fraction of predictions within that weight range.
- Lower MAE/RMSE/MSE is better; higher within-threshold accuracy is better.
