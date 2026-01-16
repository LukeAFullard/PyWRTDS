# WRTDSplus Validation Report

## Overview
WRTDSplus allows the inclusion of additional covariates (e.g., Temperature, Groundwater Level) into the regression model.
This validation checks if adding a known driver (Temperature) reduces prediction error compared to the standard Q-only model.

## Methodology
Synthetic data generated with Conc ~ Q + Temp. Models trained on 20% of data, tested on 80%.

### Python Implementation Snippet
```python
# Configure Extra Covariate
extra_cov = [{'col': 'Temp', 'log': False}]
dec_plus = Decanter(df_train, 'Date', 'Conc', 'Q', extra_covariates=extra_cov)

# Set window width for Temp
h_params_plus = {'h_time': 7, 'h_cov': 2, 'h_season': 0.5, 'h_Temp': 5}

# Train Grid (N-Dimensional) and Predict
dec_plus.decant_series(h_params_plus, use_grid=True)
res_plus = dec_plus.predict(df_test, use_grid=True)
```

**Note:** This feature is a Python-specific extension. No direct comparison with standard R EGRET is available.

## Results
| Model | RMSE (Log) |
| :--- | :--- |
| Standard WRTDS | 0.1435 |
| WRTDSplus | 0.1038 |

**Conclusion:** SUCCESS. WRTDSplus improved prediction by leveraging Temperature.