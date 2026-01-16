# WRTDS Standard Validation Report

## Overview
This report validates the Python `Decanter` implementation of Standard WRTDS against the established R `EGRET` package.
The validation uses a synthetic daily dataset with known Discharge and Concentration series.

## Methodology
1. **Data Loading:**
   - Daily Records: 4018
   - Calibration Samples: 120
2. **Model Training:**
   - Python: `Decanter.decant_series(h_params, use_grid=True)`
   - R: `modelEstimation(eList)` (Pre-computed in `egret_results.csv`)
3. **Comparison:**
   - RMSE and Percentage RMSE (PRMSE) are calculated for both Estimated Concentration (Daily) and Flow Normalized Concentration.

### Python Implementation Snippet
```python
# Initialize Decanter with Sample and Daily History
dec = Decanter(df_sample, 'Date', 'Conc', 'Q', daily_data=df_daily)

# Fit Grid and Predict
h_params = {'h_time': 7, 'h_cov': 2, 'h_season': 0.5}
dec.decant_series(h_params, use_grid=True)
results = dec.predict(df_daily, use_grid=True)
```

## Comparison Results (Python vs R)
Based on 4018 overlapping days.

| Metric | Estimated (Conc) | Flow Normalized |
| :--- | :--- | :--- |
| **RMSE** | 0.0813 | 0.0322 |
| **PRMSE** | 1.05% | 0.42% |

**Conclusion:** Moderate differences observed.