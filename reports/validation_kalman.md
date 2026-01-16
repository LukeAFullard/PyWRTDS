# WRTDS-Kalman Validation Report

## Methodology
Used synthetic AR(1) autocorrelated data. Trained on sparse subset (every 10th day) using Standard WRTDS, then applied Kalman correction (rho=0.9) to predict test days.

### Python Implementation Snippet
```python
# Initialize Decanter with combined data (Test rows are NaN)
dec_combined = Decanter(df_combined, 'Date', 'Conc', 'Q')
dec_combined.decant_series(h_params, use_grid=True)

# Get Standard Predictions
est_log = dec_combined.get_estimated_series(h_params, use_grid=True)

# Apply Kalman Correction
kalman_log = dec_combined.add_kalman_correction(est_log, rho=0.9)
```

## Results
| Model | RMSE (Log) |
| :--- | :--- |
| Standard WRTDS | 0.2089 |
| WRTDS-Kalman | 0.1555 |

**Conclusion:** SUCCESS. Kalman correction improved prediction accuracy.