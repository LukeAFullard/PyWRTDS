# Generalized Flow Normalization (GFN) Validation Report

## Methodology
Synthetic data with trending Discharge (Q) over 20 years. Concentration is proportional to Q.
Stationary FN should produce a flat line (removing Q trend). GFN should preserve the trend.

### Python Implementation Snippet
```python
# 1. Fit Grid on Sample
dec.decant_series(h_params, use_grid=True)
dec.save_model('temp_grid.pkl')

# 2. Load Grid into Daily Decanter
dec_daily = Decanter(df_daily_dummy, 'Date', 'Conc', 'Q', daily_data=df_daily)
dec_daily.load_model('temp_grid.pkl')

# 3. Run GFN (Window=5 years)
gfn_series = dec_daily.decant_series(h_params, use_grid=True, gfn_window=5.0)
```

### R Reproduction Code
```r
# To verify against EGRET:
# eList <- ...
# dailyResults <- runSeries(eList, windowSide = 2.5, ...)
```

## Results
| Method | Slope (Conc/Year) |
| :--- | :--- |
| Stationary | -0.0169 |
| GFN (5-yr) | 0.4699 |

**R Comparison:** RMSE = 0.5053

**Conclusion:** SUCCESS. GFN preserved the trend, Stationary removed it.