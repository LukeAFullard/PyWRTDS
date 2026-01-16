# WRTDS-P (Projection) Validation Report

## Methodology
Synthetic data where Conc increases with Discharge (Q).
We compare a Baseline Scenario (Historical Q) vs a High Q Scenario (Q * 1.5).

### Python Implementation Snippet
```python
# 1. Baseline Scenario
res_base = dec.decant_series(h_params, use_grid=True, integration_scenarios=df_daily)

# 2. High Q Scenario
df_high = df_daily.copy()
df_high['Q'] = df_high['Q'] * 1.5
res_high = dec.decant_series(h_params, use_grid=True, integration_scenarios=df_high)
```

## Results
- **Mean Baseline Conc:** 7.6704
- **Mean High Q Conc:**   9.3237

**Conclusion:** SUCCESS. High Q scenario correctly produced higher concentration.