# Uncertainty (Bootstrap) Validation Report

## Methodology
Ran Block and Wild Bootstrap (n=10) on synthetic data to generate confidence intervals (5th-95th percentile).

### Python Implementation Snippet
```python
# Run Bootstrap (Block method)
res_block = dec.bootstrap_uncertainty(h_params, n_bootstraps=10, method='block', use_grid=True)

# Run Bootstrap (Wild method)
res_wild = dec.bootstrap_uncertainty(h_params, n_bootstraps=10, method='wild', use_grid=True)
```

### R Reproduction Code
```r
# To verify against EGRETci:
# library(EGRETci)
# ciLower <- ciCalculations(eList, ..., method='block')
```

## Results
- **Block Bootstrap:** Consistent
- **Wild Bootstrap:** Consistent

**Conclusion:** SUCCESS. All bootstrap intervals are logically consistent.