# Generalized Flow Normalization (GFN) Validation Report

## Methodology
Synthetic data with trending Discharge (Q) over 20 years. Concentration is proportional to Q.
Stationary FN should produce a flat line (removing Q trend). GFN should preserve the trend.

## Results
| Method | Slope (Conc/Year) |
| :--- | :--- |
| Stationary | -0.0169 |
| GFN (5-yr) | 0.4601 |

**Conclusion:** SUCCESS. GFN preserved the trend, Stationary removed it.