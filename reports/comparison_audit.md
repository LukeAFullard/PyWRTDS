# WRTDS Python vs R Comparison Audit

## Overview
This audit compares the results of the Python `Decanter` implementation of WRTDS against the standard R `EGRET` package.
The goal is to verify that the Python implementation closely matches the R version, identify any discrepancies, and report standard error metrics.

## Methodology
- **Input Data:** A sample dataset `test_data_sample.csv` (Calibration) and `test_data_daily.csv` (Daily History).
- **Configuration:** Both implementations were run with `minNumObs=50` (R) / `min_obs=50` (Python) and standard half-window widths (Time=7, Cov=2, Season=0.5).
- **Comparison Script:** `audit_differences.py` merged the results and calculated residuals.

## Results Summary

| Metric | Estimated Concentration | Flow Normalized Concentration |
| :--- | :--- | :--- |
| **RMSE** | 0.0211 mg/L | 0.0196 mg/L |
| **PRMSE** | 0.26% | 0.25% |
| **Mean Relative Difference** | 0.22% | 0.22% |
| **Correlation** | 1.0000 | 1.0000 |

## Discrepancy Analysis
The discrepancies are extremely small, with PRMSE ~0.25%, indicating excellent alignment.
The largest absolute differences occur at high discharge values (e.g., Q > 50). This is expected as:
1. High discharge events are sparse, making local regression sensitive to minor differences in weighting or window expansion.
2. Grid-based interpolation (used in Python for speed) may extrapolate slightly differently than R's method at the edges of the domain.

Specifically, the maximum absolute difference in Estimated Concentration was ~0.13 mg/L (relative error ~0.6%) at a peak discharge event.

## Actions Taken
- **Audit Script:** Created `audit_differences.py` to automate the comparison and visualize residuals.
- **Logging:** Enhanced `src/wrtds.py` to support verbose logging in `fit_local_model`. This allows detailed tracing of window expansion and convergence for specific data points if deeper debugging is required in the future.
- **Verification:** Confirmed that date processing (Decimal Year) matches EGRET's logic (accounting for leap years).

## Conclusion
The Python implementation matches the R version to a very high degree of precision suitable for operational use. The small differences observed are within the expected range for numerical variations between languages and optimization routines (WLS implementations).
