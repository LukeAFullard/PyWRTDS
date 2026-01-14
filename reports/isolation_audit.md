# WRTDS Isolation Audit: Grid vs Exact

## Objective
To isolate the source of remaining discrepancies between the Python `Decanter` implementation and the R `EGRET` package. Specifically, to determine if differences arise from the regression engine itself or the grid-based approximation used for performance.

## Methodology
Three sets of results were compared for the daily time series (2000-2010):
1.  **PyGrid:** Python implementation using `compute_grid` (14 Q x 177 T nodes) and `RegularGridInterpolator`. (Matches standard EGRET configuration).
2.  **PyExact:** Python implementation calculating Weighted Linear Regression exactly at every daily point (no grid, computationally expensive).
3.  **R (EGRET):** Standard output from EGRET's `modelEstimation` function.

## Findings

| Comparison | RMSE (Estimated) | RMSE (Flow Normalized) | Interpretation |
| :--- | :--- | :--- | :--- |
| **PyGrid vs R** | **0.0211** | **0.0196** | Baseline difference. Very low. |
| **PyExact vs R** | 0.0414 | 0.0349 | **Worse** match than PyGrid. |
| **PyGrid vs PyExact** | 0.0385 | 0.0326 | Error introduced by Grid Approximation. |

## Analysis

1.  **R uses Grid Approximation:** The fact that `PyGrid` matches `R` significantly better than `PyExact` matches `R` confirms that EGRET's output is heavily influenced by its internal grid interpolation. `PyExact` represents the mathematical "truth" of the WRTDS regression equations, but `R` outputs the "grid-approximated" values.
2.  **Grid Error dominates:** The error introduced by using a grid (RMSE ~0.038) is nearly double the difference between the Python Grid and R Grid implementations (RMSE ~0.021).
3.  **High Discharge Extrapolation:** The "Exact" method handles high discharge events (extrapolation) by running the regression locally. The "Grid" method interpolates/extrapolates from the pre-computed surface. Most of the "max difference" events observed in the previous audit correspond to high-Q days where the Grid approximation deviates from the Exact solution.

## Conclusion
The Python implementation (`Decanter`) in Grid mode is a **faithful reproduction of the EGRET software**, including its grid-based approximation behaviors. The remaining small differences (0.021 RMSE) are attributable to:
1.  **Interpolation Implementation:** `scipy` (multilinear) vs R (likely bilinear/approx).
2.  **Optimization:** `numpy.linalg.lstsq` (OLS) vs R `survreg` (MLE), which may differ slightly in convergence at the boundaries.

**Recommendation:** For operational consistency with EGRET, use `use_grid=True` (default). For higher mathematical precision at the cost of computation time, use `use_grid=False`.
