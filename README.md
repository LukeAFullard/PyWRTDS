# PyWRTDS

A Python implementation of **Weighted Regressions on Time, Discharge, and Season (WRTDS)**, a statistical method for removing covariate effects (flow normalization) and extracting trends from time series data.

Originally developed for hydrology (river water quality), this framework is generalizable to any domain where a response variable is driven by:
1.  **Trend:** A long-term signal (the target).
2.  **Covariate:** A high-frequency driver (e.g., Ad Spend, Temperature, Discharge).
3.  **Season:** A cyclic forcing function.

## Features

*   **Locally Weighted Regression:** Fits a unique model for every time step to capture non-stationary relationships.
*   **Flow Normalization:** Integrates over the probability distribution of the covariate to remove its stochastic influence.
*   **Grid Optimization:** Uses interpolation surfaces to scale to large datasets ($O(1)$ lookup per point).
*   **Generalized Flow Normalization (GFN):** Handles covariates that have their own long-term trends.
*   **WRTDS-Kalman:** Applies AR(1) filtering to residuals to restore system memory/autocorrelation.
*   **WRTDSplus:** Supports multiple covariates (e.g., Temperature, Lagged Flow) beyond the standard model.
*   **WRTDS-P (Projection):** Enables "What-If" scenario analysis by integrating over simulated covariate distributions.
*   **Uncertainty Analysis:** Provides confidence intervals via Block or Wild Bootstrap.
*   **Note:** Handling of censored data (Tobit regression) is planned but not yet implemented.

## Installation

Requires `numpy`, `pandas`, and `scipy`.

```bash
pip install numpy pandas scipy
```

## Quick Start

```python
import pandas as pd
import numpy as np
from src.wrtds import Decanter

# 1. Prepare Data
# Data must be daily (or regular frequency).
# Response/Covariate must be strictly positive (for log transformation).
df = pd.DataFrame({
    'Date': pd.date_range('2020-01-01', periods=1000, freq='D'),
    'Sales': np.random.uniform(10, 100, 1000),    # Response (t0)
    'AdSpend': np.random.uniform(100, 500, 1000)  # Covariate (c0)
})

# 2. Initialize
dec = Decanter(df, date_col='Date', response_col='Sales', covariate_col='AdSpend')

# 3. Run WRTDS (Standard)
# h_time (years), h_cov (log units), h_season (years)
cleaned_series = dec.decant_series(h_params={'h_time': 1, 'h_cov': 2, 'h_season': 0.5})

# 4. Run Optimized (Grid)
# Recommended for large datasets
cleaned_grid = dec.decant_series(h_params={'h_time': 1, 'h_cov': 2, 'h_season': 0.5},
                                 use_grid=True, grid_config={'n_t': 50, 'n_q': 15})

# 5. Run Generalized Flow Normalization (GFN)
# Use a 1-year moving window for covariate integration (handles trending AdSpend)
cleaned_gfn = dec.decant_series(h_params={'h_time': 1, 'h_cov': 2, 'h_season': 0.5},
                                gfn_window=1.0)
```

## Advanced Usage

### WRTDSplus (Multiple Covariates)

If your response depends on multiple drivers (e.g., Ad Spend AND Temperature):

```python
# Add extra column
df['Temp'] = np.random.uniform(20, 30, 1000)

# Initialize with extra_covariates config
dec = Decanter(df, 'Date', 'Sales', 'AdSpend',
               extra_covariates=[{'col': 'Temp', 'log': False}])

# Run Decant (Pass window width for Temp via h_params)
# Grid optimization is disabled for WRTDSplus
cleaned_plus = dec.decant_series(h_params={'h_time': 1, 'h_cov': 2, 'h_season': 0.5, 'h_Temp': 5})
```

### WRTDS-P (Projection/Forecasting)

Perform scenario analysis (e.g., "What if Ad Spend doubles?"):

```python
# Create a scenario DataFrame
# Must contain columns for all covariates
scenario_df = pd.DataFrame({
    'AdSpend': np.ones(1000) * 1000  # High Spend Scenario
})

# Run Decant with the scenario
# The result represents the expected Sales under the scenario conditions
projected_series = dec.decant_series(h_params={'h_time': 1, 'h_cov': 2, 'h_season': 0.5},
                                     integration_scenarios=scenario_df)
```

### Uncertainty Analysis (Bootstrap)

Estimate 90% confidence intervals around the trend:

```python
# Returns DataFrame with ['mean', 'p05', 'p95']
# method='wild' is robust to changing variance (heteroscedasticity)
uncertainty_df = dec.bootstrap_uncertainty(h_params={'h_time': 1, 'h_cov': 2, 'h_season': 0.5},
                                           n_bootstraps=100,
                                           block_size=30,
                                           method='wild')
```

### WRTDS-Kalman

If you need the estimated time series (not the normalized one) to respect short-term memory (autocorrelation):

```python
# 1. Get the raw model predictions (log space) - Implementation dependent
# ...
# 2. Apply Correction
# dec.add_kalman_correction(estimated_log_series, rho=0.9)
```

## Reference

See `reports/WRTDS_Analysis_Report.md` for the theoretical background.
See `reports/WRTDS_Implementation_Plan.md` for the implementation details.
