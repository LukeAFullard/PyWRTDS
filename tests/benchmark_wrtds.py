import sys
import os
import time
import numpy as np
import pandas as pd

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from wrtds import Decanter

def generate_data(n_years=5):
    np.random.seed(42)
    dates = pd.date_range(start='2010-01-01', periods=n_years*365, freq='D')
    n = len(dates)

    # Covariate: sine wave + noise
    c0 = np.exp(np.sin(np.linspace(0, n_years*2*np.pi, n)) + np.random.normal(0, 0.5, n))

    # Response: Trend * Covariate * Seasonality * Noise
    trend = np.linspace(10, 12, n)
    t0 = trend * c0 * np.random.normal(1, 0.1, n)

    df = pd.DataFrame({'Date': dates, 'Sales': t0, 'AdSpend': c0})
    # Ensure positive
    df['Sales'] = df['Sales'].abs() + 0.1
    df['AdSpend'] = df['AdSpend'].abs() + 0.1
    return df

def run_benchmark():
    print("Generating 10 years of daily data...")
    df = generate_data(n_years=10) # 3650 points
    dec = Decanter(df, date_col='Date', response_col='Sales', covariate_col='AdSpend')

    print(f"Running Exact Method on {len(df)} points...")
    start_time = time.time()
    # Using wider windows to ensure we hit data
    # Intentionally using 'exact' loop (default implementation)
    res_exact = dec.decant_series(h_params={'h_time': 1, 'h_cov': 2, 'h_season': 0.5})
    end_time = time.time()

    print(f"Exact Method Time: {end_time - start_time:.4f} seconds")

    print(f"Running Grid Method on {len(df)} points...")
    start_time_grid = time.time()
    res_grid = dec.decant_series(h_params={'h_time': 1, 'h_cov': 2, 'h_season': 0.5}, use_grid=True, grid_config={'n_t': 50, 'n_q': 10})
    end_time_grid = time.time()
    print(f"Grid Method Time: {end_time_grid - start_time_grid:.4f} seconds")

    # Comparison
    # Convert NaNs to 0 for rough comparison or ignore
    arr_exact = np.array(res_exact)
    arr_grid = np.array(res_grid)

    mask = ~np.isnan(arr_exact) & ~np.isnan(arr_grid)
    diff = np.abs(arr_exact[mask] - arr_grid[mask])
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)

    print(f"Mean Absolute Difference: {mean_diff:.5f}")
    print(f"Max Absolute Difference: {max_diff:.5f}")

if __name__ == "__main__":
    run_benchmark()
