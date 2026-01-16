import numpy as np
import pandas as pd
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.wrtds import Decanter

def generate_synthetic_data():
    np.random.seed(42)
    dates = pd.date_range(start='2010-01-01', end='2015-12-31', freq='D')
    n = len(dates)
    t = np.arange(n) / 365.25

    # Covariate: Discharge (Q)
    # Log-normal, seasonal
    log_q = 2.0 + 0.5 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.5, n)
    q = np.exp(log_q)

    # Response: Conc
    # Model: ln(C) = 1.0 + 0.3*ln(Q) - 0.05*t + 0.2*sin(2pi*t) + noise
    log_c = 1.0 + 0.3 * log_q - 0.05 * t + 0.2 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, n)
    c = np.exp(log_c)

    df = pd.DataFrame({'Date': dates, 'Q': q, 'Conc': c})

    # Subsample for calibration (bi-weekly)
    sample_mask = np.arange(n) % 14 == 0
    df_sample = df[sample_mask].copy()

    return df, df_sample

def run_validation():
    print("--- Standard WRTDS Validation ---")

    # 1. Generate Data
    df_daily, df_sample = generate_synthetic_data()
    print(f"Generated {len(df_daily)} daily records and {len(df_sample)} samples.")

    # 2. Initialize Decanter
    # We provide df_daily as daily_history for accurate Flow Normalization
    dec = Decanter(df_sample, 'Date', 'Conc', 'Q', daily_data=df_daily)

    # 3. Run Grid Method (Standard)
    print("\nRunning WRTDS (Grid Mode)...")
    # Using defaults: h_time=7, h_cov=2, h_season=0.5
    h_params = {'h_time': 7, 'h_cov': 2, 'h_season': 0.5}
    grid_res = dec.decant_series(h_params, use_grid=True)

    # 4. Run Exact Method (Small Subset)
    # Exact mode is slow (O(N^2)), so we only validate it runs on a small subset of samples.
    print("Running WRTDS (Exact Mode) - First 10 samples...")
    df_sample_small = df_sample.head(10).copy()
    dec_short = Decanter(df_sample_small, 'Date', 'Conc', 'Q', daily_data=df_daily)
    exact_res = dec_short.decant_series(h_params, use_grid=False)
    print(f"Exact Result (Head 5): {exact_res[:5]}")

    # 5. Generate Daily Series
    print("Generating Daily Series using Grid...")
    daily_results = dec.predict(df_daily, use_grid=True)

    print("\nResults Head:")
    print(daily_results.head())

    # Save
    out_file = os.path.join(os.path.dirname(__file__), 'standard_results.csv')
    daily_results.to_csv(out_file)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    run_validation()
