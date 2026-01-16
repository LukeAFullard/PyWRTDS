import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.wrtds import Decanter

def generate_plus_data():
    np.random.seed(202)
    dates = pd.date_range(start='2010-01-01', end='2014-12-31', freq='D')
    n = len(dates)
    t = np.arange(n) / 365.25

    # Covariate 1: Q
    log_q = 2.0 + 0.5 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.5, n)
    q = np.exp(log_q)

    # Covariate 2: Temperature (Seasonally driven but with noise)
    # Temp is high in summer (mid-year)
    temp = 20 + 10 * np.sin(2 * np.pi * (t - 0.25)) + np.random.normal(0, 2, n)
    # Ensure strictly positive if we want log?
    # Temp can be 0 or negative. WRTDSplus allows linear extra covariates.
    # We'll use linear Temp.

    # Response: Conc depends on Q and Temp
    # C = Q^0.3 * exp(0.05 * Temp)
    # ln(C) = 0.3 ln(Q) + 0.05 Temp + noise
    log_c = 1.0 + 0.3 * log_q + 0.05 * temp + np.random.normal(0, 0.1, n)
    c = np.exp(log_c)

    df = pd.DataFrame({'Date': dates, 'Q': q, 'Temp': temp, 'Conc': c})

    # Train/Test split
    mask = np.random.rand(n) < 0.2 # 20% training
    df_train = df[mask].copy()
    df_test = df[~mask].copy()

    return df_train, df_test

def run_validation():
    print("--- WRTDSplus Validation ---")

    df_train, df_test = generate_plus_data()
    print(f"Train N: {len(df_train)}, Test N: {len(df_test)}")

    # 1. Standard WRTDS (Ignorant of Temp)
    print("\nFitting Standard WRTDS (Q only)...")
    dec_std = Decanter(df_train, 'Date', 'Conc', 'Q')
    h_params = {'h_time': 7, 'h_cov': 2, 'h_season': 0.5}

    # Train Grid
    dec_std.decant_series(h_params, use_grid=True)
    # Predict Test
    res_std = dec_std.predict(df_test, use_grid=True)
    rmse_std = np.sqrt(np.mean((np.log(res_std['estimated']) - np.log(df_test['Conc']))**2))
    print(f"Standard RMSE (Log): {rmse_std:.4f}")

    # 2. WRTDSplus (With Temp)
    print("\nFitting WRTDSplus (Q + Temp)...")
    # Config: Temp is linear
    extra_cov = [{'col': 'Temp', 'log': False}]
    dec_plus = Decanter(df_train, 'Date', 'Conc', 'Q', extra_covariates=extra_cov)

    # Need h_Temp in h_params
    # Window width for Temp? Range is ~10 to 30. width=5 seems reasonable.
    h_params_plus = h_params.copy()
    h_params_plus['h_Temp'] = 5

    # Train Grid
    # Note: Grid will be 4D (T, Q, Temp).
    # n_extra=7 default.
    dec_plus.decant_series(h_params_plus, use_grid=True)

    # Predict Test
    res_plus = dec_plus.predict(df_test, use_grid=True)
    rmse_plus = np.sqrt(np.mean((np.log(res_plus['estimated']) - np.log(df_test['Conc']))**2))
    print(f"WRTDSplus RMSE (Log): {rmse_plus:.4f}")

    # Report
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    report_path = os.path.join(root_dir, 'reports', 'validation_wrtds_plus.md')

    lines = []
    lines.append("# WRTDSplus Validation Report")
    lines.append("")
    lines.append("## Methodology")
    lines.append("Synthetic data where Conc depends on Discharge (Q) and Temperature. Comparison of standard model (Q only) vs WRTDSplus (Q + Temp).")
    lines.append("")
    lines.append("## Results")
    lines.append("| Model | RMSE (Log) |")
    lines.append("| :--- | :--- |")
    lines.append(f"| Standard WRTDS | {rmse_std:.4f} |")
    lines.append(f"| WRTDSplus | {rmse_plus:.4f} |")
    lines.append("")

    if rmse_plus < rmse_std:
        lines.append("**Conclusion:** SUCCESS. WRTDSplus improved prediction by leveraging Temperature.")
        print("SUCCESS: WRTDSplus improved prediction by leveraging Temperature.")
    else:
        lines.append("**Conclusion:** FAILURE. WRTDSplus did not improve prediction.")
        print("FAILURE: WRTDSplus did not improve prediction.")

    with open(report_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    run_validation()
