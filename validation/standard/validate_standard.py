import numpy as np
import pandas as pd
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.wrtds import Decanter

def generate_synthetic_data():
    # Load the same data used to generate egret_results.csv if possible
    # We use 'test_data_daily.csv' and 'test_data_sample.csv' which match egret_results.csv
    # based on file exploration.

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    daily_path = os.path.join(root_dir, 'test_data_daily.csv')
    sample_path = os.path.join(root_dir, 'test_data_sample.csv')

    if os.path.exists(daily_path) and os.path.exists(sample_path):
        df_daily = pd.read_csv(daily_path)
        df_sample = pd.read_csv(sample_path)
        # Filter sample to non-nulls
        df_sample = df_sample.dropna(subset=['Conc'])
        return df_daily, df_sample
    else:
        print("Warning: Standard test data not found. Generating fresh synthetic data (Comparison with EGRET CSV will fail).")
        np.random.seed(42)
        dates = pd.date_range(start='2010-01-01', end='2015-12-31', freq='D')
        n = len(dates)
        t = np.arange(n) / 365.25
        log_q = 2.0 + 0.5 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.5, n)
        q = np.exp(log_q)
        log_c = 1.0 + 0.3 * log_q - 0.05 * t + 0.2 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, n)
        c = np.exp(log_c)
        df_daily = pd.DataFrame({'Date': dates, 'Q': q, 'Conc': c})
        sample_mask = np.arange(n) % 14 == 0
        df_sample = df_daily[sample_mask].copy()
        return df_daily, df_sample

def run_validation():
    print("--- Standard WRTDS Validation ---")

    report_lines = []
    report_lines.append("# WRTDS Standard Validation Report")
    report_lines.append("")
    report_lines.append("## Overview")
    report_lines.append("Comparing Python implementation against R 'EGRET' package results (if available).")

    # 1. Data
    df_daily, df_sample = generate_synthetic_data()
    print(f"Data: {len(df_daily)} daily, {len(df_sample)} samples.")

    # 2. Init
    dec = Decanter(df_sample, 'Date', 'Conc', 'Q', daily_data=df_daily)

    # 3. Run Grid
    print("Running Python WRTDS (Grid)...")
    h_params = {'h_time': 7, 'h_cov': 2, 'h_season': 0.5}
    # Must fit grid first
    dec.decant_series(h_params, use_grid=True)
    py_res = dec.predict(df_daily, use_grid=True)

    # 4. Compare with R
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    egret_path = os.path.join(root_dir, 'egret_results.csv')

    if os.path.exists(egret_path):
        print("Loading EGRET results...")
        df_r = pd.read_csv(egret_path)

        # Merge
        # Ensure dates match format
        df_r['Date'] = pd.to_datetime(df_r['Date'])
        # Python results index is preserved from df_daily, which might be range or datetime
        # df_daily['Date'] is what we used.
        df_daily['Date'] = pd.to_datetime(df_daily['Date'])
        py_res['Date'] = df_daily['Date']

        merged = pd.merge(py_res, df_r, on='Date', suffixes=('_py', '_r'))

        if len(merged) == 0:
            msg = "Error: Comparison failed. Dates do not overlap."
            print(msg)
            report_lines.append(f"**Status:** Failed. {msg}")
        else:
            # Metrics
            # Estimated (ConcDay)
            est_rmse = np.sqrt(np.mean((merged['estimated'] - merged['ConcDay'])**2))
            est_prmse = est_rmse / np.mean(merged['ConcDay']) * 100

            # Decanted (FNConc)
            fn_rmse = np.sqrt(np.mean((merged['decanted'] - merged['FNConc'])**2))
            fn_prmse = fn_rmse / np.mean(merged['FNConc']) * 100

            print(f"Estimated RMSE: {est_rmse:.4f} (PRMSE: {est_prmse:.2f}%)")
            print(f"Decanted  RMSE: {fn_rmse:.4f} (PRMSE: {fn_prmse:.2f}%)")

            report_lines.append("## Comparison Results (Python vs R)")
            report_lines.append(f"Based on {len(merged)} overlapping days.")
            report_lines.append("")
            report_lines.append("| Metric | Estimated (Conc) | Flow Normalized |")
            report_lines.append("| :--- | :--- | :--- |")
            report_lines.append(f"| **RMSE** | {est_rmse:.4f} | {fn_rmse:.4f} |")
            report_lines.append(f"| **PRMSE** | {est_prmse:.2f}% | {fn_prmse:.2f}% |")

            if est_prmse < 1.0 and fn_prmse < 1.0:
                report_lines.append("")
                report_lines.append("**Conclusion:** Excellent alignment (<1% Error).")
            else:
                report_lines.append("")
                report_lines.append("**Conclusion:** Moderate differences observed.")
    else:
        print("EGRET results not found.")
        report_lines.append("## Comparison Results")
        report_lines.append("Skipped: `egret_results.csv` not found.")

    # Save Report
    report_path = os.path.join(root_dir, 'reports', 'validation_standard.md')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    run_validation()
