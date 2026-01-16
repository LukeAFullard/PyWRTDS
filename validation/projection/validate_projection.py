import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.wrtds import Decanter

def generate_data():
    np.random.seed(505)
    dates = pd.date_range(start='2010-01-01', end='2011-12-31', freq='D')
    n = len(dates)
    t = np.arange(n) / 365.25

    log_q = 2.0 + np.random.normal(0, 0.5, n)
    q = np.exp(log_q)

    # Positive relationship C ~ Q
    log_c = 1.0 + 0.5 * log_q + np.random.normal(0, 0.1, n)
    c = np.exp(log_c)

    df = pd.DataFrame({'Date': dates, 'Q': q, 'Conc': c})
    mask = np.arange(n) % 10 == 0
    df_sample = df[mask].copy()

    return df, df_sample

def run_validation():
    print("--- WRTDS-P (Projection) Validation ---")
    df_daily, df_sample = generate_data()

    dec = Decanter(df_sample, 'Date', 'Conc', 'Q', daily_data=df_daily)
    h_params = {'h_time': 7, 'h_cov': 2, 'h_season': 0.5}

    # 1. Baseline Scenario (Actual History)
    # We pass the daily dataframe as the scenario
    print("Running Baseline Projection...")
    res_base = dec.decant_series(h_params, use_grid=True, integration_scenarios=df_daily)

    # 2. High Q Scenario (+50% Discharge)
    print("Running High Q Scenario (+50%)...")
    df_high = df_daily.copy()
    df_high['Q'] = df_high['Q'] * 1.5
    res_high = dec.decant_series(h_params, use_grid=True, integration_scenarios=df_high)

    mean_base = np.mean(res_base)
    mean_high = np.mean(res_high)

    print(f"Mean Baseline Conc: {mean_base:.4f}")
    print(f"Mean High Q Conc:   {mean_high:.4f}")

    # Since C ~ Q^0.5, higher Q -> higher C

    # Report
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    report_path = os.path.join(root_dir, 'reports', 'validation_projection.md')

    lines = []
    lines.append("# WRTDS-P (Projection) Validation Report")
    lines.append("")
    lines.append("## Methodology")
    lines.append("Synthetic data where Conc increases with Discharge (Q).")
    lines.append("We compare a Baseline Scenario (Historical Q) vs a High Q Scenario (Q * 1.5).")
    lines.append("")
    lines.append("### Python Implementation Snippet")
    lines.append("```python")
    lines.append("# 1. Baseline Scenario")
    lines.append("res_base = dec.decant_series(h_params, use_grid=True, integration_scenarios=df_daily)")
    lines.append("")
    lines.append("# 2. High Q Scenario")
    lines.append("df_high = df_daily.copy()")
    lines.append("df_high['Q'] = df_high['Q'] * 1.5")
    lines.append("res_high = dec.decant_series(h_params, use_grid=True, integration_scenarios=df_high)")
    lines.append("```")
    lines.append("")
    lines.append("## Results")
    lines.append(f"- **Mean Baseline Conc:** {mean_base:.4f}")
    lines.append(f"- **Mean High Q Conc:**   {mean_high:.4f}")
    lines.append("")

    if mean_high > mean_base:
        lines.append("**Conclusion:** SUCCESS. High Q scenario correctly produced higher concentration.")
        print("SUCCESS: High Q scenario produced higher concentration.")
    else:
        lines.append("**Conclusion:** FAILURE. High Q scenario did not produce higher concentration.")
        print("FAILURE: High Q scenario did not produce higher concentration.")

    with open(report_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    run_validation()
