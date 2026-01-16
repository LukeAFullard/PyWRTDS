import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.wrtds import Decanter

def generate_data():
    np.random.seed(404)
    dates = pd.date_range(start='2010-01-01', end='2012-12-31', freq='D')
    n = len(dates)
    t = np.arange(n) / 365.25

    log_q = 2.0 + np.random.normal(0, 0.5, n)
    q = np.exp(log_q)

    # Conc
    log_c = 1.0 + 0.3 * log_q + 0.05 * t + np.random.normal(0, 0.2, n)
    c = np.exp(log_c)

    df = pd.DataFrame({'Date': dates, 'Q': q, 'Conc': c})
    # Sample
    mask = np.arange(n) % 10 == 0
    df_sample = df[mask].copy()

    return df, df_sample

def run_validation():
    print("--- Bootstrap Uncertainty Validation ---")
    df_daily, df_sample = generate_data()
    print(f"Sample N: {len(df_sample)}")

    dec = Decanter(df_sample, 'Date', 'Conc', 'Q', daily_data=df_daily)
    h_params = {'h_time': 7, 'h_cov': 2, 'h_season': 0.5}
    grid_config = {'n_t': 50, 'n_q': 15}

    # 1. Block Bootstrap
    print("\nRunning Block Bootstrap (n=10)...")
    res_block = dec.bootstrap_uncertainty(h_params, n_bootstraps=10, method='block', use_grid=True, grid_config=grid_config)
    print("Block Results Head:")
    print(res_block.head())

    # Check bounds
    if (res_block['p05'] <= res_block['mean']).all() and (res_block['mean'] <= res_block['p95']).all():
        print("SUCCESS: Block Bootstrap bounds are consistent.")
    else:
        print("FAILURE: Block Bootstrap bounds are inconsistent.")

    # 2. Wild Bootstrap
    print("\nRunning Wild Bootstrap (n=10)...")
    res_wild = dec.bootstrap_uncertainty(h_params, n_bootstraps=10, method='wild', use_grid=True, grid_config=grid_config)

    # Report
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    report_path = os.path.join(root_dir, 'reports', 'validation_uncertainty.md')

    lines = []
    lines.append("# Uncertainty (Bootstrap) Validation Report")
    lines.append("")
    lines.append("## Methodology")
    lines.append("Ran Block and Wild Bootstrap (n=10) on synthetic data.")
    lines.append("")
    lines.append("## Results")

    # Check consistency
    block_ok = (res_block['p05'] <= res_block['mean']).all() and (res_block['mean'] <= res_block['p95']).all()
    wild_ok = (res_wild['p05'] <= res_wild['mean']).all() and (res_wild['mean'] <= res_wild['p95']).all()

    lines.append(f"- **Block Bootstrap:** {'Consistent' if block_ok else 'Inconsistent'}")
    lines.append(f"- **Wild Bootstrap:** {'Consistent' if wild_ok else 'Inconsistent'}")
    lines.append("")

    if block_ok and wild_ok:
        lines.append("**Conclusion:** SUCCESS. All bootstrap intervals are logically consistent.")
        print("SUCCESS: Wild Bootstrap bounds are consistent.")
    else:
        lines.append("**Conclusion:** FAILURE. Inconsistent bounds detected.")
        print("FAILURE: Wild Bootstrap bounds are inconsistent.")

    with open(report_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    run_validation()
