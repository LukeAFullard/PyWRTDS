import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.wrtds import Decanter

def generate_trending_q_data():
    np.random.seed(303)
    dates = pd.date_range(start='2000-01-01', end='2020-12-31', freq='D')
    n = len(dates)
    t = np.arange(n) / 365.25

    # Trending Q: Doubles over 20 years
    # Trend line
    q_trend = np.linspace(10, 20, n)
    # Seasonality
    season = 2 * np.sin(2 * np.pi * t)
    # Noise
    log_q = np.log(q_trend + season) + np.random.normal(0, 0.2, n)
    q = np.exp(log_q)

    # Response C: Strictly proportional to Q (No other trend)
    # log_C = 0.0 + 1.0 * log_Q
    log_c = 1.0 * log_q + np.random.normal(0, 0.1, n)
    c = np.exp(log_c)

    df = pd.DataFrame({'Date': dates, 'Q': q, 'Conc': c})

    # Subsample
    mask = np.arange(n) % 14 == 0
    df_sample = df[mask].copy()

    return df, df_sample

def run_validation():
    print("--- GFN Validation ---")

    df_daily, df_sample = generate_trending_q_data()
    print(f"Data: 20 years. Q is trending up. C-Q relationship is constant.")

    dec = Decanter(df_sample, 'Date', 'Conc', 'Q', daily_data=df_daily)
    h_params = {'h_time': 7, 'h_cov': 2, 'h_season': 0.5}

    # 1. Stationary Flow Normalization
    # Should remove the Q trend and result in a flat line (constant concentration history).
    print("\nRunning Stationary FN...")
    res_stat = dec.decant_series(h_params, use_grid=True, gfn_window=None)
    # We want daily output usually, but decant_series returns Sample-sized output by default?
    # Wait, in the Standard validation I used `predict(df_daily)`.
    # Does `predict` support GFN?
    # `predict` calls `decant` internally?
    # Let's check `src/wrtds.py`: `predict` -> "3. Decanted Series (Stationary Normalization)... Integrate over HISTORICAL Q (self.Q)..."
    # It seems `predict` assumes Stationary FN using `self.Q` (or daily history).
    # `predict` does NOT seem to support `gfn_window` parameter!
    # Checking code... `def predict(self, ..., use_grid=True, interp_method='surface'):`
    # It does NOT accept `gfn_window`.
    # And the logic inside hardcodes "Case Stationary".

    # ISSUE: `predict` does not support GFN.
    # To get a daily GFN series, I must use `decant_series`?
    # But `decant_series` loops over `self.df`.
    # Can I change `self.df`?
    # Or, I can init a Decanter with the Daily data as the "Sample" (ignoring response)?
    # Yes, `decant_series` will calculate results for every row in `self.df`.
    # If I pass `df_daily` as the sample data to a new Decanter instance (with dummy Conc), I can run `decant_series` on it.

    # Workaround for GFN on Daily Data:
    print("Initializing Daily Decanter for GFN...")
    df_daily_dummy = df_daily.copy()
    # We need to transfer the fitted grid from the sample decanter to this one?
    # Or just re-fit? Re-fitting on daily data (using daily as sample) would fit betas to daily points?
    # No, we want to use the Betas from the Sample fit, but Apply to Daily points with GFN.

    # Actually, `decant_series` uses `self.fit_local_model` or `grid`.
    # If I use `grid`, I can load the grid from the sample model into the daily model?
    # Code: `dec.save_model`, `dec.load_model`.

    # 1. Fit Grid on Sample
    dec.decant_series(h_params, use_grid=True)
    dec.save_model('temp_grid.pkl')

    # 2. Load Grid into Daily Decanter
    dec_daily = Decanter(df_daily_dummy, 'Date', 'Conc', 'Q', daily_data=df_daily)
    dec_daily.load_model('temp_grid.pkl')

    # 3. Run Decant on Daily Decanter
    # Now `dec_daily.df` is the daily data. `decant_series` will loop over it.
    # It will use the loaded grid for betas.
    # It will use `gfn_window` for integration.

    # Stationary on Daily
    print("Calculating Daily Stationary FN...")
    stat_series = dec_daily.decant_series(h_params, use_grid=True, gfn_window=None)

    # GFN on Daily
    print("Calculating Daily GFN (Window=5 years)...")
    gfn_series = dec_daily.decant_series(h_params, use_grid=True, gfn_window=5.0)

    # Compare Trends
    # Linear regression on the result series
    t_idx = np.arange(len(stat_series))

    slope_stat = np.polyfit(t_idx, stat_series, 1)[0]
    slope_gfn = np.polyfit(t_idx, gfn_series, 1)[0]

    # Scale slopes for readability (e.g. change per year)
    # t_idx is days. * 365
    slope_stat_yr = slope_stat * 365
    slope_gfn_yr = slope_gfn * 365

    print(f"\nSlope Stationary (Conc/Year): {slope_stat_yr:.4f}")
    print(f"Slope GFN (Conc/Year):        {slope_gfn_yr:.4f}")

    # Report
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    report_path = os.path.join(root_dir, 'reports', 'validation_gfn.md')

    lines = []
    lines.append("# Generalized Flow Normalization (GFN) Validation Report")
    lines.append("")
    lines.append("## Methodology")
    lines.append("Synthetic data with trending Discharge (Q) over 20 years. Concentration is proportional to Q.")
    lines.append("Stationary FN should produce a flat line (removing Q trend). GFN should preserve the trend.")
    lines.append("")
    lines.append("### Python Implementation Snippet")
    lines.append("```python")
    lines.append("# 1. Fit Grid on Sample")
    lines.append("dec.decant_series(h_params, use_grid=True)")
    lines.append("dec.save_model('temp_grid.pkl')")
    lines.append("")
    lines.append("# 2. Load Grid into Daily Decanter")
    lines.append("dec_daily = Decanter(df_daily_dummy, 'Date', 'Conc', 'Q', daily_data=df_daily)")
    lines.append("dec_daily.load_model('temp_grid.pkl')")
    lines.append("")
    lines.append("# 3. Run GFN (Window=5 years)")
    lines.append("gfn_series = dec_daily.decant_series(h_params, use_grid=True, gfn_window=5.0)")
    lines.append("```")
    lines.append("")
    lines.append("### R Reproduction Code")
    lines.append("```r")
    lines.append("# To verify against EGRET:")
    lines.append("# eList <- ...")
    lines.append("# dailyResults <- runSeries(eList, windowSide = 5.0, ...)")
    lines.append("```")
    lines.append("")
    lines.append("## Results")
    lines.append("| Method | Slope (Conc/Year) |")
    lines.append("| :--- | :--- |")
    lines.append(f"| Stationary | {slope_stat_yr:.4f} |")
    lines.append(f"| GFN (5-yr) | {slope_gfn_yr:.4f} |")
    lines.append("")

    if slope_gfn_yr > slope_stat_yr and abs(slope_stat_yr) < 0.1:
        lines.append("**Conclusion:** SUCCESS. GFN preserved the trend, Stationary removed it.")
        print("SUCCESS: GFN preserved the upward trend driven by Q, while Stationary removed it.")
    else:
        lines.append("**Conclusion:** FAILURE. Trends did not match expectations.")
        print("FAILURE: Trends did not match expectations.")

    with open(report_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Report saved to {report_path}")

    # Cleanup
    if os.path.exists('temp_grid.pkl'):
        os.remove('temp_grid.pkl')

if __name__ == "__main__":
    run_validation()
