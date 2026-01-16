import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.wrtds import Decanter

def generate_ar1_data():
    np.random.seed(101)
    dates = pd.date_range(start='2010-01-01', end='2012-12-31', freq='D')
    n = len(dates)
    t = np.arange(n) / 365.25

    # Covariate
    log_q = 2.0 + np.random.normal(0, 0.5, n)
    q = np.exp(log_q)

    # Trend
    trend = 1.0 + 0.3 * log_q

    # AR(1) Noise
    rho = 0.9
    noise = np.zeros(n)
    epsilon = np.random.normal(0, 0.1, n)
    noise[0] = epsilon[0]
    for i in range(1, n):
        noise[i] = rho * noise[i-1] + epsilon[i]

    log_c = trend + noise
    c = np.exp(log_c)

    df = pd.DataFrame({'Date': dates, 'Q': q, 'Conc': c})

    # Train on every 10th day
    train_mask = np.arange(n) % 10 == 0
    df_train = df[train_mask].copy()

    return df, df_train

def run_validation():
    print("--- WRTDS-Kalman Validation ---")

    df_daily, df_train = generate_ar1_data()
    print(f"Total Days: {len(df_daily)}, Training Samples: {len(df_train)}")

    dec = Decanter(df_train, 'Date', 'Conc', 'Q', daily_data=df_daily)

    # 1. Fit & Predict Standard (Grid)
    print("Fitting Standard Model...")
    h_params = {'h_time': 7, 'h_cov': 2, 'h_season': 0.5}

    # We need to run decant_series(use_grid=True) once to build the grid
    dec.decant_series(h_params, use_grid=True)

    # Predict on all days
    res_df = dec.predict(df_daily, use_grid=True)
    estimated_log = np.log(res_df['estimated'].values)

    # 2. Apply Kalman
    print("Applying Kalman Correction (rho=0.9)...")
    # Note: add_kalman_correction usually takes the SERIES to be corrected.
    # It calculates residuals based on self.df (Training Data).
    # And applies correction to the series passed in?
    # Let's check src/wrtds.py
    # `add_kalman_correction(self, estimated_log_series, ...)`
    # "Interpolate residuals to all days... return estimated_log_series + interpolated_residuals"
    # But wait, 'estimated_log_series' must match the length of... self.T?
    # self.T is the Sample T? Or Daily T?
    # In __init__, if daily_data is provided, self.daily_history is set.
    # But self.T is always derived from self.df (the Sample).
    # `add_kalman_correction`:
    #   df_kalman = pd.DataFrame({'T': self.T, 'Res': residuals})
    #   It constructs residuals for `self.df`.
    #   Then it calculates forward/backward pass on `df_kalman`.
    #   Wait. If `df_kalman` only contains Sample points, `ffill` on `Res` works on the Sample rows.
    #   But we need to interpolate to *Daily* time steps.
    #   Current implementation:
    #     df_kalman only has Sample T.
    #     dt_forward uses (T - Last_T).
    #     This calculates decay *between samples*.
    #   But it returns a series of length... ?
    #   "return estimated_log_series + interpolated_residuals"
    #   It assumes `estimated_log_series` has same length/index as `self.df`?
    #   If I pass a Daily Series (len 1000) but self.df is Sample (len 100), this will crash or misalign?
    #
    #   Let's check code carefully.
    #   `add_kalman_correction` uses `self.T`. `self.T` is from `self.df` (Sample).
    #   So it seems currently `add_kalman_correction` is designed to correct the SAMPLE series?
    #   That doesn't make sense for daily estimation. We want to correct the Daily series.
    #
    #   Actually, looking at `add_kalman_correction` in `src/wrtds.py`:
    #   It builds `df_kalman` from `self.T` and `residuals`.
    #   Then it does `ffill`.
    #   If `self.T` is just the samples, `ffill` just propagates the residual to the *next sample*.
    #   It doesn't seem to support generating a Daily time series if the input `estimated_log_series` is Daily.
    #
    #   Wait, if `estimated_log_series` is passed, what determines the output shape?
    #   The code uses `estimated_log_series + interpolated_residuals`.
    #   So `interpolated_residuals` must match `estimated_log_series`.
    #   But `interpolated_residuals` is derived from `forward_res + backward_res`.
    #   `forward_res` comes from `df_kalman`.
    #   `df_kalman` is built from `self.T`.
    #   So `interpolated_residuals` has length of `self.T` (Sample).
    #
    #   Conclusion: The current `add_kalman_correction` implementation ONLY supports correcting the Sample series.
    #   It does NOT support correcting an arbitrary Daily series using the Sample residuals.
    #   This is a limitation or a bug if the intent is Daily estimation (which WRTDS-Kalman usually is).
    #
    #   However, I must work with *implemented* capabilities.
    #   If it only works for the sample series, I can only validate it on the sample series?
    #   But correcting the sample series (which are observed) just gives you back the observation (Residual + Estimate = Obs).
    #   So running it on `self.df` is trivial/identity.
    #
    #   Is there a way to use it for Daily?
    #   Maybe `self.T` should have been the target T?
    #
    #   Let's re-read the code for `add_kalman_correction`.
    #
    #   Code:
    #   residuals = Y - estimated (on valid_mask)
    #   df_kalman = pd.DataFrame({'T': self.T, 'Res': residuals})
    #   ... calculation ...
    #   interpolated_residuals = ...
    #
    #   Yes, it is hardcoded to `self.T`.
    #
    #   If I want to use it for daily, I might need to hack it?
    #   Or maybe I should verify this behavior.
    #
    #   If I cannot use it for daily, I can't really "validate" it in the useful sense (predicting unseen days).
    #
    #   Wait, if I initialize `Decanter` with the DAILY data as `data`, and pass the TRUE observed as `response_col`?
    #   But then I'm training on Daily.
    #
    #   Okay, I will write the validation script to expose this.
    #   I will try to use it on the sample data and see if it returns the observed values (Identity check).
    #   This confirms the math is "working" as coded, even if the feature scope is limited.
    #
    #   Actually, looking at `add_kalman_correction` again...
    #   If I want to validate it, I should see if it reconstructs the observations.
    #
    #   I will also add a "Note" in the output about the daily limitation.

    # 1. Fit model on Training Data.
    # 2. Get Estimated Series (Log) for Training Data.
    # 3. Run Kalman Correction.
    # 4. Compare Corrected vs Observed (Should be identical).

    # This validates the "implementation" logic (AR decay calculation) is running,
    # even if applied to the same points.

    # Wait, `rho ** dt_forward`.
    # If `dt_forward` is between samples (e.g. 10 days), the residual decays.
    # But `ffill` on the SAMPLE dataframe sets `Last_Res` to the previous sample's residual.
    # And `dt` is `T - Last_T`.
    # So for row `i`, `Last_T` is `T[i-1]`.
    # So `forward_res` at `i` is `Res[i-1] * rho^(T[i]-T[i-1])`.
    # And `backward_res` at `i` is `Res[i+1] * rho^(...)`.
    # And `interpolated` is average.
    # But at `i`, we HAVE an observation. So we calculate `residuals[i]`.
    # The code says: `interpolated_residuals[valid_mask] = residuals[valid_mask]`.
    # So at observed points, it forces the residual.
    # So `add_kalman_correction` on `self.df` will just return `self.Y`.

    # So verifying it on `self.df` is just verifying the override logic.

    # Is there ANY way to use this for interpolation?
    # Only if `self.df` contains "Target" rows with NaNs in Response?
    # YES! WRTDS allows `data` (the dataframe) to have NaNs?
    # `fit_local_model` filters: `valid_obs = ~np.isnan(self.Y)`.
    # So I can pass a DataFrame that includes both Training (Observed) and Test (NaN) rows!
    # Then `add_kalman_correction` will interpolate for the NaN rows!

    # Strategy:
    # 1. Create a DataFrame containing ALL days.
    # 2. Set 'Conc' to NaN for Test days.
    # 3. Initialize Decanter with this "Merged" dataframe.
    # 4. Run `decant_series` (or `get_estimated_series`) -> This fits the model using only Valid obs (Training).
    # 5. Run `add_kalman_correction`.
    # 6. Extract predictions for the Test rows (which were NaN).
    # 7. Compare with Truth.

    # This seems to be the intended usage pattern!

    # 1. Merge
    df_combined = df_daily.copy()

    # Recreate mask (every 10th day)
    n = len(df_combined)
    train_mask = np.arange(n) % 10 == 0

    # Mask out test data in 'Conc' column
    df_combined.loc[~train_mask, 'Conc'] = np.nan

    dec_combined = Decanter(df_combined, 'Date', 'Conc', 'Q')

    # Fit Grid
    dec_combined.decant_series(h_params, use_grid=True)

    # Get Estimates (Log)
    est_log = dec_combined.get_estimated_series(h_params, use_grid=True)

    # Kalman
    kalman_log = dec_combined.add_kalman_correction(est_log, rho=0.9)

    # Evaluate on Test set
    # Indices where train_mask is False
    test_idx = ~train_mask

    rmse_std = np.sqrt(np.mean((est_log[test_idx] - np.log(df_daily.loc[test_idx, 'Conc']))**2))
    rmse_kal = np.sqrt(np.mean((kalman_log[test_idx] - np.log(df_daily.loc[test_idx, 'Conc']))**2))

    print(f"Standard RMSE (Log): {rmse_std:.4f}")
    print(f"Kalman RMSE (Log):   {rmse_kal:.4f}")

    # Report
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    report_path = os.path.join(root_dir, 'reports', 'validation_kalman.md')

    lines = []
    lines.append("# WRTDS-Kalman Validation Report")
    lines.append("")
    lines.append("## Methodology")
    lines.append("Used synthetic AR(1) autocorrelated data. Trained on sparse subset (every 10th day), predicted test days.")
    lines.append("")
    lines.append("## Results")
    lines.append("| Model | RMSE (Log) |")
    lines.append("| :--- | :--- |")
    lines.append(f"| Standard WRTDS | {rmse_std:.4f} |")
    lines.append(f"| WRTDS-Kalman | {rmse_kal:.4f} |")
    lines.append("")

    if rmse_kal < rmse_std:
        lines.append("**Conclusion:** SUCCESS. Kalman correction improved prediction accuracy.")
        print("SUCCESS: Kalman correction improved prediction.")
    else:
        lines.append("**Conclusion:** FAILURE. Kalman correction did not improve prediction.")
        print("FAILURE: Kalman correction did not improve prediction.")

    with open(report_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    run_validation()
