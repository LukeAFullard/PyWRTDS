import sys
import os
import numpy as np
import pandas as pd
import unittest

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from wrtds import Decanter

class TestWRTDS(unittest.TestCase):
    def setUp(self):
        # 1. Generate Dummy Data
        np.random.seed(42)
        dates = pd.date_range(start='2010-01-01', end='2010-12-31', freq='D') # Shortened for test speed
        n = len(dates)

        # Covariate c0 (Random walk + Seasonality)
        # Using a simple function to make it deterministic
        c0 = np.exp(np.sin(np.linspace(0, 20, n)) + np.random.normal(0, 0.5, n))

        # True Trend (What we want to recover): Linear growth
        true_trend = np.linspace(10, 20, n)

        # Response t0 (Trend + Covariate Effect + Noise)
        # Let's say t0 is heavily influenced by c0
        t0 = true_trend * c0 * np.random.normal(1, 0.1, n)

        self.df = pd.DataFrame({'Date': dates, 'Sales': t0, 'AdSpend': c0})

    def test_decanter_init(self):
        dec = Decanter(self.df, date_col='Date', response_col='Sales', covariate_col='AdSpend')
        self.assertTrue('log_response' in dec.df.columns)
        self.assertTrue('decimal_time' in dec.df.columns)
        self.assertEqual(len(dec.T), len(self.df))

    def test_tricube(self):
        dec = Decanter(self.df, date_col='Date', response_col='Sales', covariate_col='AdSpend')
        d = np.array([0, 0.5, 1.0, 1.5])
        h = 1.0
        w = dec._tricube(d, h)
        self.assertEqual(w[0], 1.0)
        self.assertAlmostEqual(w[1], (1 - 0.5**3)**3)
        self.assertEqual(w[2], 0.0)
        self.assertEqual(w[3], 0.0)

    def test_decant_series(self):
        dec = Decanter(self.df, date_col='Date', response_col='Sales', covariate_col='AdSpend')
        # Use small windows to ensure locality in this small dataset
        # h_time=0.5 years, h_cov=1, h_season=0.5
        adjusted_series = dec.decant_series(h_params={'h_time': 0.5, 'h_cov': 1, 'h_season': 0.5})

        self.assertEqual(len(adjusted_series), len(self.df))
        self.assertFalse(np.isnan(adjusted_series).all(), "Result should not be all NaNs")

        # Check if we removed some variance
        # Raw variance (log space or linear)
        raw_std = np.std(self.df['Sales'])
        clean_std = np.std(adjusted_series)

        # WRTDS should generally reduce variance if the covariate was adding noise
        # But here the covariate (AdSpend) is part of the signal we remove.
        # The 'Trend' (10 to 20) has lower variance than the raw Sales which jumps with AdSpend
        # print(f"Raw Std: {raw_std}, Clean Std: {clean_std}")

        # Basic check: output should be positive
        self.assertTrue(np.all(np.array(adjusted_series) > 0))

    def test_gfn(self):
        """Test Generalized Flow Normalization vs Stationary"""
        # Create data where covariate has a strong trend
        np.random.seed(42)
        n = 1000 # ~3 years
        dates = pd.date_range(start='2010-01-01', periods=n, freq='D')

        # Covariate doubles over time (Trend)
        # Linear growth in log space
        c0 = np.exp(np.linspace(0, 2, n) + np.random.normal(0, 0.2, n))

        # Response is perfectly correlated with Covariate (no trend in beta0)
        t0 = c0 * 10

        df = pd.DataFrame({'Date': dates, 'Sales': t0, 'AdSpend': c0})
        dec = Decanter(df, date_col='Date', response_col='Sales', covariate_col='AdSpend')

        # 1. Stationary Normalization
        # Integrates over ALL history (including low start and high end)
        # Result should be constant average of C0 * 10
        res_stat = dec.decant_series(h_params={'h_time': 2, 'h_cov': 2, 'h_season': 0.5})

        # 2. Generalized (Windowed) Normalization (Window = 1 year)
        # Integrates over local history. Since C0 is growing, the normalization
        # base grows. So the "Decanted" result (which removes C0 effect)
        # should technically strip the C0 trend if the model fits perfectly?
        # Actually:
        # Model: ln(Sales) = ln(10) + 1 * ln(AdSpend)
        # SFN: Expectation over whole history of AdSpend. E[Sales] = 10 * E[AdSpend_all] = Constant.
        # GFN: Expectation over local AdSpend. E[Sales] = 10 * E[AdSpend_local].
        # Since AdSpend_local is increasing, GFN result will increase.
        # This reflects that the "Expected State" of the system is increasing because the DRIVER is increasing.

        res_gfn = dec.decant_series(h_params={'h_time': 2, 'h_cov': 2, 'h_season': 0.5}, gfn_window=1.0)

        # Check that they are different
        self.assertNotEqual(res_stat[0], res_gfn[0])

        # Check trends
        # SFN should be roughly flat (constant expectation)
        slope_stat = np.polyfit(np.arange(n), res_stat, 1)[0]

        # GFN should be positive (tracking the driver trend)
        slope_gfn = np.polyfit(np.arange(n), res_gfn, 1)[0]

        self.assertGreater(slope_gfn, slope_stat)

    def test_sparse_data_and_kalman(self):
        """Test handling of sparse data and WRTDS-Kalman correction"""
        np.random.seed(42)
        n = 100
        dates = pd.date_range(start='2010-01-01', periods=n, freq='D')

        # Simple signal: sine wave
        t0 = np.sin(np.linspace(0, 10, n)) + 2.0 # Ensure > 0
        c0 = np.ones(n) # Constant covariate for simplicity

        df = pd.DataFrame({'Date': dates, 'Sales': t0, 'AdSpend': c0})

        # Create sparse version
        df_sparse = df.copy()
        # Set 50% to NaN
        mask = np.random.choice([True, False], size=n)
        df_sparse.loc[mask, 'Sales'] = np.nan

        dec = Decanter(df_sparse, date_col='Date', response_col='Sales', covariate_col='AdSpend')

        # 1. Run standard Decant (should ignore NaNs)
        res = dec.decant_series(h_params={'h_time': 1, 'h_cov': 2, 'h_season': 0.5})
        self.assertFalse(np.isnan(res).all())

        # 2. Test Kalman
        # To test Kalman, we need the ESTIMATED series (not decanted/normalized),
        # because Kalman applies to the model fit.
        # Let's manually generate "estimated" series (simple model)
        # For this test, we assume the model found a flat line (mean),
        # so residuals = observed - mean.

        # In this synthetic case, Sales moves (sine), but covariate is flat.
        # WRTDS should fit the sine wave via Time/Season terms.
        # Let's say WRTDS underfits slightly.

        # Let's try to run `add_kalman_correction` on a dummy estimate
        est_log = np.log(df['Sales'].values) * 0.9 # Assume 10% error

        # Applying Kalman should bring the estimate CLOSER to the observed values
        # where we have data, and interpolate in between.

        corrected_log = dec.add_kalman_correction(est_log, rho=0.9)
        corrected = np.exp(corrected_log)

        # Check error on OBSERVED points (should be 0 or very close if we used the residuals)
        # Since we passed `est_log` and `dec.Y` is the observed log,
        # Kalman logic: residual = Y - est.
        # corrected = est + residual = est + (Y - est) = Y.
        # So on observed days, we should recover the observed value exactly.

        valid_mask = ~np.isnan(dec.Y)

        # Error before correction
        err_pre = np.mean(np.abs(np.exp(dec.Y[valid_mask]) - np.exp(est_log[valid_mask])))

        # Error after correction (Should be 0)
        err_post = np.mean(np.abs(np.exp(dec.Y[valid_mask]) - corrected[valid_mask]))

        self.assertLess(err_post, 1e-10) # Should be effectively 0
        self.assertLess(err_post, err_pre)

        # Check interpolation (on a NaN point)
        # Find a point that is NaN in input
        nan_indices = np.where(~valid_mask)[0]
        if len(nan_indices) > 0:
            idx = nan_indices[0]
            # The corrected value should NOT be equal to the initial estimate
            # It should have been pulled by neighbors
            self.assertNotEqual(corrected[idx], np.exp(est_log[idx]))

if __name__ == '__main__':
    unittest.main()
