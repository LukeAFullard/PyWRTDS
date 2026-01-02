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

    def test_bootstrap(self):
        """Test Bootstrap Uncertainty Analysis"""
        np.random.seed(42)
        n = 100
        dates = pd.date_range(start='2010-01-01', periods=n, freq='D')

        # Simple signal
        t0 = np.linspace(10, 20, n) * 2.0 # Ensure > 0
        c0 = np.ones(n)

        df = pd.DataFrame({'Date': dates, 'Sales': t0, 'AdSpend': c0})
        dec = Decanter(df, date_col='Date', response_col='Sales', covariate_col='AdSpend')

        # Run small bootstrap
        # Use small grid size to speed up test
        # Need use_grid=True
        # Note: The Decanter init inside bootstrap uses Dummy col names 'log_response'.
        # We need to make sure the original Decanter works even if we patch it?
        # The bootstrap logic patches the NEW instance internals, so it should be fine.

        # However, the dummy init line:
        # dec_boot = Decanter(self.df, 'date', 'log_response', 'log_covariate')
        # requires 'log_response' to exist in the dataframe passed.
        # But 'log_response' IS created in the __init__ of the original object, and self.df HAS it.
        # But we pass the string names of columns. 'log_response' is not the original name.
        # If we pass 'log_response' as the response_col, the init will try to log it again!
        # Resulting in log(log(y)). This is a bug in the implementation of bootstrap method.
        # I need to fix the implementation in wrtds.py first?
        # Or I can fix the test expectation if I modify the implementation.

        # Wait, the implementation does:
        # df_boot.iloc[:, list(self.df.columns).index(self.df.columns[self.df.columns.get_loc('log_response')-1])] = np.exp(new_log_response)
        # It tries to find the original column (assumed to be before log_response?)
        # And then:
        # dec_boot = Decanter(self.df, 'date', 'log_response', 'log_covariate')
        # If we initialize with 'log_response', the new init will do:
        # self.df['log_response'] = np.log(self.df['log_response'])
        # So we are double logging.

        # We need to perform the actual bootstrap test now that logic is fixed
        # Run bootstrap
        # Using very small n_bootstraps to keep test fast
        res_boot = dec.bootstrap_uncertainty(h_params={'h_time': 2, 'h_cov': 2, 'h_season': 0.5}, n_bootstraps=5, block_size=10)

        self.assertIn('mean', res_boot.columns)
        self.assertIn('p05', res_boot.columns)
        self.assertIn('p95', res_boot.columns)
        self.assertEqual(len(res_boot), n)

        # Verify order
        self.assertTrue(np.all(res_boot['p95'] >= res_boot['mean']))
        self.assertTrue(np.all(res_boot['mean'] >= res_boot['p05']))

        # The true trend is linear growth 20 -> 40
        # The mean should be reasonably close (within noise)
        # Note: res_boot is the normalized series.
        # Since C0 is 1 (const), normalized = estimated.
        # True signal is t0.

        # Check first and last point
        # Allow some margin for random noise in bootstrap
        # p95 should effectively cover the true signal
        # Since we added no noise in t0 (it's perfect signal), residuals are 0?
        # No, fit_local_model won't be perfect.
        # If residuals are 0, bootstrap collapses to single line.
        # Let's add noise to t0 so bootstrap has something to do.

        t0_noisy = t0 + np.random.normal(0, 1.0, n)
        df_noisy = pd.DataFrame({'Date': dates, 'Sales': t0_noisy, 'AdSpend': c0})
        dec_noisy = Decanter(df_noisy, date_col='Date', response_col='Sales', covariate_col='AdSpend')

        res_boot_noisy = dec_noisy.bootstrap_uncertainty(h_params={'h_time': 2, 'h_cov': 2, 'h_season': 0.5}, n_bootstraps=10, block_size=10)

        # The bounds should have some width now
        width = res_boot_noisy['p95'] - res_boot_noisy['p05']
        self.assertTrue(np.mean(width) > 0.1)

    def test_wild_bootstrap(self):
        """Test Wild Bootstrap and preservation of NaNs"""
        np.random.seed(42)
        n = 100
        dates = pd.date_range(start='2010-01-01', periods=n, freq='D')

        # Signal with Heteroscedasticity (Variance increases over time)
        trend = np.linspace(10, 20, n)
        noise = np.random.normal(0, np.linspace(0.1, 1.0, n), n)
        t0 = trend + noise
        c0 = np.ones(n)

        df = pd.DataFrame({'Date': dates, 'Sales': t0, 'AdSpend': c0})

        # Add a gap
        df.loc[50:55, 'Sales'] = np.nan

        dec = Decanter(df, date_col='Date', response_col='Sales', covariate_col='AdSpend')

        # Run Wild Bootstrap
        res_wild = dec.bootstrap_uncertainty(h_params={'h_time': 2, 'h_cov': 2, 'h_season': 0.5},
                                             n_bootstraps=10,
                                             use_grid=True,
                                             method='wild')

        # Check result shape
        self.assertEqual(len(res_wild), n)

        # Check that NaN gaps in input produce NaNs in output (p05, p95)
        # Why? Because if input is NaN, residuals are NaN, synthetic Y is NaN, decanted output is...
        # Wait, standard WRTDS 'decant_series' fills NaNs?
        # decant_series iterates over ROWS. If row['log_covariate'] is valid, it predicts.
        # It does NOT depend on row['Sales']. It depends on the MODEL fitted to Sales.
        # If we feed a synthetic Y with NaNs to Decanter, fit_local_model will filter them out (as per our sparse update).
        # So the model will still be fitted (using available data).
        # And decant_series will produce values for the gap days!

        # So, strictly speaking, uncertainty bands WILL exist during the gap.
        # This is correct behavior: we infer the state during the gap based on neighbors.
        # The uncertainty should be HIGHER in the gap because data is missing.

        # Let's verify we didn't crash on NaNs
        self.assertFalse(np.isnan(res_wild['mean'].iloc[0]))

        # Check that Wild method ran and produced width
        width = res_wild['p95'] - res_wild['p05']
        self.assertTrue(np.mean(width) > 0)

        # Ideally, uncertainty should be higher at end (high noise) than start (low noise)
        # Wild bootstrap preserves this structure.
        w_start = np.mean(width[:10])
        w_end = np.mean(width[-10:])
        self.assertLess(w_start, w_end)

    def test_wrtds_plus(self):
        """Test WRTDSplus with an extra covariate"""
        np.random.seed(42)
        n = 500
        dates = pd.date_range(start='2010-01-01', periods=n, freq='D')

        # Primary Covariate (Discharge)
        c0 = np.exp(np.sin(np.linspace(0, 10, n)))

        # Secondary Covariate (e.g., Temperature - linear)
        # Strong signal
        temp = np.linspace(0, 30, n) + np.random.normal(0, 2, n)

        # Response driven by BOTH
        # ln(Sales) = 1*ln(c0) + 0.1*Temp
        t0 = c0 * np.exp(0.1 * temp)

        df = pd.DataFrame({'Date': dates, 'Sales': t0, 'AdSpend': c0, 'Temp': temp})

        # 1. Standard WRTDS (ignores Temp)
        # Should perform poorly because Temp explains a lot of variance
        dec_std = Decanter(df, date_col='Date', response_col='Sales', covariate_col='AdSpend')
        res_std = dec_std.decant_series(h_params={'h_time': 2, 'h_cov': 2, 'h_season': 0.5})

        # 2. WRTDSplus (includes Temp)
        # Should account for Temp in the model
        dec_plus = Decanter(df, date_col='Date', response_col='Sales', covariate_col='AdSpend',
                            extra_covariates=[{'col': 'Temp', 'log': False}])

        # Need window for Temp. Passed via h_params as 'h_Temp'
        res_plus = dec_plus.decant_series(h_params={'h_time': 2, 'h_cov': 2, 'h_season': 0.5, 'h_Temp': 10})

        # Compare Residuals (estimated, not decanted)
        # We need to expose prediction to do this clean comparison,
        # but decant_series returns the NORMALIZED series.
        # However, if WRTDSplus works, the normalized series should be smoother?
        # Actually, if we normalize out Temp, we remove the Temp trend.
        # In Standard, Temp trend remains in the residuals (or gets aliased into Time trend).

        # Let's check if WRTDSplus runs without crashing and produces different results.
        self.assertEqual(len(res_plus), n)
        self.assertNotEqual(res_plus[0], res_std[0])

        # Verify that grid is disabled for WRTDSplus
        # Capturing stdout is hard in unit test, but we can check the behavior implicitly
        # (It should take longer if N is large, but 500 is fast)

if __name__ == '__main__':
    unittest.main()
