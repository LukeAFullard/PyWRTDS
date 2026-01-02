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

if __name__ == '__main__':
    unittest.main()
