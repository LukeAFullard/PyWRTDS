import numpy as np
import pandas as pd
from datetime import datetime

class Decanter:
    def __init__(self, data, date_col, response_col, covariate_col):
        """
        data: pandas DataFrame
        date_col: string name of date column
        response_col: string name of t0 (target)
        covariate_col: string name of c0 (driver)
        """
        self.df = data.copy()

        # Convert dates to Decimal Time (e.g., 2023.5)
        # This simplifies the math for 't'
        self.df['date'] = pd.to_datetime(self.df[date_col])
        self.df['decimal_time'] = self.df['date'].dt.year + (self.df['date'].dt.dayofyear - 1) / 365.25
        self.df['season'] = self.df['decimal_time'] % 1  # 0 to 1

        # Transform strictly positive data to Log Space as per WRTDS standard
        # Note: If your data has zeros/negatives, you must apply an offset or remove log
        if (self.df[response_col] <= 0).any():
            raise ValueError(f"Column '{response_col}' contains non-positive values. WRTDS requires strictly positive response data for log transformation.")
        if (self.df[covariate_col] <= 0).any():
            raise ValueError(f"Column '{covariate_col}' contains non-positive values. WRTDS requires strictly positive covariate data for log transformation.")

        self.df['log_response'] = np.log(self.df[response_col])
        self.df['log_covariate'] = np.log(self.df[covariate_col])

        # Store vectors for fast vectorized access
        self.T = self.df['decimal_time'].values
        self.Q = self.df['log_covariate'].values
        self.S = self.df['season'].values
        self.Y = self.df['log_response'].values

    def _tricube(self, d, h):
        """
        Calculates tricube weight.
        d: array of distances
        h: half-window width
        """
        # Normalize distance by window width
        # Avoid division by zero
        if h == 0:
            return np.where(d == 0, 1.0, 0.0)

        x = np.abs(d) / h

        # Filter: if distance > window, weight is 0
        w = np.zeros_like(x)
        mask = x <= 1
        w[mask] = (1 - x[mask]**3)**3
        return w

    def get_weights(self, t_target, q_target, s_target, h_time=7, h_cov=2, h_season=0.5):
        """
        Calculates the combined weight for all historical points relative to a
        single target point (t_target, q_target, s_target).

        h_time: window in years (default 7)
        h_cov: window in log-covariate units (default 2)
        h_season: window in years (0.5 = 6 months)
        """
        # 1. Time Distance
        dist_t = self.T - t_target
        w_t = self._tricube(dist_t, h_time)

        # 2. Covariate Distance
        dist_q = self.Q - q_target
        w_q = self._tricube(dist_q, h_cov)

        # 3. Seasonal Distance (Circular)
        dist_s_raw = np.abs(self.S - s_target)
        dist_s = np.minimum(dist_s_raw, 1 - dist_s_raw)
        w_s = self._tricube(dist_s, h_season)

        # Combine weights (Section 3.2)
        W = w_t * w_q * w_s

        return W

    def fit_local_model(self, t_target, q_target, s_target, h_params):
        """
        Performs Weighted Least Squares for a specific target point.
        Returns the coefficients [beta_0, beta_1, beta_2, beta_3, beta_4]
        """
        # Get weights
        W = self.get_weights(t_target, q_target, s_target, **h_params)

        # Filter out zero-weight points to speed up matrix math
        mask = W > 0
        if np.sum(mask) < 10: # Safety check for sparse data
            return None

        weights_active = W[mask]
        y_active = self.Y[mask]

        # Construct Design Matrix X for the active points
        # 1, ln(c0), t, sin, cos
        n_active = np.sum(mask)
        X = np.zeros((n_active, 5))
        X[:, 0] = 1 # Intercept
        X[:, 1] = self.Q[mask] # log covariate
        X[:, 2] = self.T[mask] # decimal time
        X[:, 3] = np.sin(2 * np.pi * self.T[mask])
        X[:, 4] = np.cos(2 * np.pi * self.T[mask])

        # Solve WLS: (X.T * W * X)^-1 * X.T * W * Y
        # We use a diagonal weight matrix trick for numpy
        sqrt_W = np.sqrt(weights_active)
        X_w = X * sqrt_W[:, np.newaxis]
        y_w = y_active * sqrt_W

        try:
            betas, residuals, rank, s = np.linalg.lstsq(X_w, y_w, rcond=None)
            return betas
        except np.linalg.LinAlgError:
            return None

    def predict_point(self, t_target, q_target, betas):
        """
        Predicts value using the locally fitted betas.
        """
        if betas is None: return np.nan

        # prediction_log = beta0 + beta1*q + beta2*t + beta3*sin + beta4*cos
        # Note: sin/cos depend on t_target, not q_target
        prediction_log = (betas[0] +
                          betas[1] * q_target +
                          betas[2] * t_target +
                          betas[3] * np.sin(2*np.pi*t_target) +
                          betas[4] * np.cos(2*np.pi*t_target))

        return np.exp(prediction_log) # Return to linear space

    def decant_series(self, h_params={'h_time':7, 'h_cov':2, 'h_season':0.5}):
        """
        Generates the cleaned time series t1 (Flow Normalized).
        """
        results = []

        # Iterate through every time step in the original data
        # Note: In production (Section 6), you would do this on a Grid
        # and interpolate to save time. This loop is O(N^2).
        print("Starting Decanting Process (this may take time)...")

        for i, row in self.df.iterrows():
            t_current = row['decimal_time']
            s_current = row['season']

            # 1. Fit the model specific to this day's conditions
            # We use the current time/season, but we center the covariate window
            # generally or use the actual covariate to find the local relationship.
            # Usually, we center the model estimation on the actual observed Q
            # to get the most accurate betas for this domain.
            q_current = row['log_covariate']

            betas = self.fit_local_model(t_current, q_current, s_current, h_params)

            if betas is None:
                results.append(np.nan)
                continue

            # 2. Flow Normalization (Integration)
            # Instead of predicting with q_current, we predict with ALL historical Qs
            # This effectively integrates out the covariate effect.

            # Method: Monte Carlo Integration over historical distribution
            # We use the vector of all historical log_covariates: self.Q

            # Construct prediction matrix for integration
            # We need to predict for t_current, but varying Q
            # Formula: beta0 + beta1*Q_hist + beta2*t + ...

            # Constant parts of the prediction equation
            const_part = (betas[0] +
                          betas[2] * t_current +
                          betas[3] * np.sin(2*np.pi*t_current) +
                          betas[4] * np.cos(2*np.pi*t_current))

            # Variable part (the covariate)
            cov_part = betas[1] * self.Q

            # Predictions for this specific day, under all historical covariate conditions
            predictions_log = const_part + cov_part
            predictions = np.exp(predictions_log)

            # The "Decanted" value is the mean of these expectations
            t1_val = np.mean(predictions)
            results.append(t1_val)

        return results
