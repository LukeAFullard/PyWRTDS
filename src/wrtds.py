import numpy as np
import pandas as pd
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator

class Decanter:
    def __init__(self, data, date_col, response_col, covariate_col, extra_covariates=None):
        """
        data: pandas DataFrame
        date_col: string name of date column
        response_col: string name of t0 (target)
        covariate_col: string name of c0 (driver)
        extra_covariates: list of dicts, e.g. [{'col': 'Temp', 'log': False}]
        """
        self.orig_date_col = date_col
        self.orig_response_col = response_col
        self.orig_covariate_col = covariate_col
        self.extra_cov_config = extra_covariates if extra_covariates else []

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

        # Handle Extra Covariates
        self.extra_cov_names = []
        for conf in self.extra_cov_config:
            col = conf['col']
            is_log = conf.get('log', False) # Default to Linear space

            if is_log:
                if (self.df[col] <= 0).any():
                    raise ValueError(f"Extra Covariate '{col}' contains non-positive values but log=True requested.")
                name = f"log_{col}"
                self.df[name] = np.log(self.df[col])
            else:
                name = f"lin_{col}"
                self.df[name] = self.df[col] # Linear space

            self.extra_cov_names.append(name)

        # Store vectors for fast vectorized access
        self.T = self.df['decimal_time'].values
        self.Q = self.df['log_covariate'].values
        self.S = self.df['season'].values
        self.Y = self.df['log_response'].values

        # Matrix of Extra Covariates (N_samples x N_extras)
        if self.extra_cov_names:
            self.X_extras = self.df[self.extra_cov_names].values
        else:
            self.X_extras = None

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

    def get_weights(self, t_target, q_target, s_target, h_params, extras_target=None):
        """
        Calculates the combined weight for all historical points relative to a
        single target point (t_target, q_target, s_target).
        """
        h_time = h_params.get('h_time', 7)
        h_cov = h_params.get('h_cov', 2)
        h_season = h_params.get('h_season', 0.5)

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

        # 4. Extra Covariates
        # extras_target must be a list of values corresponding to self.X_extras columns
        if self.X_extras is not None and extras_target is not None:
             for i, conf in enumerate(self.extra_cov_config):
                 h_extra = h_params.get(f"h_{conf['col']}", 2) # Default window 2 units
                 target_val = extras_target[i]
                 obs_vals = self.X_extras[:, i]

                 dist_ex = obs_vals - target_val
                 w_ex = self._tricube(dist_ex, h_extra)

                 W = W * w_ex

        return W

    def fit_local_model(self, t_target, q_target, s_target, h_params, extras_target=None):
        """
        Performs Weighted Least Squares for a specific target point.
        Returns the coefficients.
        """
        # Get weights
        W = self.get_weights(t_target, q_target, s_target, h_params, extras_target)

        # Filter out zero-weight points to speed up matrix math
        # Also ensure we only use valid (non-NaN) observations
        valid_obs = ~np.isnan(self.Y)
        mask = (W > 0) & valid_obs

        # Dimension of model: 5 (Standard) + N_extras
        n_extras = self.X_extras.shape[1] if self.X_extras is not None else 0
        n_params = 5 + n_extras

        # Heuristic: Need at least 2 * params data points? Or just > params
        if np.sum(mask) < max(10, n_params + 2):
            return None

        weights_active = W[mask]
        y_active = self.Y[mask]

        # Construct Design Matrix X for the active points
        # 1, ln(c0), t, sin, cos, [extra1, extra2...]
        n_active = np.sum(mask)
        X = np.zeros((n_active, n_params))
        X[:, 0] = 1 # Intercept
        X[:, 1] = self.Q[mask] # log covariate
        X[:, 2] = self.T[mask] # decimal time
        X[:, 3] = np.sin(2 * np.pi * self.T[mask])
        X[:, 4] = np.cos(2 * np.pi * self.T[mask])

        if n_extras > 0:
            X[:, 5:] = self.X_extras[mask, :]

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

    def predict_point(self, t_target, q_target, betas, extras_target=None):
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

        # Add contributions from extra covariates
        # extras_target: [val1, val2...]
        if extras_target is not None:
            # Betas for extras start at index 5
            for i, val in enumerate(extras_target):
                prediction_log += betas[5 + i] * val

        return np.exp(prediction_log) # Return to linear space

    def compute_grid(self, h_params, n_t=100, n_q=15):
        """
        Computes the regression surfaces (betas) on a regular grid.
        Returns a tuple of (t_grid, q_grid, betas_grid)
        """
        print(f"Computing Grid ({n_t} x {n_q})...")

        # Define grid boundaries
        t_min, t_max = self.T.min(), self.T.max()
        q_min, q_max = self.Q.min(), self.Q.max()

        # Handle constant covariate/time edge cases (linspace needs range)
        if t_min == t_max:
            t_min -= 0.01
            t_max += 0.01
        if q_min == q_max:
            q_min -= 0.01
            q_max += 0.01

        # Add small buffer to grid
        t_grid = np.linspace(t_min, t_max, n_t)
        q_grid = np.linspace(q_min, q_max, n_q)

        # Shape: (n_t, n_q, 5 coefficients)
        betas_grid = np.zeros((n_t, n_q, 5))

        for i, t_val in enumerate(t_grid):
            for j, q_val in enumerate(q_grid):
                s_val = t_val % 1
                betas = self.fit_local_model(t_val, q_val, s_val, h_params)
                if betas is not None:
                    betas_grid[i, j, :] = betas
                else:
                    # Fill with NaNs or nearest?
                    # For now, NaNs, which might propagate.
                    # Ideally, we should interpolate gaps, but WRTDS usually assumes dense enough data.
                    betas_grid[i, j, :] = np.nan

        return t_grid, q_grid, betas_grid

    def decant_series(self, h_params={'h_time':7, 'h_cov':2, 'h_season':0.5}, use_grid=False, grid_config={'n_t':100, 'n_q':15}, gfn_window=None, integration_scenarios=None):
        """
        Generates the cleaned time series t1 (Flow Normalized).
        If use_grid=True, it computes a regression surface and interpolates betas.

        gfn_window: float (years). If provided, performs Generalized Flow Normalization (GFN)
                    by integrating over covariate values only within [t - gfn_window/2, t + gfn_window/2].

        integration_scenarios: pandas DataFrame (optional). WRTDS-P (Projection).
                    If provided, these values are used for the integration step instead of the historical
                    covariate record. Must contain columns matching the original covariate names.
        """
        results = []

        # Pre-process integration scenarios if provided
        Q_scenario = None
        Extras_scenario = None

        if integration_scenarios is not None:
            # Transform scenario to Log Space
            # Primary Covariate
            q_raw = integration_scenarios[self.orig_covariate_col]
            if (q_raw <= 0).any():
                 raise ValueError("Integration scenario primary covariate must be strictly positive.")
            Q_scenario = np.log(q_raw).values

            # Extra Covariates
            if self.extra_cov_config:
                Extras_scenario_list = []
                for conf in self.extra_cov_config:
                    col = conf['col']
                    is_log = conf.get('log', False)
                    vals = integration_scenarios[col]
                    if is_log:
                        if (vals <= 0).any():
                            raise ValueError(f"Integration scenario extra covariate '{col}' contains non-positive values.")
                        Extras_scenario_list.append(np.log(vals).values)
                    else:
                        Extras_scenario_list.append(vals.values)
                Extras_scenario = np.column_stack(Extras_scenario_list)

        # Grid not supported for WRTDSplus yet
        if use_grid and self.X_extras is not None:
            print("Warning: Grid optimization not supported for WRTDSplus (extra covariates). Falling back to Exact method.")
            use_grid = False

        interpolator = None
        if use_grid:
            t_grid, q_grid, betas_grid = self.compute_grid(h_params, **grid_config)

            # Create an interpolator for each of the 5 beta coefficients
            # RegularGridInterpolator requires points to be strictly ascending (linspace ensures this)
            # We create a list of 5 interpolators
            interpolators = []
            for k in range(5):
                # Handle NaNs in grid?
                # RegularGridInterpolator doesn't like NaNs.
                # Simple fill: Forward fill / Backward fill along Time axis
                # This is a naive patch for sparse grids
                grid_slice = pd.DataFrame(betas_grid[:, :, k])
                grid_slice = grid_slice.ffill(axis=0).bfill(axis=0).ffill(axis=1).bfill(axis=1)

                interp = RegularGridInterpolator((t_grid, q_grid), grid_slice.values, bounds_error=False, fill_value=None)
                interpolators.append(interp)

        print(f"Starting Decanting Process (Method: {'Grid' if use_grid else 'Exact'})...")

        for i, row in self.df.iterrows():
            t_current = row['decimal_time']
            s_current = row['season']
            q_current = row['log_covariate']

            # Extract extras for this row if they exist
            extras_current = None
            if self.X_extras is not None:
                extras_current = self.X_extras[i, :]

            betas = None

            if use_grid:
                # Interpolate betas
                # Input to interpolator is (t, q)
                pts = np.array([[t_current, q_current]])
                betas = np.array([interp(pts)[0] for interp in interpolators])
                if np.isnan(betas).any():
                    betas = None
            else:
                # Exact calculation
                betas = self.fit_local_model(t_current, q_current, s_current, h_params, extras_current)

            if betas is None:
                results.append(np.nan)
                continue

            # Flow Normalization (Integration)
            # Formula: beta0 + beta1*Q_hist + beta2*t + ... + beta_ext * Ext_hist
            const_part = (betas[0] +
                          betas[2] * t_current +
                          betas[3] * np.sin(2*np.pi*t_current) +
                          betas[4] * np.cos(2*np.pi*t_current))

            # Select covariates for integration
            # Priority:
            # 1. Custom Scenario (WRTDS-P) -> Uses provided distribution (Stationary over the scenario)
            # 2. GFN (Windowed History)
            # 3. SFN (Full History)

            if integration_scenarios is not None:
                # WRTDS-P
                Q_integration = Q_scenario
                Extras_integration = Extras_scenario
            elif gfn_window is not None:
                # Generalized Flow Normalization (Windowed)
                h_window = gfn_window / 2
                mask = np.abs(self.T - t_current) <= h_window
                Q_integration = self.Q[mask]
                if self.X_extras is not None:
                    Extras_integration = self.X_extras[mask, :]

                # If window is empty (shouldn't happen with reasonable data), fallback to full
                if len(Q_integration) == 0:
                     Q_integration = self.Q
                     if self.X_extras is not None:
                         Extras_integration = self.X_extras
            else:
                # Stationary Flow Normalization (Full Record)
                Q_integration = self.Q
                if self.X_extras is not None:
                    Extras_integration = self.X_extras

            cov_part = betas[1] * Q_integration

            # Add Extra Covariates Contribution
            extra_part = 0
            if self.X_extras is not None:
                # betas[5:] are for extras
                # We need to dot product betas[5:] with columns of Extras_integration
                # Note: If we have custom scenarios, Extras_integration is set correctly above
                for k in range(len(self.extra_cov_config)):
                    beta_val = betas[5 + k]
                    col_vals = Extras_integration[:, k]
                    extra_part += beta_val * col_vals

            predictions_log = const_part + cov_part + extra_part
            predictions = np.exp(predictions_log)
            t1_val = np.mean(predictions)
            results.append(t1_val)

        return results

    def bootstrap_uncertainty(self, h_params, n_bootstraps=100, block_size=30, use_grid=True, method='block'):
        """
        Estimates uncertainty bands using Bootstrap on residuals.

        method: 'block' (Standard Moving Block Bootstrap) or 'wild' (Wild Bootstrap for heteroscedasticity).

        Returns: DataFrame with columns ['mean', 'p05', 'p95'] for the decanted series.
        """
        print(f"Starting Bootstrap Uncertainty Analysis ({method} method, {n_bootstraps} runs)...")

        # 1. Calculate Initial Residuals (from Grid model)
        # We need the estimated values (NOT flow normalized) to get residuals
        # Ideally we reuse a grid if computed
        if use_grid:
            t_grid, q_grid, betas_grid = self.compute_grid(h_params)
            # Create Interpolators
            interpolators = []
            for k in range(5):
                grid_slice = pd.DataFrame(betas_grid[:, :, k])
                grid_slice = grid_slice.ffill(axis=0).bfill(axis=0).ffill(axis=1).bfill(axis=1)
                interp = RegularGridInterpolator((t_grid, q_grid), grid_slice.values, bounds_error=False, fill_value=None)
                interpolators.append(interp)

            # Predict for every point to get residuals
            preds_log = []
            for i, row in self.df.iterrows():
                pts = np.array([[row['decimal_time'], row['log_covariate']]])
                betas = np.array([interp(pts)[0] for interp in interpolators])
                if np.isnan(betas).any():
                     preds_log.append(np.nan)
                else:
                    val = (betas[0] +
                           betas[1] * row['log_covariate'] +
                           betas[2] * row['decimal_time'] +
                           betas[3] * np.sin(2*np.pi*row['decimal_time']) +
                           betas[4] * np.cos(2*np.pi*row['decimal_time']))
                    preds_log.append(val)
            preds_log = np.array(preds_log)
        else:
            # Non-grid path not implemented for bootstrap to save complexity
            raise NotImplementedError("Bootstrap requires use_grid=True for performance.")

        residuals = self.Y - preds_log

        # 2. Bootstrap Loop
        n = len(residuals)

        # Identify valid residuals (for Block method source)
        valid_res_indices = np.where(~np.isnan(residuals))[0]
        valid_res = residuals[valid_res_indices]

        results_matrix = np.zeros((n_bootstraps, n))

        for b in range(n_bootstraps):
            boot_res = np.zeros(n)

            if method == 'wild':
                # Wild Bootstrap: Preserve local residual, multiply by random variable
                # Rademacher: +1 or -1 with p=0.5
                # Allows handling heteroscedasticity
                flips = np.random.choice([-1, 1], size=n)
                # Keep NaNs as NaNs (multiplication by +/-1 preserves NaN)
                boot_res = residuals * flips

            elif method == 'block':
                # Block Bootstrap: Resample blocks from valid residuals
                # Note: This destroys correspondence with Time if gaps exist,
                # effectively assuming homoscedasticity.
                current_idx = 0
                while current_idx < n:
                    # Pick random start from VALID residuals
                    start = np.random.randint(0, len(valid_res) - block_size)
                    chunk = valid_res[start : start + block_size]

                    # Fill the bootstrap vector
                    # Note: We are filling sequentially. If the original series had gaps,
                    # this "compacts" the residuals. But we map them to the full time series 1-to-1?
                    # No, usually we map index i -> index i.
                    # Ideally, we should resample blocks of INDICES from the original series,
                    # keeping the (NaN) structure if a block contains NaNs.
                    # But standard practice often just resamples the errors.

                    end_idx = min(current_idx + block_size, n)
                    needed = end_idx - current_idx
                    boot_res[current_idx : end_idx] = chunk[:needed]
                    current_idx += block_size
            else:
                 raise ValueError(f"Unknown bootstrap method: {method}")

            # Create synthetic response
            # New Y = Pred + Resampled Residual
            new_log_response = preds_log + boot_res

            # Enforce Missingness Pattern:
            # If original Y was NaN, synthetic Y must be NaN.
            # This prevents "filling in" gaps which would artificially lower uncertainty.
            new_log_response[np.isnan(self.Y)] = np.nan

            # Create temporary Decanter for this run
            # We assume Covariate and Time are fixed (Fixed-X resampling)
            df_boot = self.df.copy()

            # Update the ORIGINAL response column with the new synthetic data (inverse log)
            # This ensures that when we re-init Decanter, it logs it back correctly to 'log_response'
            # and performs the positive check.
            # Handle NaNs: exp(NaN) = NaN.
            df_boot[self.orig_response_col] = np.exp(new_log_response)

            # Re-initialize using original column names
            dec_boot = Decanter(df_boot, self.orig_date_col, self.orig_response_col, self.orig_covariate_col)

            # Run Decant
            # We pass the PRE-COMPUTED grid config to save time?
            # No, we must re-compute grid because Y changed.
            res = dec_boot.decant_series(h_params, use_grid=True)
            results_matrix[b, :] = res

        # 3. Aggregate
        df_results = pd.DataFrame({
            'mean': np.nanmean(results_matrix, axis=0),
            'p05': np.nanpercentile(results_matrix, 5, axis=0),
            'p95': np.nanpercentile(results_matrix, 95, axis=0)
        })

        return df_results

    def add_kalman_correction(self, estimated_log_series, rho=0.9):
        """
        Applies WRTDS-Kalman correction (AR1 filtering of residuals).
        estimated_log_series: array of log-space predictions from WRTDS (NOT flow normalized)
        rho: autocorrelation coefficient (0.8 - 0.95 typical)

        Returns: series with Kalman correction applied (in log space).
        """
        # Calculate residuals where we have observations
        # res = log_obs - log_model

        n = len(self.df)
        residuals = np.full(n, np.nan)

        # Only calculate for valid observations
        valid_mask = ~np.isnan(self.Y)
        residuals[valid_mask] = self.Y[valid_mask] - estimated_log_series[valid_mask]

        # Interpolate residuals to all days using AR(1) decay
        # For a day t, finding prev obs t_a and next obs t_b
        # res(t) = res(t_a) * rho^(t-t_a) + ...

        # To do this efficiently without a slow loop, we can use forward/backward pass

        # Forward Pass
        forward_res = np.zeros(n)
        last_res = 0
        last_t = -9999

        for i in range(n):
            if valid_mask[i]:
                last_res = residuals[i]
                last_t = self.T[i]
                forward_res[i] = last_res
            else:
                # Decay
                dt = self.T[i] - last_t
                # If dt is huge (start of record), decay is effectively 0
                if dt > 10: # Optimization: stop calculating if correlation is gone
                    forward_res[i] = 0
                else:
                    # T is in years. rho is typically daily correlation?
                    # WRTDS-K usually defines rho as DAILY correlation.
                    # self.T is years. dt_days = dt * 365.25
                    dt_days = dt * 365.25
                    forward_res[i] = last_res * (rho ** dt_days)

        # Backward Pass
        backward_res = np.zeros(n)
        last_res = 0
        last_t = 9999

        for i in range(n-1, -1, -1):
            if valid_mask[i]:
                last_res = residuals[i]
                last_t = self.T[i]
                backward_res[i] = last_res
            else:
                dt = last_t - self.T[i]
                if dt > 10:
                    backward_res[i] = 0
                else:
                    dt_days = dt * 365.25
                    backward_res[i] = last_res * (rho ** dt_days)

        # Combined Estimate
        # If we have obs, it's just the residual.
        # If not, it's a weighted average?
        # Standard simple smoother: If we have forward and backward, take average?
        # Actually, if we are strictly between two points, the 'best' estimate is roughly linear combination.
        # But (rho^a + rho^b) is a reasonable approximation for simply adding the information content.
        # Let's use the average of the two propagated signals.

        interpolated_residuals = np.zeros(n)
        for i in range(n):
            if valid_mask[i]:
                interpolated_residuals[i] = residuals[i]
            else:
                # Average the forward and backward projections
                # This ensures continuity
                interpolated_residuals[i] = (forward_res[i] + backward_res[i]) / 2.0

        return estimated_log_series + interpolated_residuals
