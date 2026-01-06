import numpy as np
import pandas as pd
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator
import itertools
import pickle
import warnings

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

        # State for persistence
        self.grid_axes = None
        self.betas_grid = None
        self.interpolators = None
        self.h_params_last = None

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
            # Check for rank deficiency
            # Note: rank is returned as scalar if singular, but it is scalar here.
            if rank < n_params:
                return None
            return betas
        except np.linalg.LinAlgError:
            return None

    def predict_point(self, t_target, q_target, betas, extras_target=None, return_log=False):
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

        if return_log:
            return prediction_log
        return np.exp(prediction_log) # Return to linear space

    def compute_grid(self, h_params, n_t=100, n_q=15, n_extra=7):
        """
        Computes the regression surfaces (betas) on a regular grid.
        Returns a tuple of (grid_axes_tuple, betas_grid)

        grid_axes_tuple: (t_grid, q_grid, extra1_grid, ...)
        betas_grid: N-D array of coefficients
        """
        n_extras = self.X_extras.shape[1] if self.X_extras is not None else 0
        grid_dims_str = f"{n_t} x {n_q}"
        if n_extras > 0:
            grid_dims_str += f" x {n_extra}" * n_extras

        print(f"Computing Grid ({grid_dims_str})...")

        # 1. Define grid boundaries for Time and Primary Covariate
        t_min, t_max = self.T.min(), self.T.max()
        q_min, q_max = self.Q.min(), self.Q.max()

        if t_min == t_max:
            t_min -= 0.01
            t_max += 0.01
        if q_min == q_max:
            q_min -= 0.01
            q_max += 0.01

        t_grid = np.linspace(t_min, t_max, n_t)
        q_grid = np.linspace(q_min, q_max, n_q)

        # 2. Define grid boundaries for Extra Covariates
        extra_grids = []
        if n_extras > 0:
            for k in range(n_extras):
                vals = self.X_extras[:, k]
                e_min, e_max = vals.min(), vals.max()
                if e_min == e_max:
                    e_min -= 0.01
                    e_max += 0.01
                # Use n_extra points for each extra dimension
                e_grid = np.linspace(e_min, e_max, n_extra)
                extra_grids.append(e_grid)

        # 3. Create List of Axes
        # Axes: [T, Q, E1, E2...]
        grid_axes = [t_grid, q_grid] + extra_grids

        # 4. Prepare Betas Grid
        # Shape: (n_t, n_q, n_e1, ..., n_params)
        shape = tuple(len(g) for g in grid_axes) + (5 + n_extras,)
        betas_grid = np.zeros(shape)

        # 5. Iterate over all grid points
        # itertools.product(*grid_axes) yields tuples (t, q, e1, ...)
        # We also need indices to fill the array.

        ranges = [range(len(g)) for g in grid_axes]

        for indices in itertools.product(*ranges):
            # indices is tuple (i_t, i_q, i_e1, ...)

            # Get values
            values = [grid_axes[k][indices[k]] for k in range(len(grid_axes))]
            t_val = values[0]
            q_val = values[1]
            extras_val = values[2:] if n_extras > 0 else None

            s_val = t_val % 1

            betas = self.fit_local_model(t_val, q_val, s_val, h_params, extras_val)

            if betas is not None:
                betas_grid[indices] = betas
            else:
                betas_grid[indices] = np.nan

        return tuple(grid_axes), betas_grid

    def _prepare_grid_interpolators(self, h_params, grid_config):
        """
        Helper to compute grid and create interpolators.
        Stores them in self.interpolators.
        """
        # If we already have a grid and it matches (assuming h_params unchanged if not checking),
        # but for safety we recompute if called via public methods unless explicitly managed.
        # But here we just compute.

        self.grid_axes, self.betas_grid = self.compute_grid(h_params, **grid_config)
        self.h_params_last = h_params
        self._build_interpolators_from_grid()

        return self.interpolators

    def _build_interpolators_from_grid(self):
        """
        Reconstructs interpolators from self.grid_axes and self.betas_grid.
        """
        if self.grid_axes is None or self.betas_grid is None:
            return

        n_params_grid = self.betas_grid.shape[-1]
        self.interpolators = []

        for k in range(n_params_grid):
            grid_slice = self.betas_grid[..., k]

            # Legacy 2D filling
            if len(self.grid_axes) == 2:
                grid_slice_df = pd.DataFrame(grid_slice)
                grid_slice = grid_slice_df.ffill(axis=0).bfill(axis=0).ffill(axis=1).bfill(axis=1).values

            interp = RegularGridInterpolator(self.grid_axes, grid_slice, bounds_error=False, fill_value=None)
            self.interpolators.append(interp)

    def save_model(self, filepath):
        """
        Saves the fitted grid and parameters to a file.
        Only saves the lightweight grid data, not the full original dataframe.
        """
        if self.grid_axes is None:
            raise ValueError("No model fitted. Run decant_series with use_grid=True first.")

        state = {
            'grid_axes': self.grid_axes,
            'betas_grid': self.betas_grid,
            'h_params': self.h_params_last,
            'extra_cov_config': self.extra_cov_config,
            'orig_covariate_col': self.orig_covariate_col,
            'orig_response_col': self.orig_response_col,
            'orig_date_col': self.orig_date_col,
            # We need Q (historical distribution) for decanting, so we must save it.
            'history_Q': self.Q,
            'history_X_extras': self.X_extras
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_model(self, filepath):
        """
        Loads a fitted grid from a file.
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.grid_axes = state['grid_axes']
        self.betas_grid = state['betas_grid']
        self.h_params_last = state['h_params']
        self.extra_cov_config = state.get('extra_cov_config', [])

        # Restore historical distribution for integration
        self.Q = state.get('history_Q')
        self.X_extras = state.get('history_X_extras')

        # Validation: Check that extra_cov_config matches grid dimensions
        n_extras_config = len(self.extra_cov_config)
        n_extras_grid = len(self.grid_axes) - 2

        if n_extras_config != n_extras_grid:
            # Try to recover if config was not saved or different
            # But here we warn or raise
             warnings.warn(f"Loaded model grid has {n_extras_grid} extra dimensions, but config specifies {n_extras_config}. Predictions may be incorrect.", UserWarning)

        # Rebuild interpolators
        self._build_interpolators_from_grid()

    def get_estimated_series(self, h_params, use_grid=False, grid_config={'n_t':100, 'n_q':15}):
        """
        Returns the model predictions (in log space) for the observed data points.
        This represents the "trend + seasonality + flow effect" component.
        """
        if use_grid:
            if self.interpolators is None:
                self._prepare_grid_interpolators(h_params, grid_config)

            # Vectorized Interpolation
            # Construct points array (N, D)
            cols = [self.T, self.Q]
            if self.X_extras is not None:
                for k in range(self.X_extras.shape[1]):
                    cols.append(self.X_extras[:, k])

            pts = np.column_stack(cols)

            # Predict all betas at once: (N, n_betas)
            # List comprehension over interpolators
            betas_cols = [interp(pts) for interp in self.interpolators]
            all_betas = np.column_stack(betas_cols)

            # Calculate predictions vectorized
            # beta0 + beta1*Q + beta2*t + beta3*sin + beta4*cos
            pred = (all_betas[:, 0] +
                    all_betas[:, 1] * self.Q +
                    all_betas[:, 2] * self.T +
                    all_betas[:, 3] * np.sin(2 * np.pi * self.T) +
                    all_betas[:, 4] * np.cos(2 * np.pi * self.T))

            # Extras
            if self.X_extras is not None:
                for k in range(self.X_extras.shape[1]):
                    pred += all_betas[:, 5 + k] * self.X_extras[:, k]

            return pred

        else:
            # Exact method (Loop)
            preds_log = []
            for i, row in self.df.iterrows():
                t_current = row['decimal_time']
                q_current = row['log_covariate']
                s_current = row['season']

                extras_current = None
                if self.X_extras is not None:
                    extras_current = self.X_extras[i, :]

                betas = self.fit_local_model(t_current, q_current, s_current, h_params, extras_current)

                if betas is None:
                    preds_log.append(np.nan)
                else:
                    # Predict in log space
                    val = self.predict_point(t_current, q_current, betas, extras_current, return_log=True)
                    preds_log.append(val)

            return np.array(preds_log)

    def decant_series(self, h_params={'h_time':7, 'h_cov':2, 'h_season':0.5}, use_grid=False, grid_config={'n_t':100, 'n_q':15}, gfn_window=None, integration_scenarios=None):
        """
        Generates the cleaned time series t1 (Flow Normalized).
        """
        # Pre-process integration scenarios
        Q_scenario = None
        Extras_scenario = None

        if integration_scenarios is not None:
            # Transform scenario to Log Space
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

        if use_grid:
            print(f"Starting Decanting Process (Method: Grid)...")
            if self.interpolators is None:
                self._prepare_grid_interpolators(h_params, grid_config)

            # 1. Vectorized Beta Interpolation
            cols = [self.T, self.Q]
            if self.X_extras is not None:
                for k in range(self.X_extras.shape[1]):
                    cols.append(self.X_extras[:, k])
            pts = np.column_stack(cols)

            # (N_samples, N_betas)
            all_betas = np.column_stack([interp(pts) for interp in self.interpolators])

            # 2. Vectorized Integration
            # Formula: beta0 + beta1*Q_hist + beta2*t + ...

            # Constant part per time step
            const_part = (all_betas[:, 0] +
                          all_betas[:, 2] * self.T +
                          all_betas[:, 3] * np.sin(2*np.pi*self.T) +
                          all_betas[:, 4] * np.cos(2*np.pi*self.T))

            # Identify Integration Source (Q_integration)
            # Three cases: Stationary, Scenarios, GFN

            # Case A: Scenarios or Stationary (Constant Integration Vector)
            if gfn_window is None:
                if integration_scenarios is not None:
                    Q_integ = Q_scenario
                    Extras_integ = Extras_scenario
                else:
                    Q_integ = self.Q
                    Extras_integ = self.X_extras

                # Covariate Part: Matrix Multiplication
                # (N_samples, 1) * (1, N_integration_points) -> (N_samples, N_integration_points)
                # But to save memory, we can do it row-wise or broadcast if N*M is small.
                # Let's try broadcasting but be mindful of memory.

                # If N*M > 10^8 (e.g. 10k * 10k), we should loop/batch.
                # Assuming typical use (daily data for 10-20 years ~ 7000 points).
                # 7000*7000 = 49M doubles = ~400MB. Safe.

                term_cov = all_betas[:, 1][:, np.newaxis] * Q_integ[np.newaxis, :]

                term_extras = 0
                if Extras_integ is not None:
                    for k in range(Extras_integ.shape[1]):
                        # beta_ext * Ext_hist
                        # Beta index is 5 + k
                        b = all_betas[:, 5 + k][:, np.newaxis]
                        e = Extras_integ[:, k][np.newaxis, :]
                        term_extras += b * e

                log_preds = const_part[:, np.newaxis] + term_cov + term_extras
                results = np.nanmean(np.exp(log_preds), axis=1)

                # If betas were nan, results are nan.
                return results

            else:
                # Case B: GFN (Windowed) - Integration set changes per row
                # We have pre-computed betas, but we must loop for integration
                # to select the window.
                h_window = gfn_window / 2
                results = []
                n = len(self.df)

                for i in range(n):
                    if np.isnan(all_betas[i, 0]):
                        results.append(np.nan)
                        continue

                    t_curr = self.T[i]
                    # Select window
                    # This boolean masking is somewhat slow but much faster than fitting
                    mask = np.abs(self.T - t_curr) <= h_window

                    if not np.any(mask):
                        # Fallback to full record if empty window
                        Q_local = self.Q
                        Extras_local = self.X_extras
                        weights = None
                    else:
                        Q_local = self.Q[mask]
                        Extras_local = self.X_extras[mask, :] if self.X_extras is not None else None

                        # Weighted Integration (GFN uses tricube weights based on time distance)
                        dist_t = self.T[mask] - t_curr
                        w_t = self._tricube(dist_t, h_window)

                        # Normalize weights
                        if np.sum(w_t) > 0:
                            weights = w_t / np.sum(w_t)
                        else:
                            # Fallback if sum is 0 (unlikely with tricube in window unless all at edge)
                            weights = np.ones_like(w_t) / len(w_t)

                    # Calculate mean
                    # const_part[i] + beta1[i]*Q + ...
                    pred_log = const_part[i] + all_betas[i, 1] * Q_local

                    if Extras_local is not None:
                        for k in range(Extras_local.shape[1]):
                            pred_log += all_betas[i, 5 + k] * Extras_local[:, k]

                    if weights is not None:
                         results.append(np.sum(np.exp(pred_log) * weights))
                    else:
                         results.append(np.mean(np.exp(pred_log)))

                return np.array(results)

        else:
            # Exact Method (Loop)
            print(f"Starting Decanting Process (Method: Exact)...")
            results = []
            for i, row in self.df.iterrows():
                t_current = row['decimal_time']
                s_current = row['season']
                q_current = row['log_covariate']

                # Extract extras for this row if they exist
                extras_current = None
                if self.X_extras is not None:
                    extras_current = self.X_extras[i, :]

                betas = self.fit_local_model(t_current, q_current, s_current, h_params, extras_current)

                if betas is None:
                    results.append(np.nan)
                    continue

                # Flow Normalization (Integration)
                const_part = (betas[0] +
                              betas[2] * t_current +
                              betas[3] * np.sin(2*np.pi*t_current) +
                              betas[4] * np.cos(2*np.pi*t_current))

                # Select covariates for integration
                if integration_scenarios is not None:
                    Q_integration = Q_scenario
                    Extras_integration = Extras_scenario
                elif gfn_window is not None:
                    h_window = gfn_window / 2
                    mask = np.abs(self.T - t_current) <= h_window
                    Q_integration = self.Q[mask]
                    if self.X_extras is not None:
                        Extras_integration = self.X_extras[mask, :]
                    if len(Q_integration) == 0:
                         Q_integration = self.Q
                         if self.X_extras is not None:
                             Extras_integration = self.X_extras
                else:
                    Q_integration = self.Q
                    if self.X_extras is not None:
                        Extras_integration = self.X_extras

                cov_part = betas[1] * Q_integration

                extra_part = 0
                if self.X_extras is not None:
                    for k in range(len(self.extra_cov_config)):
                        beta_val = betas[5 + k]
                        col_vals = Extras_integration[:, k]
                        extra_part += beta_val * col_vals

                predictions_log = const_part + cov_part + extra_part
                predictions = np.exp(predictions_log)
                t1_val = np.mean(predictions)
                results.append(t1_val)

            return results

    def predict(self, new_data_df, use_grid=True):
        """
        Out-of-sample prediction using the fitted model.
        Returns a DataFrame with 'estimated' (point prediction) and 'decanted' (flow normalized) columns.
        """
        if use_grid and self.interpolators is None:
             raise ValueError("Model grid not found. Run decant_series(use_grid=True) or load_model() first.")

        # Prepare new data
        # We need derived columns: decimal_time, log_covariate, extras

        df_new = new_data_df.copy()

        # Check required columns
        req_cols = [self.orig_date_col, self.orig_covariate_col]
        for c in req_cols:
            if c not in df_new.columns:
                raise ValueError(f"New data missing column: {c}")

        # Transform
        df_new['date'] = pd.to_datetime(df_new[self.orig_date_col])
        df_new['decimal_time'] = df_new['date'].dt.year + (df_new['date'].dt.dayofyear - 1) / 365.25

        if (df_new[self.orig_covariate_col] <= 0).any():
             raise ValueError("New data contains non-positive covariate values.")
        df_new['log_covariate'] = np.log(df_new[self.orig_covariate_col])

        T_new = df_new['decimal_time'].values
        Q_new = df_new['log_covariate'].values

        X_extras_new = None
        if self.extra_cov_config:
            extra_vals = []
            for conf in self.extra_cov_config:
                col = conf['col']
                if col not in df_new.columns:
                     raise ValueError(f"New data missing extra covariate: {col}")

                is_log = conf.get('log', False)
                vals = df_new[col].values
                if is_log:
                     if (vals <= 0).any(): raise ValueError("Non-positive extra covariate.")
                     vals = np.log(vals)
                extra_vals.append(vals)
            X_extras_new = np.column_stack(extra_vals)

        # Prediction

        if use_grid:
            # Check for Extrapolation
            # Grid boundaries are in self.grid_axes
            # grid_axes[0] is T, [1] is Q, etc.

            # Check Time
            t_min, t_max = self.grid_axes[0].min(), self.grid_axes[0].max()
            if (T_new < t_min).any() or (T_new > t_max).any():
                warnings.warn("New data contains time values outside the training grid. Extrapolation will be used.", UserWarning)

            # Check Covariate
            q_min, q_max = self.grid_axes[1].min(), self.grid_axes[1].max()
            if (Q_new < q_min).any() or (Q_new > q_max).any():
                 warnings.warn("New data contains covariate values outside the training grid. Extrapolation will be used.", UserWarning)

            # 1. Interpolate Betas
            cols = [T_new, Q_new]
            if X_extras_new is not None:
                for k in range(X_extras_new.shape[1]):
                    cols.append(X_extras_new[:, k])
            pts = np.column_stack(cols)

            # (N_new, N_betas)
            all_betas = np.column_stack([interp(pts) for interp in self.interpolators])

            # 2. Estimated Series (Point prediction)
            est_log = (all_betas[:, 0] +
                       all_betas[:, 1] * Q_new +
                       all_betas[:, 2] * T_new +
                       all_betas[:, 3] * np.sin(2 * np.pi * T_new) +
                       all_betas[:, 4] * np.cos(2 * np.pi * T_new))

            if X_extras_new is not None:
                for k in range(X_extras_new.shape[1]):
                    est_log += all_betas[:, 5 + k] * X_extras_new[:, k]

            estimated = np.exp(est_log)

            # 3. Decanted Series (Stationary Normalization)
            # Integrate over HISTORICAL Q (self.Q), not new data Q.
            # Assuming Stationary Flow Normalization.

            const_part = (all_betas[:, 0] +
                          all_betas[:, 2] * T_new +
                          all_betas[:, 3] * np.sin(2*np.pi*T_new) +
                          all_betas[:, 4] * np.cos(2*np.pi*T_new))

            # Broadcasting: (N_new, 1) + (N_new, 1) * (1, M_hist) ...
            term_cov = all_betas[:, 1][:, np.newaxis] * self.Q[np.newaxis, :]

            term_extras = 0
            if self.X_extras is not None:
                for k in range(self.X_extras.shape[1]):
                    b = all_betas[:, 5 + k][:, np.newaxis]
                    e = self.X_extras[:, k][np.newaxis, :]
                    term_extras += b * e

            log_preds = const_part[:, np.newaxis] + term_cov + term_extras
            decanted = np.nanmean(np.exp(log_preds), axis=1)

            return pd.DataFrame({'estimated': estimated, 'decanted': decanted}, index=df_new.index)

        else:
             raise NotImplementedError("Out-of-sample prediction only supported for Grid method.")

    def bootstrap_uncertainty(self, h_params, n_bootstraps=100, block_size=30, use_grid=True, grid_config={'n_t':100, 'n_q':15}, method='block'):
        """
        Estimates uncertainty bands using Bootstrap on residuals.

        method: 'block' (Standard Moving Block Bootstrap) or 'wild' (Wild Bootstrap for heteroscedasticity).

        Returns: DataFrame with columns ['mean', 'p05', 'p95'] for the decanted series.
        """
        print(f"Starting Bootstrap Uncertainty Analysis ({method} method, {n_bootstraps} runs)...")

        # 1. Calculate Initial Residuals (from Grid model)
        # We need the estimated values (NOT flow normalized) to get residuals
        preds_log = self.get_estimated_series(h_params, use_grid=use_grid, grid_config=grid_config)

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
                flips = np.random.choice([-1, 1], size=n)
                boot_res = residuals * flips

            elif method == 'block':
                current_idx = 0
                while current_idx < n:
                    start = np.random.randint(0, len(valid_res) - block_size)
                    chunk = valid_res[start : start + block_size]
                    end_idx = min(current_idx + block_size, n)
                    needed = end_idx - current_idx
                    boot_res[current_idx : end_idx] = chunk[:needed]
                    current_idx += block_size
            else:
                 raise ValueError(f"Unknown bootstrap method: {method}")

            # Create synthetic response
            # Ensure we don't propagate NaNs from preds_log if model failed
            new_log_response = preds_log + boot_res

            # If original data was NaN, keep it NaN
            new_log_response[np.isnan(self.Y)] = np.nan

            # If prediction failed (NaN), we cannot create valid synthetic data for that point
            # We treat it as NaN (missing) in the synthetic dataset
            # (Decanter handles missing response by ignoring it in fit_local_model)

            df_boot = self.df.copy()
            df_boot[self.orig_response_col] = np.exp(new_log_response)

            dec_boot = Decanter(df_boot, self.orig_date_col, self.orig_response_col, self.orig_covariate_col, extra_covariates=self.extra_cov_config)

            # Re-run decant
            # Important: dec_boot will recompute its own grid since we re-init
            res = dec_boot.decant_series(h_params, use_grid=use_grid, grid_config=grid_config)
            results_matrix[b, :] = res

        # 3. Aggregate
        df_results = pd.DataFrame({
            'mean': np.nanmean(results_matrix, axis=0),
            'p05': np.nanpercentile(results_matrix, 5, axis=0),
            'p95': np.nanpercentile(results_matrix, 95, axis=0)
        })

        return df_results

    def add_kalman_correction(self, estimated_log_series=None, h_params=None, rho=0.9, use_grid=False, grid_config={'n_t':100, 'n_q':15}):
        """
        Applies WRTDS-Kalman correction (AR1 filtering of residuals).

        Args:
            estimated_log_series: array of log-space predictions. If None, computed via h_params.
            h_params: dict of WRTDS parameters. Required if estimated_log_series is None.
            rho: autocorrelation coefficient (0.8 - 0.95 typical)
            use_grid: bool, for on-the-fly estimation
            grid_config: dict, for on-the-fly estimation

        Returns: series with Kalman correction applied (in log space).
        """
        if estimated_log_series is None:
             if h_params is None:
                 raise ValueError("Must provide either estimated_log_series or h_params")
             estimated_log_series = self.get_estimated_series(h_params, use_grid, grid_config)

        # Calculate residuals where we have observations
        # res = log_obs - log_model

        n = len(self.df)
        residuals = np.full(n, np.nan)

        # Only calculate for valid observations
        valid_mask = ~np.isnan(self.Y)
        residuals[valid_mask] = self.Y[valid_mask] - estimated_log_series[valid_mask]

        # Interpolate residuals to all days using AR(1) decay via Vectorized operations
        df_kalman = pd.DataFrame({
            'T': self.T,
            'Res': residuals
        })

        # Forward Pass
        df_kalman['Last_Res'] = df_kalman['Res'].ffill()
        df_kalman['Last_T'] = df_kalman['T'].where(~df_kalman['Res'].isna()).ffill()
        df_kalman['Last_T'] = df_kalman['Last_T'].fillna(-9999)
        dt_forward = (df_kalman['T'] - df_kalman['Last_T']) * 365.25
        forward_res = (df_kalman['Last_Res'].fillna(0) * (rho ** dt_forward)).values

        # WRTDS-K uses Forward-Only AR(1) propagation (Causal)
        interpolated_residuals = forward_res

        # Ensure observed residuals are exact
        interpolated_residuals[valid_mask] = residuals[valid_mask]

        return estimated_log_series + interpolated_residuals
