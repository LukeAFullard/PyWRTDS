import numpy as np
import pandas as pd
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator
import itertools
import pickle

def to_decimal_date(date_series):
    """
    Converts pandas datetime series to decimal year, respecting leap years.
    Matches EGRET/lubridate: year + (doy - 1) / days_in_year
    """
    dt = pd.to_datetime(date_series)
    year = dt.dt.year
    doy = dt.dt.dayofyear
    is_leap = dt.dt.is_leap_year
    days_in_year = np.where(is_leap, 366, 365)
    return year + (doy - 1) / days_in_year

def get_adjusted_doy(date_series):
    """
    Returns day of year (1-366).
    For non-leap years, days >= 60 (Mar 1+) are shifted by +1.
    This ensures Mar 1 is always 61, Feb 29 is 60.
    """
    dt = pd.to_datetime(date_series)
    doy = dt.dt.dayofyear
    is_leap = dt.dt.is_leap_year

    vals = doy.values.copy()
    # Shift non-leap days after Feb 28 (doy 59)
    # If not leap, doy 60 is Mar 1. We want it to be 61.
    mask = (~is_leap.values) & (vals >= 60)
    vals[mask] += 1
    return vals

class Decanter:
    def __init__(self, data, date_col, response_col, covariate_col, extra_covariates=None, daily_data=None):
        """
        data: pandas DataFrame (Sample Data for Calibration)
        date_col: string name of date column
        response_col: string name of t0 (target)
        covariate_col: string name of c0 (driver)
        extra_covariates: list of dicts, e.g. [{'col': 'Temp', 'log': False}]
        daily_data: pandas DataFrame (Full Daily Record for Flow Normalization) - Optional but recommended for accurate FN.
        """
        self.orig_date_col = date_col
        self.orig_response_col = response_col
        self.orig_covariate_col = covariate_col
        self.extra_cov_config = extra_covariates if extra_covariates else []

        self.df = data.copy()
        self.daily_history = daily_data.copy() if daily_data is not None else None

        if self.daily_history is not None:
             if self.orig_date_col not in self.daily_history.columns:
                  raise ValueError(f"Daily data must contain date column '{self.orig_date_col}'")
             if self.orig_covariate_col not in self.daily_history.columns:
                  raise ValueError(f"Daily data must contain covariate column '{self.orig_covariate_col}'")

             # Process daily history
             self.daily_history['date'] = pd.to_datetime(self.daily_history[self.orig_date_col])
             # Use new decimal date logic
             self.daily_history['decimal_time'] = to_decimal_date(self.daily_history['date'])

             if (self.daily_history[self.orig_covariate_col] <= 0).any():
                  raise ValueError("Daily history contains non-positive covariate values.")
             self.daily_history['log_covariate'] = np.log(self.daily_history[self.orig_covariate_col])

             # Extra covariates in daily history
             if self.extra_cov_config:
                 for conf in self.extra_cov_config:
                     col = conf['col']
                     if col not in self.daily_history.columns:
                         raise ValueError(f"Daily history missing extra covariate '{col}'")

                     is_log = conf.get('log', False)
                     if is_log:
                         if (self.daily_history[col] <= 0).any():
                             raise ValueError(f"Daily history extra covariate '{col}' contains non-positive values.")
                         self.daily_history[f"log_{col}"] = np.log(self.daily_history[col])
                     else:
                         self.daily_history[f"lin_{col}"] = self.daily_history[col]

        # Convert dates to Decimal Time (e.g., 2023.5)
        # This simplifies the math for 't'
        self.df['date'] = pd.to_datetime(self.df[date_col])
        self.df['decimal_time'] = to_decimal_date(self.df['date'])
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
        self.conc_interpolator = None # For surface interpolation
        self.h_params_last = None

        # Determine boundaries for edge adjustment
        # Use daily history if available (EGRET logic uses Daily range), else Sample range
        if self.daily_history is not None:
            self.t_min_bound = np.min(self.daily_history['decimal_time'].values)
            self.t_max_bound = np.max(self.daily_history['decimal_time'].values)
        else:
            self.t_min_bound = np.min(self.T)
            self.t_max_bound = np.max(self.T)

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

        # Edge Adjustment removed here as it is handled in fit_local_model
        # But wait, if get_weights is called directly (not via fit_local_model), we might want it?
        # Typically get_weights is internal.
        # But we loop fit_local_model calling get_weights.
        # fit_local_model applies edge adjustment logic to h_time before loop.
        # So we should use the passed h_time directly here.
        pass

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

    def fit_local_model(self, t_target, q_target, s_target, h_params, extras_target=None, min_obs=100, min_uncen=50):
        """
        Performs Weighted Least Squares for a specific target point.
        Returns the coefficients.
        """
        # Iterative Window Expansion (EGRET Logic)
        h_time = h_params.get('h_time', 7)
        h_cov = h_params.get('h_cov', 2)
        h_season = h_params.get('h_season', 0.5)

        # Determine edge-adjusted initial h_time (logic previously in get_weights, now here/loop)
        # Note: get_weights implemented edge adjustment. But if we loop, we need to apply logic to temp h_time?
        # EGRET applies edge adjustment logic to the initial h_time?
        # run_WRTDS:
        # distTime = min(distLow, distHigh)
        # if (edgeAdjust) tempWindowY <- if (distTime > tempWindowY) tempWindowY else ((2 * tempWindowY) - distTime)
        # Then loop: tempWindowY *= 1.1

        # So I should handle edge adjustment logic here based on current h_time
        t_min, t_max = self.t_min_bound, self.t_max_bound
        dist_to_edge = min(t_target - t_min, t_max - t_target)

        # Initial adjustment
        if dist_to_edge < h_time:
             h_time = 2 * h_time - dist_to_edge

        # Loop
        max_iter = 100
        for _ in range(max_iter):
            # Calculate weights
            # We call _tricube directly or get_weights?
            # get_weights has the combination logic.
            # But get_weights currently has edge adjustment logic baked in.
            # I should refactor get_weights to take explicit h values, or pass params.
            # Passing params is cleaner.

            curr_params = h_params.copy()
            curr_params['h_time'] = h_time
            curr_params['h_cov'] = h_cov
            curr_params['h_season'] = h_season
            # Note: extra covariates h params are not expanded in EGRET? EGRET doesn't have them.
            # We will keep them fixed or expand? Let's keep fixed for now.

            # Temporarily disable edge adjustment in get_weights because we handled it?
            # Or assume get_weights re-calculates edge adjustment?
            # If we update h_time here, get_weights will apply edge adjustment to the NEW h_time?
            # EGRET: `tempWindowY` is the state.
            # `run_WRTDS` sets `tempWindowY` (possibly edge adjusted) at start.
            # Then loop multiplies `tempWindowY`.
            # So edge adjustment happens ONCE at start.
            # My `get_weights` calculates edge adjustment dynamically based on passed `h_time`.
            # If I pass `h_time` that is already adjusted (or not), `get_weights` will adjust it AGAIN?
            # Yes, currently `get_weights` logic: `if dist < h_time: h_time = ...`.
            # If I want to control it here, I should modify `get_weights` to NOT do it, or pass a flag.
            # Or rely on `get_weights` doing it correctly for every iteration?
            # EGRET: `tempWindowY` grows. The "edge adjustment" is just initializing `tempWindowY` to a larger value if near edge.
            # The expansion is multiplicative on that larger value.
            # If `get_weights` recalculates:
            # Iter 1: h=7. Dist=1. New_h = 13.
            # Iter 2: h=7.7. Dist=1. New_h = 2*7.7 - 1 = 14.4.
            # 13 * 1.1 = 14.3. Close.
            # So letting `get_weights` handle it seems fine/consistent.

            W = self.get_weights(t_target, q_target, s_target, curr_params, extras_target)

            valid_obs = ~np.isnan(self.Y)
            mask = (W > 0) & valid_obs

            num_pos_wt = np.sum(mask)
            # Assuming all valid obs are "Uncen" for now (since we don't track censoring column explicitly yet)
            num_uncen = num_pos_wt

            if num_pos_wt >= min_obs and num_uncen >= min_uncen:
                break

            # Expand
            h_time *= 1.1
            h_cov *= 1.1
            if h_season < 0.5:
                h_season = min(h_season * 1.1, 0.5)

        # Proceed with WLS
        weights_active = W[mask]

        # Normalize weights to sum to N (approx), matching EGRET logic
        # EGRET: weight <- weight / mean(weight). sum(weight) = numPosWt.
        # This affects the magnitude of residuals and thus Sigma.
        mean_w = np.mean(weights_active)
        if mean_w > 0:
            weights_active = weights_active / mean_w

        y_active = self.Y[mask]

        # Dimension of model: 5 (Standard) + N_extras
        n_extras = self.X_extras.shape[1] if self.X_extras is not None else 0
        n_params = 5 + n_extras

        if np.sum(mask) < max(10, n_params + 2):
             return None

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

            # Calculate standard error of the regression (sigma) for bias correction
            # residuals from lstsq is sum of squared residuals (weighted SSE)
            if residuals.size > 0:
                sse = residuals[0]
            else:
                 # If residuals empty (perfect fit or undetermined), calculate manually
                 # or return nan if rank issue
                 sse = np.sum((y_w - X_w @ betas)**2)

            # Degrees of freedom
            # EGRET uses survreg which performs MLE, using N instead of N-p for variance.
            # To match EGRET exactly, we use dof = n_active.
            dof = n_active
            if dof > 0:
                sigma = np.sqrt(sse / dof)
            else:
                sigma = np.nan

            # Append sigma to betas
            return np.append(betas, sigma)

        except np.linalg.LinAlgError:
            return None

    def predict_point(self, t_target, q_target, betas, extras_target=None, return_log=False):
        """
        Predicts value using the locally fitted betas.
        Betas array is expected to have sigma as the last element.
        """
        if betas is None: return np.nan

        # Extract sigma (last element) and regression betas
        sigma = betas[-1]
        reg_betas = betas[:-1]

        # prediction_log = beta0 + beta1*q + beta2*t + beta3*sin + beta4*cos
        # Note: sin/cos depend on t_target, not q_target
        prediction_log = (reg_betas[0] +
                          reg_betas[1] * q_target +
                          reg_betas[2] * t_target +
                          reg_betas[3] * np.sin(2*np.pi*t_target) +
                          reg_betas[4] * np.cos(2*np.pi*t_target))

        # Add contributions from extra covariates
        # extras_target: [val1, val2...]
        if extras_target is not None:
            # Betas for extras start at index 5
            for i, val in enumerate(extras_target):
                prediction_log += reg_betas[5 + i] * val

        if return_log:
            return prediction_log

        # Apply Bias Correction for linear space
        # exp(y + sigma^2/2)
        if not np.isnan(sigma):
             bias = np.exp(sigma**2 / 2)
        else:
             bias = 1.0

        return np.exp(prediction_log) * bias

    def compute_grid(self, h_params, n_t=None, n_q=None, n_extra=7, min_obs=100):
        """
        Computes the regression surfaces (betas) on a regular grid.
        Returns a tuple of (grid_axes_tuple, betas_grid)

        Defaults match EGRET logic if n_t/n_q are None.
        grid_axes_tuple: (t_grid, q_grid, extra1_grid, ...)
        betas_grid: N-D array of coefficients
        """
        n_extras = self.X_extras.shape[1] if self.X_extras is not None else 0

        # 1. Define grid boundaries for Time and Primary Covariate
        t_min, t_max = self.T.min(), self.T.max()
        q_min, q_max = self.Q.min(), self.Q.max()

        # EGRET Logic for Grid Generation
        # LogQ: 14 points, min-0.05 to max+0.05
        # Time: 16 steps per year.

        # Q Grid
        if n_q is None:
             q_bottom = q_min - 0.05
             q_top = q_max + 0.05
             n_q = 14
             q_grid = np.linspace(q_bottom, q_top, n_q)
        else:
            if q_min == q_max:
                q_min -= 0.01
                q_max += 0.01
            q_grid = np.linspace(q_min, q_max, n_q)

        # T Grid
        if n_t is None:
            t_bottom = np.floor(t_min)
            t_top = np.ceil(t_max)
            step_year = 1/16.0
            t_grid = np.arange(t_bottom, t_top + step_year/1000, step_year)
            n_t = len(t_grid)
        else:
             if t_min == t_max:
                t_min -= 0.01
                t_max += 0.01
             t_grid = np.linspace(t_min, t_max, n_t)

        grid_dims_str = f"{n_t} x {n_q}"
        if n_extras > 0:
            grid_dims_str += f" x {n_extra}" * n_extras

        print(f"Computing Grid ({grid_dims_str})...")

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
        # Shape: (n_t, n_q, n_e1, ..., n_params + 1) -> +1 for sigma
        shape = tuple(len(g) for g in grid_axes) + (5 + n_extras + 1,)
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

            betas = self.fit_local_model(t_val, q_val, s_val, h_params, extras_val, min_obs=min_obs)

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

        # grid_config might contain min_obs?
        # Typically grid_config has n_t, n_q. min_obs is usually separate or in grid_config?
        # Let's support passing min_obs via grid_config if present, or default.

        self.grid_axes, self.betas_grid = self.compute_grid(h_params, **grid_config)
        self.h_params_last = h_params
        self._build_interpolators_from_grid()

        return self.interpolators

    def _build_interpolators_from_grid(self):
        """
        Reconstructs interpolators from self.grid_axes and self.betas_grid.
        Also builds Concentration Surface interpolator.
        """
        if self.grid_axes is None or self.betas_grid is None:
            return

        n_params_grid = self.betas_grid.shape[-1]
        self.interpolators = []

        for k in range(n_params_grid):
            grid_slice = self.betas_grid[..., k]

            # Legacy 2D filling (if needed, but usually grid is complete or has NaNs)
            # RegularGridInterpolator handles NaNs with fill_value=nan? No, it propagates NaNs.
            # EGRET fills holes? We assume compute_grid returns full grid (maybe with NaNs).

            interp = RegularGridInterpolator(self.grid_axes, grid_slice, bounds_error=False, fill_value=None)
            self.interpolators.append(interp)

        # Build Concentration Surface Grid
        # Evaluate model at every grid point
        # Grid axes: (T, Q, E...)
        # We need to construct meshgrid logic to evaluate the polynomial

        # Helper to generate meshgrid points
        # meshgrid with indexing='ij' matches array layout
        grids = np.meshgrid(*self.grid_axes, indexing='ij')

        # Extract components
        T_grid = grids[0]
        Q_grid = grids[1]

        # Betas slices
        # betas_grid shape: (N_T, N_Q, ..., N_Params)
        beta0 = self.betas_grid[..., 0]
        beta1 = self.betas_grid[..., 1]
        beta2 = self.betas_grid[..., 2]
        beta3 = self.betas_grid[..., 3]
        beta4 = self.betas_grid[..., 4]
        sigma = self.betas_grid[..., -1]

        yHat_grid = (beta0 +
                     beta1 * Q_grid +
                     beta2 * T_grid +
                     beta3 * np.sin(2 * np.pi * T_grid) +
                     beta4 * np.cos(2 * np.pi * T_grid))

        # Extras
        n_extras = len(self.extra_cov_config)
        if n_extras > 0:
            for k in range(n_extras):
                E_grid = grids[2 + k]
                beta_k = self.betas_grid[..., 5 + k]
                yHat_grid += beta_k * E_grid

        # Bias Correction
        bias_grid = np.exp(sigma**2 / 2)
        # Handle NaNs in sigma (bias=1)
        bias_grid = np.nan_to_num(bias_grid, nan=1.0)

        # Concentration Grid
        # We store yHat and bias separately to interpolate in log space if needed
        self.yHat_grid = yHat_grid
        self.bias_grid = bias_grid

        # Legacy direct concentration interpolator (Default Surface)
        conc_grid = np.exp(yHat_grid) * bias_grid
        self.conc_interpolator = RegularGridInterpolator(self.grid_axes, conc_grid, bounds_error=False, fill_value=None)

        # New: Log Surface Interpolators
        self.log_conc_interpolator = RegularGridInterpolator(self.grid_axes, yHat_grid, bounds_error=False, fill_value=None)
        self.bias_interpolator = RegularGridInterpolator(self.grid_axes, bias_grid, bounds_error=False, fill_value=None)

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

        # Rebuild interpolators
        self._build_interpolators_from_grid()

    def get_estimated_series(self, h_params, use_grid=False, grid_config={'n_t':None, 'n_q':None}, min_obs=100):
        """
        Returns the model predictions (in log space) for the observed data points.
        This represents the "trend + seasonality + flow effect" component.
        """
        # Inject min_obs into grid_config if using grid
        if use_grid:
            gc = grid_config.copy()
            if 'min_obs' not in gc:
                gc['min_obs'] = min_obs

            if self.interpolators is None:
                self._prepare_grid_interpolators(h_params, gc)

            # Vectorized Interpolation
            # Construct points array (N, D)
            cols = [self.T, self.Q]
            if self.X_extras is not None:
                for k in range(self.X_extras.shape[1]):
                    cols.append(self.X_extras[:, k])

            pts = np.column_stack(cols)

            # Predict all betas at once: (N, n_betas + 1)
            # List comprehension over interpolators
            betas_cols = [interp(pts) for interp in self.interpolators]
            all_params = np.column_stack(betas_cols)
            all_betas = all_params[:, :-1]

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

                betas = self.fit_local_model(t_current, q_current, s_current, h_params, extras_current, min_obs=min_obs)

                if betas is None:
                    preds_log.append(np.nan)
                else:
                    # Predict in log space
                    val = self.predict_point(t_current, q_current, betas, extras_current, return_log=True)
                    preds_log.append(val)

            return np.array(preds_log)

    def decant_series(self, h_params={'h_time':7, 'h_cov':2, 'h_season':0.5}, use_grid=False, grid_config={'n_t':None, 'n_q':None}, gfn_window=None, integration_scenarios=None, min_obs=100, interp_method='surface'):
        """
        Generates the cleaned time series t1 (Flow Normalized).
        interp_method: 'surface' (interpolate Conc), 'log_surface' (interpolate LogConc then Exp), or 'coefficients'.
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
            gc = grid_config.copy()
            if 'min_obs' not in gc:
                gc['min_obs'] = min_obs

            if self.interpolators is None:
                self._prepare_grid_interpolators(h_params, gc)

            # Pre-calculation for Coefficient Method
            all_betas = None
            bias_factors = None
            const_part = None

            if interp_method == 'coefficients' or (integration_scenarios is not None):
                # Fallback to coefficients for scenarios if surface not implemented for scenarios yet
                # Or simply always compute betas if we might need them.
                # 1. Vectorized Beta Interpolation
                cols = [self.T, self.Q]
                if self.X_extras is not None:
                    for k in range(self.X_extras.shape[1]):
                        cols.append(self.X_extras[:, k])
                pts = np.column_stack(cols)

                # (N_samples, N_betas + 1)
                all_params = np.column_stack([interp(pts) for interp in self.interpolators])

                # Split betas and sigma
                all_betas = all_params[:, :-1]
                all_sigmas = all_params[:, -1]

                # Calculate Bias Correction Factor per point
                bias_factors = np.exp(all_sigmas**2 / 2)
                bias_factors = np.nan_to_num(bias_factors, nan=1.0)

                # Constant part per time step
                const_part = (all_betas[:, 0] +
                              all_betas[:, 2] * self.T +
                              all_betas[:, 3] * np.sin(2*np.pi*self.T) +
                              all_betas[:, 4] * np.cos(2*np.pi*self.T))

            # Identify Integration Source (Q_integration)
            # Three cases: Stationary (Day-of-Year Specific), Scenarios, GFN

            # Case A: Scenarios or Stationary
            if gfn_window is None:
                if integration_scenarios is not None:
                    # If scenarios provided, integrate over scenarios (Stationary assumption over scenarios?)
                    # Usually scenarios provide Q for specific dates?
                    # If scenario is a single vector (distribution), we use it for all T.
                    Q_integ = Q_scenario
                    Extras_integ = Extras_scenario

                    term_cov = all_betas[:, 1][:, np.newaxis] * Q_integ[np.newaxis, :]
                    term_extras = 0
                    if Extras_integ is not None:
                        for k in range(Extras_integ.shape[1]):
                            b = all_betas[:, 5 + k][:, np.newaxis]
                            e = Extras_integ[:, k][np.newaxis, :]
                            term_extras += b * e

                    log_preds = const_part[:, np.newaxis] + term_cov + term_extras
                    linear_preds = np.exp(log_preds) * bias_factors[:, np.newaxis]
                    results = np.nanmean(linear_preds, axis=1)
                    return results

                else:
                    # EGRET Stationary Flow Normalization
                    # Integrate over historical Qs conditioned on Day of Year.
                    # Q_integ changes for each row (depends on day of year).

                    # Pre-calculate day of year for historical data
                    if self.daily_history is not None:
                        source_df = self.daily_history
                        doy_hist_vals = get_adjusted_doy(source_df['date'])
                        source_Q = source_df['log_covariate'].values
                        if self.X_extras is not None:
                             extras_cols = [self.daily_history[col].values for col in self.extra_cov_names]
                             source_Extras = np.column_stack(extras_cols)
                        else:
                             source_Extras = None
                    else:
                        doy_hist_vals = get_adjusted_doy(self.df[self.orig_date_col])
                        source_Q = self.Q
                        source_Extras = self.X_extras

                    # Target days
                    doy_target = get_adjusted_doy(self.df[self.orig_date_col])

                    # Optimization: Group by DOY
                    # Unique DOYs (1-366)
                    # For each unique DOY in target, find matching historical Qs

                    # Create a map of DOY -> Indices in history
                    # This avoids searching every time
                    doy_map = {}
                    for d in range(1, 368):
                        doy_map[d] = np.where(doy_hist_vals == d)[0]

                    results = np.full(len(self.df), np.nan)

                    # We can iterate over unique DOYs in target to vectorize chunks
                    unique_doys = np.unique(doy_target)

                    for d in unique_doys:
                        target_indices = np.where(doy_target == d)[0]
                        hist_indices = doy_map.get(d, np.array([]))

                        # Special handling for Feb 28 (59) and Feb 29 (60)
                        # Merge them to ensure sufficient data and continuity
                        if d == 59 or d == 60:
                            hist_indices = np.concatenate([doy_map.get(59, []), doy_map.get(60, [])])

                        if len(hist_indices) == 0:
                            continue # Should not happen with valid data

                        hist_indices = hist_indices.astype(int)

                        Q_local = source_Q[hist_indices]
                        Extras_local = source_Extras[hist_indices, :] if source_Extras is not None else None

                        # Calculation for this chunk
                        # target_indices are the rows in self.df
                        n_chunk = len(target_indices)
                        n_hist = len(hist_indices)

                        if interp_method == 'coefficients':
                            # betas: (N_chunk, N_params)
                            chunk_betas = all_betas[target_indices]
                            chunk_const = const_part[target_indices]
                            chunk_bias = bias_factors[target_indices]

                            # (N_chunk, 1) * (1, N_hist) -> (N_chunk, N_hist)
                            term_cov = chunk_betas[:, 1][:, np.newaxis] * Q_local[np.newaxis, :]

                            term_extras = 0
                            if Extras_local is not None:
                                for k in range(Extras_local.shape[1]):
                                    b = chunk_betas[:, 5 + k][:, np.newaxis]
                                    e = Extras_local[:, k][np.newaxis, :]
                                    term_extras += b * e

                            log_preds = chunk_const[:, np.newaxis] + term_cov + term_extras
                            linear_preds = np.exp(log_preds) * chunk_bias[:, np.newaxis]

                            chunk_results = np.nanmean(linear_preds, axis=1)
                            results[target_indices] = chunk_results

                        elif interp_method == 'surface' or interp_method == 'log_surface':
                            # Construct all pairs (T_target, Q_hist)
                            # T_target: self.T[target_indices]
                            T_chunk = self.T[target_indices]

                            # Meshgrid logic manual broadcast
                            # T: (N_chunk, 1) -> (N_chunk, N_hist)
                            T_eval = np.broadcast_to(T_chunk[:, np.newaxis], (n_chunk, n_hist)).ravel()
                            # Q: (1, N_hist) -> (N_chunk, N_hist)
                            Q_eval = np.broadcast_to(Q_local[np.newaxis, :], (n_chunk, n_hist)).ravel()

                            eval_cols = [T_eval, Q_eval]
                            if Extras_local is not None:
                                # Extras correspond to Q_hist.
                                # Extra k: (1, N_hist) -> (N_chunk, N_hist)
                                for k in range(Extras_local.shape[1]):
                                    E_k = Extras_local[:, k]
                                    E_eval = np.broadcast_to(E_k[np.newaxis, :], (n_chunk, n_hist)).ravel()
                                    eval_cols.append(E_eval)

                            eval_pts = np.column_stack(eval_cols)

                            if interp_method == 'surface':
                                # Interpolate Conc directly
                                conc_vals = self.conc_interpolator(eval_pts)
                            else: # log_surface
                                log_vals = self.log_conc_interpolator(eval_pts)
                                bias_vals = self.bias_interpolator(eval_pts)
                                conc_vals = np.exp(log_vals) * bias_vals

                            # Reshape to (N_chunk, N_hist) and mean
                            conc_matrix = conc_vals.reshape(n_chunk, n_hist)
                            chunk_results = np.nanmean(conc_matrix, axis=1)
                            results[target_indices] = chunk_results

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
                    else:
                        Q_local = self.Q[mask]
                        Extras_local = self.X_extras[mask, :] if self.X_extras is not None else None

                    # Calculate mean
                    # const_part[i] + beta1[i]*Q + ...
                    pred_log = const_part[i] + all_betas[i, 1] * Q_local

                    if Extras_local is not None:
                        for k in range(Extras_local.shape[1]):
                            pred_log += all_betas[i, 5 + k] * Extras_local[:, k]

                    pred_linear = np.exp(pred_log) * bias_factors[i]
                    results.append(np.mean(pred_linear))

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

                betas = self.fit_local_model(t_current, q_current, s_current, h_params, extras_current, min_obs=min_obs)

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

                # Bias correction
                sigma = betas[-1]
                if not np.isnan(sigma):
                     bias = np.exp(sigma**2 / 2)
                else:
                     bias = 1.0

                predictions_log = const_part + cov_part + extra_part
                predictions = np.exp(predictions_log) * bias
                t1_val = np.mean(predictions)
                results.append(t1_val)

            return results

    def predict(self, new_data_df, use_grid=True, interp_method='surface'):
        """
        Out-of-sample prediction using the fitted model.
        Returns a DataFrame with 'estimated' (point prediction) and 'decanted' (flow normalized) columns.
        interp_method: 'surface', 'log_surface', or 'coefficients'.
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
        # Use a temporary standard column for processing, but rely on orig_date_col for logic
        df_new['_date_processed'] = pd.to_datetime(df_new[self.orig_date_col])
        df_new['decimal_time'] = to_decimal_date(df_new['_date_processed'])

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
            # 1. Estimated Series (Point prediction)
            if interp_method == 'coefficients':
                cols = [T_new, Q_new]
                if X_extras_new is not None:
                    for k in range(X_extras_new.shape[1]):
                        cols.append(X_extras_new[:, k])
                pts = np.column_stack(cols)

                all_params = np.column_stack([interp(pts) for interp in self.interpolators])
                all_betas = all_params[:, :-1]
                all_sigmas = all_params[:, -1]

                bias_factors = np.exp(all_sigmas**2 / 2)
                bias_factors = np.nan_to_num(bias_factors, nan=1.0)

                est_log = (all_betas[:, 0] +
                           all_betas[:, 1] * Q_new +
                           all_betas[:, 2] * T_new +
                           all_betas[:, 3] * np.sin(2 * np.pi * T_new) +
                           all_betas[:, 4] * np.cos(2 * np.pi * T_new))

                if X_extras_new is not None:
                    for k in range(X_extras_new.shape[1]):
                        est_log += all_betas[:, 5 + k] * X_extras_new[:, k]

                estimated = np.exp(est_log) * bias_factors

                # Pre-calc for FN
                const_part = (all_betas[:, 0] +
                              all_betas[:, 2] * T_new +
                              all_betas[:, 3] * np.sin(2*np.pi*T_new) +
                              all_betas[:, 4] * np.cos(2*np.pi*T_new))

            elif interp_method == 'surface' or interp_method == 'log_surface':
                cols = [T_new, Q_new]
                if X_extras_new is not None:
                    for k in range(X_extras_new.shape[1]):
                        cols.append(X_extras_new[:, k])
                pts = np.column_stack(cols)

                if interp_method == 'surface':
                    estimated = self.conc_interpolator(pts)
                else:
                    log_est = self.log_conc_interpolator(pts)
                    bias_est = self.bias_interpolator(pts)
                    estimated = np.exp(log_est) * bias_est

            # 3. Decanted Series (Stationary Normalization)
            # Integrate over HISTORICAL Q (self.Q) conditioned on Day of Year.

            # Need DOY for new data
            doy_new = get_adjusted_doy(df_new['_date_processed'])

            # Need DOY for history
            # Use daily_history if available, otherwise fallback to self.df (Sample) with warning
            if self.daily_history is not None:
                source_df = self.daily_history
                # daily_history has 'date' column created in __init__
                doy_hist = get_adjusted_doy(source_df['date'])
                source_Q = source_df['log_covariate'].values
                if self.X_extras is not None:
                     # We need to access X_extras from daily_history
                     # We didn't store X_extras as a matrix in __init__ for daily_history,
                     # but we added columns. Reconstruct matrix.
                     extras_cols = [self.daily_history[col].values for col in self.extra_cov_names]
                     source_Extras = np.column_stack(extras_cols)
                else:
                     source_Extras = None
            else:
                # Fallback to self.Q (Sample data) - Incorrect for FN but prevents crash if no daily data
                # print("Warning: No daily history provided. Flow Normalization using sample data (sparse). Results will be inaccurate.")
                source_df = self.df
                doy_hist = get_adjusted_doy(source_df['date'])
                source_Q = self.Q
                source_Extras = self.X_extras

            # Map DOY -> Hist Indices
            doy_map = {}
            for d in range(1, 368):
                doy_map[d] = np.where(doy_hist == d)[0]

            decanted = np.full(len(df_new), np.nan)

            unique_doys = np.unique(doy_new)

            for d in unique_doys:
                target_indices = np.where(doy_new == d)[0]
                hist_indices = doy_map.get(d, np.array([]))

                # Special handling for Feb 28 (59) and Feb 29 (60)
                if d == 59 or d == 60:
                     hist_indices = np.concatenate([doy_map.get(59, []), doy_map.get(60, [])])

                if len(hist_indices) == 0:
                    continue

                hist_indices = hist_indices.astype(int)

                Q_local = source_Q[hist_indices]
                Extras_local = source_Extras[hist_indices, :] if source_Extras is not None else None

                n_chunk = len(target_indices)
                n_hist = len(hist_indices)

                if interp_method == 'coefficients':
                    chunk_betas = all_betas[target_indices]
                    chunk_const = const_part[target_indices]
                    chunk_bias = bias_factors[target_indices]

                    term_cov = chunk_betas[:, 1][:, np.newaxis] * Q_local[np.newaxis, :]

                    term_extras = 0
                    if Extras_local is not None:
                        for k in range(Extras_local.shape[1]):
                            b = chunk_betas[:, 5 + k][:, np.newaxis]
                            e = Extras_local[:, k][np.newaxis, :]
                            term_extras += b * e

                    log_preds = chunk_const[:, np.newaxis] + term_cov + term_extras
                    linear_preds = np.exp(log_preds) * chunk_bias[:, np.newaxis]

                    chunk_results = np.nanmean(linear_preds, axis=1)
                    decanted[target_indices] = chunk_results

                elif interp_method == 'surface' or interp_method == 'log_surface':
                    T_chunk = T_new[target_indices]

                    T_eval = np.broadcast_to(T_chunk[:, np.newaxis], (n_chunk, n_hist)).ravel()
                    Q_eval = np.broadcast_to(Q_local[np.newaxis, :], (n_chunk, n_hist)).ravel()

                    eval_cols = [T_eval, Q_eval]
                    if Extras_local is not None:
                        for k in range(Extras_local.shape[1]):
                            E_k = Extras_local[:, k]
                            E_eval = np.broadcast_to(E_k[np.newaxis, :], (n_chunk, n_hist)).ravel()
                            eval_cols.append(E_eval)

                    eval_pts = np.column_stack(eval_cols)

                    if interp_method == 'surface':
                        conc_vals = self.conc_interpolator(eval_pts)
                    else:
                        log_vals = self.log_conc_interpolator(eval_pts)
                        bias_vals = self.bias_interpolator(eval_pts)
                        conc_vals = np.exp(log_vals) * bias_vals

                    conc_matrix = conc_vals.reshape(n_chunk, n_hist)
                    decanted[target_indices] = np.nanmean(conc_matrix, axis=1)

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
            new_log_response = preds_log + boot_res
            new_log_response[np.isnan(self.Y)] = np.nan

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

    def add_kalman_correction(self, estimated_log_series=None, h_params=None, rho=0.9, use_grid=False, grid_config={'n_t':None, 'n_q':None}, min_obs=100):
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
             estimated_log_series = self.get_estimated_series(h_params, use_grid, grid_config, min_obs=min_obs)

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

        # Backward Pass
        df_kalman['Next_Res'] = df_kalman['Res'].bfill()
        df_kalman['Next_T'] = df_kalman['T'].where(~df_kalman['Res'].isna()).bfill()
        df_kalman['Next_T'] = df_kalman['Next_T'].fillna(9999)
        dt_backward = (df_kalman['Next_T'] - df_kalman['T']) * 365.25
        backward_res = (df_kalman['Next_Res'].fillna(0) * (rho ** dt_backward)).values

        # Combined Estimate
        interpolated_residuals = (forward_res + backward_res) / 2.0
        interpolated_residuals[valid_mask] = residuals[valid_mask]

        return estimated_log_series + interpolated_residuals
