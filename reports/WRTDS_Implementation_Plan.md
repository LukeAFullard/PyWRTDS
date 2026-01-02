# WRTDS Python Implementation Guide

This guide translates the "WRTDS" hydrological framework into a general-purpose Python workflow for removing covariate effects from any time series.

We will refer to your variables as:

* $t_0$: **The Response** (Raw Time Series, e.g., Sales, Water Quality, Server Load).
* $c_0$: **The Primary Covariate** (High-frequency driver, e.g., Ad Spend, Discharge, Temperature).
* $t$: **Decimal Time** (The temporal coordinate).
* $t_1$: **The Decanted Series** (The adjusted/normalized result).

---

### **Phase 1: Theoretical Setup & Variable Mapping**

**Justification (Report Section 2.2):**
Standard regression assumes relationships are static (stationary). WRTDS assumes relationships evolve. To capture this, we model the response in log-space using a locally weighted regression.

The local model for any specific moment in time $t$ is:

$$ \ln(t_0) = \beta_0(t) + \beta_1(t) \ln(c_0) + \beta_2(t) t + \beta_3(t) \sin(2\pi t) + \beta_4(t) \cos(2\pi t) + \epsilon $$

We need to implement the weighting logic to solve this regression locally.

#### **Python Setup**

First, let's create a class structure and handle the data transformation (Log space).

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize
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
        self.df['log_response'] = np.log(self.df[response_col])
        self.df['log_covariate'] = np.log(self.df[covariate_col])

        # Store vectors for fast vectorized access
        self.T = self.df['decimal_time'].values
        self.Q = self.df['log_covariate'].values
        self.S = self.df['season'].values
        self.Y = self.df['log_response'].values

```

---

### **Phase 2: The Weighting Kernel (The "Locality")**

**Justification (Report Section 3.1 & 3.2):**
To solve the regression "locally," we need to decide which historical data points are relevant to a specific prediction date. WRTDS uses a **Tricube** window in three dimensions: Time, Covariate magnitude, and Season.

The weight $w$ for a distance $d$ and window width $h$ is:

$$ w(d) = \begin{cases} \left( 1 - \left( \frac{|d|}{h} \right)^3 \right)^3 & \text{if } |d| \le h \\ 0 & \text{if } |d| > h \end{cases} $$

#### **Step-by-Step Implementation**

1. **Distance Calculation:** Calculate distance between the *target point* and *all observed points*.
2. **Seasonality Circularity:** January is close to December. We must handle the "wrap around" logic: $d_{season} = \min(|S_p - S_i|, 1 - |S_p - S_i|)$.
3. **Window Widths ($h$):** You must define how "wide" your window is.

```python
    def _tricube(self, d, h):
        """
        Calculates tricube weight.
        d: array of distances
        h: half-window width
        """
        # Normalize distance by window width
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

```

---

### **Phase 3: Weighted Least Squares (WLS)**

**Justification (Report Section 2.2):**
Now that we have weights, we run a regression. In a global model, all weights are 1. Here, weights are specific to the target date.
The design matrix $X$ typically includes: constant, $\ln(c_0)$, time, $\sin(2\pi t)$, $\cos(2\pi t)$.

```python
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

        s_target = t_target % 1
        prediction_log = (betas[0] +
                          betas[1] * q_target +
                          betas[2] * t_target +
                          betas[3] * np.sin(2*np.pi*t_target) +
                          betas[4] * np.cos(2*np.pi*t_target))

        return np.exp(prediction_log) # Return to linear space

```

---

### **Phase 4: "Decanting" (Flow Normalization)**

**Justification (Report Section 5.1 & 5.2):**
This is the core deliverable. We don't just want to predict the value; we want to remove the specific effect of the covariate $c_0$.
Instead of inputting the *actual* $c_0$ for today, we integrate over the probability distribution of $c_0$.
**Procedurally:** We take the model fitted for "today" ($t, c_0$) and run *every historical covariate value* through it, then average the results.

```python
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

```

---

### **Phase 5: Execution Example**

Here is how you would run this pipeline.

```python
# 1. Generate Dummy Data
np.random.seed(42)
dates = pd.date_range(start='2010-01-01', end='2020-01-01', freq='D')
n = len(dates)

# Covariate c0 (Random walk + Seasonality)
c0 = np.exp(np.sin(np.linspace(0, 20, n)) + np.random.normal(0, 0.5, n))

# True Trend (What we want to recover): Linear growth
true_trend = np.linspace(10, 20, n)

# Response t0 (Trend + Covariate Effect + Noise)
# Let's say t0 is heavily influenced by c0
t0 = true_trend * c0 * np.random.normal(1, 0.1, n)

df = pd.DataFrame({'Date': dates, 'Sales': t0, 'AdSpend': c0})

# 2. Initialize
# Map Sales -> t0, AdSpend -> c0
dec = Decanter(df, date_col='Date', response_col='Sales', covariate_col='AdSpend')

# 3. Decant
# Note: h_time=2 years, h_cov=1 log unit
adjusted_series = dec.decant_series(h_params={'h_time': 2, 'h_cov': 1, 'h_season': 0.5})

# 4. Result
df['Decanted_Sales'] = adjusted_series

print(df[['Date', 'Sales', 'Decanted_Sales']].head())

```

---

### **Key Considerations for General Application**

1. **Stationary vs. Generalized (GFN):**
The code above implements **Stationary Flow Normalization** (using `self.Q`—the entire history—for integration).
* *Adjustment:* If your covariate $c_0$ has a strong trend (e.g., market size is growing, so Ad Spend is naturally higher in 2024 than 2010), using the 2010 data to normalize 2024 is invalid. You must restrict `self.Q` in the integration step to a window around the current year (Report Section 5.3).


2. **Performance ($O(N^2)$ vs Surfaces):**
The `decant_series` function above loops through every row and re-calculates the regression. For 3,000 days, this is fine. For 100,000 rows, this will be slow.
* *Optimization:* Implement the **Grid approach** mentioned in Report Section 6.1. Calculate betas for a grid of 100 times $T$ 15 covariate levels, store them, and interpolate.


3. **Data Limits:**
The report mentions **Tobit Regression** (Section 4). If your data has "Floor" limits (e.g., zeros that are actually "below detection"), standard WLS (used above) will bias the results. You would need to replace `np.linalg.lstsq` with a Maximum Likelihood Estimator that accounts for censoring.
