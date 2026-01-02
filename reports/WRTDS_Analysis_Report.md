# A Comprehensive Analysis of Weighted Regressions on Time, Discharge, and Season (WRTDS) for Signal Extraction and Covariate Normalization in Time Series Analysis

## 1. Introduction: The Statistical Challenge of Covariate Removal

The analysis of time series data in complex systems is frequently complicated by the presence of exogenous drivers—covariates that induce high-frequency variability, thereby masking the underlying low-frequency trends that are often the primary subject of inquiry. Your project, which seeks to transform a raw time series $t_0$ influenced by covariates $c_0, c_1, \dots$ into a "cleaned" series $t_1$, addresses a fundamental problem in statistical signal processing: the separation of an anthropogenic or systemic signal from natural or stochastic noise. In the domain of environmental statistics, specifically hydrology, this problem is canonical. River water quality (the response variable) is heavily influenced by river discharge (the covariate), which is driven by stochastic weather events. To discern whether watershed management policies are effective, statisticians must remove the influence of the random weather (flow) to reveal the underlying trend in water quality.

The method developed to solve this specific problem is Weighted Regressions on Time, Discharge, and Season (WRTDS). Introduced by Hirsch, Moyer, and Archfield in 2010, WRTDS represents a significant departure from traditional parametric regression.1 Instead of assuming a fixed functional relationship between the response and the covariates that holds true for the entire period of record, WRTDS employs a locally weighted regression approach. This allows the relationship between the signal and the noise to evolve over time, accommodating the non-stationarity inherent in complex systems.3

This report provides an exhaustive examination of WRTDS, its mathematical architecture, its extensions, and its applicability to your general time series project. While the nomenclature of WRTDS is specific to hydrology—referring to "Discharge" ($Q$) as the primary covariate and "Concentration" ($C$) as the response—the statistical framework is entirely generalizable. For your project, "Discharge" serves as a proxy for your primary high-variance covariate $c_0$, "Season" represents cyclic forcing functions $c_1$, and "Concentration" is your target time series $t_0$. The "cleaned" series $t_1$ you seek corresponds to the Flow-Normalized Concentration (FNConc), a statistical product of WRTDS obtained through integration over the probability distribution of the covariate.3

By dissecting the mechanics of WRTDS, including its handling of censored data (Tobit regression), its weighting strategies (tricube functions), and its advanced extensions like WRTDS-Kalman and Generalized Flow Normalization, this report will demonstrate how these concepts can be transposed to non-hydrological domains to robustly remove covariate effects and isolate systemic trends.

## 2. The Theoretical Architecture of WRTDS

To understand why WRTDS is suitable for your project, it is necessary to contrast it with the methods it replaced. Traditional approaches to removing covariate effects typically rely on global parametric regression. In such models, a single equation (e.g., $\ln(C) = \beta_0 + \beta_1 \ln(Q) + \beta_2 T + \epsilon$) is fitted to the entire dataset. The "cleaned" series is then derived either by examining the coefficient of time ($\beta_2$) or by analyzing the residuals.

These global models suffer from a fatal flaw when applied to long-term records: they assume the relationship between the covariate and the response is static (stationary).4 In reality, the interaction between a driver and a response often changes. For example, in a river, high flow might dilute a pollutant in the 1980s (negative correlation), but due to changes in land use, high flow might scour sediment and increase pollution in the 2000s (positive correlation). A global model averages these contradictory behaviors, resulting in a poor fit and biased trend estimation.2

### 2.1 The Local Regression Paradigm

WRTDS abandons the global model in favor of Locally Weighted Scatterplot Smoothing (LOWESS) extended into three dimensions. The central premise is that the coefficients of the regression equation are not constants but are continuous functions of time. For every single point in the estimation space (typically every day of the record), WRTDS estimates a unique regression model based only on data points that are "close" in terms of time, season, and covariate magnitude.5

This local flexibility is the method's primary statistical strength. It allows the model to capture:
* **Evolving Trends:** Changes in the baseline value of the response variable over time (the intercept $\beta_0(t)$).
* **Evolving Covariate Effects:** Changes in how the covariate $c_0$ impacts the response (the slope $\beta_1(t)$).
* **Evolving Seasonality:** Changes in the amplitude or phase of cyclical patterns (the harmonic coefficients $\beta_3(t), \beta_4(t)$).2

For your project, this means that if the way your covariate $c_0$ affects your time series $t_0$ changes halfway through the dataset (e.g., a structural break or gradual regime shift), WRTDS will naturally adapt to this change without requiring you to manually specify interaction terms or split the record.8

### 2.2 The Four-Component Decomposition

WRTDS conceptually decomposes the time series $t_0$ into four additive components (in log space):
* **Trend ($T$):** The low-frequency signal you wish to isolate.
* **Discharge ($Q$):** The high-frequency noise driven by the covariate $c_0$.
* **Season ($S$):** The cyclic variation driven by $c_1$.
* **Random ($\epsilon$):** The unstructured residual noise.2

The mathematical formulation for the estimated value at any specific time $t$ is:

$$\ln(t_0) = \beta_0(t) + \beta_1(t) \cdot \ln(c_0) + \beta_2(t) \cdot t + \beta_3(t) \sin(2\pi t) + \beta_4(t) \cos(2\pi t) + \epsilon$$

Crucially, the coefficients $\beta_i$ are indexed by $t$, emphasizing that they are re-estimated for every point in time. This equation is solved using Weighted Least Squares (WLS), where the weights determine the "local" neighborhood.5

## 3. Mathematical Formulation and Weighting Strategy

The core of the WRTDS algorithm is the definition of "locality." How does the model decide which data points are relevant for estimating the state of the system on a specific day? This is governed by a distance metric in a three-dimensional space comprising Time, Discharge (Covariate), and Season.

### 3.1 The Tricube Weighting Function

To ensure smooth transitions between local models (differentiability), WRTDS employs the Tricube weight function. For a given distance $d$ between an observation and the target estimation point, and a specified window width (bandwidth) $h$, the weight $w$ is calculated as:

$$w(d) = \begin{cases} \left( 1 - \left( \frac{|d|}{h} \right)^3 \right)^3 & \text{if } |d| \le h \\ 0 & \text{if } |d| > h \end{cases}$$

The tricube function is chosen because it assigns high weight to points near the center of the window and decays smoothly to zero at the edge ($h$), with a derivative of zero at the boundary. This prevents numerical artifacts or sudden jumps in the cleaned time series $t_1$ as data points enter or leave the moving window.7

### 3.2 Multidimensional Weight Determination

For a target prediction point $P$ defined by coordinates $(T_p, \ln Q_p, S_p)$—representing the prediction Time, Covariate magnitude, and Season—and an observation point $O_i$ with coordinates $(T_i, \ln Q_i, S_i)$, the total weight $W_i$ assigned to observation $O_i$ is the product of three independent weights:

$$W_i = w_{time} \times w_{flow} \times w_{season}$$

Each component weight is calculated using the tricube function applied to the distance in that specific dimension, scaled by a user-defined half-window width:

**Time Weight ($w_{time}$):**
$$d_{time} = T_p - T_i$$
$$w_{time} = \text{tricube}\left( \frac{d_{time}}{h_{time}} \right)$$

The default half-window $h_{time}$ is typically 7 to 10 years. This wide window ensures that the "Trend" component is smooth and robust to short-term anomalies, effectively serving as a low-pass filter.9

**Covariate (Discharge) Weight ($w_{flow}$):**
$$d_{flow} = \ln(Q_p) - \ln(Q_i)$$
$$w_{flow} = \text{tricube}\left( \frac{d_{flow}}{h_{flow}} \right)$$

The default half-window $h_{flow}$ is typically 2 natural log units. This ensures that the prediction for a "high covariate" day is derived primarily from other "high covariate" days, respecting the physical relationship between $c_0$ and $t_0$.7

**Seasonal Weight ($w_{season}$):**
$$d_{season} = \min( |S_p - S_i|, 1 - |S_p - S_i| ) \quad (\text{accounting for circularity})$$
$$w_{season} = \text{tricube}\left( \frac{d_{season}}{h_{season}} \right)$$

The default half-window $h_{season}$ is typically 0.5 years. This implies that data from January is highly relevant for predicting January, moderately relevant for February, and irrelevant for July.2

**Implication for Your Project:** You must carefully select these window widths ($h$). A narrower time window makes the cleaned series $t_1$ more volatile and responsive to rapid changes (lower bias, higher variance). A wider window produces a smoother long-term trend (higher bias, lower variance). The WRTDS software (EGRET package in R) allows these to be tuned, but defaults are set to balance these trade-offs for decadal-scale environmental data.11

## 4. Handling Censored Data: The Tobit Likelihood Approach

A critical "key feature" you requested is the handling of data limitations. In many time series, the response variable $t_0$ may be censored—that is, reported as "less than $L$" (left-censored) or "greater than $U$" (right-censored) rather than a precise value. In water quality, this corresponds to concentrations below the laboratory detection limit.
Standard methods (like substituting $L/2$ or $0$) introduce significant bias. WRTDS integrates Survival Analysis methods directly into the local regression framework using Tobit Regression (or Censored Likelihood Estimation).12

### 4.1 The Likelihood Function

Instead of minimizing the sum of squared errors (OLS), WRTDS maximizes the log-likelihood of the data given the local model parameters. For a dataset containing both uncensored observations ($U$) and left-censored observations ($C$), the log-likelihood function $\mathcal{L}$ is defined as:

$$\mathcal{L}(\beta, \sigma) = \sum_{i \in U} W_i \cdot \ln \left[ \frac{1}{\sigma} \phi \left( \frac{y_i - X_i \beta}{\sigma} \right) \right] + \sum_{j \in C} W_j \cdot \ln \left[ \Phi \left( \frac{L_j - X_j \beta}{\sigma} \right) \right]$$

Where:
* $W$ are the tricube weights described above.
* $\phi$ is the probability density function (PDF) of the standard normal distribution.
* $\Phi$ is the cumulative distribution function (CDF) of the standard normal distribution.
* $y_i$ is the observed value (log-transformed).
* $L_j$ is the censoring limit (log-transformed).
* $X \beta$ is the linear predictor from the local regression equation.
* $\sigma$ is the standard deviation of the residuals (locally estimated).12

### 4.2 Adjusted Maximum Likelihood Estimation (AMLE)

The optimization of this likelihood function is performed for every single point in the grid. This allows WRTDS to extract information even from "non-detects." For example, if a "less than" value occurs during a period of low covariate values, it still contributes probabilistic information that constrains the lower tail of the regression relationship. This ensures that the cleaned time series $t_1$ is not artificially inflated by detection limits, a common problem in simpler cleaning methods.15

For your project, if your time series $t_0$ has floor or ceiling constraints (e.g., cannot be negative, or capped at a sensor maximum), this feature of WRTDS provides a rigorous statistical solution that preserves the integrity of the trend.14

## 5. Flow Normalization: The Mechanism of Covariate Removal

The most distinct innovation of WRTDS, and the direct answer to your request for a "cleaned time series," is the concept of Flow Normalization (FN). In most regression analyses, the "cleaned" value is often the residual or the intercept. WRTDS takes a different approach: Integration.

### 5.1 The Definition of the Cleaned Series $t_1$

WRTDS defines the "clean" value not as the value observed in the absence of the covariate, but as the expected value of the system averaged over the typical probability distribution of the covariate.
The Flow-Normalized Concentration (your $t_1$) on a specific day $T$ is defined as the integral of the estimated regression surface over the probability density function (PDF) of the covariate $Q$:

$$t_1(T) = E = \int_{-\infty}^{\infty} \hat{C}(T, Q) \cdot f_Q(Q) \, dQ$$

Here, $\hat{C}(T, Q)$ is the model's prediction of the response variable for time $T$ and covariate $Q$. The function $f_Q(Q)$ is the probability density of the covariate.2

### 5.2 The Monte Carlo Integration Algorithm

Since the true PDF of the covariate $f_Q(Q)$ is unknown and likely complex (e.g., non-Gaussian, multimodal), WRTDS approximates this integral using a discrete summation over the historical record of the covariate. This acts as a form of non-parametric Monte Carlo integration.

For a specific date $T$ (e.g., June 15, 2023), the algorithm proceeds as follows:
1. **Isolate the Model:** Retrieve the regression coefficients estimated for June 15, 2023. This defines the relationship between $c_0$ and $t_0$ for this specific moment in history.
2. **Retrieve Covariate History:** Extract the set of all covariate values $c_0$ observed on June 15 across the entire period of record (e.g., 30 years of June 15s). This empirical distribution represents the climatology of the covariate for that specific season.7
3. **Predict Across History:** Input every historical $c_0$ value into the current model (Step 1) to generate a set of predicted values. This answers the question: "What would the response be today if we experienced the conditions of 1990? 1991?... 2023?"
4. **Average:** Compute the mean of these predictions.

$$t_1(T) \approx \frac{1}{Y} \sum_{y=1}^{Y} \hat{C}(T, Q_{T_{day}, y})$$

**Statistical Implication:** This process effectively marginalizes out the covariate. The resulting value $t_1$ reflects the state of the system ($t_0$) driven by time and management factors, independent of the specific random covariate realization ($c_0$) that happened to occur on that day. It removes the "noise" (weather) while preserving the "signal" (climate).3

### 5.3 Stationary vs. Generalized Flow Normalization (SFN vs. GFN)

A critical nuance in this process is whether the probability distribution of the covariate $c_0$ is itself changing over time.
* **Stationary Flow Normalization (SFN):** Assumes the covariate's statistical properties (mean, variance) are constant over the long term. The integration uses the entire historical record of $c_0$ for every prediction. This was the original 2010 formulation.2
* **Generalized Flow Normalization (GFN):** Acknowledges that the covariate $c_0$ may have its own trend (e.g., climate change reducing river flows, or market volume increasing over decades). If $c_0$ is trending, using a 30-year average to normalize the current year might be biased. GFN uses a locally weighted integral, where the historical $c_0$ values are weighted based on their temporal proximity to the prediction year. This effectively normalizes the response relative to "contemporary" covariate conditions rather than "historic" ones.8

**Recommendation for Your Project:** If your covariates $c_0$ exhibit a long-term trend (non-stationarity), you should implement the GFN approach. This ensures that your cleaned series $t_1$ reflects the trend in the response variable's relationship to the covariate, rather than conflating it with the trend in the covariate itself.3

## 6. Computational Implementation: The EGRET Framework

Implementing WRTDS requires handling significant computational complexity. A naïve implementation involves running a weighted regression for every point in the time series. If your time series has length $N$, calculating the weights for every other point leads to a complexity of roughly $O(N^2)$.18 For daily data over decades, this is computationally expensive.

### 6.1 The Grid (Surfaces) Optimization

The USGS implementation of WRTDS, the EGRET (Exploration and Graphics for RivEr Trends) package in R, solves this efficiency problem using a grid-based look-up table approach.
Instead of calculating the regression for every daily point:
* **Grid Generation:** The software defines a grid of 14-16 values for Log-Discharge ($Q$) and hundreds of values for Time ($T$) (typically monthly steps).
* **Surface Estimation:** The weighted regression is performed only at these grid nodes. This results in a matrix (the surfaces object) containing the estimated coefficients or predicted values.10
* **Interpolation:** For any specific daily observation in the record, the estimated value is derived via bicubic interpolation from this pre-computed surface.

This optimization reduces the complexity from quadratic relative to the number of days to linear relative to the grid size. For your project, if you are processing high-frequency data (e.g., minute-by-minute), adopting this "Surface Interpolation" strategy is mandatory to maintain performance. You would compute the "cleaning surface" on a coarse grid and then map your fine-resolution data onto it.10

### 6.2 Data Structures

EGRET organizes data into three primary dataframes, a structure you might emulate:
* **Daily:** The continuous, complete time series of the covariate $c_0$ (required for normalization integration).
* **Sample:** The discrete, often sparse observations of the response variable $t_0$ (used to fit the model).
* **INFO:** Metadata and hyperparameters (window widths, station details).

This separation is crucial because WRTDS allows $t_0$ to be sparse (e.g., monthly samples) while producing a daily cleaned series $t_1$ by leveraging the complete record of $c_0$.20

## 7. Extensions: Addressing Autocorrelation and Forecasting

Since the original 2010 publication, WRTDS has been extended to address limitations regarding serial correlation and forecasting capabilities. These extensions may be highly relevant depending on the specific dynamics of your time series.

### 7.1 WRTDS-Kalman (WRTDS-K): Restoring System Memory

Standard WRTDS is a deterministic smoother. It assumes that deviations from the trend are purely random noise ($\epsilon$). However, in many physical and economic systems, errors are autocorrelated (serial correlation). If the value is high today, it is likely to be high tomorrow due to system memory (e.g., groundwater storage, market sentiment).
WRTDS-K augments the WRTDS estimate with a Kalman filter-like correction based on an AR(1) (first-order autoregressive) process.
* **Mechanism:** It calculates the residuals ($observed - estimated$) for the days with samples.
* **Interpolation:** It interpolates these residuals to non-sample days using the correlation coefficient $\rho$ (typically 0.8 to 0.95).
* **Result:** This creates a "best estimate" of the actual daily value that includes the short-term memory of the system.21

**Relevance:** If your "cleaned" series $t_1$ needs to reflect the actual historical state with high precision (including short-term anomalies that are real but unexplained by covariates), use WRTDS-K. If you want a purely smoothed long-term trend, standard WRTDS is sufficient.

### 7.2 WRTDS-P: Projection and Forecasting

While WRTDS is primarily a hindcasting (historical analysis) tool, the WRTDS-P (Projection) extension adapts the framework for forecasting. It uses the probabilistic nature of Flow Normalization to project future scenarios.
* **Mechanism:** Instead of integrating over historical covariate distributions, WRTDS-P integrates over simulated or scenario-based covariate distributions (e.g., "What if the covariate variance doubles in the next decade?").
* **Application:** This allows for risk assessment under changing covariate regimes, projecting the "cleaned" trend forward into hypothetical futures.22

### 7.3 WRTDSplus: Custom Covariates

Standard WRTDS hardcodes "Discharge" and "Time." WRTDSplus generalizes the inputs to allow for:
* **Antecedent Conditions:** Including a moving average of past covariate values (e.g., 30-day mean flow) to account for lag or hysteresis.
* **Flashiness:** Including the rate of change of the covariate.
This extension effectively increases the dimensionality of the weighting function, allowing for more complex covariate removal.23

## 8. Comparative Analysis: WRTDS vs. Alternative Methods

To confirm that WRTDS is the optimal choice for your project, it is instructive to compare it with other statistical cleaning methods identified in the research.

### 8.1 WRTDS vs. Generalized Additive Models (GAMs)

GAMs are the closest competitor to WRTDS. Both are non-parametric smoothers.
* **GAM Structure:** $t_0 = s(t) + s(c_0) + s(c_1) + \epsilon$. GAMs use splines to fit smooth functions for each predictor.
* **WRTDS Structure:** $t_0 = \beta(t, c_0) \dots$. WRTDS fits a local model for every point.
* **Comparison:** Research indicates that while both produce similar results for general trends, WRTDS is often superior in capturing interaction effects (e.g., when the shape of the covariate response changes over time). GAMs typically require explicit interaction terms (tensor products) to match this, which can be difficult to specify correctly. However, GAMs are computationally much faster ($O(N)$ vs $O(N^2)$).16
* **Decision:** Use WRTDS if the interaction between time and covariate is complex and unknown. Use GAMs if computational speed is paramount and interactions are simple.

### 8.2 WRTDS vs. Deep Learning (LSTM)

Long Short-Term Memory (LSTM) networks are powerful for time series prediction.
* **Comparison:** A study evaluating water quality prediction across 500 catchments found that LSTM did not markedly outperform WRTDS.25
* **Interpretability:** WRTDS provides explicit components (Trend, Season, Covariate) that are interpretable. LSTMs are "black boxes." For a project where "cleaning" implies explaining why the series changed (attribution), WRTDS offers higher transparency.
* **Data Requirements:** LSTMs generally require massive datasets to converge. WRTDS is robust with smaller datasets (e.g., hundreds of samples) because it imposes structural priors (smoothness).25

### 8.3 WRTDS vs. Parametric Regression (LOADEST)

* **Comparison:** WRTDS consistently outperforms fixed-parameter models (like LOADEST) in reducing bias. Parametric models fail when the system exhibits non-stationarity (e.g., a change in the slope of the covariate relationship), whereas WRTDS adapts locally to these changes.4

## 9. Generalizing to Your Project: A Blueprint for Implementation

The following section synthesizes the research into a practical roadmap for applying WRTDS to your general time series $t_0$ and covariates $c_0, c_1$.

### 9.1 Data Preparation and Variable Mapping

* **Response Variable ($t_0$):** Map to "Concentration." WRTDS operates in log-space ($\ln(t_0)$). Ensure your data is strictly positive. If your data contains zeros or negatives (e.g., profit/loss, temperature), you must apply an offset (add a constant) or modify the regression kernel to be linear rather than log-linear.24
* **Primary Covariate ($c_0$):** Map to "Discharge." This should be the continuous, high-frequency driver you wish to remove.
* **Secondary Covariate ($c_1$):** Map to "Season" if it is cyclical (0 to 1 scale). If $c_1$ is not cyclical, you must treat it as a second "Discharge" dimension, effectively moving to a 4D weighting space (Time, $c_0$, $c_1$, Season).

### 9.2 Parameter Selection (The "Art" of Cleaning)

You must select three window widths ($h$) that define the scale of smoothing.
* **$h_{time}$ (Trend smoothness):** If you want to remove short-term volatility and see decadal trends, use a wide window (e.g., 20% of record length). If you want to see rapid shifts, use a narrow window (e.g., 5%).
* **$h_{covariate}$ (Covariate locality):** Controls how "local" the covariate relationship is. A width of 0.5 to 1.0 standard deviations of $c_0$ is a good starting point.
* **$h_{season}$:** Controls how much data from adjacent "seasons" influences the current prediction.

### 9.3 The Cleaning Workflow

To produce your cleaned series $t_1$:
1. **Fit the Surface:** Generate the regression surface $\hat{t_0}(t, c_0)$ using the tricube-weighted local regression.
2. **Integrate (Normalize):** For each time step $i$:
    * Construct the probability distribution of $c_0$ relevant to time $i$ (using GFN if $c_0$ has a trend).
    * Integrate the surface $\hat{t_0}(i, c_0)$ over this distribution.
    * The result is $t_1[i]$.

### 9.4 Table 1: Summary of WRTDS Features for General Time Series

| Feature | WRTDS Implementation | Benefit for General Time Series |
| :--- | :--- | :--- |
| Local Regression | Unique $\beta$ coefficients for every time step. | Handles non-stationary relationships (e.g., changing correlation regimes). |
| Flow Normalization | Integration over covariate PDF. | Removes covariate noise without losing signal; conceptually superior to residual analysis. |
| Tricube Weighting | Smooth decay of influence with distance. | Prevents discontinuities/jumps in the cleaned series $t_1$. |
| Tobit/Survival | Maximum Likelihood for censored data. | Robustly handles floor/ceiling constraints (e.g., detection limits, zero-bound). |
| WRTDS-K | AR(1) correction of residuals. | Preserves system memory and autocorrelation in the cleaned series. |
| GFN | Time-varying integration weights. | Correctly cleans series even when the covariate itself has a long-term trend. |

## 10. Conclusion

Weighted Regressions on Time, Discharge, and Season (WRTDS) offers a robust, statistically rigorous framework for separating signal from noise in time series data influenced by strong covariates. Its value for your project lies in its rejection of the static assumptions that plague traditional regression. By viewing the cleaning process as a local integration problem rather than a global subtraction problem, WRTDS allows you to derive a trend $t_1$ that is truly independent of the stochastic fluctuations of your covariates $c_0, c_1$.

While the method requires careful attention to computational complexity ($O(N^2)$) and parameter tuning (window widths), its ability to handle censored data, evolving seasonality, and non-stationary covariate relationships makes it a superior choice for analyzing complex systems. Whether your domain is environmental, financial, or industrial, the "Flow Normalization" concept—replacing the observed driver with its probability distribution—provides a powerful definition of what it means to "clean" a time series.
