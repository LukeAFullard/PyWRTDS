import pandas as pd
import numpy as np

# Create synthetic data
# 10 years
dates = pd.date_range(start='2000-01-01', end='2010-12-31', freq='D')
n = len(dates)

# Discharge (Q): Log-normal distribution with some seasonality
t = np.arange(n) / 365.25
seasonality_q = 0.5 * np.sin(2 * np.pi * t)
log_q = 2 + seasonality_q + np.random.normal(0, 0.5, n)
q = np.exp(log_q)

# Concentration (C): Depends on Q, Time, Seasonality
b0, b1, b2 = 1.0, 0.5, -0.02
seasonality_c = 0.3 * np.sin(2 * np.pi * t) + 0.2 * np.cos(2 * np.pi * t)
log_c = b0 + b1 * log_q + b2 * t + seasonality_c + np.random.normal(0, 0.2, n)
c = np.exp(log_c)

df = pd.DataFrame({
    'Date': dates,
    'Q': q,
    'Conc': c
})

# Subsample Conc ~ monthly (approx 120-130 samples)
mask = np.random.rand(n) < (12 / 365.25)
df_sample = df.copy()
df_sample.loc[~mask, 'Conc'] = np.nan

# Save
df.to_csv('test_data_daily.csv', index=False)
df_sample.to_csv('test_data_sample.csv', index=False)

print("Data generated. N_days:", n, "N_samples:", mask.sum())
