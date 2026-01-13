import pandas as pd
import numpy as np
from src.wrtds import Decanter
import sys

# 1. Load Data
df = pd.read_csv('test_data_sample.csv')
df_daily = pd.read_csv('test_data_daily.csv')

# 2. Initialize Decanter
decanter = Decanter(df, date_col='Date', response_col='Conc', covariate_col='Q')

# 3. Decant Series
h_params = {'h_time': 7, 'h_cov': 2, 'h_season': 0.5}

print(f"Running pyWRTDS with default EGRET-style grid...")
grid_config = {'n_t': None, 'n_q': None} # Trigger EGRET defaults

# Fit grid
decanter.decant_series(h_params, use_grid=True, grid_config=grid_config)

# Predict daily
print("Predicting daily series...")
daily_results = decanter.predict(df_daily, use_grid=True)

# 4. Save Results
results = pd.DataFrame({
    'Date': df_daily['Date'],
    'Conc': daily_results['estimated'],
    'FNConc': daily_results['decanted']
})

results.to_csv('pywrtds_results.csv', index=False)
