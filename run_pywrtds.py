import pandas as pd
import numpy as np
from src.wrtds import Decanter
import sys

# 1. Load Data
df = pd.read_csv('test_data_sample.csv')
df_daily = pd.read_csv('test_data_daily.csv')

# 2. Initialize Decanter
decanter = Decanter(df, date_col='Date', response_col='Conc', covariate_col='Q', daily_data=df_daily)

# 3. Decant Series
h_params = {'h_time': 7, 'h_cov': 2, 'h_season': 0.5}

print(f"Running pyWRTDS with default EGRET-style grid...")
# Match EGRET run_egret.R which uses minNumObs=50
grid_config = {'n_t': None, 'n_q': None, 'min_obs': 50}

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
