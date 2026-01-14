import pandas as pd
import numpy as np
from src.wrtds import Decanter, to_decimal_date

# Load Data
daily_df = pd.read_csv('test_data_daily.csv')
sample_df = pd.read_csv('test_data_sample.csv')

# Target Point 14 (Index 13)
idx = 13
target_row = daily_df.iloc[idx]
# Ensure consistent date parsing
date_target = pd.to_datetime(target_row['Date'])
# Manual decimal date check
t_target = to_decimal_date(pd.Series([date_target]))[0]
q_target = np.log(target_row['Q'])
s_target = t_target % 1

print(f"Target T: {t_target}")
print(f"Target Q (Log): {q_target}")

# Init Decanter
decanter = Decanter(sample_df, date_col='Date', response_col='Conc', covariate_col='Q', daily_data=daily_df)

h_params = {'h_time': 7, 'h_cov': 2, 'h_season': 0.5}
min_obs = 50
min_uncen = 50

# Inspect Internals via fit_local_model (which returns coeffs)
betas = decanter.fit_local_model(t_target, q_target, s_target, h_params, min_obs=min_obs, min_uncen=min_uncen)

if betas is not None:
    sigma = betas[-1]
    coeffs = betas[:-1]
    print("Coefficients:", coeffs)
    print(f"Scale (Sigma): {sigma}")
else:
    print("Fit failed")
