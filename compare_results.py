import pandas as pd
import numpy as np
import sys

# Load Results
try:
    egret = pd.read_csv("egret_results.csv")
    egret['Date'] = pd.to_datetime(egret['Date'])
    egret.set_index('Date', inplace=True)
except FileNotFoundError:
    print("egret_results.csv not found.")
    sys.exit(1)

try:
    pywrtds = pd.read_csv("pywrtds_results.csv")
    pywrtds['Date'] = pd.to_datetime(pywrtds['Date'])
    pywrtds.set_index('Date', inplace=True)
except FileNotFoundError:
    print("pywrtds_results.csv not found.")
    sys.exit(1)

# Merge
combined = pywrtds.join(egret, lsuffix='_py', rsuffix='_R')

# Columns
col_est_py = 'Conc' if 'Conc' in combined.columns else 'Conc_py'
col_fn_py = 'FNConc' if 'FNConc' in combined.columns else 'FNConc_py'
col_est_R = 'ConcDay'
col_fn_R = 'FNConc' if 'FNConc' in combined.columns else 'FNConc_R'

# Calculate metrics
corr_est = combined[col_est_py].corr(combined[col_est_R])
corr_fn = combined[col_fn_py].corr(combined[col_fn_R])

rmse_est = np.sqrt(((combined[col_est_py] - combined[col_est_R])**2).mean())
rmse_fn = np.sqrt(((combined[col_fn_py] - combined[col_fn_R])**2).mean())

# Percentage RMSE (normalized by R result)
prmse_est = np.sqrt((( (combined[col_est_py] - combined[col_est_R]) / combined[col_est_R] )**2).mean()) * 100
prmse_fn = np.sqrt((( (combined[col_fn_py] - combined[col_fn_R]) / combined[col_fn_R] )**2).mean()) * 100

print(f"Correlation (Estimated): {corr_est:.4f}")
print(f"Correlation (FN): {corr_fn:.4f}")
print(f"RMSE (Estimated): {rmse_est:.4f}")
print(f"RMSE (FN): {rmse_fn:.4f}")
print(f"PRMSE (Estimated): {prmse_est:.2f}%")
print(f"PRMSE (FN): {prmse_fn:.2f}%")
