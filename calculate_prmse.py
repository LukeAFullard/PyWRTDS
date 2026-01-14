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

# Calculate RMSE
rmse_est = np.sqrt(((combined[col_est_py] - combined[col_est_R])**2).mean())
rmse_fn = np.sqrt(((combined[col_fn_py] - combined[col_fn_R])**2).mean())

# Calculate Means (Reference: EGRET)
mean_est = combined[col_est_R].mean()
mean_fn = combined[col_fn_R].mean()

# Calculate Percentage RMSE (Normalized RMSE)
p_rmse_est = (rmse_est / mean_est) * 100
p_rmse_fn = (rmse_fn / mean_fn) * 100

print(f"Mean Estimated Conc (EGRET): {mean_est:.4f} mg/L")
print(f"Mean FN Conc (EGRET): {mean_fn:.4f} mg/L")
print(f"RMSE (Estimated): {rmse_est:.4f} mg/L")
print(f"RMSE (FN): {rmse_fn:.4f} mg/L")
print("-" * 30)
print(f"Percentage RMSE (Estimated): {p_rmse_est:.4f}%")
print(f"Percentage RMSE (FN): {p_rmse_fn:.4f}%")
