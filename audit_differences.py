import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def audit():
    # Load Results
    try:
        egret = pd.read_csv("egret_results.csv")
        egret['Date'] = pd.to_datetime(egret['Date'])
        egret.set_index('Date', inplace=True)
    except FileNotFoundError:
        print("egret_results.csv not found.")
        return

    try:
        pywrtds = pd.read_csv("pywrtds_results.csv")
        pywrtds['Date'] = pd.to_datetime(pywrtds['Date'])
        pywrtds.set_index('Date', inplace=True)
    except FileNotFoundError:
        print("pywrtds_results.csv not found.")
        return

    try:
        daily = pd.read_csv("test_data_daily.csv")
        daily['Date'] = pd.to_datetime(daily['Date'])
        daily.set_index('Date', inplace=True)
    except FileNotFoundError:
        print("test_data_daily.csv not found.")
        return

    # Merge
    combined = pywrtds.join(egret, lsuffix='_py', rsuffix='_R')
    combined = combined.join(daily[['Q']], rsuffix='_daily')

    # Columns
    col_est_py = 'Conc' if 'Conc' in combined.columns else 'Conc_py'
    col_fn_py = 'FNConc' if 'FNConc' in combined.columns else 'FNConc_py'
    col_est_R = 'ConcDay'
    col_fn_R = 'FNConc' if 'FNConc' in combined.columns else 'FNConc_R'

    # Calculate differences
    combined['Diff_Est'] = combined[col_est_py] - combined[col_est_R]
    combined['Diff_FN'] = combined[col_fn_py] - combined[col_fn_R]
    combined['RelDiff_Est'] = (combined['Diff_Est'] / combined[col_est_R]) * 100
    combined['RelDiff_FN'] = (combined['Diff_FN'] / combined[col_fn_R]) * 100

    combined['Year'] = combined.index.year
    combined['Month'] = combined.index.month
    combined['DOY'] = combined.index.dayofyear

    # Summary Stats
    print("Audit Summary:")
    print("-" * 20)
    print(f"Mean Abs Diff (Est): {combined['Diff_Est'].abs().mean():.6f}")
    print(f"Max Abs Diff (Est): {combined['Diff_Est'].abs().max():.6f}")
    print(f"Mean Rel Diff (Est) %: {combined['RelDiff_Est'].abs().mean():.4f}%")
    print(f"Mean Abs Diff (FN): {combined['Diff_FN'].abs().mean():.6f}")
    print(f"Max Abs Diff (FN): {combined['Diff_FN'].abs().max():.6f}")
    print(f"Mean Rel Diff (FN) %: {combined['RelDiff_FN'].abs().mean():.4f}%")

    # Top 10 discrepancies
    print("\nTop 10 Largest Absolute Differences (Estimated):")
    print(combined.loc[combined['Diff_Est'].abs().nlargest(10).index, [col_est_py, col_est_R, 'Diff_Est', 'RelDiff_Est', 'Q']])

    print("\nTop 10 Largest Absolute Differences (FN):")
    print(combined.loc[combined['Diff_FN'].abs().nlargest(10).index, [col_fn_py, col_fn_R, 'Diff_FN', 'RelDiff_FN', 'Q']])

    # Plotting
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))

    # Time Series of Differences
    axes[0, 0].plot(combined.index, combined['Diff_Est'], label='Est Diff')
    axes[0, 0].set_title('Estimated Concentration Difference (Py - R) over Time')
    axes[0, 0].set_ylabel('Difference (mg/L)')
    axes[0, 0].legend()

    axes[0, 1].plot(combined.index, combined['Diff_FN'], label='FN Diff', color='orange')
    axes[0, 1].set_title('Flow Normalized Concentration Difference (Py - R) over Time')
    axes[0, 1].set_ylabel('Difference (mg/L)')
    axes[0, 1].legend()

    # Difference vs Log(Q)
    axes[1, 0].scatter(np.log(combined['Q']), combined['Diff_Est'], alpha=0.5, s=10)
    axes[1, 0].set_title('Est Diff vs Log(Q)')
    axes[1, 0].set_xlabel('Log(Q)')
    axes[1, 0].set_ylabel('Difference')

    axes[1, 1].scatter(np.log(combined['Q']), combined['Diff_FN'], alpha=0.5, s=10, color='orange')
    axes[1, 1].set_title('FN Diff vs Log(Q)')
    axes[1, 1].set_xlabel('Log(Q)')
    axes[1, 1].set_ylabel('Difference')

    # Difference vs Season (DOY)
    axes[2, 0].scatter(combined['DOY'], combined['Diff_Est'], alpha=0.5, s=10)
    axes[2, 0].set_title('Est Diff vs Day of Year')
    axes[2, 0].set_xlabel('DOY')
    axes[2, 0].set_ylabel('Difference')

    axes[2, 1].scatter(combined['DOY'], combined['Diff_FN'], alpha=0.5, s=10, color='orange')
    axes[2, 1].set_title('FN Diff vs Day of Year')
    axes[2, 1].set_xlabel('DOY')
    axes[2, 1].set_ylabel('Difference')

    plt.tight_layout()
    plt.savefig('audit_plots.png')
    print("\nPlots saved to audit_plots.png")

if __name__ == "__main__":
    audit()
