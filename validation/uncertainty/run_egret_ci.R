library(EGRET)
library(EGRETci)

# Load Data
daily_df <- read.csv("../../test_data_daily.csv")
sample_df <- read.csv("../../test_data_sample.csv")

# Ensure Dates are Dates
daily_df$Date <- as.Date(daily_df$Date)
sample_df$Date <- as.Date(sample_df$Date)
sample_df <- sample_df[!is.na(sample_df$Conc),]

# Create eList
# We manually construct Sample and Daily dataframes as expected by EGRET
# Sample: Date, ConcLow, ConcHigh, Uncen, ConcAve, Julian, Month, Day, DecYear, MonthSeq, SinDY, CosDY
Sample <- data.frame(
  Date = sample_df$Date,
  ConcLow = sample_df$Conc,
  ConcHigh = sample_df$Conc,
  Uncen = 1,
  ConcAve = sample_df$Conc,
  Julian = as.numeric(sample_df$Date - as.Date("1850-01-01")),
  Month = as.numeric(format(sample_df$Date, "%m")),
  Day = as.numeric(format(sample_df$Date, "%d")),
  DecYear = decimalDate(sample_df$Date),
  MonthSeq = NA, # Will compute
  SinDY = sin(2*pi*decimalDate(sample_df$Date)),
  CosDY = cos(2*pi*decimalDate(sample_df$Date)),
  LogQ = log(sample_df$Q)
)
Sample$MonthSeq <- Sample$Month + (as.numeric(format(Sample$Date, "%Y")) - 1850) * 12

# Daily: Date, Q, Julian, Month, Day, DecYear, MonthSeq, LogQ
Daily <- data.frame(
  Date = daily_df$Date,
  Q = daily_df$Q,
  Julian = as.numeric(daily_df$Date - as.Date("1850-01-01")),
  Month = as.numeric(format(daily_df$Date, "%m")),
  Day = as.numeric(format(daily_df$Date, "%d")),
  DecYear = decimalDate(daily_df$Date),
  MonthSeq = NA,
  LogQ = log(daily_df$Q)
)
Daily$MonthSeq <- Daily$Month + (as.numeric(format(Daily$Date, "%Y")) - 1850) * 12

INFO <- data.frame(
  param.nm = "Concentration",
  shortName = "Conc",
  constitAbbrev = "C",
  staAbbrev = "Test"
)

eList <- as.egret(INFO, Daily, Sample, NA)

# Standard Estimate first
surfaces <- estSurfaces(eList, minNumObs=10, minNumUncen=5, verbose=FALSE)
eList <- as.egret(eList$INFO, eList$Daily, eList$Sample, surfaces)
daily_resp <- estDailyFromSurfaces(eList)
if (inherits(daily_resp, "eList")) {
  eList <- daily_resp
} else {
  eList$Daily <- daily_resp
}

# Run CI Calculations
# We use standard bootstrap settings (nBoot=10 for speed in validation, blockLength=200)
# Note: EGRETci allows specifying nBoot.
# Block bootstrap logic in EGRETci is specific.
set.seed(42)
# ciCalculations is computationally intensive.
# We will run a small number for validation demo.
# Pass minNumObs/Uncen to ensure inner modelEstimation calls work on small dataset
CIAnnualResults <- ciCalculations(eList, nBoot = 10, blockLength = 200, widthCI = 90, minNumObs=10, minNumUncen=5)

# Extract and Save
# We want the daily CI or annual? The user python script validates daily/sample level uncertainty usually?
# Python script `validate_bootstrap.py` returns daily time series of CIs.
# EGRETci `ciCalculations` returns ANNUAL CIs in `CIAnnualResults`.
# Does EGRETci support Daily CIs?
# `bootAnnual` is internal.
# `ciCalculations` basically runs WRTDS `nBoot` times.
# If we want daily comparison, we should save the daily mean/p05/p95 from the boots?
# EGRETci doesn't expose daily series easily in the public API return of `ciCalculations` (it returns a table).
# But we can replicate the logic: run `run_WRTDS` nBoot times on resampled data.

# Replicate Python "Block Bootstrap" on Daily Series
# Generate N bootstraps
n_boot <- 10
boot_results <- matrix(NA, nrow=length(Daily$Date), ncol=n_boot)
# This populates Daily$ConcDay, Daily$FNConc

# We need the residuals to bootstrap
# residuals = log(Sample$Conc) - Predicted_at_Sample
# EGRET calculates this.

# For this validation script, we will simply export the Standard Model results to double check.
# But wait, we want Uncertainty comparison.
# If EGRETci only gives annual, we can compare Annual Uncertainty?
# Python `bootstrap_uncertainty` returns a daily dataframe.
# Let's save the Annual CI results to `r_results_uncertainty.csv`
# and update Python script to aggregate its daily results to Annual for comparison.

write.csv(CIAnnualResults, "r_results_uncertainty.csv", row.names=FALSE)
print("Saved r_results_uncertainty.csv")
