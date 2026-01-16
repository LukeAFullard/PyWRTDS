library(EGRET)

# Load Data
daily_df <- read.csv("../../test_data_daily.csv")
sample_df <- read.csv("../../test_data_sample.csv")

daily_df$Date <- as.Date(daily_df$Date)
sample_df$Date <- as.Date(sample_df$Date)
sample_df <- sample_df[!is.na(sample_df$Conc),]

# Create eList (Simplified)
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
  SinDY = sin(2*pi*decimalDate(sample_df$Date)),
  CosDY = cos(2*pi*decimalDate(sample_df$Date)),
  LogQ = log(sample_df$Q)
)

Daily <- data.frame(
  Date = daily_df$Date,
  Q = daily_df$Q,
  Julian = as.numeric(daily_df$Date - as.Date("1850-01-01")),
  Month = as.numeric(format(daily_df$Date, "%m")),
  Day = as.numeric(format(daily_df$Date, "%d")),
  DecYear = decimalDate(daily_df$Date),
  LogQ = log(daily_df$Q)
)
Daily$MonthSeq <- Daily$Month + (as.numeric(format(Daily$Date, "%Y")) - 1850) * 12
Sample$MonthSeq <- Sample$Month + (as.numeric(format(Sample$Date, "%Y")) - 1850) * 12

INFO <- data.frame(
  param.nm = "Concentration",
  shortName = "Conc",
  constitAbbrev = "C",
  staAbbrev = "Test"
)

eList <- as.egret(INFO, Daily, Sample, NA)

# Standard Estimation
# Use low-level functions to skip CrossVal (which seems to fail on this small dataset)
surfaces <- estSurfaces(eList, minNumObs=10, minNumUncen=5, verbose=FALSE)
eList <- as.egret(eList$INFO, eList$Daily, eList$Sample, surfaces)
eList <- estDailyFromSurfaces(eList)

# Export standard results (Daily)
# We need `ConcDay` (Estimated) and `FNConc` (Flow Normalized)
out_df <- eList$Daily[, c("Date", "ConcDay", "FNConc")]
write.csv(out_df, "egret_results.csv", row.names=FALSE)
print("Saved egret_results.csv")
