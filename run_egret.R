library(EGRET)

# 1. Load Data
daily_df <- read.csv("test_data_daily.csv")
sample_df <- read.csv("test_data_sample.csv")

# Prepare inputs
daily_input <- data.frame(
  dateTime = as.character(daily_df$Date),
  value = daily_df$Q,
  code = rep("", nrow(daily_df))
)

Daily <- populateDaily(daily_input, qConvert=1, verbose=FALSE)

sample_df <- sample_df[!is.na(sample_df$Conc), ]
Sample <- data.frame(
    Date = as.Date(sample_df$Date),
    ConcLow = sample_df$Conc,
    ConcHigh = sample_df$Conc,
    Uncen = 1,
    ConcAve = sample_df$Conc
)
# Add required columns
Sample$Julian <- as.numeric(Sample$Date - as.Date("1850-01-01"))
Sample$Month <- as.numeric(format(Sample$Date, "%m"))
Sample$Day <- as.numeric(format(Sample$Date, "%d"))
Sample$DecYear <- decimalDate(Sample$Date)
Sample$MonthSeq <- Sample$Month + (as.numeric(format(Sample$Date, "%Y")) - 1850) * 12
Sample$SinDY <- sin(2 * pi * Sample$DecYear)
Sample$CosDY <- cos(2 * pi * Sample$DecYear)

INFO <- data.frame(
  station.nm = "Test Station",
  shortName = "Test",
  station.id = "00000000",
  dec_lat_va = 0,
  dec_long_va = 0,
  constituent = "Test Constituent",
  param.nm = "Test Param",
  param.units = "mg/L",
  drainSqKm = 100
)

eList <- mergeReport(INFO, Daily, Sample, verbose=FALSE)

# 2. Run WRTDS
# Use minNumObs=50 to correspond to our small dataset
eList <- modelEstimation(eList, verbose=FALSE, minNumObs=50)

# 4. Extract Results
results <- eList$Daily
output <- results[, c("Date", "ConcDay", "FNConc")]

write.csv(output, "egret_results.csv", row.names=FALSE)
