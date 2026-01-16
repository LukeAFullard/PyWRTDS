library(EGRET)
sample_df <- read.csv("test_data_sample.csv")
daily_df <- read.csv("test_data_daily.csv")
sample_df$Date <- as.Date(sample_df$Date)
daily_df$Date <- as.Date(daily_df$Date)
sample_df <- sample_df[!is.na(sample_df$Conc),]

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
Sample$MonthSeq <- Sample$Month + (as.numeric(format(Sample$Date, "%Y")) - 1850) * 12

# Simulate CrossVal on point 1
# Target is point 1
estY <- Sample$DecYear[1]
estLQ <- Sample$LogQ[1]
DecLow <- decimalDate(min(daily_df$Date))
DecHigh <- decimalDate(max(daily_df$Date))

# Remove point 1 from Sample
Sample_loo <- Sample[-1, ]

res <- run_WRTDS(estY, estLQ, Sample_loo,
                 windowY=7, windowQ=2, windowS=0.5,
                 minNumObs=10, minNumUncen=5, edgeAdjust=TRUE,
                 DecLow=DecLow, DecHigh=DecHigh)
print("LOO run_WRTDS succeeded")
