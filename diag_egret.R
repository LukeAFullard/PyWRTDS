library(EGRET)

# Load Data
daily_df <- read.csv("test_data_daily.csv")
sample_df <- read.csv("test_data_sample.csv")

# Point 14 (Index 14 in R 1-based)
target_row <- daily_df[14, ]
target_date <- as.Date(target_row$Date)
target_Q <- target_row$Q
target_LogQ <- log(target_Q)
target_DecYear <- decimalDate(target_date)

print(paste("Target Date:", target_date))
print(paste("Target DecYear:", target_DecYear))
print(paste("Target LogQ:", target_LogQ))

# Prepare Sample
sample_df$Date <- as.Date(sample_df$Date)
sample_df <- sample_df[!is.na(sample_df$Conc), ]
Sample <- data.frame(
    Date = sample_df$Date,
    ConcLow = sample_df$Conc,
    ConcHigh = sample_df$Conc,
    Uncen = 1,
    ConcAve = sample_df$Conc
)
Sample$Julian <- as.numeric(Sample$Date - as.Date("1850-01-01"))
Sample$Month <- as.numeric(format(Sample$Date, "%m"))
Sample$Day <- as.numeric(format(Sample$Date, "%d"))
Sample$DecYear <- decimalDate(Sample$Date)
Sample$MonthSeq <- Sample$Month + (as.numeric(format(Sample$Date, "%Y")) - 1850) * 12
Sample$SinDY <- sin(2 * pi * Sample$DecYear)
Sample$CosDY <- cos(2 * pi * Sample$DecYear)
Sample$LogQ <- log(sample_df$Q) # Ensure LogQ is present

# DecLow / DecHigh (Range of Daily)
DecLow <- decimalDate(as.Date(daily_df$Date[1]))
DecHigh <- decimalDate(as.Date(daily_df$Date[nrow(daily_df)]))

# Run WRTDS for single point
print("Running run_WRTDS...")
set.seed(42)

res <- run_WRTDS(
    estY = target_DecYear,
    estLQ = target_LogQ,
    localSample = Sample,
    DecLow = DecLow,
    DecHigh = DecHigh,
    minNumObs = 50, # Match Py default override
    minNumUncen = 50,
    windowY = 7,
    windowQ = 2,
    windowS = 0.5,
    edgeAdjust = TRUE
)

print("Result (survReg):")
print(res$survReg)

# Replicate Logic to get Internals
windowY <- 7
windowQ <- 2
windowS <- 0.5
minNumObs <- 50
minNumUncen <- 50
edgeAdjust <- TRUE

estY <- target_DecYear
estLQ <- target_LogQ

# Edge Adjust
distLow <- estY - DecLow
distHigh <- DecHigh - estY
distTime <- min(distLow, distHigh)
tempWindowY <- windowY
if (edgeAdjust & !is.na(distTime)) {
    tempWindowY <- if (distTime > tempWindowY) tempWindowY else ((2 * tempWindowY) - distTime)
}
print(paste("Initial Edge Adjusted WindowY:", tempWindowY))

# Loop
tempWindowQ <- windowQ
tempWindowS <- windowS
k <- 1
repeat {
    Sam <- Sample[abs(Sample$DecYear - estY) <= tempWindowY, ]
    diffY <- abs(Sam$DecYear - estY)
    weightY <- triCube(diffY, tempWindowY)
    weightQ <- triCube(Sam$LogQ - estLQ, tempWindowQ)
    diffUpper <- ceiling(diffY)
    diffLower <- floor(diffY)
    diffSeason <- pmin(abs(diffUpper - diffY), abs(diffY - diffLower))
    weightS <- triCube(diffSeason, tempWindowS)
    Sam$weight <- weightY * weightQ * weightS
    Sam <- subset(Sam, weight > 0)
    numPosWt <- length(Sam$weight)
    numUncen <- sum(Sam$Uncen)

    if (numPosWt >= minNumObs & numUncen >= minNumUncen | k > 10000) break

    tempWindowY <- tempWindowY * 1.1
    tempWindowQ <- tempWindowQ * 1.1
    tempWindowS <- if (windowS <= 0.5) min(tempWindowS * 1.1, 0.5) else windowS
    k <- k + 1
}

print(paste("Final WindowY:", tempWindowY))
print(paste("Num Active:", numPosWt))
print(paste("Sum Weights (Raw):", sum(Sam$weight)))

# Run SurvReg
weight <- Sam$weight
aveWeight <- sum(weight)/numPosWt
weight <- weight/aveWeight
Sam <- data.frame(Sam)

survModel <- survival::survreg(survival::Surv(log(ConcLow), log(ConcHigh), type = "interval2") ~ DecYear + LogQ + SinDY + CosDY,
                               data = Sam, weights = weight, dist = "gaus")

print("Coefficients:")
print(coef(survModel))
print(paste("Scale (Sigma):", survModel$scale))
