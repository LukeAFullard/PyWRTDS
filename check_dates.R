library(EGRET)
dates <- c("2000-01-01", "2000-02-29", "2000-03-01", "2001-01-01")
decs <- decimalDate(as.Date(dates))
print(data.frame(Date=dates, Dec=decs))
