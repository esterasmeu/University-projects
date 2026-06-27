
# APPLICATION 2 — VAR / VECM ANALYSIS
# VARIABLES:
# 1. RON/EUR Exchange Rate
# 2. ROBOR 3M
# 3. Inflation (HICP)

# LOAD PACKAGES


install.packages(c(
  "readxl",
  "vars",
  "urca",
  "tseries",
  "forecast",
  "lmtest",
  "tsDyn"
))

library(readxl)
library(vars)
library(urca)
library(tseries)
library(forecast)
library(lmtest)
library(tsDyn)
library(readxl)


install.packages("tseries", dependencies = TRUE)
library(tseries)
# IMPORT XLSX FILES



# Exchange Rate

#Select your exchange rate file:
print("Please select your Exchange Rate file:")
exchange <- read_excel(file.choose())


#Select your Inflation (HICP) file:
print("Please select your Inflation (HICP) file:")
inflation <- read_excel(file.choose())

hcip_analysis <- read_excel(file.choose())
exchange_analysis<-read_excel(file.choose())



# CHECK COLUMN NAMES


colnames(exchange_analysis)[1]
colnames(exchange_analysis)[2]
colnames(hcip_analysis_cleaned)[1]

# SELECT VARIABLES



EXR <- exchange_analysis$Euro
HCPI <- hcip_analysis_cleaned$TIME



# CREATE COMBINED DATASET


combined_data <- data.frame(
  EXR,
  HCPI
)

# Remove missing values

combined_data <- na.omit(combined_data)



# DATA ALIGNMENT & EXTRACTION (2010-01 to 2025-12)


# 1. Process Exchange Rate (Filter 2010-01 to 2025-12 and sort ascending)
exchange_filtered <- exchange_analysis[exchange_analysis$Date >= "2010-01" & exchange_analysis$Date <= "2025-12", ]
exchange_filtered <- exchange_filtered[order(exchange_filtered$Date), ]
EXR <- as.numeric(exchange_filtered$Euro)

# 2. Process Horizontal HICP Data for Romania
romania_row <- hcip_analysis_cleaned[hcip_analysis_cleaned$TIME == "Romania", ]
start_col   <- which(colnames(hcip_analysis_cleaned) == "2010-01")
end_col     <- which(colnames(hcip_analysis_cleaned) == "2025-12")

HCPI <- as.numeric(romania_row[ , start_col:end_col])
# Find the columns corresponding to our target timeframe
start_col <- which(colnames(hcip_analysis_cleaned) == "2010-01")
end_col   <- which(colnames(hcip_analysis_cleaned) == "2025-12")
HCPI <- as.numeric(romania_row[ , start_col:end_col])

combined_data <- data.frame(EXR, HCPI)
combined_data <- na.omit(combined_data)

# CONVERT TO TIME SERIES
# MONTHLY DATA => frequency = 12



ts_data <- ts(
  combined_data,
  start = c(2010,1),
  frequency = 12
)

# TASK 1 - NON-STATIONARITY ANALYSIS

# ADF TESTS - LEVELS


adf_exr <- adf.test(ts_data[, "EXR"])
print(adf_exr)

adf_hcpi <- adf.test(ts_data[, "HCPI"])
print(adf_hcpi)


# FIRST DIFFERENCE


diff_data <- diff(ts_data)

diff_data <- na.omit(diff_data)

# ADF TESTS - FIRST DIFFERENCES


adf_diff_exr <- adf.test(diff_data[, "EXR"])
print(adf_diff_exr)

adf_diff_hcpi <- adf.test(diff_data[, "HCPI"])
print(adf_diff_hcpi)


# TASK 2 - COINTEGRATION ANALYSIS

# LAG LENGTH SELECTION


lag_selection <- VARselect(
  ts_data,
  lag.max = 12,
  type = "const"
)

print(lag_selection)

print(lag_selection$selection)

# Example chosen lag

optimal_lag <- 2


# JOHANSEN COINTEGRATION TEST


johansen_test <- ca.jo(
  ts_data,
  type = "trace",
  ecdet = "const",
  K = optimal_lag
)

summary(johansen_test)

# TASK 3 - GRANGER CAUSALITY


var_model <- VAR(
  diff_data,
  p = optimal_lag,
  type = "const"
)



# EXCHANGE RATE -> INFLATION


granger_exr_inf <- causality(
  var_model,
  cause = "EXR"
)

print(granger_exr_inf)




# TASK 4 - VAR / VECM MODELING

# CASE A - VAR MODEL
# USE IF NO COINTEGRATION EXISTS


var_model <- VAR(
  diff_data,
  p = optimal_lag,
  type = "const"
)

summary(var_model)

# CASE B - VECM MODEL
# USE IF COINTEGRATION EXISTS

vecm_model <- cajorls(
  johansen_test,
  r = 1
)

summary(vecm_model$rlm)

# CONVERT VECM TO VAR FORM

vec_var <- vec2var(
  johansen_test,
  r = 1
)

# TASK 5 - IMPULSE RESPONSE FUNCTION

# EXCHANGE RATE SHOCK -> INFLATION RESPONSE

irf_exr_inf <- irf(
  vec_var,
  impulse = "EXR",
  response = "HCPI",
  n.ahead = 12,
  boot = TRUE
)
png("irf_exchange_rate_to_inflation.png", width = 800, height = 600)
par(mar = c(5, 5, 4, 2))
plot(irf_exr_inf)
dev.off()



print(irf_exr_inf)


