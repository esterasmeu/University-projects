library(fpp2)
library(tseries)
library(urca)
library(stats)

# 1. Grab the dataset that is already in your environment
raw_data <- exchange_rate_on_forex_market_monthly_values_2026.05.11 

# 2. Clean the 9th column (RON/EUR): force to text, fix decimals, and make numeric
eur_text <- as.character(raw_data[[9]])
eur_text <- gsub(",", ".", trimws(eur_text)) # Remove spaces and fix Romanian commas
eur_clean <- na.omit(as.numeric(eur_text))   # Convert to numbers and drop text rows

# 3. Flip chronologically (oldest to newest) and build the Time Series
y <- ts(rev(eur_clean), start = c(2005, 1), frequency = 12)


# =========================================================
# TASK 1: EXPLORATORY ANALYSIS
# =========================================================
# Plot 1: Trend and Volatility
autoplot(y) +
  ggtitle("Monthly Evolution of RON/EUR Exchange Rate") +
  xlab("Year") +
  ylab("Exchange Rate (RON/EUR)") +
  theme_bw()

# Plot 2: Seasonal Subseries
ggsubseriesplot(y) +
  ylab("Rate") +
  ggtitle("Seasonal Subseries Plot – Exchange Rate") +
  theme_bw()


# =========================================================
# TASK 2: STATIONARITY ANALYSIS (INITIAL SERIES)
# =========================================================
# ADF TEST (Augmented Dickey-Fuller)
adf_trend <- ur.df(y, type = "trend", selectlags = "AIC")
summary(adf_trend)

# KPSS TEST
y %>% ur.kpss() %>% summary()

# PHILLIPS-PERRON TEST
PP.test(y)


# =========================================================
# TASK 3: TRANSFORMATION OF THE SERIES
# =========================================================
# Apply First Difference (d=1) to make it stationary
y_d1 <- diff(y, differences = 1)

# Plot the stationary differenced series
autoplot(y_d1) +
  ggtitle("First Difference of Exchange Rate (Stationary)") +
  theme_bw()

# Retest stationarity to prove it worked
summary(ur.df(y_d1, type = "none", selectlags = "AIC"))
PP.test(y_d1)


# =========================================================
# TASK 4: MODEL IDENTIFICATION
# =========================================================
# Plot ACF and PACF of the differenced series to guess the AR and MA terms
ggtsdisplay(y_d1, main="ACF and PACF for Model Identification")


# =========================================================
# TASK 5: MODEL SELECTION
# =========================================================
# Testing 3 candidate models based on standard financial series behavior
fit1 <- Arima(y, order = c(1,1,1))
fit2 <- Arima(y, order = c(2,1,0))
fit3 <- Arima(y, order = c(0,1,1))

# Compare AIC and BIC scores (Look for the lowest numbers!)
summary(fit1)
summary(fit2)
summary(fit3)

# Use the algorithm to find the absolute best mathematical fit
best_model <- auto.arima(y, ic="aic", trace=TRUE)
summary(best_model)

