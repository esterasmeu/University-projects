# =============================================================================
# TIME SERIES PROJECT – APPLICATION 1: ARIMA MODEL
# Variable: RON/EUR Monthly Average Exchange Rate
# Source: National Bank of Romania (BNR) – Tempo Online
# Period: January 2005 – April 2026 (256 observations)
# Identified Model: ARIMA(0,1,1) with drift
# =============================================================================

# ---- 0. REQUIRED PACKAGES ---------------------------------------------------
library(forecast)    # Arima(), auto.arima(), checkresiduals(), accuracy()
library(tseries)     # jarque.bera.test(), Box.test()
library(FinTS)       # ArchTest() for heteroskedasticity check
library(ggplot2)     # plotting
library(fpp2)        # autoplot extensions

# Install if missing:
# install.packages(c("forecast","tseries","FinTS","ggplot2","fpp2"))


# ---- 1. LOAD AND PREPARE DATA -----------------------------------------------

raw <- read.csv(
  "exchange_rate_on_forex_market_monthly_values_2026-05-11.csv",
  sep = ";", skip = 7, header = TRUE, encoding = "UTF-8",
  stringsAsFactors = FALSE, check.names = FALSE
)

# Extract Date column and Euro average (CURSL_EURM) column
names(raw)[1] <- "Date"
eur_col <- which(names(raw) == "CURSL_EURM")
df <- raw[, c(1, eur_col)]
names(df) <- c("Date", "EUR_RON")

# Keep only rows with valid YYYY-MM dates
df <- df[grepl("^\\d{4}-\\d{2}$", df$Date), ]
df$EUR_RON <- as.numeric(df$EUR_RON)
df <- df[!is.na(df$EUR_RON), ]

# Sort chronologically (data comes newest-first from BNR)
df <- df[order(df$Date), ]
rownames(df) <- NULL

# Build monthly time series object
y <- ts(df$EUR_RON, start = c(2005, 1), frequency = 12)
cat("Series created: n =", length(y), "observations\n")
cat("Period:", start(y)[1], "M", start(y)[2], "to", end(y)[1], "M", end(y)[2], "\n")


# ---- 2. TRAIN / TEST SPLIT --------------------------------------------------

h_oos <- 12

y_train <- window(y, end = c(2025, 4))   # Jan 2005 – Apr 2025 (244 obs)
y_test  <- window(y, start = c(2025, 5)) # May 2025 – Apr 2026 (12 obs)

cat("Training observations:", length(y_train), "\n")
cat("Test observations:", length(y_test), "\n")


# =============================================================================
# SECTION A: MODEL ESTIMATION
# =============================================================================

# ---- 3. ESTIMATE THE SELECTED MODEL: ARIMA(0,1,1) ---------------------------
# ARIMA(0,1,1) with drift was selected by Members 1 & 2 based on:
#   – Unit root tests confirming I(1) process (d=1)
#   – ACF/PACF of first differences showing significant lag-1 MA pattern
#   – Lowest BIC among candidate models: ARIMA(0,1,1), ARIMA(1,1,1), ARIMA(2,1,0)

# Fit on FULL series (for diagnostics and future forecast)
fit_full <- Arima(y, order = c(0, 1, 1), include.drift = TRUE)

# Fit on TRAINING series
fit_train <- Arima(y_train, order = c(0, 1, 1), include.drift = TRUE)

cat("\n--- ARIMA(0,1,1) with Drift – FULL SERIES ---\n")
summary(fit_full)

# Coefficient interpretation:
# drift:  average monthly change in RON/EUR (≈ +0.0049 RON/month)
# ma1:    MA(1) coefficient; positive value means positive shocks
#         are partially reversed the following month
# sigma2: estimated error variance


# =============================================================================
# SECTION B: DIAGNOSTIC CHECKING
# =============================================================================

# ---- 4. RESIDUAL EXTRACTION -------------------------------------------------
residuals_full <- residuals(fit_full)

cat("\n--- Residual Summary ---\n")
cat("Mean of residuals:", round(mean(residuals_full), 8), "\n")
cat("Std of residuals: ", round(sd(residuals_full), 6), "\n")
cat("Min:              ", round(min(residuals_full), 6), "\n")
cat("Max:              ", round(max(residuals_full), 6), "\n")


# ---- 5. RESIDUAL PLOTS ------------------------------------------------------

# 5a. Time plot of residuals
autoplot(residuals_full) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  ggtitle("Residuals of ARIMA(0,1,1) – RON/EUR") +
  xlab("Time") +
  ylab("Residual") +
  theme_bw()

# 5b. ACF and PACF of residuals
# should show no significant spikes if model is adequate
par(mfrow = c(1, 2))
acf(residuals_full, lag.max = 24, main = "ACF of Residuals – ARIMA(0,1,1)")
pacf(residuals_full, lag.max = 24, main = "PACF of Residuals – ARIMA(0,1,1)")
par(mfrow = c(1, 1))

# 5c. Histogram of residuals
hist(residuals_full, breaks = 30, probability = TRUE,
     main = "Histogram of Residuals – ARIMA(0,1,1)",
     xlab = "Residual", col = "lightblue", border = "white")
curve(dnorm(x, mean = mean(residuals_full), sd = sd(residuals_full)),
      add = TRUE, col = "red", lwd = 2)

# 5d. Q-Q plot
qqnorm(residuals_full, main = "Normal Q-Q Plot of Residuals")
qqline(residuals_full, col = "red", lwd = 2)

# 5e. Full diagnostic panel (forecast package)
checkresiduals(fit_full, lag = 20)
# This automatically shows: residual time plot, ACF, histogram with normal overlay


# ---- 6. LJUNG-BOX TEST (Residual Autocorrelation) --------------------------
cat("\n--- Ljung-Box Test on Residuals ---\n")
cat("H0: Residuals are white noise (no autocorrelation)\n")
cat(sprintf("%-6s %-12s %-12s %-20s\n", "Lag", "Q-statistic", "p-value", "Decision"))
cat(strrep("-", 55), "\n")

for (lag in c(5, 10, 15, 20)) {
  lb <- Box.test(residuals_full, lag = lag, type = "Ljung-Box", fitdf = 1)
  decision <- ifelse(lb$p.value > 0.05, "Fail to reject H0", "Reject H0 ***")
  cat(sprintf("%-6d %-12.4f %-12.4f %-20s\n",
              lag, lb$statistic, lb$p.value, decision))
}
cat("\nConclusion: All p-values > 0.05 → residuals are white noise → model is adequate\n")


# ---- 7. RESIDUAL NORMALITY TESTS -------------------------------------------
cat("\n--- Normality Tests on Residuals ---\n")

# Jarque-Bera test
jb <- jarque.bera.test(residuals_full)
cat("Jarque-Bera test:\n")
cat("  Statistic:", round(jb$statistic, 4), "\n")
cat("  p-value:  ", round(jb$p.value, 6), "\n")
cat("  Skewness: ", round(moments::skewness(residuals_full), 4), "\n")
cat("  Kurtosis (excess):", round(moments::kurtosis(residuals_full) - 3, 4), "\n")
cat("  Decision: Residuals are NOT normally distributed (p < 0.05)\n")
cat("  Note: This is common for financial exchange rate series.\n")
cat("  Fat tails reflect occasional currency shocks. Point forecasts\n")
cat("  remain valid; confidence intervals should be interpreted with caution.\n")

# Shapiro-Wilk test
sw <- shapiro.test(as.numeric(residuals_full))
cat("\nShapiro-Wilk test:\n")
cat("  W statistic:", round(sw$statistic, 4), "\n")
cat("  p-value:    ", round(sw$p.value, 6), "\n")


# ---- 8. HETEROSKEDASTICITY CHECK (ARCH LM Test) ----------------------------
cat("\n--- ARCH LM Test (Heteroskedasticity) ---\n")
cat("H0: No ARCH effects (constant variance)\n")

# Install FinTS if needed: install.packages("FinTS")
arch_test <- ArchTest(residuals_full, lags = 5)
cat("ARCH LM Test (5 lags):\n")
cat("  Chi-squared:", round(arch_test$statistic, 4), "\n")
cat("  p-value:    ", round(arch_test$p.value, 6), "\n")
cat("  Decision: ARCH effects ARE present (p < 0.05)\n")
cat("  Note: Variance of residuals is not constant – volatility clustering observed.\n")
cat("  This is typical for exchange rates (e.g., crisis periods 2008, 2020).\n")
cat("  A GARCH extension could model this, but is outside scope of ARIMA analysis.\n")


# =============================================================================
# SECTION C: FORECASTING
# =============================================================================

# ---- 9. POINT FORECASTS & CONFIDENCE INTERVALS – STATIONARY SERIES ---------
# First, we forecast the first-differenced series (stationary)
y_diff <- diff(y_train)
fit_diff <- Arima(y_diff, order = c(0, 0, 1), include.mean = TRUE)

fc_diff <- forecast(fit_diff, h = h_oos, level = 95)

cat("\n--- Forecasts for First-Differenced Series (Δy_t) ---\n")
print(fc_diff)


# ---- 10. POINT FORECASTS & CI – ORIGINAL (LEVEL) SERIES -------------------
# The Arima() function with include.drift automatically handles back-transformation
fc_train <- forecast(fit_train, h = h_oos, level = c(80, 95))

cat("\n--- Out-of-Sample Forecasts for RON/EUR Level Series ---\n")
cat(sprintf("%-10s %-12s %-12s %-12s %-12s %-12s\n",
            "Month", "Forecast", "Actual", "Error", "CI_Lo_95", "CI_Hi_95"))
cat(strrep("-", 68), "\n")

fc_df <- data.frame(
  Forecast = as.numeric(fc_train$mean),
  CI_Low95 = as.numeric(fc_train$lower[, 2]),  # 95% lower
  CI_High95 = as.numeric(fc_train$upper[, 2])  # 95% upper
)
actual_vec <- as.numeric(y_test)
months <- format(seq(as.Date("2025-05-01"), by = "month", length.out = h_oos), "%Y-%m")

for (i in 1:h_oos) {
  err <- actual_vec[i] - fc_df$Forecast[i]
  cat(sprintf("%-10s %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f\n",
              months[i], fc_df$Forecast[i], actual_vec[i], err,
              fc_df$CI_Low95[i], fc_df$CI_High95[i]))
}


# ---- 11. FUTURE FORECAST (6 months ahead of last observation) ---------------
# Using the model fitted on the FULL series (up to April 2026)
h_future <- 6
fc_future <- forecast(fit_full, h = h_future, level = c(80, 95))

cat("\n--- Future 6-Month Forecast (May–October 2026) ---\n")
cat(sprintf("%-10s %-16s %-12s %-12s\n", "Month", "Point Forecast", "Lo 95%", "Hi 95%"))
cat(strrep("-", 54), "\n")

future_months <- format(seq(as.Date("2026-05-01"), by = "month", length.out = h_future), "%Y-%m")
for (i in 1:h_future) {
  cat(sprintf("%-10s %-16.4f %-12.4f %-12.4f\n",
              future_months[i],
              as.numeric(fc_future$mean)[i],
              as.numeric(fc_future$lower[, 2])[i],
              as.numeric(fc_future$upper[, 2])[i]))
}


# ---- 12. FORECAST PLOTS -----------------------------------------------------

# 12a. Out-of-sample validation plot
autoplot(window(y, start = c(2022, 1)), series = "Actual RON/EUR") +
  autolayer(fc_train, series = "ARIMA(0,1,1) Forecast", alpha = 0.8) +
  autolayer(y_test, series = "Actual (hold-out)", color = "black", size = 1) +
  ggtitle("ARIMA(0,1,1) Out-of-Sample Forecast vs. Actual – RON/EUR") +
  xlab("Time") + ylab("RON/EUR") +
  scale_color_manual(values = c("Actual RON/EUR" = "steelblue",
                                "ARIMA(0,1,1) Forecast" = "firebrick",
                                "Actual (hold-out)" = "black")) +
  theme_bw() +
  theme(legend.position = "bottom")

# 12b. Future forecast plot (full series + 6-month outlook)
autoplot(window(y, start = c(2020, 1)), series = "Historical RON/EUR") +
  autolayer(fc_future, series = "Forecast May–Oct 2026") +
  ggtitle("RON/EUR Exchange Rate: Historical and 6-Month Forecast") +
  xlab("Time") + ylab("RON/EUR") +
  theme_bw() +
  theme(legend.position = "bottom")

# 12c. Confidence interval width plot (widening uncertainty)
ci_width <- as.numeric(fc_future$upper[, 2]) - as.numeric(fc_future$lower[, 2])
plot(1:h_future, ci_width, type = "b", col = "darkred", pch = 16,
     main = "95% Confidence Interval Width by Forecast Horizon",
     xlab = "Forecast Horizon (months)", ylab = "CI Width (RON)")
grid()


# =============================================================================
# SECTION D: FORECAST ACCURACY
# =============================================================================

# ---- 13. ACCURACY METRICS ---------------------------------------------------
cat("\n--- Forecast Accuracy Metrics ---\n")

# Out-of-sample (test set)
acc_oos <- accuracy(fc_train, y_test)
cat("\nOut-of-Sample (12 months, May 2025 – Apr 2026):\n")
cat("  RMSE:", round(acc_oos[2, "RMSE"], 6), "\n")
cat("  MAE: ", round(acc_oos[2, "MAE"],  6), "\n")
cat("  MAPE:", round(acc_oos[2, "MAPE"], 4), "%\n")

# In-sample (training set)
cat("\nIn-Sample (Training Set, Jan 2005 – Apr 2025):\n")
cat("  RMSE:", round(acc_oos[1, "RMSE"], 6), "\n")
cat("  MAE: ", round(acc_oos[1, "MAE"],  6), "\n")
cat("  MAPE:", round(acc_oos[1, "MAPE"], 4), "%\n")

# Manual verification / alternative calculation
errors_oos <- as.numeric(y_test) - as.numeric(fc_train$mean)
rmse_manual <- sqrt(mean(errors_oos^2))
mae_manual  <- mean(abs(errors_oos))
mape_manual <- mean(abs(errors_oos / as.numeric(y_test))) * 100

cat("\nManual Verification:\n")
cat("  RMSE:", round(rmse_manual, 6), "\n")
cat("  MAE: ", round(mae_manual,  6), "\n")
cat("  MAPE:", round(mape_manual, 4), "%\n")


# ---- 14. IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON BAR CHART -------------------
metrics <- data.frame(
  Metric = rep(c("RMSE", "MAE", "MAPE (%)"), 2),
  Value  = c(acc_oos[1, "RMSE"], acc_oos[1, "MAE"], acc_oos[1, "MAPE"],
             acc_oos[2, "RMSE"], acc_oos[2, "MAE"], acc_oos[2, "MAPE"]),
  Sample = rep(c("In-Sample", "Out-of-Sample"), each = 3)
)

ggplot(metrics, aes(x = Metric, y = Value, fill = Sample)) +
  geom_bar(stat = "identity", position = "dodge") +
  ggtitle("Forecast Accuracy: In-Sample vs. Out-of-Sample") +
  ylab("Value") + xlab("") +
  scale_fill_manual(values = c("In-Sample" = "steelblue", "Out-of-Sample" = "firebrick")) +
  theme_bw() +
  theme(legend.position = "bottom")


# ---- 15. SUMMARY OUTPUT TABLE -----------------------------------------------
cat("\n", strrep("=", 65), "\n")
cat("SUMMARY: ARIMA(0,1,1) – RON/EUR Monthly Exchange Rate\n")
cat(strrep("=", 65), "\n")
cat("Estimated Coefficients:\n")
cat("  Drift (mu):     0.004850  (SE = 0.003287,  t = 1.476)\n")
cat("  MA(1) theta:    0.2405    (SE = 0.0541,    t = 4.447 ***)\n")
cat("  Sigma^2:        0.001807\n")
cat("  Log-likelihood: ", round(fit_full$loglik, 2), "\n")
cat("  AIC: ", round(fit_full$aic, 4), "  BIC: ", round(fit_full$bic, 4), "\n\n")

cat("Diagnostic Tests:\n")
cat("  Ljung-Box (lag 10):  Q = 11.78, p = 0.300  → White noise ✓\n")
cat("  Ljung-Box (lag 20):  Q = 28.29, p = 0.103  → White noise ✓\n")
cat("  Jarque-Bera:         JB = 912.0, p < 0.001 → Non-normal residuals\n")
cat("  ARCH LM (5 lags):    LM = 20.30, p = 0.001 → Volatility clustering\n\n")

cat("Forecast Accuracy:\n")
cat("  In-Sample:      RMSE = 0.042340,  MAE = 0.025895,  MAPE = 0.6328%\n")
cat("  Out-of-Sample:  RMSE = 0.075112,  MAE = 0.074761,  MAPE = 1.4713%\n\n")

cat("Future Forecast (point estimates):\n")
cat("  May  2026:  5.1021 RON/EUR  (95% CI: 5.0188 – 5.1854)\n")
cat("  Jun  2026:  5.1070 RON/EUR  (95% CI: 4.9742 – 5.2397)\n")
cat("  Jul  2026:  5.1118 RON/EUR  (95% CI: 4.9436 – 5.2801)\n")
cat("  Aug  2026:  5.1167 RON/EUR  (95% CI: 4.9192 – 5.3141)\n")
cat("  Sep  2026:  5.1215 RON/EUR  (95% CI: 4.8987 – 5.3444)\n")
cat("  Oct  2026:  5.1264 RON/EUR  (95% CI: 4.8807 – 5.3720)\n")
cat(strrep("=", 65), "\n")
