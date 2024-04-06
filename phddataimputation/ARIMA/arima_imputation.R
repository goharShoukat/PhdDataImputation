if (!requireNamespace("imputeTS", quietly = TRUE)) {
  install.packages("imputeTS")
}

if (!requireNamespace("forecast", quietly = TRUE)) {
  install.packages("forecast")
}
library(tseries)
library(imputeTS)
library(forecast)
library(tidyr)
setwd("~/Documents/GitHub/PhdDataImputation/phddataimputation")
file_path <- "data/trainingData/M2_1hour_Gaps_10%_Missing.csv"

#data <- read.csv(file_path)$WindSpeed_artificial_gaps[1:672]
data <- read.csv(file_path)$WindSpeed_original[1:672]
plot_acf<-acf(data, lag.max = 672, )
pdf("acf_plot.pdf")
plot(plot_acf, main = "Autocorrelation Function (ACF)", xlab = "Lag", ylab = "Autocorrelation", mar = c(5, 5, 4, 2))
dev.off()

plot_pacf<-pacf(data, lag.max = 10, )
pdf("pacf_plot.pdf")
plot(plot_pacf, main = "Partial-Autocorrelation Function (ACF)", xlab = "Lag", ylab = "Autocorrelation", mar = c(5, 5, 4, 2))
dev.off()


########################################################################################################################
diff_data <- diff(data)
diff_data2 <- diff(diff_data)
adf.test(diff_data)
plot_acf <- acf(diff_data2, lag.max = 5)
pdf("acf_diff_plot.pdf")
plot(plot_acf, main = "Autocorrelation Function (ACF)", xlab = "Lag", ylab = "Autocorrelation", mar = c(5, 5, 4, 2))
dev.off()


arima_model <- forecast::auto.arima(data)

acf(diff((na.remove(data))))
order <- arima_model$arma[1:3]

imputed_data <- imputeTS::na_kalman(data, model = 'arima_model', smooth = TRUE)


write.csv(data.frame(imputed_data), file = "output/arima/imputed_series.csv", row.names = FALSE)
