if (!requireNamespace("imputeTS", quietly = TRUE)) {
  install.packages("imputeTS")
}

if (!requireNamespace("forecast", quietly = TRUE)) {
  install.packages("forecast")
}

library(imputeTS)
library(forecast)

file_path <- "data/trainingData/M2_1hour_Gaps_10%_Missing.csv"

data <- read.csv(file_path)[1:672, 2]


arima_model <- forecast::auto.arima(data, seasonal = TRUE)


order <- arima_model$arma[1:3]

imputed_data <- imputeTS::na_kalman(data, model = arima_model, smooth = TRUE)


write.csv(data.frame(imputed_data), file = "output/arima/imputed_series.csv", row.names = FALSE)
