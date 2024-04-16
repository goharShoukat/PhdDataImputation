import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

original = pd.read_csv("data/trainingData/M2_1hour_Gaps_10%_Missing.csv").iloc[
    :672
]
missing_values = original[original["WindSpeed_artificial_gaps"].isna()].index

arima = pd.read_csv("output/arima/imputed_series.csv")
arima_imputed = arima.to_numpy()[missing_values]
arima_mse = mean_squared_error(
    original["WindSpeed_original"][missing_values], arima_imputed
)
# arima_mse = mean_squared_error(
#     original["WindSpeed_original"], arima["imputed_data"]
# )


# path = "output/2/{}/full-reconstruction.csv".format(m)
path = "output/1_dropout_0.025/stateful.csv"
lstm = pd.read_csv(path)
lstm_imputed = lstm["Predicted_WindSpeed"].to_numpy()[missing_values]
mse_lstm = (
    mean_squared_error(
        original["WindSpeed_original"][missing_values].to_numpy(),
        lstm_imputed,
    ),
)


original_imputed = original["WindSpeed_original"][missing_values]

plt.figure(figsize=(30, 30))
plt.scatter(
    original_imputed, lstm_imputed, label="LSTM", color="red", alpha=0.5
)
plt.scatter(
    original_imputed, arima_imputed, label="ARIMA", color="blue", alpha=0.5
)
plt.axline((0, 0), slope=1, color="black")
plt.legend()
plt.xlabel("True Speed (m/s)")
plt.ylabel("Imputed speed (m/s)")
plt.savefig("QQ Comparison.png", dpi=600)
# lstm = []
# for con in config1():
#     print(con)
#     model_path = "output/{}/Model1-{}Neurons{}/".format(
#         con["features"],
#         con["neurons"],
#         {True: "Scaled", False: "WithoutScale"}[con["scaling"]],
#     )
#     # path = "output/{}/Model1-{}Neurons{}/".format(1, 128, "Scaled")
#     df = pd.read_csv(
#         model_path + "/reconstructed.csv", index_col=False, header=None
#     )
#     lstm_mse = mean_squared_error(
#         original["WindSpeed_artificial_gaps"]
#         .dropna()
#         .reset_index(drop=True)[con["features"] :],
#         df,
#     )
#     lstm.append(lstm_mse)
