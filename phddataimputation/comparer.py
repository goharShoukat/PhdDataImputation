import pandas as pd
from sklearn.metrics import mean_squared_error

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

lstm_error = []
for m in [2, 3, 4, 5, 6, 7]:
    path = "output/2/{}/full-reconstruction.csv".format(m)
    lstm = pd.read_csv(path, header=None)
    lstm_imputed = lstm.to_numpy()[missing_values]
    lstm_error.append(
        [
            m,
            mean_squared_error(
                original["WindSpeed_original"][missing_values], lstm_imputed
            ),
        ]
    )

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
