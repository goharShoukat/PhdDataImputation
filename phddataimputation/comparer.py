import pandas as pd
from sklearn.metrics import mean_squared_error
from utils import plot
from config import config1, config2, config3, config4

original = pd.read_csv("data/trainingData/M2_1hour_Gaps_10%_Missing.csv").iloc[
    :672
]
arima = pd.read_csv("output/arima/imputed_series.csv")
arima_mse = mean_squared_error(
    original["WindSpeed_original"], arima["imputed_data"]
)

lstm = []
for con in config1():
    print(con)
    model_path = "output/{}/Model1-{}Neurons{}/".format(
        con["features"],
        con["neurons"],
        {True: "Scaled", False: "WithoutScale"}[con["scaling"]],
    )
    # path = "output/{}/Model1-{}Neurons{}/".format(1, 128, "Scaled")
    df = pd.read_csv(
        model_path + "/reconstructed.csv", index_col=False, header=None
    )
    lstm_mse = mean_squared_error(
        original["WindSpeed_artificial_gaps"]
        .dropna()
        .reset_index(drop=True)[con["features"] :],
        df,
    )
    lstm.append(lstm_mse)
