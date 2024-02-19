from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd


# https://medium.com/@vineet.pandya/use-tensorflow-lstm-for-time-series-forecasting-770ec789d2ce
# https://github.com/Vineet214/TimeSeriesForecasting-LSTM/blob/main/Univariate%20Time%20Series%20Forecasting%20-%20LSTM.ipynb
def Sequential_Input_LSTM(df, input_sequence):
    df_np = df.to_numpy()
    X = []
    y = []

    for i in range(len(df_np) - input_sequence):
        row = [a for a in df_np[i : i + input_sequence]]
        X.append(row)
        label = df_np[i + input_sequence]
        y.append(label)

    return np.array(X), np.array(y)


n_input = 2
model1 = Sequential()
n_features = 1
model1.add(InputLayer((n_input, n_features)))
model1.add(LSTM(100, return_sequences=True))
model1.add(LSTM(100, return_sequences=True))
model1.add(LSTM(100, return_sequences=True))
model1.add(LSTM(100, return_sequences=True))
model1.add(LSTM(100, return_sequences=True))
model1.add(LSTM(100, return_sequences=True))
model1.add(LSTM(100, return_sequences=True))
model1.add(LSTM(50))
model1.add(Dense(8, activation="relu"))
model1.add(Dense(1, activation="linear"))
# df_min_model_data = df_hour_lvl['T']

df = (
    pd.read_csv("data/trainingData/M2_1hour_Gaps_10%_Missing.csv").iloc[:672]
    # .dropna()
    # .reset_index(drop=True)
)

X, y = Sequential_Input_LSTM(
    df["WindSpeed_artificial_gaps"].dropna().reset_index(drop=True), n_input
)
model_save_callback = ModelCheckpoint(
    filepath="models/7_layers_lstm_2_FFN/{}/".format(n_input),
    monitor="loss",
    mode="min",
    save_best_only=True,
    # period=10,
)
model1.compile(
    loss=MeanSquaredError(),
    optimizer=Adam(learning_rate=0.0001),
    metrics=RootMeanSquaredError(),
)

history = model1.fit(
    X,
    y,
    epochs=3,
    validation_split=0.1,
    callbacks=model_save_callback,
)
pd.DataFrame(model1.history.history).to_csv(
    "models/7_layers_lstm_2_FFN/{}/loss.csv".format(n_input)
)
