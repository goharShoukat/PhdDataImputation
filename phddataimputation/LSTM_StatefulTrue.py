from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd


# Function to create stateful input sequences
def Sequential_Input_LSTM_Stateful(df, input_sequence, batch_size):
    df_np = df.to_numpy()
    X = []
    y = []

    # Calculate the total number of batches
    total_batches = (len(df_np) - input_sequence) // batch_size

    for i in range(total_batches):
        batch_X = []
        batch_y = []

        # Create a batch of sequences
        for j in range(batch_size):
            idx = i * batch_size + j
            row = [a for a in df_np[idx : idx + input_sequence]]
            batch_X.append(row)
            label = df_np[idx + input_sequence]
            batch_y.append(label)

        X.append(batch_X)
        y.append(batch_y)

    return np.array(X), np.array(y)


# Define batch size
batch_size = 32

n_input = 2
model1 = Sequential()
n_features = 1
model1.add(
    LSTM(
        100,
        return_sequences=True,
        stateful=True,
        batch_input_shape=(batch_size, n_input, n_features),
    )
)  # Fix input layer
model1.add(LSTM(100, return_sequences=True, stateful=True))
model1.add(LSTM(100, return_sequences=True, stateful=True))
model1.add(LSTM(100, return_sequences=True, stateful=True))
model1.add(LSTM(50, stateful=True))
model1.add(Dense(8, activation="relu"))
model1.add(Dense(1, activation="linear"))

df = pd.read_csv("data/trainingData/M2_1hour_Gaps_10%_Missing.csv").iloc[:672]

X, y = Sequential_Input_LSTM_Stateful(
    df["WindSpeed_artificial_gaps"].dropna().reset_index(drop=True), n_input, batch_size
)

model_save_callback = ModelCheckpoint(
    filepath="models/Stateful/{}/".format(n_input),
    monitor="loss",
    mode="min",
    save_best_only=True,
    period=10,
)

model1.compile(
    loss=MeanSquaredError(),
    optimizer=Adam(learning_rate=0.0001),
    metrics=RootMeanSquaredError(),
)

# Train the model with stateful data
for i in range(3000):
    for j in range(len(X)):
        history = model1.train_on_batch(X[j], y[j])

    # Reset states at the end of each batch
    model1.reset_states()

    if i % 10 == 0:
        model1.save("models/Stateful/{}/model_{}.h5".format(n_input, i))
        print(f"Epoch {i}, Loss: {history}")

hist = pd.DataFrame(history.history).to_csv(
    "models/Stateful/{}/loss.csv".format(n_input)
)
