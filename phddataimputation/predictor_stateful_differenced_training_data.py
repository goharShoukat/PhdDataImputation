import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from utils import generateDirectory

batch_size = 1

n_input = 2
n_features = 1
dropout = 0.0


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


# Load the trained model

saved_model_path = "models/Stateful_dropout_{}V2/{}/model_{}.h5".format(
    dropout, n_input, 574
)  # Change the epoch number as needed
trained_model = load_model(saved_model_path)

# Assuming df_test is your test dataset
df_test = pd.read_csv("data/trainingData/M2_1hour_Gaps_10%_Missing.csv").iloc[:672]

# Prepare input sequences for prediction
X_test, _ = Sequential_Input_LSTM_Stateful(
    df_test["WindSpeed_original"],
    n_input,
    batch_size,
)

# Make predictions
predictions = []
for x in X_test:
    prediction = trained_model.predict(np.array(x))
    predictions.append(prediction[0][0])

# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(predictions, columns=["Predicted_WindSpeed"])

# Optional: Append the predictions to the original test dataset
df_test_with_predictions = pd.concat([df_test, predictions_df], axis=1)

# Print or save the predictions
print(df_test_with_predictions)
outdir = "output/{}_dropout_{}V2".format(n_input, dropout)
generateDirectory(outdir)
df_test_with_predictions.to_csv(outdir + "/stateful.csv", index=False)
