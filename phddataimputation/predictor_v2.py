from utils import featureGeneration, PostProcessing, generateDirectory
import pandas as pd
import numpy as np
import tensorflow as tf

df = (
    pd.read_csv("data/trainingData/M2_1hour_Gaps_10%_Missing.csv").iloc[:672]
    # .dropna()
    # .reset_index(drop=True)
)


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


# for m in [2, 3, 4, 5, 6, 7]:
for m in [1]:
    path = "models/2/{}".format(m)
    outdir = "output/2/{}/".format(m)
    generateDirectory(outdir)
    X, _ = Sequential_Input_LSTM(
        df["WindSpeed_original"],
        input_sequence=m,
    )
    model = tf.keras.models.load_model(path)
    if m == 1:
        predictions = []
        for x in X:
            print(x.reshape(1, 1))
            predict = model.predict(x).flatten()
            predictions.append(predict)
    else:
        predictions = model.predict(X).flatten()
    np.savetxt(
        outdir + "full-reconstruction.csv",
        predictions,
        fmt="%f",
        delimiter=",",
    )
