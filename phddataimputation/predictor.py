from utils import featureGeneration, PostProcessing, generateDirectory
import pandas as pd
import numpy as np
import tensorflow as tf
from config import config1, config2, config3, config4

for con in config4():  # change here
    path = "models/{}/Model1-{}Neurons{}".format(
        con["features"],
        con["neurons"],
        {True: "Scaled", False: "WithoutScale"}[con["scaling"]],
    )
    outDir = "output/{}/Model1-{}Neurons{}/".format(
        con["features"],
        con["neurons"],
        {True: "Scaled", False: "WithoutScale"}[con["scaling"]],
    )
    generateDirectory(outDir)

    model = tf.keras.models.load_model(path)
    x, y = featureGeneration(
        pd.read_csv("data/trainingData/M2_1hour_Gaps_10%_Missing.csv")
        .iloc[:672]
        .reset_index(drop=True)
        .dropna()
        .to_numpy(),
        con["features"],
        True,
    )

    foo = PostProcessing(path)
    model = foo.load()
    foo.normalise(x, y, 1)
    if con["features"] == 1:
        scaled_predictions = []
        i = 0
        for bar in foo.X:
            i += 1
            print(i)
            scaled_predictions.append(foo.predict(bar))

    else:
        scaled_predictions = foo.predict(x)

    if con["scaling"]:
        denormalised = foo.denormalise(np.array(scaled_predictions).reshape(-1, 1))

        np.savetxt(
            outDir + "reconstructed.csv",
            denormalised,
            fmt="%f",
            delimiter=",",
        )

    else:
        np.savetxt(
            outDir + "reconstructed.csv",
            np.array(scaled_predictions).reshape(-1, 1),
            fmt="%f",
            delimiter=",",
        )
