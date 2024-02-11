from utils import (
    featureGeneration,
    PostProcessing,
    generateDirectory,
    reconstruction_input_generator,
)
import pandas as pd
import numpy as np
import tensorflow as tf
from config import config1, config2, config3, config4

for con in config2():  # change here
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
    df = pd.read_csv("data/trainingData/M2_1hour_Gaps_10%_Missing.csv").iloc[:672]
    x, y = featureGeneration(
        df.reset_index(drop=True).dropna().to_numpy(),
        con["features"],
        True,
    )

    testX = reconstruction_input_generator(
        df, "WindSpeed_artificial_gaps", con["features"]
    )

    foo = PostProcessing(path)
    model = foo.load()

    if con["scaling"]:
        scaled_testX = foo.normalise(x, y, con["features"], testX)
    else:
        scaled_testX = testX

    if con["features"] == 1:
        predictions = []
        i = 0
        for bar in scaled_testX:
            i += 1
            print(i)
            predictions.append(foo.predict(np.array(bar).reshape(1, 1)))

    else:
        predictions = foo.predict(scaled_testX)

    if con["scaling"]:
        denormalised = foo.denormalise(np.array(predictions).reshape(-1, 1))

        np.savetxt(
            outDir + "reconstructed.csv",
            denormalised,
            fmt="%f",
            delimiter=",",
        )

    else:
        np.savetxt(
            outDir + "reconstructed.csv",
            np.array(predictions).reshape(-1, 1),
            fmt="%f",
            delimiter=",",
        )
