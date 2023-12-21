from LSTMNet import LSTMNet
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils import generateDirectory, featureGeneration
from config import config1, config2, config3, config4
import logging


logging.basicConfig(filename="config1.log", level=logging.INFO, filemode="w")

for con in config1():
    x1, y = featureGeneration(
        pd.read_csv("data/trainingData/M2_1hour_Gaps_10%_Missing.csv")
        .dropna()
        .to_numpy(),
        con["features"],
        False,
    )

    if con["scaling"]:
        scalar = MinMaxScaler(feature_range=(0, 1))
        X1 = scalar.fit_transform(x1).T.reshape(-1, con["features"], 1)
        Y = scalar.fit_transform(y.reshape(-1, 1))
    else:
        X1 = x1.T.reshape(-1, con["features"], 1)
        Y = y.reshape(-1, 1)

    model = LSTMNet(con["neurons"])

    optimizer = "adam"
    loss = "mean_squared_error"
    metrics = ["mean_absolute_error"]

    input_shape = (con["features"], 1)
    model.build(input_shape)

    pathToSaveModel = "models/{}/Model1-{}Neurons{}".format(
        con["features"],
        con["neurons"],
        {True: "Scaled", False: "WithoutScale"}[con["scaling"]],
    )
    generateDirectory(pathToSaveModel)

    model.summary(path=pathToSaveModel)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    model.train(X1, Y, epochs=20, batch_size=8)

    model.save_model(pathToSaveModel, format="tf")

    logging.info(
        "Features: {}, Neurons: {}, Scaling: {}".format(
            con["features"], con["neurons"], con["scaling"]
        )
    )
    logging.info("Loss: {}".format(model.getLoss()))
    logging.info("Mean Absolute Error: {}".format(model.getMeanAbsoluteError()))
    logging.info("Validation Loss: {}".format(model.getValLoss()))
    logging.info(
        "Validation Mean Absolute Error: {}\n\n\n\n".format(
            model.getValMeanAbsoluteError()
        )
    )
