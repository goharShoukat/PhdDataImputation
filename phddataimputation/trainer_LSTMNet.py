from LSTMNet import LSTMNet
from calbacks import callbacks
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils import generateDirectory, featureGeneration
from config import config1, config2, config3, config4
import logging


logging.basicConfig(filename="config3.log", level=logging.INFO, filemode="w")

for con in config3():
    x1, y = featureGeneration(
        pd.read_csv("data/trainingData/M2_1hour_Gaps_10%_Missing.csv")
        .iloc[:672]
        .reset_index(drop=True)
        .dropna()
        .to_numpy(),
        con["features"],
        False,
    )

    if con["scaling"]:
        scalarX = MinMaxScaler(feature_range=(0, 1))
        X1 = scalarX.fit_transform(x1).T.reshape(-1, con["features"], 1)

        scalarY = MinMaxScaler(feature_range=(0, 1))
        Y = scalarY.fit_transform(y.reshape(-1, 1))
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

    model.train(X1, Y, epochs=200, batch_size=8, checkpoint_save_dir=pathToSaveModel)

    # model.save_model(pathToSaveModel, format="tf")

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
