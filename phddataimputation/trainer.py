from Model import ConvAndLSTMNet
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

x1 = pd.read_csv("data/trainingData/x1.csv", header=None)
x2 = pd.read_csv("data/trainingData/x2.csv", header=None)
y = pd.read_csv("data/trainingData/y.csv", header=None)

scalarX = MinMaxScaler(feature_range=(0, 1))
X1 = scalarX.fit_transform(x1).T.reshape(-1, 24, 1)
X2 = scalarX.fit_transform(x2).T.reshape(-1, 24, 1)
Y = scalarX.fit_transform(y)

model = ConvAndLSTMNet()

optimizer = "adam"
loss = "mean_squared_error"
metrics = ["mean_absolute_error"]

input_shape = (24, 1)
model.build(input_shape)
pathToSaveModel = "models/Model1"
model.summary(path=pathToSaveModel)

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics,
)

model.train(x, y, epochs=5, batch_size=32)

model.save_model(pathToSaveModel, format="tf")
