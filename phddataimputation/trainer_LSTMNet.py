from LSTMNet import LSTMNet
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

x1 = pd.read_csv("data/trainingData/x__feature.csv", header=None)
y = pd.read_csv("data/trainingData/y.csv", header=None)

scalarX = MinMaxScaler(feature_range=(0, 1))
X1 = scalarX.fit_transform(x1).T.reshape(-1, 1, 1)
Y = scalarX.fit_transform(y)

model = LSTMNet()

optimizer = "adam"
loss = "mean_squared_error"
metrics = ["mean_absolute_error"]

input_shape = (1, 1)
model.build(input_shape)
pathToSaveModel = "models/1/Model1"
model.summary(path=pathToSaveModel)

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics,
)

model.train(X1, y, epochs=100, batch_size=32)

model.save_model(pathToSaveModel, format="tf")
