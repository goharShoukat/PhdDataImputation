from LSTMNet import LSTMNet
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils import generateDirectory, featureGeneration

features = 4
neurons = 128
scaling: bool = False


x1, y = featureGeneration(
    pd.read_csv("data/trainingData/M2_1hour_Gaps_10%_Missing.csv")
    .dropna()
    .to_numpy(),
    features,
    False,
)

if scaling:
    scalar = MinMaxScaler(feature_range=(0, 1))
    X1 = scalar.fit_transform(x1).T.reshape(-1, features, 1)
    Y = scalar.fit_transform(y.reshape(-1, 1))
else:
    X1 = x1.T.reshape(-1, features, 1)
    Y = y.reshape(-1, 1)

model = LSTMNet(neurons)

optimizer = "adam"
loss = "mean_squared_error"
metrics = ["mean_absolute_error"]

input_shape = (features, 1)
model.build(input_shape)

pathToSaveModel = "models/{}/Model1-{}Neurons{}".format(
    features, neurons, {True: "Scaled", False: "WithoutScale"}[scaling]
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
