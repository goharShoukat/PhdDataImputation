from LSTMNet import LSTMNet
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils import generateDirectory, featureGeneration

features = 1
x1, y = featureGeneration(
    pd.read_csv("data/trainingData/M2_1hour_Gaps_10%_Missing.csv")
    .dropna()
    .to_numpy(),
     features,
     False
)

scalarX = MinMaxScaler(feature_range=(0, 1))
X1 = scalarX.fit_transform(x1).T.reshape(-1, 1, 1)
Y = scalarX.fit_transform(y.reshape(-1,1))

model = LSTMNet()

optimizer = "adam"
loss = "mean_squared_error"
metrics = ["mean_absolute_error"]

input_shape = (features, 1)
model.build(input_shape)
pathToSaveModel = "models/{}/Model1-64Neurons".format(features)
generateDirectory(pathToSaveModel)
model.summary(path=pathToSaveModel)

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics,
)

model.train(X1, Y, epochs=20, batch_size=8)

model.save_model(pathToSaveModel, format="tf")
