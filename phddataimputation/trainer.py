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
