import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('data/trainingData/M2_1hour_Gaps_10%_Missing.csv').dropna().to_numpy()

y = []

for i in range(len(df) - 24 - 24):
    #if (i + 1 + 24 + 24) < len(df)
    if i == 0:
        x1 = np.reshape(df[i:i+24, 1].astype(float), [1, -1])
        x2 = np.reshape(df[i + 1 + 24 : i + 1 + 24 + 24, 1].astype(float), [1, -1])
    else: 
        x1 = np.vstack((np.reshape(df[i:i+24, 1].astype(float), [1, -1]), x1))
        x2 = np.vstack((np.reshape(df[i + 1 + 24 : i + 1 + 24 + 24, 1].astype(float), [1, -1]), x2))
    y.extend([float(df[i+24, 1])])
x1 = np.round(x1, decimals=2)
x2 = np.round(x2, decimals=2)
y = np.round(y, decimals=2)
np.savetxt('data/trainingData/x1.csv', x1, fmt='%f', delimiter=',')
np.savetxt('data/trainingData/x2.csv', x2, fmt='%f', delimiter=',')
np.savetxt('data/trainingData/y.csv', y, fmt='%f', delimiter=',')
