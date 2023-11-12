#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 21:27:36 2023

@author: goharshoukat
"""
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(
    "data/semiprocessed/M2/M2WindSpeedAfter2012.csv", usecols=["WindSpeed"]
)
df["lag_1"] = df["WindSpeed"].shift(1)
df["lag_2"] = df["WindSpeed"].shift(2)
df["lag_3"] = df["WindSpeed"].shift(3)
df["lag_4"] = df["WindSpeed"].shift(4)
df["lag_5"] = df["WindSpeed"].shift(5)
df["lag_6"] = df["WindSpeed"].shift(6)
df["lag_7"] = df["WindSpeed"].shift(7)
df["lag_8"] = df["WindSpeed"].shift(8)
df["lag_9"] = df["WindSpeed"].shift(9)
df["lag_10"] = df["WindSpeed"].shift(10)
df["lag_11"] = df["WindSpeed"].shift(11)
df["lag_12"] = df["WindSpeed"].shift(12)
df["lag_13"] = df["WindSpeed"].shift(13)
df["lag_14"] = df["WindSpeed"].shift(14)
df["lag_15"] = df["WindSpeed"].shift(15)
df["lag_16"] = df["WindSpeed"].shift(16)
df["lag_17"] = df["WindSpeed"].shift(17)
df["lag_18"] = df["WindSpeed"].shift(18)
df["lag_19"] = df["WindSpeed"].shift(19)
df["lag_20"] = df["WindSpeed"].shift(20)
df = df.dropna()
res = []
for col in df.columns[1:]:
    res.append(
        mutual_info_regression(
            np.reshape(df[col].tolist(), (-1, 1)),
            np.reshape(df.WindSpeed.tolist(), (-1, 1)),
        )[0]
    )

plt.plot(res)

plt.plot(df.corr())
corr = df.corr()

plt.plot(corr["lag_1"])
