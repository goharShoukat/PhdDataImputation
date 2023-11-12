#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 7 Nov 18:54:30 2023

@author: goharshoukat

calculates the autocorelation
"""

from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.feature_selection import mutual_info_regression


x1 = pd.read_csv("data/trainingData/x1.csv")
x2 = pd.read_csv("data/trainingData/x2.csv")
y = pd.read_csv("data/trainingData/y.csv")
df = pd.read_csv("data/semiprocessed/M2/M2WindSpeedAfter2012.csv")
df1 = df.iloc[:19]
df2 = df.iloc[10000:20000]
df3 = df.iloc[20000:30000]
df4 = df.iloc[30000:40000]
df5 = df.iloc[40000:50000]
df6 = df.iloc[50000:60000]
df7 = df.iloc[60000:70000]
df8 = df.iloc[70000:]
plot_acf(df.WindSpeed, lags=20)
plot_acf(df1.WindSpeed, alpha=0.05)
plot_acf(df1.WindSpeed.diff(), lags=50, alpha=0.05)
plot_acf(df2.WindSpeed - df1.WindSpeed.mean(), lags=len(df2) - 1)
plot_acf(df3.WindSpeed - df1.WindSpeed.mean(), lags=len(df3) - 1)
plot_acf(df4.WindSpeed - df1.WindSpeed.mean(), lags=len(df4) - 1)
plot_acf(df5.WindSpeed - df1.WindSpeed.mean(), lags=len(df5) - 1)
plot_acf(df6.WindSpeed - df1.WindSpeed.mean(), lags=len(df6) - 1)
plot_acf(df7.WindSpeed - df1.WindSpeed.mean(), lags=len(df7) - 1)
plot_acf(df8.WindSpeed - df1.WindSpeed.mean(), lags=len(df8) - 1)

detrended = df.WindSpeed - df.WindSpeed.mean()
plot_acf(detrended, lags=len(df) - 1, alpha=0.05)

detrended = df.WindSpeed.diff()
plt.plot(detrended)
plot_acf(detrended, lags=5, alpha=0.05)
