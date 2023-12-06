#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 19:18:41 2023

@author: goharshoukat

Decompose entire series into trend, and seasonal components and test ACF
"""

import numpy as np
import pandas as pd
import statsmodels
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL, MSTL
from scipy.fft import fft, ifft
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

df = (
    pd.read_csv("data/trainingData/M2_1hour_Gaps_10%_Missing.csv")
    .dropna()
    .iloc[:672, 2]
    .reset_index(drop=True)
)
df.plot()
# df = pd.read_csv('data/trainingData/M2_1hour_Gaps_10%_Missing.csv' ).dropna().iloc[:,1]
# df = pd.Series(df.values, index=pd.date_range("31-1-2012 11:00:00", periods=len(df), freq="H") )
result = adfuller(df, autolag="AIC")
print("ADF Statistic: %f" % result[0])

print("p-value: %f" % result[1])

print("Critical Values:")

for key, value in result[4].items():
    print("\t%s: %.3f" % (key, value))
if result[0] < result[4]["5%"]:
    print("Reject Ho - Time Series is Stationary")
else:
    print("Failed to Reject Ho - Time Series is Non-Stationary")
# df.index = pd.to_datetime(df.index)

plot_acf(df, lags=100)
plot_pacf(df, lags=100)


foo = 1.0 / len(df) * np.abs(np.fft.fft(df)) ** 2
plt.plot(foo[1 : int(len(foo) / 2)])  # peak periods = 6, 8, 11, 16


model = ARIMA(df, order=(2, 0, 0))
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())

residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()

ypred = []
for i in df.index:
    ypred.append(model_fit.predict(i))

bar = model_fit.predict()

plt.plot(bar, label="predicted")
plt.plot(df, label="actual")
plt.legend()


# =============================================================================
# auto arima
# =============================================================================
model = pm.auto_arima(
    df.iloc[:-5],
    m=12,  # frequency of series
    seasonal=False,  # TRUE if seasonal series
    d=None,  # let model determine 'd'
    test="adf",  # use adftest to find optimal 'd'
    start_p=0,
    start_q=0,  # minimum p and q
    max_p=12,
    max_q=12,  # maximum p and q
    D=None,  # let model determine 'D'
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True,
)

# print model summary
print(model.summary())
qux = model.predict(5)


plt.plot(qux, label="predicted")
plt.plot(df, label="actual")
plt.legend()
