#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 01:06:30 2024

@author: goharshoukat
"""

import pandas as pd
from sklearn.metrics import mean_squared_error

arima = pd.read_csv("output/arima/imputed_series.csv")
df = pd.read_csv("data/trainingData/M2_1hour_Gaps_10%_Missing.csv").iloc[:672]

column = "WindSpeed_artificial_gaps"
indices = df[df[column].isna()].index
foo = arima.loc[indices.tolist(), "imputed_data"]
bar = df.loc[indices.tolist(), "WindSpeed_original"]
mean_squared_error(bar, foo)

lstm = pd.read_csv(
    "output/1/Model1-256NeuronsScaled/reconstructed.csv", header=None
)

mean_squared_error(bar, lstm)
