#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 19:25:31 2023

@author: goharshoukat

Function to segment data if the gaps exceed a user defined value
return a dict with a list of segmented data bins
"""
import pandas as pd
import numpy as np

from phddataimputation.timeSeriesPlotter import timeSeriesPlotter

from phddataimputation.dataExtracter import dataExtractor, stripTimeSeries

df = dataExtractor(
    pd.read_csv("data/raw/m2.csv", skiprows=[1]),
    "data/semiprocessed/M2/M2WindSpeed.csv",
    "WindSpeed",
).dropna()

df = pd.concat([
        df,
        (
            df.WindSpeed.isnull().astype(int)
            .groupby(df.WindSpeed.notnull().astype(int).cumsum())
            .cumsum().to_frame('consec_count')
        )
    ],
    axis=1
)



df['Date'] = pd.to_datetime(df['Date'])

# Calculate time differences
df['time_diff'] = df['Date'].diff()

# Define your threshold for gap length (e.g., 2 months)
df.to_csv('review.csv')
threshold = pd.Timedelta('60 days')
threshold = pd.Timedelta('12 hours')

# Identify gaps exceeding the threshold
gaps_exceeding_threshold = df[df['time_diff'] > threshold]
