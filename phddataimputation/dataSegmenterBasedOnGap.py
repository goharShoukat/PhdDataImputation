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
)

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



df['match'] = df['consec_count'].diff()
df.to_csv('review.csv')
