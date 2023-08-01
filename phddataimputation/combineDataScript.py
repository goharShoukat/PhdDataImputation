#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 22:06:38 2023

@author: goharshoukat

Script to combine all the different functions to test it out
"""

import pandas as pd
from phddataimputation.timeSeriesPlotter import timeSeriesPlotter
from phddataimputation.dataExtracter import dataExtractor, stripTimeSeries

dfm2 = dataExtractor(
    pd.read_csv("data/raw/m2.csv", skiprows=[1]),
    "data/semiprocessed/M2/M2WindSpeed.csv",
    "WindSpeed",
)
timeSeriesPlotter(
    dfm2,
    "WindSpeed",
    unit="m/s",
    plot_direc="/Users/goharshoukat/Documents/GitHub/PhdDataImputation/output/m2/",
    Coordinates="",
    title="M2 Wind Speed",
)


dfm3 = dataExtractor(
    pd.read_csv("data/raw/m3.csv", skiprows=[1]),
    "data/semiprocessed/M3/M3WindSpeed.csv",
    "WindSpeed",
)
timeSeriesPlotter(
    dfm3,
    "WindSpeed",
    unit="m/s",
    plot_direc="/Users/goharshoukat/Documents/GitHub/PhdDataImputation/output/m3/",
    Coordinates="",
    title="M3 Wind Speed",
)

dfm4 = dataExtractor(
    pd.read_csv("data/raw/m4.csv", skiprows=[1]),
    "data/semiprocessed/M4/M4WindSpeed.csv",
    "WindSpeed",
)
timeSeriesPlotter(
    dfm4,
    "WindSpeed",
    unit="m/s",
    plot_direc="/Users/goharshoukat/Documents/GitHub/PhdDataImputation/output/m4/",
    Coordinates="",
    title="M4 Wind Speed",
)

dfm5 = dataExtractor(
    pd.read_csv("data/raw/m5.csv", skiprows=[1]),
    "data/semiprocessed/M5/M4WindSpeed.csv",
    "WindSpeed",
)
timeSeriesPlotter(
    dfm5,
    "WindSpeed",
    unit="m/s",
    plot_direc="/Users/goharshoukat/Documents/GitHub/PhdDataImputation/output/m5/",
    Coordinates="",
    title="M5 Wind Speed",
)


dfm6 = dataExtractor(
    pd.read_csv("data/raw/m6.csv", skiprows=[1]),
    "data/semiprocessed/M6/M6WindSpeed.csv",
    "WindSpeed",
)
timeSeriesPlotter(
    dfm6,
    "WindSpeed",
    unit="m/s",
    plot_direc="/Users/goharshoukat/Documents/GitHub/PhdDataImputation/output/m6/",
    Coordinates="",
    title="M6 Wind Speed",
)


# =============================================================================
# Concat each bouys time series into a single df
# =============================================================================

rows = [
    dfm2["WindSpeed"].to_numpy(),
    dfm3["WindSpeed"].to_numpy(),
    dfm4["WindSpeed"].to_numpy(),
    dfm5["WindSpeed"].to_numpy(),
    dfm6["WindSpeed"].to_numpy(),
]
df_wind = pd.DataFrame(data=rows).transpose()
df_wind = df_wind.rename(columns={0: "M2", 1: "M3", 2: "M4", 3: "M5", 4: "M6"})
df_wind.to_csv("data/semiprocessed/wind.csv", index=False)

# =============================================================================
# Strip time series to remove faulty data before certain year
# write the data into new csv
# =============================================================================
df = dataExtractor(
    pd.read_csv("data/raw/m2.csv", skiprows=[1]),
    "data/semiprocessed/M2/M2WindSpeedFullTimeSeries.csv",
    "WindSpeed",
)

df = stripTimeSeries(df)
df.to_csv("data/semiprocessed/M2/M2WindSpeedAfter2012.csv")
