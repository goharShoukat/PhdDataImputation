#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 19:24:06 2023

@author: goharshoukat

This script extracts wind speed from raw data files
- strips the wrong data prior to 2012
- converts knots to m/s
- combines data from all buoys to one df
- writes the data to semi-processed folder
"""

from phddataimputation.timeSeriesPlotter import timeSeriesPlotter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# =============================================================================
# Load each time series individually and plot for wind speed
# =============================================================================
dfm2=pd.read_csv('/Users/goharshoukat/Documents/GitHub/PhdDataImputation/data/raw/m2.csv', skiprows=[1])
dfm2['WindSpeed'] = dfm2['WindSpeed'] * 0.514444
dfm2['Date'] = pd.to_datetime(dfm2['time'], errors='coerce')
timeSeriesPlotter(dfm2, 'WindSpeed', unit='m/s', plot_direc='/Users/goharshoukat/Documents/GitHub/PhdDataImputation/output/m2/', Coordinates='',
                  title='M2 Wind Speed')

dfm3=pd.read_csv('/Users/goharshoukat/Documents/GitHub/PhdDataImputation/data/raw/m3.csv', skiprows=[1])
dfm3['WindSpeed'] = dfm3['WindSpeed'] * 0.514444
dfm3['Date'] = pd.to_datetime(dfm3['time'], errors='coerce')
timeSeriesPlotter(dfm3, 'WindSpeed', unit='m/s', plot_direc='/Users/goharshoukat/Documents/GitHub/PhdDataImputation/output/m3/', Coordinates='',
                  title='M3 Wind Speed')

dfm4=pd.read_csv('/Users/goharshoukat/Documents/GitHub/PhdDataImputation/data/raw/m4.csv', skiprows=[1])
dfm4['WindSpeed'] = dfm4['WindSpeed'] * 0.514444
dfm4['Date'] = pd.to_datetime(dfm4['time'], errors='coerce')
timeSeriesPlotter(dfm4, 'WindSpeed', unit='m/s', plot_direc='/Users/goharshoukat/Documents/GitHub/PhdDataImputation/output/m4/', Coordinates='',
                  title='M4 Wind Speed')

dfm5=pd.read_csv('/Users/goharshoukat/Documents/GitHub/PhdDataImputation/data/raw/m5.csv', skiprows=[1])
dfm5['WindSpeed'] = dfm5['WindSpeed'] * 0.514444
dfm5['Date'] = pd.to_datetime(dfm5['time'], errors='coerce')
timeSeriesPlotter(dfm5, 'WindSpeed', unit='m/s', plot_direc='/Users/goharshoukat/Documents/GitHub/PhdDataImputation/output/m5/', Coordinates='',
                  title='M5 Wind Speed')


dfm6=pd.read_csv('/Users/goharshoukat/Documents/GitHub/PhdDataImputation/data/raw/m6.csv', skiprows=[1])
dfm6['WindSpeed'] = dfm6['WindSpeed'] * 0.514444
dfm6['Date'] = pd.to_datetime(dfm6['time'], errors='coerce')
timeSeriesPlotter(dfm6, 'WindSpeed', unit='m/s', plot_direc='/Users/goharshoukat/Documents/GitHub/PhdDataImputation/output/m6/', Coordinates='',
                  title='M6 Wind Speed')


# =============================================================================
# Concat each bouys time series into a single df
# =============================================================================

rows=[dfm2['WindSpeed'].to_numpy(), dfm3['WindSpeed'].to_numpy(), dfm4['WindSpeed'].to_numpy(),
 dfm5['WindSpeed'].to_numpy(), dfm6['WindSpeed'].to_numpy()]
df_wind = pd.DataFrame(data=rows).transpose()
df_wind=df_wind.rename(columns={0:'M2', 1:'M3', 2:'M4', 3:'M5', 4:'M6'})
df_wind.to_csv('data/semiprocessed/wind.csv', index=False)