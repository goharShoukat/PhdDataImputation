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
import pandas as pd

def dataExtractor(df, outputPath, variable):
    df[variable] = df[variable] * 0.514444 #convert to m/s
    df['Date']      = pd.to_datetime(df['time'], errors='coerce')
    df_variable = pd.DataFrame({'Date': df['Date'].T, variable: df[variable].T})
    df_variable.to_csv(outputPath, index=False)
    return df_variable

df = dataExtractor(pd.read_csv('data/raw/m2.csv', skiprows=[1]), 
              'data/semiprocessed/M2/M2WindSpeed.csv', 'WindSpeed')