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
    return df_variable
