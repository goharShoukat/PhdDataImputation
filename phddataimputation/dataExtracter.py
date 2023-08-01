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


def dataExtractor(df, output_path, variable):
    df[variable] = df[variable] * 0.514444  # convert to m/s
    df["Date"] = pd.to_datetime(df["time"], errors="coerce")
    df_variable = pd.DataFrame({"Date": df["Date"].T, variable: df[variable].T})
    df_variable.to_csv(output_path, index=False)
    return df_variable


def stripTimeSeries(df, year="2012"):
    year = "2012"
    year = pd.to_datetime(year, format="%Y")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df = df[df["Date"] > year]
    return df
