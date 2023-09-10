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
    """
    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    output_path : TYPE
        DESCRIPTION.
    variable : TYPE
        DESCRIPTION.

    Returns
    -------
    df_variable : TYPE
        DESCRIPTION.

    """
    df[variable] = df[variable] * 0.514444  # convert to m/s
    df["Date"] = pd.to_datetime(df["time"], errors="coerce")
    df_variable = pd.DataFrame({"Date": df["Date"].T, variable: df[variable].T})
    df_variable.to_csv(output_path, index=False)
    return df_variable


def stripTimeSeries(df, year="2012"):
    """
    Parameters
    ----------
    df : pd.DataFrame
        Time Series read from buoys.
    year : str, optional
        DESCRIPTION. The default is "2012".

    Returns
    -------
    df : pd.DataFrame
        Returns a dataframe after removing segment of the time series prior to the year specified.

    """
    year1 = pd.to_datetime(year)
    # df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] > year].reset_index(drop=True)
    return df
