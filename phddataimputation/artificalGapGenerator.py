#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:35:41 2023

@author: goharshoukat

Creates artificial gaps in the data
"""

import math
import random
import pandas as pd
from dataExtracter import dataExtractor, stripTimeSeries


def remove_n_consecutive_rows(
    frame: pd.DataFrame, n: int, percent: float
) -> pd.DataFrame:
    """


    Parameters
    ----------
    frame : pd.DataFrame
        dataframe to create gaps in.
        Ideally should not contain any nans.
    n : int
        Number of successive rows to drop
    percent : float
        percentage of actual data to be deleted.
    Returns
    -------
    pd.DataFrame
        With artificially created datagraps.

    """
    chunks_to_remove = int(math.ceil(percent / 100 * frame.shape[0] / n))
    # split the indices into chunks of length n+2
    chunks = [list(range(i, i + n + 2)) for i in range(0, frame.shape[0] - n)]
    drop_indices = list()
    for i in range(chunks_to_remove):
        indices = random.choice(chunks)
        drop_indices += indices[1:-1]
        # remove all chunks which contain overlapping values with indices
        chunks = [c for c in chunks if not any(n in indices for n in c)]
    return frame.drop(drop_indices)


def concatenateDeletedWithOriginalDFWithDroppedNA(
    df: pd.DataFrame, n: int, percent: float
):
    """

    frame : pd.DataFrame
        dataframe to create gaps in.
        Ideally should not contain any nans.
    n : int
        Number of successive rows to drop
    percent : float
        percentage of actual data to be deleted.
    Returns
    -------
    pd.DataFrame
        With artificially created datagaps and the original series too for comparison
    """
    df_miss = remove_n_consecutive_rows(df, n, percent)
    return pd.merge(
        df_miss,
        df,
        on="Date",
        how="right",
        suffixes=["_artificial_gaps", "_original"],
    )


df = dataExtractor(
    pd.read_csv("data/raw/m2.csv", skiprows=[1]),
    "data/semiprocessed/M2/M2WindSpeedFullTimeSeries.csv",
    "WindSpeed",
)

df2 = df.dropna().reset_index(drop=True)
df3 = stripTimeSeries(df2).iloc[:20]
df4 = concatenateDeletedWithOriginalDFWithDroppedNA(df3, 1, 30)
# df4.to_csv('data/trainingData/M2_1hour_Gaps_30%_Missing.csv', index=False)
