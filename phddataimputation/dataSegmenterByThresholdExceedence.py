#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 19:25:31 2023

@author: goharshoukat

Function to segment data if the gaps exceed a user defined value
return a dict with a list of segmented data bins
"""
import pandas as pd
from typing import Dict


def dataSegmenterByThresholdExceedence(
    df: pd.DataFrame, threshold: str
) -> Dict[str, pd.DataFrame]:
    """

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with a column for dates and property.
    threshold : str
        the time duration for gaps in the data.

    Returns
    -------
    dict
        original with an additional column containing gap duration and
        segmented containing rows where gap is below a threshold

    """
    df = df.dropna()  # preserve index
    df["Date"] = pd.to_datetime(df["Date"])
    df["time_diff"] = df["Date"].diff()
    threshold = pd.Timedelta(threshold)
    gaps_exceeding_threshold = df[df["time_diff"] < threshold]

    return {"original": df, "segmented": gaps_exceeding_threshold}


# df = pd.concat([
#         df,
#         (
#             df.WindSpeed.isnull().astype(int)
#             .groupby(df.WindSpeed.notnull().astype(int).cumsum())
#             .cumsum().to_frame('consec_count')
#         )
#     ],
#     axis=1
# )
