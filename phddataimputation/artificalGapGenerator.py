#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:35:41 2023

@author: goharshoukat
"""

import math
import random
import pandas as pd


def remove_n_consecutive_rows(frame, n, percent):
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


def concatenateDeletedWithOriginalDFWithDroppedNA(df, n, percent):
    df_miss = remove_n_consecutive_rows(df, n, percent)
    return pd.merge(
        df_miss, df, on="Date", how="right", suffixes=["_artificial_gaps", "_original"]
    )
