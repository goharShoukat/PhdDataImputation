#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:16:30 2024

@author: goharshoukat
"""

import pandas as pd
import numpy as np


def diff_inv(series_diff, first_value):
    series_inverted = (
        np.r_[first_value, series_diff].cumsum().astype("float64")
    )
    return series_inverted


# =============================================================================
# example implementation
# =============================================================================
# df = pd.DataFrame({'A': np.random.randint(0, 10, 10)})
# df_diff = df.diff().dropna().reset_index(drop=True)

# inversed_series = diff_inv(df_diff.A , df.A[0])
# inversed_series
