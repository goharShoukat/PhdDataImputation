#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 16:23:39 2023

@author: goharshoukat

Plots the bar chart for gap analysis in different properites. 
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def barChartGapQuantification(df, title):
    cols = df.columns
    bins = [0, 1, 2, 3, 24, 48, 365 * 24]

    gaps = {}
    # extract gaps for each column and stack them in a nest list
    for col in cols:
        na_groups = df[col].notna().cumsum()[df[col].isna()]
        nans = (
            na_groups.groupby(na_groups).agg(len).reset_index(drop=True).rename("gaps")
        )

        grouped = list(nans.groupby(pd.cut(nans, bins)).count())
        gaps[col] = grouped

    width = 0.17
    barm2 = np.arange(len(bins) - 1)
    barm3 = [x + width for x in barm2]
    barm4 = [x + width for x in barm3]
    barm5 = [x + width for x in barm4]
    barm6 = [x + width for x in barm5]

    fig, ax = plt.subplots(figsize=(30, 30))

    rects = ax.bar(barm2, gaps["M3"], width=width, label="M2")
    ax.bar_label(rects, padding=1)

    rects = ax.bar(barm3, gaps["M2"], width=width, label="M3")
    ax.bar_label(rects, padding=1)

    rects = ax.bar(barm4, gaps["M4"], width=width, label="M4")
    ax.bar_label(rects, padding=1)

    rects = ax.bar(barm5, gaps["M5"], width=width, label="M5")
    ax.bar_label(rects, padding=1)

    rects = ax.bar(barm6, gaps["M6"], width=width, label="M6")
    ax.bar_label(rects, padding=1)

    ax.set_xticks(
        [r + 2 * width for r in range(len(bins) - 1)],
        [
            "1H",
            "2H",
            "3H",
            r"$4H \,\less \,time \,\leq\, 1D$",
            r"$1D \,\less \,time \,\leq\, 2D$",
            r"$2D \,\less\, time \,\leq\, 1Y$",
        ],
    )

    ax.set_xlabel("Gaps")
    ax.set_ylabel("Occurance")
    ax.set_title(title)
    plt.legend()
    plt.savefig("output/{} gap bars.png".format(title), dpi=200)
