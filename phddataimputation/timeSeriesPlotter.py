#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 10:03:56 2023

@author: goharshoukat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from matplotlib.ticker import MaxNLocator, MultipleLocator


def timeSeriesPlotter(df, variable, Coordinates, unit, plot_direc, title):
    # inputs
    # df : pd.DataFrame : the entire dataframe as it is passed through from the run_code script
    # x_variable: str : x variable name for the heatmap. like mwp
    # y_variable: ndarray : input array like swh
    # the titles are used in plotting

    # Coordinates : str : The coordinates for which this data is extracted
    # date_range : str : The date interval for which this data corresponds to
    # units : pd.DataFrame : df of units with columns as variable names
    # plot_direc : str : output directory entered by user
    # output:

    df["new_date"] = df["Date"].dt.strftime("%Y-%m-%d")
    availability = 100 - df[variable].isnull().sum() / len(df) * 100
    left = min(df["Date"])
    right = max((df["Date"]))

    fig, ax = plt.subplots(figsize=(60, 60))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%Y"))
    loc = MultipleLocator(base=2.0)
    ax.yaxis.set_major_locator(loc)
    plt.setp(ax.get_xticklabels())
    plt.gca().xaxis.set_tick_params(rotation=90)

    ax.set(
        xlabel="Year",
        ylabel=variable + " (" + unit + ")",
        # =============================================================================
        #            title = '{}\n{} Time Series: ({} - {})'.format(Coordinates, variable, df['new_date'][0], df.iloc[-1]['new_date']) , xlim=[left , right])
        # =============================================================================
        title="{} Time Series: ({} - {})".format(
            title, df["new_date"][0], df.iloc[-1]["new_date"]
        ),
        xlim=[left, right],
    )
    ax.tick_params(
        direction="in",
        length=6,
        width=1.2,
        grid_alpha=0.5,
        bottom=True,
        top=True,
        left=True,
        right=True,
    )
    ax.scatter(df["Date"], df[variable], s=0.01)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    ax.grid(which="both", linestyle="--")
    if variable == "Bearing":
        ax.yaxis.set_ticks(np.arange(0, 360, 30))

    # plt.text(0.5, 0.5, 'matplotlib', ha='right', va='top', transform=ax.transAxes)
    plt.text(
        0.87,
        0.87,
        "Average \n{:.2f}".format(np.mean(df[variable])),
        fontsize=9,
        transform=plt.gcf().transFigure,
    )
    plt.text(
        0.93,
        0.87,
        "Availibility \n{:.2f}".format(availability),
        fontsize=9,
        transform=plt.gcf().transFigure,
    )

    plt.text(
        0.87,
        0.80,
        "Minimum \n{:.2f}".format(np.min(df[variable])),
        fontsize=9,
        transform=plt.gcf().transFigure,
    )
    plt.text(
        0.93,
        0.80,
        "Maximum \n{:.2f}".format(np.max(df[variable])),
        fontsize=9,
        transform=plt.gcf().transFigure,
    )
    plt.text(
        0.87,
        0.73,
        "Std \n{:.2f}".format(np.std(df[variable])),
        fontsize=9,
        transform=plt.gcf().transFigure,
    )
    plt.savefig(plot_direc + "{} Time Series.pdf".format(title))
