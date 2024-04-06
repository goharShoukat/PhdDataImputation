#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 20:22:30 2024

@author: goharshoukat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from matplotlib.ticker import MaxNLocator, MultipleLocator


df = pd.read_csv(
    "m2.csv", usecols=["WindSpeed", "time", "WaveHeight", "Hmax"]
).iloc[1:]
df.isnull().sum() / len(df) * 100
na_groups = df["WindSpeed"].notna().cumsum()[df["WindSpeed"].isna()]
foo = na_groups.groupby(na_groups).agg(len)
bar = foo.value_counts()

index = bar.index.values
vals = bar.values
vals[5] = np.sum(vals[4:])
vals = vals[:5]

plt.bar(["1", "2", "3", "4", ">5"], [668, 64, 8, 5, 7])

# =============================================================================
#
# =============================================================================
df["time"] = pd.to_datetime(df["time"])
# df['time'] = df['time'].dt.strftime("%Y-%m-%d")
y2 = df[["WindSpeed"]].astype(float).fillna(0)
y2 = np.ma.masked_where((0), df["WindSpeed"].astype(float))

fig, ax = plt.subplots(figsize=(60, 60))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%Y"))

ax.plot(df["time"], y2, "o-", markersize=0.5, linewidth=0.3)
ax.set(xlabel="Year", ylabel="Speed (kn)")
plt.savefig("Wind Speed Time Series.png", dpi=100)
# plt.xticks(np.linspace(2001, 2023, 1))
# plt.figure()
# plt.plot(df["time"], y2, "o-")
# plt.show()

variable = "WindSpeed"
unit = "kn"
availability = 100 - df[variable].isnull().sum() / len(df) * 100
left = min(df["time"])
right = max((df["time"]))

fig, ax = plt.subplots(figsize=(60, 60))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%Y"))
loc = MultipleLocator(base=1.0)
ax.yaxis.set_major_locator(loc)
plt.setp(ax.get_xticklabels())
plt.gca().xaxis.set_tick_params(rotation=90)

# ax.set(
#     xlabel="Year",
#     ylabel=variable + " (" + unit + ")",
#     # title="{} Time Series: ({} - {})".format(
#     #     variable, df["new_date"][0], df.iloc[-1]["new_date"]
#     # ),
#     # xlim=[left, right],
# )
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
ax.plot(df["time"], y2, linewidth=0.3, markersize=1)
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
# ax.grid(b=True, which="both", linestyle="--")
ax.yaxis.set_ticks(np.arange(0, 50, 5))
