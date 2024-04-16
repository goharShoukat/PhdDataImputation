#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 16:36:36 2024

@author: goharshoukat
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

imputed = pd.read_csv("output/arima/imputed_series.csv")
orig = pd.read_csv("data/trainingData/M2_1hour_Gaps_10%_Missing.csv").iloc[
    :672
]
orig["WindSpeed_original"].mean()
df = pd.concat([imputed, orig], axis=1)
foo = df[["imputed_data", "WindSpeed_original"]][
    df["WindSpeed_artificial_gaps"].isna()
]
mse = mean_squared_error(foo["imputed_data"], foo["WindSpeed_original"])
# =============================================================================
# scatter plot
# =============================================================================

plt.figure()
plt.scatter(
    foo["WindSpeed_original"],
    foo["imputed_data"],
    label="ARIMA (2, 2, 0)",
    color="red",
    alpha=0.5,
)
plt.axline((0, 0), slope=1, color="black")
plt.legend()
plt.xlabel("True Speed (m/s)")
plt.ylabel("Imputed speed (m/s)")
plt.savefig("ARIMA QQ Comparison.png", dpi=600)


# =============================================================================
# pdf
# =============================================================================
counts, bin_edges = np.histogram(
    imputed["imputed_data"], bins=10, density=True
)
pdf = counts / sum(counts)

plt.hist(orig["WindSpeed_original"].to_list(), bins, alpha=0.5, label="x")
plt.hist(imputed["imputed_data"].to_list(), bins, alpha=0.5, label="y")
plt.legend(loc="upper right")
pyplot.show()

plt.hist(
    foo["WindSpeed_original"],
    bins=20,
    label="Original",
    alpha=0.5,
    density=True,
)
plt.hist(
    foo["imputed_data"], bins=20, label="Imputed", alpha=0.5, density=True
)
plt.legend()
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Density")
plt.savefig("PDF - ARIMA.png", dpi=100)
