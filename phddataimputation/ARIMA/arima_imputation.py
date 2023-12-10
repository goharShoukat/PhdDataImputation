import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

utils = importr("utils")
# package_names = ('imputeTS')
# utils.install_packages('imputeTS')
imputeTS = importr("imputeTS")

kalman_auto_arima = robjects.r["na.kalman"]
df = (
    pd.read_csv("phddataimputation/data/trainingData/M2_1hour_Gaps_10%_Missing.csv")
    .iloc[:672]
    .reset_index(drop=True)
)
print(df.head())
this_value = np.ndarray.tolist(df["WindSpeed_artificial_gaps"].values)
this_value = robjects.FloatVector(this_value)

fit_arima = kalman_auto_arima(this_value, model="auto.arima")
print(fit_arima)

plt.plot(fit_arima, label="imputed")
plt.plot(df["WindSpeed_artificial_gaps"], label="original")
plt.legend()
plt.savefig("output.pdf")

outputDf = pd.DataFrame(
    {
        "Imputed": fit_arima,
        "Artificial Gaps": df.WindSpeed_artificial_gaps,
        "Original": df.WindSpeed_original,
    }
)
outputDf.to_csv("imputed.csv")
