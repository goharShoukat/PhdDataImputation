"""
Feature generation depending on number of timesteps to consider
"""
import numpy as np


def featureGeneration(df, features, save: bool):
    """
    df: pandas.DataFrame
    features: int: number of features to be generated with lag

    returns:
        x: np.array: 2D array with each row containing the feature set
        y: np.array: output for each row of x
    """
    y = []

    for i in range(len(df) - features):
        if i == 0:
            x = np.reshape(df[i : i + features, 1].astype(float), [1, -1])
        else:
            x = np.vstack(
                (
                    np.reshape(df[i : i + features, 1].astype(float), [1, -1]),
                    x,
                )
            )
        y.extend([float(df[i + features, 1])])
    x = np.flip(np.round(x, decimals=3), axis=0)
    y = np.round(y, decimals=3)

    if save:
        np.savetxt(
            "data/trainingData/x_{}_feature.csv".format(features),
            x,
            fmt="%f",
            delimiter=",",
        )
        np.savetxt(
            "data/trainingData/y_{}.csv".format(features),
            y,
            fmt="%f",
            delimiter=",",
        )
    return x, y.reshape(-1, 1)
