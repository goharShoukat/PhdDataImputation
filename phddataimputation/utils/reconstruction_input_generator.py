import numpy as np
import pandas as pd


def reconstruction_input_generator(df, column, features):
    indices = df[df[column].isna()].index

    x = []
    for index in indices:
        x.append(
            df[column]
            .iloc[index - features : index]
            .reset_index(drop=True)
            .tolist()
        )

    x = np.array(x)

    return x


def reconstruct_artificial_with_imputation(df, column, imputations):
    indices = df[df[column].isna()].index
    for index, impute in zip(indices, imputations[0]):
        df.loc[index, column] = impute

    return df
