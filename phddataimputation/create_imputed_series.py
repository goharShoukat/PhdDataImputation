import pandas as pd
import numpy as np

from utils import reconstruct_artificial_with_imputation
from config import config1, config2, config3, config4


for con in config3():
    df = pd.read_csv("data/trainingData/M2_1hour_Gaps_10%_Missing.csv").iloc[:672]
    outDir = "output/{}/Model1-{}Neurons{}/".format(
        con["features"],
        con["neurons"],
        {True: "Scaled", False: "WithoutScale"}[con["scaling"]],
    )
    imputations = pd.read_csv(outDir + "reconstructed.csv", header=None)

    imputed = reconstruct_artificial_with_imputation(
        df, "WindSpeed_artificial_gaps", imputations
    )
    imputed.to_csv(outDir + "imputed-df.csv")
