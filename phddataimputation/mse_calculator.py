import pandas as pd
from utils import mse
from config import config1, config2, config3, config4

for con in config1():  # change here
    inDir = "output/{}/Model1-{}Neurons{}/".format(
        con["features"],
        con["neurons"],
        {True: "Scaled", False: "WithoutScale"}[con["scaling"]],
    )
    df = pd.read_csv(inDir + "imputed-df.csv").dropna()
    print(mse(df, "WindSpeed_original", "WindSpeed_artificial_gaps"))
