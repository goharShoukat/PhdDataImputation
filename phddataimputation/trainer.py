from Model import ConvAndLSTMNet
import os
import pandas as pd
import numpy as np

x1 = pd.read_csv('data/trainingData/x1.csv')
x2 = pd.read_csv('data/trainingData/x2.csv')
y = pd.ready_csv('data/trainingData/y.csv')