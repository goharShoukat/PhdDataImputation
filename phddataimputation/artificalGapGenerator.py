#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:35:41 2023

@author: goharshoukat
"""

import numpy as np
import pandas as pd

df = pd.read_csv('data/semiprocessed/wind_full.csv')
m2 = df.M2.dropna()

drop_indices = np.random.choice(m2.index, 1, replace=False)

