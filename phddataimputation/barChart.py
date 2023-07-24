#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 16:23:39 2023

@author: goharshoukat

Plots the bar chart for gap analysis in different properites. 
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('data/semiprocessed/wind.csv')
cols = df.columns
bins = [0, 1, 6, 12, 24, 48, 72, 365*24, 365*48]

gaps = {}
#extract gaps for each column and stack them in a nest list
for col in cols: 
    na_groups = df[col].notna().cumsum()[df[col].isna()]
    nans = na_groups.groupby(na_groups).agg(len).reset_index(drop=True).rename('gaps')
    
    
    grouped = list(nans.groupby(pd.cut(nans, bins)).count().rename('WindSpeed'))
    gaps[col] = grouped


width = 0.15
barm2 = np.arange(len(bins) - 1)
barm3 = [x + width for x in barm2]
barm4 = [x + width for x in barm3]
barm5 = [x + width for x in barm4]
barm6 = [x + width for x in barm5]


fig, ax = plt.subplots(figsize=(30,30))
ax.bar(barm2, gaps['M3'], width=width, label='M2')
ax.bar(barm3, gaps['M2'], width=width, label='M3')
ax.bar(barm4, gaps['M4'], width=width, label='M4')
ax.bar(barm5, gaps['M5'], width=width, label='M5')
ax.bar(barm6, gaps['M6'], width=width, label='M6')
ax.set_xticks([r + 2*width for r in range(len(bins) - 1)],
              ['1', '2 - 6', '7 - 12', '13 - 24', '25-48', '49 - 72', '73 - 1Y', '> 1Y'] )
ax.set_xlabel('Gaps (in Hour duration)')
ax.set_ylabel('Occurance')
ax.set_title('Wind Speed')
plt.legend()