# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 17:44:03 2022

@author: shafi
"""

import pandas as pd
import numpy as np
from sklearn import linear_model 
import matplotlib.pyplot as plt


df=pd.read_csv('E:/CV (ALL FOR JOB)/ALL PROJECTS/5. IMAGE DETECTION/canada.csv')
df
df.head(3)

%matplotlib inline 
plt.xlabel('year',fontsize=20)
plt.ylabel('per capita income (US$)',fontsize=20)

plt.scatter(df.year, df['per capita income (US$)'], color='blue', marker='+')
new_df = df.drop('per capita income (US$)',axis='columns')
new_df
per=df['per capita income (US$)']
per
reg = linear_model.LinearRegression()
reg.fit(new_df,df['per capita income (US$)'])
reg.predict([[2020]])
