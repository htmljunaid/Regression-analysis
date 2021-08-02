# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 00:27:27 2021

@author: Junaid M04000018
"""

#%%1. IMPORT PACKAGES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


#%% 2. GET THE DATA
dLentil = pd.read_csv("lentil-data.csv")
dLentil.info()

#%% 3.1 DEFINE Y (RESPONSE) VARIABLE
y = dLentil['Customers']


#%% 3.2  DEFINE X (EXPLANATORY) VARIABLE
dX = dLentil.drop(['Customers'],axis=1)

#%% 3.3 PREPARE CATEGORICAL VARIABLES
dumTheme = pd.get_dummies(dLentil['Theme'], drop_first=True, prefix="Theme")
dX = pd.concat([dX,dumTheme],axis=1)
dX.drop('Theme', axis=1, inplace=True)

#%% 3.4 DEFINE X (EXPLANATORY) VARIABLES
X=dX

#%% 5. FIT A MODEL
#%% 5.1 USING STATSMODELS
X_sm = sm.add_constant(X)
smResults = sm.OLS(y, X_sm).fit()

#%% 5.2 EVALUATE EXPLANATIONS
print(smResults.summary())
write_path = 'output/sm_results.csv'
with open(write_path,'w') as f:
     f.write(smResults.summary().as_csv())

#the summary object has useful attributes for direct access.
print(smResults.pvalues)
print(smResults.aic)
print(smResults.rsquared)
print(smResults.rsquared_adj)

#%% heatmap for correlation
    
heatmap=sns.heatmap(X.corr(),annot=True)
plt.savefig('heatmap of correlation.png', dpi=300)

smResults.bse

std_error = np.sqrt(smResults.scale) 
print(std_error)

 
