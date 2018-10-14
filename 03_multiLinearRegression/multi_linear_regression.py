

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# import dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values #first 4 columns
y = dataset.iloc[:, 4].values #last column

print(dataset)


# encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3]) # state column --> 0 or 1
onehotencoder = OneHotEncoder(categorical_features=[3]) # create dummy variable representation, i.e. 0 --> 1 0 0; 1 --> 0 1 0
X = onehotencoder.fit_transform(X).toarray()



# avoid the dummy variable Trap
X = X[:, 1:]
print(X)


#splitting the dataset in to the traning set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

#perform feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# fitting multi linear regression to traning set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting the Test set results
y_pred = regressor.predict(X_test)


#optimization with backward elimination model
import statsmodels.formula.api as sm

#adding columns of 1s to account for contant b0 in the formula
#y = b0 + b1*X1 + b2*X2 + ... bn*Xn

# orginal
# X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
# X_opt = X[:, [0, 1, 2, 3, 4 ,5]]
# #SL is .05, any model above would be eliminated
# # ordinary least squares
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# print(regressor_OLS.summary())

# removal of variable 2
# X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
# X_opt = X[:, [0, 1, 3, 4 ,5]]
# #SL is .05, any model above would be eliminated
# # ordinary least squares
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# print(regressor_OLS.summary())

# removal of variable 1
# X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
# X_opt = X[:, [0, 3, 4 ,5]]
# #SL is .05, any model above would be eliminated
# # ordinary least squares
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# print(regressor_OLS.summary())


# removal of variable last 2 since they are both above the .05%
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 3 ]]
#SL is .05, any model above would be eliminated
# ordinary least squares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())



# 1. the lower the p value, the more signficant independent variable is with respect to
# your depedent variable

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.951
# Model:                            OLS   Adj. R-squared:                  0.945
# Method:                 Least Squares   F-statistic:                     169.9
# Date:                Sun, 14 Oct 2018   Prob (F-statistic):           1.34e-27
# Time:                        00:35:13   Log-Likelihood:                -525.38
# No. Observations:                  50   AIC:                             1063.
# Df Residuals:                      44   BIC:                             1074.
# Df Model:                           5
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const       5.013e+04   6884.820      7.281      0.000    3.62e+04     6.4e+04
# x1           198.7888   3371.007      0.059      0.953   -6595.030    6992.607
# x2           -41.8870   3256.039     -0.013      0.990   -6604.003    6520.229
# x3             0.8060      0.046     17.369      0.000       0.712       0.900
# x4            -0.0270      0.052     -0.517      0.608      -0.132       0.078
# x5             0.0270      0.017      1.574      0.123      -0.008       0.062
# ==============================================================================
# Omnibus:                       14.782   Durbin-Watson:                   1.283
# Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.266
# Skew:                          -0.948   Prob(JB):                     2.41e-05
# Kurtosis:                       5.572   Cond. No.                     1.45e+06
# ==============================================================================
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# [2] The condition number is large, 1.45e+06. This might indicate that there are
# strong multicollinearity or other numerical problems.