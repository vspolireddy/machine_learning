# =============================================================================
# Regression models (both linear and non-linear) are used for 
# predicting a real value, like salary for example. If your
# independent variable is time, then you are forecasting 
# future values, otherwise your model is predicting present but 
# unknown values. Regression technique vary from Linear Regression to 
# SVR and Random Forests Regression.
# =============================================================================



# Simple linear Regression
#y =b + b * x
#    0   1   1
# y = dependent variable
# b1 = coefficent
# x1 = independent variable
# b0 = constant

#best fit of line = Sum (yi - yi^)^2 -> min
#ordinary leasts squares methods

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("data/Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#training
# regression is prediction of continous variable
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)



#predicting the test set results
y_pred = regressor.predict(X_test)


#visualing the traning set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color ='blue')
plt.title('Salary v. Experience (Traning set)')
plt.xlabel('Years of experience')
plt.ylabel('salary')
plt.show()


#visualing the test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color ='blue')
plt.title('Salary v. Experience (test set)')
plt.xlabel('Years of experience')
plt.ylabel('salary')
plt.show()



