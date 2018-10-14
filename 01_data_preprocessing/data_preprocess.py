# data preprocessing


#dataset
#    Country   Age   Salary Purchased
# 0   France  44.0  72000.0        No
# 1    Spain  27.0  48000.0       Yes
# 2  Germany  30.0  54000.0        No
# 3    Spain  38.0  61000.0        No
# 4  Germany  40.0      NaN       Yes
# 5   France  35.0  58000.0       Yes
# 6    Spain   NaN  52000.0        No
# 7   France  48.0  79000.0       Yes
# 8  Germany  50.0  83000.0        No
# 9   France  37.0  67000.0       Yes

# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# import dataset
dataset = pd.read_csv("data/Data.csv")
X = dataset.iloc[:, :-1].values #first 3 columns
y = dataset.iloc[:, 3].values #last column

# missing data - fill in mean value
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1:3]) #1:3 <-- upper bound excluded, i.e. extract column idx 1 and 2
X[:, 1:3] = imputer.transform(X[:, 1:3])

print(dataset)
print(X[:, 1:3])

# encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) # name --> 0 or 1 or 2
onehotencoder = OneHotEncoder(categorical_features=[0]) # create dummy variable representation, i.e. 0 --> 1 0 0; 1 --> 0 1 0
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y) # Purchased (Yes/No)  --> 0 or 1

print(y)

#
from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

#perform feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)





