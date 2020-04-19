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
import pandas as pd


# import dataset
dataset = pd.read_csv("data/Data.csv")
X = dataset.iloc[:, :-1].values #first 3 columns
y = dataset.iloc[:, 3].values #last column

# missing data - fill in mean value
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy='mean')

# skip column zero because non numerical
imputer.fit(X[:, 1:3 ])
X[:,1:3] = imputer.transform(X[:,1:3])

# =============================================================================
# France	44.0	72000.0
# Spain	27.0	48000.0
# Germany	30.0	54000.0
# Spain	38.0	61000.0
# Germany	40.0	63777.77777777778
# France	35.0	58000.0
# Spain	38.77777777777778	52000.0
# France	48.0	79000.0
# Germany	50.0	83000.0
# France	37.0	67000.0
# =============================================================================


## encoding 3 countries
## one hot enconder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

### France, Spain and Germany
# =============================================================================
# 1.0	0.0	0.0	44.0	72000.0
# 0.0	0.0	1.0	27.0	48000.0
# 0.0	1.0	0.0	30.0	54000.0
# 0.0	0.0	1.0	38.0	61000.0
# 0.0	1.0	0.0	40.0	63777.77777777778
# 1.0	0.0	0.0	35.0	58000.0
# 0.0	0.0	1.0	38.77777777777778	52000.0
# 1.0	0.0	0.0	48.0	79000.0
# 0.0	1.0	0.0	50.0	83000.0
# 1.0	0.0	0.0	37.0	67000.0
# =============================================================================


## encode the dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# =============================================================================
# 0
# 1
# 0
# 0
# 1
# 1
# 0
# 1
# 0
# 1
# 
# =============================================================================

### feature scaling
#salary since its higer numerical will dominate the result on depedent variable
# important to put variables on same scale

#standardisation: each value of the feature minus the mean and divide by standard deviation
# same range + or - 3 
# =============================================================================
# xstand = x - mean(x)
#        ----------------------
#        standard deviation(x)
# 
# # Normalisation: all the values are between 0 and 1
# Xnorm  =  X - min(X)
#         --------------
#         max(x) - min(x)
# =============================================================================


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# =============================================================================
# 
# 1.22474	-0.654654	-0.654654	0.758874	0.749473
# -0.816497	-0.654654	1.52753	-1.7115	-1.43818
# -0.816497	1.52753	-0.654654	-1.27555	-0.891265
# -0.816497	-0.654654	1.52753	-0.113024	-0.2532
# -0.816497	1.52753	-0.654654	0.177609	6.63219e-16
# 1.22474	-0.654654	-0.654654	-0.548973	-0.526657
# -0.816497	-0.654654	1.52753	0	-1.07357
# 1.22474	-0.654654	-0.654654	1.34014	1.38754
# -0.816497	1.52753	-0.654654	1.63077	1.75215
# 1.22474	-0.654654	-0.654654	-0.25834	0.293712
# =============================================================================

### splitting the dataset into Traning set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state=0)
































