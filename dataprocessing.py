#importing important libraries
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;

#importing dataset
dataset = pd.read_csv('Data.csv');

#creating dependent and independent variables
x= dataset.iloc[:, :-1].values;
y= dataset.iloc[:, 3].values

#taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN',strategy = 'mean', axis=0)
imputer = imputer.fit(x[:, 1:3])
x[:,1:3] = imputer.transform(x[:, 1:3]);

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:, 0] = labelencoder_X.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)