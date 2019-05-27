# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:54:40 2019

@author: gaurav.pandey1
"""

#importing important libraries
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pds;

#importing dataset
url = "https://raw.githubusercontent.com/psahni/Training/master/sprintdata.csv"
completedataset = pds.read_csv(url);

#creating dependent and independent variables
ind_var_hours= completedataset.iloc[:, 2:3].values;
dep_var_hours= completedataset.iloc[:,3:4].values;

ind_var_points= completedataset.iloc[:, 1:2].values;
dep_var_points= completedataset.iloc[:,4:5].values;
#dep_var_points= jiradataset.iloc[:,4:5].values;

#mis_data_var = jiradataset.iloc[:,1:2].values;

#cat_data_var = jiradataset.iloc[:,0].values;


#taking care of missing data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'NaN',strategy = 'mean', axis=0)
#imputer = imputer.fit(dep_var)
#dep_var= imputer.transform(dep_var);

#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'NaN',strategy = 'mean', axis=0)
#imputer = imputer.fit(ind_var)
#ind_var = imputer.transform(ind_var);

### Encoding categorical data
### Encoding the Independent Variable
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#cat_data_var = labelencoder_X.fit_transform(cat_data_var)
#onehotencoder = OneHotEncoder(categorical_features = [0])
#cat_data_var = onehotencoder.fit_transform(cat_data_var).toarray();
## Encoding the Dependent Variable
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)

##Splitting the dataset into training set and testing set for hours
#from sklearn.cross_validation import train_test_split
#ind_var_hours_train, ind_var_hours_test,dep_var_hours_train, dep_var_hours_test = train_test_split(ind_var,dep_var_hours,test_size=0.2,random_state=0);
#
#
##Splitting the dataset into training set and testing set for hours
#from sklearn.cross_validation import train_test_split
#ind_var_points_train, ind_var_points_test,dep_var_points_train, dep_var_points_test = train_test_split(ind_var,dep_var_points,test_size=0.2,random_state=0);

#Fitting Sample Linear Regression Modle to the Training set for hours
from sklearn.linear_model import LinearRegression
regressor = LinearRegression();
regressor.fit(ind_var_hours,dep_var_hours);
#Predicting the test set result for hours
hour_predict=regressor.predict(ind_var_hours);
print(hour_predict);

#Predicting the test set result for points
from sklearn.linear_model import LinearRegression
regressorp = LinearRegression();
regressorp.fit(ind_var_points,dep_var_points);
point_predict=regressorp.predict(ind_var_points);
##Fitting Sample Linear Regression Modle to the Training set for hours
#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression();
#regressor.fit(ind_var,dep_var_points);
##Predicting the test set result for points
#dep_var_points_predict=regressor.predict(ind_var);

#Visiualizing the training set
#plt.scatter(ind_var_hours_train,dep_var_hours_train,color='red')
#plt.plot(ind_var_hours_train,regressor.predict(ind_var_hours_train),color='blue')
#plt.title('Avaliable Bandwidth Vs Predicted Hours')
#plt.xlabel('Available Bandwidth')
#plt.ylabel('Hours')
#plt.show()

from sklearn.externals import joblib
joblib.dump(regressor, 'model.pkl')
regressor = joblib.load('model.pkl')

joblib.dump(regressorp, 'modelp.pkl')
regressorp = joblib.load('modelp.pkl')

# Saving the data columns from training
model_hours_columns = list(hour_predict)
model_points_columns = list(point_predict)

joblib.dump(model_hours_columns, 'model_hours_columns.pkl')
print("Models hours columns dumped!")
joblib.dump(model_points_columns, 'model_points_columns.pkl')
print("Models points columns dumped!")
