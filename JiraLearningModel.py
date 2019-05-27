# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 14:14:09 2019

@author: gaurav.pandey1
"""

#importing important libraries
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;
from sklearn import linear_model

# Load the dataset in a dataframe object and include only four features as mentioned
url = "https://raw.githubusercontent.com/psahni/Training/master/sprintdata.csv"
df = pd.read_csv(url)
include = ['Sprint', 'Estimated_Points', 'Available_Bandwidth', 'Bandwidth_Consumed','Achieved_Points'] # Only four features
df_ = df[include]

x = ['Estimated_Points','Available_Bandwidth']     #Features
y = ['Bandwidth_Consumed','Achieved_Points']    #Target variable
ind_x=df[x]
dep_y=df[y]

#taking care of missing data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'NaN',strategy = 'mean', axis=0)
#imputer = imputer.fit(dep_y)
#dep_y = imputer.transform(dep_y);

##Splitting the dataset into training set and testing set for hours
#from sklearn.cross_validation import train_test_split
#ind_x_train, ind_x_test,dep_y_train, dep_y_test = train_test_split(ind_x,dep_y,test_size=0.2,random_state=0);

reg = linear_model.LinearRegression();
reg.fit(ind_x,dep_y)
#predicted_data=reg.predict([25,400])
#print(reg)
#print(predicted_data)
#predicted_data=reg.predict(df)
from sklearn.externals import joblib
joblib.dump(reg, 'model.pkl')
reg = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(ind_x)

joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")

## import the class
#from sklearn.linear_model import LogisticRegression
## instantiate the model (using the default parameters)
#logreg = LogisticRegression()
## fit the model with data
#logreg.fit(ind_x,dep_y)
##predict
#y_pred=logreg.predict(ind_x)