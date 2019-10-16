# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 09:16:09 2019

@author: Pooja
"""
#Load Libraries
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')

def one_hot_encode(data,column):
    one_hot= pd.get_dummies(data[column])
    data = data.drop(column,axis='columns')
    return pd.concat([data,one_hot],axis='columns')

#Load Data
dataset = pd.read_csv("tcd ml 2019-20 income prediction training (with labels).csv")
data = pd.read_csv("tcd ml 2019-20 income prediction test (without labels).csv")

#merge data
frames = [dataset, data]
result = pd.concat(frames, keys=['train', 'test'])

# Check for Null Data
dataset.isnull().sum()
data.isnull().sum()
result.isnull().sum()

#Analyze Data
#fill null data for numeric data with mean
result["Year of Record"] = result["Year of Record"].fillna(result["Year of Record"].mean())
result["Year of Record"] = np.round_(result["Year of Record"])

result["Age"] = result["Age"].fillna(result["Age"].mean())
result["Age"] = np.round_(result["Age"])

#fill null data for categorical data with forward fill
result["Gender"] = result["Gender"].ffill(axis = 0)
result["Country"] = result["Country"].ffill(axis = 0)
result["Profession"] = result["Profession"].ffill(axis = 0)
result["University Degree"] = result["University Degree"].ffill(axis = 0)
result["Hair Color"] = result["Hair Color"].ffill(axis = 0)

#apply one hot encoder to categorical data
result=one_hot_encode(result,'Gender')
result=one_hot_encode(result,'Country')
result=one_hot_encode(result,'Profession')
result=one_hot_encode(result,'University Degree')
result=one_hot_encode(result,'Hair Color')

#split train and test data and check of null values
train = result.loc['train']
test = result.loc['test']

Y_train = train['Income']
train = train.drop(['Instance','Income'], axis = 1)
X_train = train

test = test.drop(['Instance','Income'], axis = 1)
X_pred = test

train.isnull().sum()
train.dtypes

test.isnull().sum()
test.dtypes

#from sklearn.model_selection import train_test_split
#_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression 
regr = LinearRegression()
regr.fit(X_train, Y_train) 

#from sklearn.ensemble import RandomForestRegressor
#rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
#rf.fit(X_train, Y_train)
#y_pred = rf.predict(X_pred)

y_pred = regr.predict(X_pred)
np.savetxt('out.csv',y_pred)

def positive_if_negative(x):
    if x < 0:
        return ((-1)*x)
    return x
   
y_pred2 = [positive_if_negative(x) for x in y_pred]
np.savetxt('out2.csv',y_pred2)
