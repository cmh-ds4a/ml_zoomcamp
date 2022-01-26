#!/usr/bin/env python
# coding: utf-8

# # Red Wine Quality Prediction

# ## Project Goal

# The goal of this project is to determine which features of red wine variants of the Portuguese "Vinho Verde" wines contribute to the quality of a good wine and apply a model to predict the quality of a Portuguese "Vinho Verde" wine.
# 
# More information and related data can be found here:  https://www.kaggle.com/ucim/red-wine-quality-cortez-et-al-2009 .

# ## Libraries

import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import math

import pickle

# turn off warnings
import warnings
warnings.filterwarnings('ignore')

# ## Load the Data
print('Preparing Data...')
df = pd.read_csv('winequality-red.csv', delimiter=';', index_col=False)

# lowercase all column names
df.columns = df.columns.str.lower()

corr_features = ['alcohol', 'sulphates', 'citric acid', 'volatile acidity', 'quality']
df_corr = df[corr_features]

# ### Split the Data
df_full_train, df_test = train_test_split(df_corr, test_size=0.2, shuffle=True, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, shuffle=True, random_state=1)

print(f'Train Data: {round(df_train.shape[0]/df.shape[0],2)*100}%; Validation Data: {round(df_val.shape[0]/df.shape[0],2)*100}%; Test Data: {round(df_test.shape[0]/df.shape[0],2)*100}%')

df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train.quality.values
y_train = df_train.quality.values
y_val = df_val.quality.values
y_test = df_test.quality.values

del df_full_train['quality']
del df_train['quality']
del df_val['quality']
del df_test['quality']

print('Training Model...')
dv = DictVectorizer(sparse=False)

train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

print('validation results:')    
print("Random Forest Score: {:.5}".format(rf.score(X_val, y_val)))


print('Training the Final Model...')
dv = DictVectorizer(sparse=False)

full_train_dict = df_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(full_train_dict)

test_dict = df_test.to_dict(orient='records')
X_test = dv.transform(test_dict)

rf = RandomForestRegressor()
rf.fit(X_full_train, y_full_train)

y_pred_val = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred_val)

print("RMSE: {:.5}".format(np.sqrt(mse)))


# ## Save the Model
output_file = 'model_rf.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, rf), f_out)

print(f'Model saved to {output_file}!')    