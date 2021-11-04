#!/usr/bin/env python
# coding: utf-8

# # Predict 2021 Residential Home Sales Price in Durham, NC

# This code answers the questions of what is the value of a home in Durham, NC?


import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.tree import export_text
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import math

import pickle

import warnings

warnings.filterwarnings("ignore")


# ## Read the Data

print("Loading the data...")
df = pd.read_csv('Durham_homes_sold_2021_YTD.csv')


# ## Clean the Data

print()
print("Preparing the data...")
df.columns = df.columns.str.lower().str.replace(' ', '_')

for c in ['list_price', 'sold_price', 'total_living_area_sqft', 'approx_lot_sqft']:
    df[c] = df[c].str.replace('$', '')
    df[c] = df[c].str.replace(',', '')

for i in ['list_price', 'sold_price', 'total_living_area_sqft']:
    df[i] = df[i].astype(int)
    
for f in ['approx_lot_sqft']:
    df[f] = df[f].astype(float)



df['closing_date'] = pd.to_datetime(df['closing_date'])
df['list_date'] = pd.to_datetime(df['list_date'])
df.sort_values(by=['closing_date'], inplace=True)
df['days_on_market'] = (df['closing_date'] - df['list_date']).dt.days
df['closing_month'] = df['closing_date'].dt.month
df['closing_day'] = df['closing_date'].dt.day
del df['closing_date']
del df['list_date']
df['fireplace'] = df['fireplace'].replace({'4+':'4'})
df['fireplace'].astype(int)
df['fireplace'].value_counts()
df['zip'] = df['zip'].str[:5]
df['zip'].unique()

del df['city']

# ### Handle Missing Values

df.isna().sum()
df['hoa_y/n'].value_counts()
df['hoa_1_fees_required'].isnull().sum()
df.reset_index(drop=True)
df['hoa_1_fees_required'].fillna(df['hoa_y/n'], inplace=True)
df['hoa_1_fees_required'] = np.where(df['subdivision'] == 'Not in a Subdivision', 'No', df['hoa_1_fees_required'])
hoa_yes = list(df[df['hoa_1_fees_required'] == 'Yes']['subdivision'])
hoa_no = list(df[df['hoa_1_fees_required'] == 'No']['subdivision'])
df['hoa_1_fees_required'] = np.where((df['hoa_1_fees_required'].isnull() & df['subdivision'].isin(hoa_yes)), 'Yes', df['hoa_1_fees_required'])
df['hoa_1_fees_required'] = np.where((df['hoa_1_fees_required'].isnull() & df['subdivision'].isin(hoa_no)), 'No', df['hoa_1_fees_required'])
df['hoa_1_fees_required'] = np.where(df['hoa_1_fees_required'].isnull(), 'No', df['hoa_1_fees_required'])
df['hoa_1_fees_required'].isnull().sum()
del df['hoa_y/n']
del df['list_price']


# ## Split the Data

df_full_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, shuffle=False)


df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = np.log1p(df_full_train.sold_price.values)
y_train = np.log1p(df_train.sold_price.values)
y_val = np.log1p(df_val.sold_price.values)
y_test = np.log1p(df_test.sold_price.values)

del df_full_train['sold_price']
del df_train['sold_price']
del df_val['sold_price']
del df_test['sold_price']


# ## Train the Models

print()
print("Training the model...")
dv = DictVectorizer(sparse=False)

train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)


# ### XGBoost

xgbr = XGBRegressor(eta=0.3, max_depth=5, min_child_weight=1, objective='reg:squarederror', nthread=8)
xgbr.fit(X_train, y_train)

y_pred = xgbr.predict(X_val)
mse = mean_squared_error(y_val, y_pred)

print()
print("Model trained:")
print("Score: {:.5}".format(xgbr.score(X_val, y_val)))
print("RMSE: {:.5}".format(np.sqrt(mse)))


# ## Validate the Model

print()
print("Validating the final model:")
dv = DictVectorizer(sparse=False)

full_train_dict = df_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(full_train_dict)

test_dict = df_test.to_dict(orient='records')
X_test = dv.transform(test_dict)


xgbr = XGBRegressor(eta=0.3, max_depth=5, min_child_weight=1, objective='reg:squarederror', nthread=8)
xgbr.fit(X_full_train, y_full_train)

y_pred = xgbr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print()
print("Final model validated:")
print("Score: {:.5}".format(xgbr.score(X_test, y_test)))
print("RMSE: {:.5}".format(np.sqrt(mse)))


# ## Save the Model

output_file = 'model_xgb.bin'

print()
print("Saving model as " + output_file)
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, xgbr), f_out)
