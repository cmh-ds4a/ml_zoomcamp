#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np

with open('model_rf.bin', 'rb') as f_in:
    dv, rf = pickle.load(f_in)
    
def predict(wine): 

    X = dv.transform([wine])
    y_pred = rf.predict(X)[0]

    return y_pred

def lambda_handler(event, context):

    wine = event["wine"]
    result = predict(wine)
    if result >= 7.0:
        return "This is a GOOD wine"
    else:
        return "This is NOT a good wine"