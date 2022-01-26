#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np

from flask import Flask

with open('model_rf.bin', 'rb') as f_in:
    dv, rf = pickle.load(f_in)
    
app =  Flask('wine')    

wine = {
    'alcohol': 20.5,
    'sulphates': 0.74,
    'citric acid': 0.66,
    'volatile acidity': 0.04
}
    
@app.route('/predict', methods=['GET'])    
def predict():    
    X = dv.transform([wine])
    y_pred = round(rf.predict(X)[0])
    
    qualStr = "The quality of this sample wine is " + str(y_pred) 
    	
    return qualStr
    
if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)