#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np

from flask import Flask
from flask import request
from flask import jsonify

with open('model_rf.bin', 'rb') as f_in:
    dv, rf = pickle.load(f_in)
    
app =  Flask('wine')    

    
@app.route('/predict', methods=['POST'])    
def predict(): 
    wine = request.get_json(	)

    X = dv.transform([wine])
    y_pred = rf.predict(X)[0]
    good_wine = y_pred >= 7.0
    
    result = {
        'quality': float(y_pred),
        'good_wine': bool(good_wine)
    }    

    return jsonify(result)

# print('Quality of wine', y_pred)
    
if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)