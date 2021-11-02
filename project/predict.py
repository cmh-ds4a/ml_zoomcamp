#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np

from flask import Flask
from flask import request
from flask import jsonify

with open('model_xgb.bin', 'rb') as f_in:
    dv, xgbr = pickle.load(f_in)
    
app =  Flask('sold_price')    
    
@app.route('/predict', methods=['POST'])    
def predict():
    home = request.get_json()
    
    X = dv.transform([home])
    y_pred = np.expm1(xgbr.predict(X))[0]
    
    result = {
    	'home_price': float(y_pred)
    }

    return jsonify(result)

#print('Estimated value of home: ${:,.2f}'.format(y_pred[0]))

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)