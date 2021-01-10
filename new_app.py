# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 13:43:15 2020

@author: Person
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('final_prediction_2', 'rb')) # loading the trained model

@app.route('/') # Homepage
def home():
    return render_template('C:\Users\Person\Documents\GitHub\Breast-tumor-predictive-analysis\html_me.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    # retrieving values from form
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]

    prediction = model.predict(final_features) # making prediction


    return render_template('C:\Users\Person\Documents\GitHub\Breast-tumor-predictive-analysis\html_me.html', prediction_text='Predicted Class: {}'.format(prediction)) # rendering the predicted result

if __name__ == "__main__":
    app.run(debug=True)
#PythonCopy

 


