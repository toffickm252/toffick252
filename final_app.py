# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 00:34:52 2021

@author: Person
"""

#import libraries
import numpy as np
from flask import Flask, render_template,request
import pickle

#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('final_prediction_2.pickle', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('html_me.html')


#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 4) 
    return render_template('html_me.html', prediction_text='CO2 Emission of the vehicle is :{}'.format(output))

if __name__ =='__main__':
    app.run(debug=True)
