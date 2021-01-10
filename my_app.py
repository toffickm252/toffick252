from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np
#import pandas as pd
#from flask import Flask, jsonify, request
# Starting up flask
app = Flask(__name__)

#Loading the pickel- the saved model
model = pickle.load(open('final_prediction_2.pickle','rb'))
#Specifying the columns in the model
cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

#showing the route of the app
@app.route('/')
#Uploading the html file
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(model, data=data_unseen, round = 0)
    prediction = int(prediction.Label[0])
    return render_template('home.html',pred='Expected Bill will be {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
