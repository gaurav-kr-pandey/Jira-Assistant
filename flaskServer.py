# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 23:43:14 2019

@author: gaurav.pandey1
"""

from flask import Flask

app = Flask(__name__)


# Dependencies
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
import array
# Your API definition
app = Flask(__name__)
#@app.route('/sprintdata', methods=['GET'])
#def data() :
#    url = "https://raw.githubusercontent.com/psahni/Training/master/sprintdata.csv"
#    cartData=pd.read_csv(url)
#    lastFiveRow=cartData.tail(5)
#    lastFiveRow=lastFiveRow.iloc[:, 0:3].values;
#    try:
#        lastFiveRow=pd.DataFrame(lastFiveRow)
#        return jsonify({z(lastFiveRow)})
#    except:
#        return jsonify({'trace': traceback.format_exc()})
            
@app.route('/predict/<hours>/<points>', methods=['POST'])
def predict(hours,points):
    if regressor:
        try:
            json_ = request.json
            print(json_)
            #query = pd.get_dummies(pd.DataFrame(json_))
            hours=float(hours)
            points=float(points)
            query = [hours,points]
            #query = query.reindex(columns=model_columns, fill_value=0)

            prediction = regressor.predict(query)

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    regressor = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=8081,debug=True)