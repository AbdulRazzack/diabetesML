import pandas as pd
import pickle
import numpy as np
from flask import Flask,request,app,render_template,jsonify,url_for

app=Flask(__name__)
#Load the model
reg_pred = pickle.load(open('diabetes.pkl','rb'))
scaler_pred = pickle.load(open('scaler.pkl','rb'))
 

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/pred_api',methods=['POST'])
def pred_api():
    data =request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler_pred.transform(np.array(list(data.values())).reshape(1,-1))
    output = reg_pred.predict(new_data)
    print(output[0])
    return jsonify(output[0])


if __name__=="__main__":
    app.run(debug=True)