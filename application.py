from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

# Import ridge reg and standard scaler pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
Standard_Scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route("/")
def index():
  return render_template('index.html')

@app.route("/predictData", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Extract form data
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        #DC = float(request.form.get('DC'))
        ISI = float(request.form.get('ISI'))
        #BUI = float(request.form.get('BUI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # Create data variable (list or DataFrame depending on model input)
        data = [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
        new_data_scaled=Standard_Scaler.transform(data)

        result=ridge_model.predict(new_data_scaled)

        # Example: Predict using your trained model
        # prediction = model.predict(data)
        # output = round(prediction[0], 2)

        # For now, return simple response
        return render_template("home.html", results=result[0])
    
    else:
        return render_template("home.html")



if __name__=="__main__":
  app.run(host="0.0.0.0",debug=True)