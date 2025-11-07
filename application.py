import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask,request,jsonify,render_template
import sys
import types

application=Flask(__name__)
app=application
if not hasattr(np, "_core"):
    np._core = types.ModuleType("numpy._core")
    sys.modules["numpy._core"] = np._core

if not hasattr(np._core, "multiarray"):
    np._core.multiarray = np.core.multiarray
    sys.modules["numpy._core.multiarray"] = np._core.multiarray

# import lasso regressor and standard scaler pickle
lasso_model=pickle.load(open("models/lassocvv.pkl","rb"))
Standard_scaler=pickle.load(open("models/scaler.pkl","rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="POST":
        Temperature=float(request.form.get('temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('WS'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_data_scaled=Standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=lasso_model.predict(new_data_scaled)
        return render_template('home.html',results=result[0])
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
