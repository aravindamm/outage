import numpy as np
import pickle
import sklearn
import cloudpickle
from flask import Flask, request, render_template
app = Flask(__name__,template_folder='flask-app/templates')

@app.route('/')
def home():
   return render_template('home.html')

@app.route('/OutageFlag',methods=['POST','GET'])
def OutageFlag():
    if request.method=='POST':
        result=request.form
    
    #prepare teh feature vector for prediction
        pkl_file=open('cat','rb')
        index_dict=pickle.load(pkl_file)
        new_vector=np.array(np.zeros(len(index_dict))).reshape(1,-1)
        
        try:
            new_vector[index_dict['VectorGroup_'+str(result['VectorGroup'])]]=1
        except:
            pass
        try:
            new_vector[index_dict['Insulation_'+str(result['Insulation'])]]=1
        except:
            pass
        try:
            new_vector[index_dict['EnergyLosses_'+str(result['EnergyLosses'])]]=1
        except:
            pass
        try:
            new_vector[index_dict['PressureRelay_'+str(result['PressureRelay'])]]=1
        except:
            pass
        try:
            new_vector[index_dict['CoolingOperation_'+str(result['CoolingOperation'])]]=1
        except:
            pass
        try:
            new_vector[index_dict['Bushing_'+str(result['Bushing'])]]=1
        except:
            pass
        try:
            new_vector[index_dict['OverCurrentProtection_'+str(result['OverCurrentProtection'])]]=1
        except:
            pass
        try:
            new_vector[index_dict['FireFightingSystems_'+str(result['FireFightingSystems'])]]=1
        except:
            pass
        try:
            new_vector[index_dict['Breakdownvoltage_'+str(result['Breakdownvoltage'])]]=1
        except:
            pass
        try:
            new_vector[index_dict['Watercontent_'+str(result['Watercontent'])]]=1
        except:
            pass
        try:
            new_vector[index_dict['OilAcidity_'+str(result['OilAcidity'])]]=1
        except:
            pass

        pkl_file=open('rfmmodel.pkl','rb')
        rfmmodel=pickle.load(pkl_file)
        test_prediction=rfmmodel.predict(np.array(new_vector).reshape(1,-1))
        
        return render_template('result.html',prediction=test_prediction)

if __name__ == '__main__':
    app.run()
