import pandas as pd
#import the data
outage = pd.read_excel("C:\\Users\\vkunkalagunt\\outage\\ETO_YN.xlsx")
#transform Outage_Flag as 1/0 format
outage['Outage_Flag']=outage['Outage_Flag'].apply(lambda Outage_Flag:1 if Outage_Flag=='Yes' else 0)
#convert our Categorical Variables into Dummy Variables using pandas, and drop the original categorical variables
outage=pd.concat([outage,pd.get_dummies(outage['VectorGroup'],drop_first=True,prefix="VectorGroup")],axis=1)
outage=pd.concat([outage,pd.get_dummies(outage['Insulation'],drop_first=True,prefix="Insulation")],axis=1)
outage=pd.concat([outage,pd.get_dummies(outage['EnergyLosses'],drop_first=True,prefix="EnergyLosses")],axis=1)
outage=pd.concat([outage,pd.get_dummies(outage['PressureRelay'],drop_first=True,prefix="PressureRelay")],axis=1)
outage=pd.concat([outage,pd.get_dummies(outage['CoolingOperation'],drop_first=True,prefix="CoolingOperation")],axis=1)
outage=pd.concat([outage,pd.get_dummies(outage['Bushing'],drop_first=True,prefix="Bushing")],axis=1)
outage=pd.concat([outage,pd.get_dummies(outage['OverCurrentProtection'],drop_first=True,prefix="OverCurrentProtection")],axis=1)
outage=pd.concat([outage,pd.get_dummies(outage['FireFightingSystems'],drop_first=True,prefix="FireFightingSystems")],axis=1)
outage=pd.concat([outage,pd.get_dummies(outage['Breakdownvoltage'],drop_first=True,prefix="Breakdownvoltage")],axis=1)
outage=pd.concat([outage,pd.get_dummies(outage['Watercontent'],drop_first=True,prefix="Watercontent")],axis=1)
outage=pd.concat([outage,pd.get_dummies(outage['OilAcidity'],drop_first=True,prefix="OilAcidity")],axis=1)
outage.drop(['VectorGroup','Insulation','EnergyLosses','PressureRelay','CoolingOperation','Bushing','OverCurrentProtection','FireFightingSystems','Breakdownvoltage','Watercontent','OilAcidity'],axis=1,inplace=True)
#Split the input and output variables from train and test dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(outage.drop('Outage_Flag',axis=1),outage['Outage_Flag'], test_size=0.2)
#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfm=RandomForestClassifier(criterion='entropy')
rfm.fit(x_train,y_train)
y_pred=rfm.predict(x_test)
rfm_train_accuracy=rfm.score(x_train,y_train)
rfm_train_accuracy
rfm_test_accuracy=rfm.score(x_test,y_test)
rfm_test_accuracy
#saving the pickle file
import os
os.getcwd()
os.chdir("C:\\Users\\vkunkalagunt\\outage")
import pickle
with open("rfmmodel.pkl","wb") as fid:
    pickle.dump(rfm, fid,2)
#Create a dataframe with only the dummy variables
cat=outage.drop('Outage_Flag',axis=1) 
index_dict=dict(zip(cat.columns,range(cat.shape[1])))    
with open ('cat','wb') as fid:
    pickle.dump(index_dict,fid,2)