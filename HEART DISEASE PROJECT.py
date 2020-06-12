#IMORTING SOME LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#NOW READ THE FILE
df=pd.read_csv("Heart Disease.csv")

#HANDELING THE DATASET
#REPLACING THE NAME OF COLUMN 'MALE' TO 'GENDER'
df.rename(columns={'male':'gender'},inplace=True)

#REMOVING IRRELEVANT COLUMN FROM THE DATASET
df.drop(['education','BPMeds','prevalentHyp'],axis='columns',inplace=True)
print(df.columns)
#REMOVING THE ROWS HAVING NA ACCORDING TO THE NEED
df.dropna(axis=0,inplace=True)
df.isnull().sum()

#CORRELATION
print(df.corr())

#SAVING THE FILE EXISTING FILE AS NEW_FILE
df.to_csv("heart_diseases.csv")
print("SAVED SUCCESSFULLY")

#AFTER THIS EDA PART IS DONE IN TABLEAU 
#AFTER THAT APPLYING LOGISTIC REGGRESSION

new_features=df[['age','gender','cigsPerDay','totChol','sysBP','glucose','TenYearCHD']]
X=new_features.iloc[:,:-1]
y=new_features.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)

#FITTING THE MODEL
reg= LogisticRegression()
reg.fit(X_train,y_train)

#PREDICTION
pred=reg.predict(X_test)
print(pred)
random_values  =reg.predict([[1,75,1,0,207,120]])
print("PREDICTED RESULT OF RANDOM VALUES :",random_values)

#PROBABILITY OF PREDICTION
prob=reg.predict_proba(X_test)
print(prob)
#ACCURACY OF PREDICTION
accuracy=reg.score(X_test,y_test)
print("\nACCURACY :",accuracy*100)

