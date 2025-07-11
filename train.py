import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


df=pd.read_csv('bank-additional-ready.csv',sep=';')


train,valid,test =np.split(df.sample(frac=1),(int(0.6*len(df)),int(0.8*len(df))))

def scale_dataset(df,oversample=False):
   
  x =df[df.columns[:-1]].values
  y =df[df.columns[-1]].values

  if oversample:
      ros=RandomOverSampler()
      x,y =  ros.fit_resample(x,y)



   
  scaler=StandardScaler()
  x=scaler.fit_transform(x)

  data=np.hstack((x,np.reshape(y,(-1,1))))

  return data,x,y

print(len(train[train["y"]==1]))
print(len(train[train["y"]==0]))


train,x_train,y_train = scale_dataset(train,oversample=True)



print(len(y_train))
print( sum(y_train ==1 ))
print( sum(y_train ==0))


valid, x_valid, y_valid = scale_dataset(valid, oversample=False)
test, x_test, y_test = scale_dataset(test, oversample=False)

#KNN

knn_model = KNeighborsClassifier(n_neighbors=30)
knn_model.fit(x_train , y_train)

y_pred=knn_model.predict(x_test)

print(classification_report(y_test,y_pred))
