import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


df=pd.read_csv(r"C:\Users\LAB\Documents\GitHub\ML_Project\coustmer_journey_funnel\bank-additional-cleaned.csv",sep=';')

df.head()
print(df.head())

#################### already has a header no need to add header sperate using ';' for clean table

duplicates = df[df.duplicated()]
df_cleaned=df.drop_duplicates()
df_cleaned.to_csv("bank-additional-cleaned.csv", index=False,sep=";")
################# removed duplicate and saved to new csv file

no_empty=df.dropna(axis=1,how='all')
print(no_empty)
if no_empty.shape[0]<1 :
 print("empty")
   
else:print("no empty")

##################blank coloums

missing_counts=df.isnull().sum()
missing_summary=missing_counts[missing_counts>0]
print('missing',missing_summary)

###########missing values 

df["y"]=(df["y"] == "yes" ).astype(int)

############# hardcoded yes=1 

unkowwn_count=df.apply(lambda col: (col == 'unknown').sum())
print(unkowwn_count)
################## known the count of unkown 

binary_cols = ['default', 'housing', 'loan']
df[binary_cols] = df[binary_cols].replace({'yes': 1, 'no': 0, 'unknown': -1})

df.head()
print(df.head())

###################

from sklearn.preprocessing import LabelEncoder

multi_cols = ['job', 'marital', 'education', 'contact', 'month', 'day_of_week', 'poutcome']
le = LabelEncoder()

for col in multi_cols:
    df[col] = le.fit_transform(df[col])

######################################################



df.to_csv("bank-additional-ready.csv",index=False,sep=";")