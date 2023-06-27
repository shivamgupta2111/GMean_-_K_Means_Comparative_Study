import numpy as np
import pandas as pd
cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']
df  = pd.read_csv(r"C:\Users\shiva\OneDrive\Documents\Minor project\KmeanApp\processed.cleveland.data",names=cols,header=None)
print(df.head())
columsize = len(cols)
#age in years
age = df["age"]

# sex (1 = male; 0 = female)
sex= df["sex"]

# cp: chest pain type
#         -- Value 1: typical angina
#         -- Value 2: atypical angina
#         -- Value 3: non-anginal pain
#         -- Value 4: asymptomatic
cp=df["cp"]  


#  trestbps: resting blood pressure (in mm Hg on admission to the 
#         hospital)
trestbps =df["trestbps"]

# chol: serum cholestoral in mg/dl
chol=df["chol"]  

# fbs: (fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)
fbs= df["fbs"]

# restecg: resting electrocardiographic results
#         -- Value 0: normal
#         -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST 
#                     elevation or depression of > 0.05 mV)
#         -- Value 2: showing probable or definite left ventricular hypertrophy
#                     by Estes' criteria
restecg = df["restecg"]  


# thalach: maximum heart rate achieved
thalach= df["thalach"]  


# exang: exercise induced angina (1 = yes; 0 = no)
exang = df["exang"]  



# oldpeak = ST depression induced by exercise relative to rest
oldpeak= df["oldpeak"]  



# slope: the slope of the peak exercise ST segment
#         -- Value 1: upsloping
#         -- Value 2: flat
#         -- Value 3: downsloping
slope = df["slope"]   


# ca: number of major vessels (0-3) colored by flourosopy
ca= df["ca"] 

# thal: 3 = normal; 6 = fixed defect; 7 = reversable defect, thal-> thalassemia
thal = df["thal"] 



# num: diagnosis of heart disease (angiographic disease status)
#         -- Value 0: < 50% diameter narrowing
#         -- Value 1: > 50% diameter narrowing
num = df["num"]
# print(df)

# print(thal.dtype)


# replace '?' with 'np.nan' as thal carry ? in few entries

df = df.replace({'?':np.nan})

# checking NaN values
# print(df.isnull().sum())

NullValueList = df.isnull().sum()

# print(NullValueList.dtype)
# print(df.isnull().values.any())
# Nullvalue = df.isnull().values.any()

df.fillna(method ='bfill')
# to change the target data to 0 and 1
# 0 means 'No heart disease', 1 means 'heart disease'
df['num'] = df['num']>0
df['num'] = df['num'].map({False:0, True:1})

# print(df)


num = df[cols[13]]
print(num)
name= cols[1]+" vs "+cols[2]
print(name)
print(df.iat[0,0])
