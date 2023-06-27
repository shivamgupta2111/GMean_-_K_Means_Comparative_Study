from cProfile import label
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np
cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']
url ='https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
df = pd.read_csv(url,index_col=False, header=None, names=cols)
# print(df)
# df.head()
# print(df)
# plt.scatter(df['Age'],df['Income($)'])
# plt.show()
df = df.replace({'?':np.nan})
df.fillna(method ='bfill')
df = df.astype('float64')

    # to change the target data to 0 and 1
    # 0 means 'No heart disease', 1 means 'heart disease'
df['num'] = df['num']>0
df['num'] = df['num'].map({False:0, True:1})
print(df)
km = KMeans(n_clusters = 2)
y_predicted= km.fit_predict(df[['chol','fbs','cp','sex','age','trestbps']])
print("---------------------------Predicted values------------------------------")
print(y_predicted)
print("---------------------------Actual Value----------------------------------")
num = df['num']
list=[]
for i in range(302):
    list.append(num[i])
print(list)
df['cluster'] = y_predicted
# print(df.head())


# df1 = df[df.cluster==0]
# df2 = df[df.cluster==1]
# df3 = df[df.cluster==2]

# plt.scatter(df1.Age,df1['Income($)'], color='green', label='cluster-1')
# plt.scatter(df2.Age,df2['Income($)'], color='green',label='cluster-2')
# plt.scatter(df3.Age,df3['Income($)'], color='black',label='cluster-3')
# plt.xlabel('Age')
# plt.ylabel('Income')
# plt.legend(loc ="upper left" )
# plt.show()