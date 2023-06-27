import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import math
# Use matplotlib in notebook output
# %matplotlib inline
# Data - [average passes, average goals (player goals - opponent goals)]
cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']
df  = pd.read_csv(r"C:\Users\shiva\OneDrive\Documents\Minor project\KmeanApp\processed.cleveland.data",names=cols,header=None)
print(df.head())

df = df.replace({'?':np.nan})
df=df.fillna(method ='bfill')
# df.dropna(inplace=True)

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

# df = df.replace({'?':np.nan})
# df.dropna(inplace=True)

# checking NaN values
# alues.any()
# print(df.isnull().sum())

# NullValueList = df.isnull().sum()

# print(NullValueList.dtype)-
# print(df.isnull().values.any())
# Nullvalue = df.isnull().v
# df.fillna(method ='bfill')
#
df['num'] = df['num']>0
df['num'] = df['num'].map({False:0, True:1})
cluster = [None]*len(df.index)
# df = df.astype('float64')

# print(df)




def Scatter_Plot():
    for i in range(len(num)):
        for j in range(len(num)):
            if i==j:
                j=j+1
            plotx = []
            ploty = []
            for i in range(len(num)):
                plotx.append(df[cols[j]])
                ploty.append(df[cols[i]])
            name= cols[j]+" vs "+cols[i] 
            plt.plot(plotx,ploty, name)

#__________________________________________Helping functions__________________________________________
# def random_centers(dim,k):
#     centers = []
#     for i in range(k):
#         center = []
#         for d in range(dim):
#             rand = random.randint(0,100)
#             center.append(rand)
#         centers.append(center)
#     return centers
# def random_centers(dim,k):
#     centers=[]
#     for i in range(k):
#         rand = random.randint(0,len(df.index))
#         # print("index",rand)
#         center=[]
#         center.append(age[rand])
#         center.append(sex[rand])
#         center.append(cp[rand])
#         center.append(trestbps[rand])
#         center.append(chol[rand])
#         center.append(fbs[rand])
#         center.append(restecg[rand])
#         center.append(thalach[rand])
#         center.append(exang[rand])
#         center.append(oldpeak[rand])
#         center.append(slope[rand])
#         center.append(ca[rand])
#         center.append(thal[rand])
#         centers.append(center)
#     # print(centers)
#     return centers
def random_centers(k,axis1,axis2):
    centers=[]
    for i in range(k):
        center=[]
        rand = random.randint(0,len(df.index))
        center.append(df.iat[rand,axis1])
        center.append(df.iat[rand,axis2])
        centers.append(center)
    return centers



def point_clustering(df,centers,dim1,dim2):
    # print("Center-",centers)
    for i in range(0, len(df.index)):
        nearest_center = 0
        nearest_center_dist = None
        # dim1 = 0
        # dim2 = 4
        for j in range(0,len(centers)):
            euclidean_dist = 0
            value = float(df.iat[i, dim1])
            cent = float(centers[j][0])
            dist = abs(value-cent)
            dist = dist**2
            euclidean_dist += dist
            value = float(df.iat[i, dim2])
            cent = float(centers[j][1])
            dist = abs(value-cent)
            dist = dist**2
            euclidean_dist += dist
            euclidean_dist = np.sqrt(euclidean_dist)
            if nearest_center_dist == None:
                nearest_center_dist = euclidean_dist
                nearest_center = j
            elif nearest_center_dist > euclidean_dist:
                nearest_center_dist = euclidean_dist
                nearest_center = j
        cluster[i] = nearest_center
    # print("cluster", cluster)




def mean_center(df, centers, axis1, axis2):
    new_center = []
    cluster_group = []
    no_cluster = []
    for i in range(0, len(cluster)):
        no = []
        total = 0
        for j in range(0,len(df.index)):
            if (cluster[j] == i):
                no.append(j)
                total += 1
        cluster_group.append(no)
        no_cluster.append(total)

    # print("cluster-",cluster)    
    # print("no of cluster",no_cluster)
    # print("group of cluster",cluster_group)

    # for axis 1
    for i in range(0,len(centers)):
        center = []
        sum = float(0)
        for j in range(0,len(cluster_group[i])):
           sum += float(df.iat[cluster_group[i][j],axis1])
        sum = float(sum)/float(no_cluster[i])
        center.append(sum)
        sum = float(0)
        for j in range(0,len(cluster_group[i])):
           sum += float(df.iat[cluster_group[i][j],axis2])
        sum = float(sum)/float(no_cluster[i])
        center.append(sum)
        new_center.append(center)
    
    return new_center


dimen1=0
dimen2 = 4
#  _________________________________________________K means Algorithm_________________________________________________
# Gets data and k, returns a list of center points.
def train_k_means_clustering(df, k, epochs=5):
    dims = len(df.axes[1])
    # print("Dimensions=",dims)
    # print('data[0]:',data[0])
    axis1 = dimen1
    axis2 = dimen2
    
    
    centers = random_centers(k,axis1,axis2)
    # print("train Center dim",len(centers[0]))
    # clustered_data = point_clustering(df, centers, dims, first_cluster=True)
    point_clustering(df,centers,axis1,axis2)

    for i in range(epochs):
        # print("echo",i)
        centers = mean_center(df,centers,axis1,axis2)
        # print("means Center dim",len(centers[0]))
        # print("NewCenter-",centers)
        point_clustering(df,centers,axis1,axis2)
    
    return centers

def predict_k_means_clustering(point, centers):
    dims = len(point)
    center_dims = len(centers[0])
    # predict = None
    
    if dims != center_dims:
        raise ValueError('Point given for prediction have', dims, 'dimensions but centers have', center_dims, 'dimensions')

    nearest_center = None
    nearest_dist = None
    
    for i in range(len(centers)):
        euclidean_dist = 0
        for dim in range(1, dims):
            dist = point[dim] - centers[i][dim]
            euclidean_dist += dist**2
        euclidean_dist = np.sqrt(euclidean_dist)
        if nearest_dist == None:
            nearest_dist = euclidean_dist
            nearest_center = i
        elif nearest_dist > euclidean_dist:
            nearest_dist = euclidean_dist
            nearest_center = i
        print('center:',i, 'dist:',euclidean_dist)
    if(nearest_center == 0):
        print("No Heart Risk")   
    elif(nearest_center == 1):
        print("Heart Risk")
            

def scatterplot(center):
    x = []
    y= []
    x2=[]
    y2=[]
    x3=[]
    y3=[]
    
    for i in range(0,len(cluster)):
        if(cluster[i]==0):
            x.append(df.iat[i,dimen1])
            y.append(df.iat[i,dimen2])
        elif(cluster[i]==1):
            x2.append(df.iat[i,dimen1])
            y2.append(df.iat[i,dimen2])
        elif(cluster[i]==2):
            x3.append(df.iat[i,dimen1])
            y3.append(df.iat[i,dimen2])
    

    plt.scatter(x,y,c="blue",label="cluster1")
    plt.scatter(x2,y2,c="green",label="cluster1")
    plt.scatter(x3,y3,c="orange",label="cluster1")
    plt.xlabel(cols[dimen1])
    plt.ylabel(cols[dimen2])
    plt.title("Kmeans")
    plt.legend(bbox_to_anchor = (1 , 1))
    plt.show()


def Euclidien_distance(x1, x2, y1, y2):
    diff = x1-x2
    Euclidien = (diff)**2
    diff = y1-y2
    Euclidien = Euclidien + ((diff)**2)
    Euclidien = math.sqrt(Euclidien)
    return Euclidien


def scatter_cofficient(center,axis1,axis2):
    new_center = []
    cluster_group = []
    no_cluster = []
    for i in range(0, len(cluster)):
        no = []
        total = 0
        for j in range(len(df.index)):
            if (cluster[j] == i):
                no.append(j)
                total += 1
        cluster_group.append(no)
        no_cluster.append(total)
    av_12_inter = 0
    av_23_inter =0
    av_13_inter =0
    # print(cluster_group)
    # print(no_cluster)
    for i in range(0,len(cluster_group[0])):
        for j in range(len(cluster_group[1])):
            av_12_inter = float(av_12_inter)+float(Euclidien_distance(df.iat[cluster_group[0][i],axis1],df.iat[cluster_group[1][j],axis1],df.iat[cluster_group[0][i],axis2],df.iat[cluster_group[1][j],axis2]))
    for i in range(len(cluster_group[1])):
        for j in range(len(cluster_group[2])):
            av_23_inter = float(av_23_inter)+float(Euclidien_distance(df.iat[cluster_group[1][i],axis1],df.iat[cluster_group[2][j],axis1],df.iat[cluster_group[1][i],axis2],df.iat[cluster_group[2][j],axis2]))
    for i in range(len(cluster_group[0])):
        for j in range(len(cluster_group[2])):
            av_13_inter = float(av_13_inter)+float(Euclidien_distance(df.iat[cluster_group[0][i],axis1],df.iat[cluster_group[2][j],axis1],df.iat[cluster_group[0][i],axis2],df.iat[cluster_group[2][j],axis2]))
    
    avg_inter = float(av_12_inter+av_13_inter+av_23_inter)/float(no_cluster[0]*no_cluster[1]+no_cluster[1]*no_cluster[2]+no_cluster[0]*no_cluster[2])
    intra1=0
    intra2=0
    intra3=0
    for i in range(len(cluster_group[0])):
        for j in range(len(cluster_group[0])):
            intra1 = intra1 +Euclidien_distance(df.iat[cluster_group[0][i],axis1],df.iat[cluster_group[0][j],axis1],df.iat[cluster_group[0][i],axis2],df.iat[cluster_group[0][j],axis2])
    
    for i in range(len(cluster_group[1])):
        for j in range(len(cluster_group[1])):
            intra2 = intra2 +Euclidien_distance(df.iat[cluster_group[1][i],axis1],df.iat[cluster_group[1][j],axis1],df.iat[cluster_group[1][i],axis2],df.iat[cluster_group[1][j],axis2])
    
    for i in range(len(cluster_group[2])):
        for j in range(len(cluster_group[2])):
            intra3 = intra3 +Euclidien_distance(df.iat[cluster_group[2][i],axis1],df.iat[cluster_group[2][j],axis1],df.iat[cluster_group[2][i],axis2],df.iat[cluster_group[2][j],axis2])
    
    intra = float(intra1+intra2+intra3)/float(((no_cluster[0])**2)+((no_cluster[1])**2)+((no_cluster[2])**2))
    sc_cofficient = avg_inter/intra
    print("Scatter cofficient-",sc_cofficient)

# k number of clusters we want 2:
# 0 means - no heart disease
# 1 means - risk of heart disease
# An epoch is when all the training data is used at once and is defined as the total number of iterations of 
# all the training data in one cycle for training the machine learning model. 
# epochs = input("Enter the number of iterations-")
# epochs = int(epochs)
epochs = 35
centers = train_k_means_clustering(df, k=3, epochs=5)
# num= df['num']
scatterplot(centers)
scatter_cofficient(centers,dimen1,dimen2)
for i in range(len(centers)):
    for j in range(0,2):
        value = centers[i][j]
        value = round(value,2)
        centers[i][j] = value
print("values       cluster1    cluster2    cluster3")
value = cols[dimen1]
print(value,"       ",centers[0][0],"       ",centers[1][0],"      ",centers[2][0])
value = cols[dimen2]
print(value,"       ",centers[0][1],"       ",centers[1][1],"       ",centers[2][1])


