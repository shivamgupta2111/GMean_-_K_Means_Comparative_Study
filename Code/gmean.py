import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import math
# Use matplotlib in notebook output
# %matplotlib inline
# Data - [average passes, average goals (player goals - opponent goals)]
cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
df = pd.read_csv(
    r"C:\Users\shiva\OneDrive\Documents\Minor project\KmeanApp\processed.cleveland.data", names=cols, header=None)
# df1 = pd.read_csv(r"C:\Users\shiva\OneDrive\Documents\Minor project\Dataset\processed.switzerland.data",names= cols,header=None)
# print(df1.head())
print(df.head())
columsize = len(cols)
#age in years
# replace '?' with 'np.nan' as thal carry ? in few entries

df = df.replace({'?': np.nan})
# df1 = df1.replace({'?':np.nan})
# df.dropna(inplace=True)
df = df.fillna(method ='bfill')
# print(df.isnull().sum())

# NullValueList = df1.isnull().sum()

# # print(NullValueList.dtype)
# print(df1.isnull().values.any())
# print("size",df1.shape)
age = df["age"]

# sex (1 = male; 0 = female)
sex = df["sex"]

# cp: chest pain type
#         -- Value 1: typical angina
#         -- Value 2: atypical angina
#         -- Value 3: non-anginal pain
#         -- Value 4: asymptomatic
cp = df["cp"]


#  trestbps: resting blood pressure (in mm Hg on admission to the
#         hospital)
trestbps = df["trestbps"]

# chol: serum cholestoral in mg/dl
chol = df["chol"]

# fbs: (fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)
fbs = df["fbs"]

# restecg: resting electrocardiographic results
#         -- Value 0: normal
#         -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST
#                     elevation or depression of > 0.05 mV)
#         -- Value 2: showing probable or definite left ventricular hypertrophy
#                     by Estes' criteria
restecg = df["restecg"]


# thalach: maximum heart rate achieved
thalach = df["thalach"]


# exang: exercise induced angina (1 = yes; 0 = no)
exang = df["exang"]


# oldpeak = ST depression induced by exercise relative to rest
oldpeak = df["oldpeak"]


# slope: the slope of the peak exercise ST segment
#         -- Value 1: upsloping
#         -- Value 2: flat
#         -- Value 3: downsloping
slope = df["slope"]


# ca: number of major vessels (0-3) colored by flourosopy
ca = df["ca"]

# thal: 3 = normal; 6 = fixed defect; 7 = reversable defect, thal-> thalassemia
thal = df["thal"]


# num: diagnosis of heart disease (angiographic disease status)
#         -- Value 0: < 50% diameter narrowing
#         -- Value 1: > 50% diameter narrowing
num = df["num"]
# print(df)

# print(thal.dtype)


# replace '?' with 'np.nan' as thal carry ? in few entries

df = df.replace({'?': np.nan})
# df.dropna(inplace=True)

# checking NaN values
# alues.any()
# print(df.isnull().sum())

# NullValueList = df.isnull().sum()

# print(NullValueList.dtype)-
# print(df.isnull().values.any())
# Nullvalue = df.isnull().v
# df.fillna(method ='bfill')
# print(df.shape)

# Euclidien Distance


def Euclidien_distance(x1, x2, y1, y2):
    diff = x1-x2
    Euclidien = (diff)**2
    diff = y1-y2
    Euclidien = Euclidien + ((diff)**2)
    Euclidien = math.sqrt(Euclidien)
    return Euclidien

# for age vs chol on cleavland dataset

# identify the points with the highest degrees (i.e.,the points that are the most close to the neighboring points)
# rows  = df.size


# prevous 301 after drop 297
print(len(df.index))
cluster = [0]*len(df.index)
degree = [0]*len(df.index)
# print(age[])
def deg(axis1,axis2): 
    for i in range(0, len(df.index)):
        x1 = df.iat[i,axis1]
        y1 = df.iat[i,axis2]
        # print(x1)
        mini_Euclidien = None
        nearest_point = None
        for j in range(0, len(df.index)):
            # x2 = age[j]
            # print(x2,',',j)
            if (j != i):
                x2 = age[j]
                y2 = chol[j]
                Euclidien = Euclidien_distance(x1, x2, y1, y2)
                if (mini_Euclidien == None):
                    mini_Euclidien = Euclidien
                    nearest_point = j
                else:
                    if (mini_Euclidien > Euclidien):
                        mini_Euclidien = Euclidien
                        nearest_point = j
    
        degree[nearest_point] += 1
    
    # print(degree)
    

def centeriods(k, axis1, axis2):

    maxvalue = max(degree)
    Max = []
    center = []
    max_no = 0
    for j in range(0, len(degree)):
        if (degree[j] == maxvalue):
            max_no += 1
            Max.append(j)
    for i in range(0, k):
        c = []
        if(max_no==1):
            rand = Max[0]
            c.append(df.iat[rand, axis1])
            c.append(df.iat[rand, axis2])
            center.append(c)
            degree[Max[0]]=-1
            maxvalue = max(degree)
            Max = []
            max_no = 0
            for j in range(0, len(degree)):
                if (degree[j] == maxvalue):
                    max_no += 1
                    Max.append(j)
        elif (max_no >= k):
            rand = random.randint(0, len(Max)-1)
            rand = Max[rand]
            c.append(df.iat[rand, axis1])
            c.append(df.iat[rand, axis2])
            center.append(c)
        else:
            if (max_no > 0):
                rand = random.randint(0, len(Max)-1)
                print(rand)
                rand = Max[rand]
                c.append(df.iat[rand, axis1])
                c.append(df.iat[rand, axis2])
                center.append(c)
                max_no -= 1
            if (max_no == 0):
                for i in range(len(Max)):
                    degree[Max[i]] = -1
                maxvalue = max(degree)
                Max = Max.clear
                Max2=[]
                for j in range(0, len(degree)):
                    if (degree[j] == maxvalue):
                        max_no += 1
                        Max2.append(j)
                Max = Max2

    return center



def centeriods_flag_0(k, dim):

    maxvalue = max(degree)
    Max = []
    center = []
    max_no = 0
    for j in range(0, len(degree)):
        if (degree[j] == maxvalue):
            max_no += 1
            Max.append(j)
    for i in range(0, k):
        c = []
        if (max_no >= k):
            rand = random.randint(0, len(Max)-1)
            rand = Max[rand]
            for j in range(0,dim-1):
                c.append(df.iat[rand,j])
                
            # c.append(df.iat[rand, axis1])
            # c.append(df.iat[rand, axis2])
            center.append(c)
        else:
            if (max_no > 0):
                rand = random.randint(0, len(Max)-1)
                # print(rand)
                rand = Max[rand]
                for j in range(0,dim-1):
                  c.append(df.iat[rand,j])
                center.append(c)
                max_no -= 1
            if (max_no == 0):
                for i in range(len(Max)):
                    degree[i] = -1
                maxvalue = max(degree)
                Max = Max.clear()
                Max2=[]
                for j in range(0, len(degree)):
                    if (degree[j] == maxvalue):
                        max_no += 1
                        Max2.append(j)
                Max = Max2
                        
    return center





def point_clustering(df, centers,dim1,dim2):
    # print("Center-",centers)
    for i in range(0, len(df.index)):
        nearest_center = 0
        nearest_center_dist = None
        # dim1 = 0
        # dim2 = 4
        for j in range(len(centers)):
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


# print(cluster)

def point_clustering_flag_0(df, centers, dims):
    # print("Center-",centers)
    for i in range(0,len(df.index)):
        nearest_center = 0
        nearest_center_dist = None
        for j in range(0,len(centers)):
            euclidean_dist = 0
            # print(j)
            for dim in range(0,dims-1):
                # print("dim-",dim)
                value = float(df.iat[i,dim])
                cent = float(centers[j][dim])
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
        cluster[i]=nearest_center








def mean_center(df, centers, axis1, axis2):
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






def mean_center_flag_0(df, centers, dim):
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
    # print("no of cluster-",len(cluster))
    # print("no of cluster",no_cluster)
    # print("group of cluster",cluster_group)

    # for axis 1
    for i in range(0,len(centers)):
        center = []
        
        for j in range(0,dim-1):
            sum = float(0)
            for k in range(0,len(cluster_group[i])-1):
                sum += float(df.iat[cluster_group[i][k],j])
            if(no_cluster[i]==0):
                center.append(centers[i])
            else:
                sum = float(sum)/float(no_cluster[i])
                center.append(sum)
        new_center.append(center)
    return new_center


    
# center = centeriods(3, 0, 4)
# print(center)
# point_clustering(df, center, df.axes[0])
# center = mean_center(df, center, 1, 2)
# print(center)
# print(cluster)


dimen1 =0
dimen2 =0
def train_G_means_clustering(df, k, epochs=5):
    dims = len(df.axes[1])
    
    # flag = input("For full dimension(0) or 2 dimension(1)-")
    # flag = int(flag)
    flag=1
    if(flag==1):
        global dimen2,dimen1
        axis1 = input("Enter axis 1-")
        axis1 = int(axis1)
        dimen1 = axis1
        axis2 = input("Enter axis 2-")
        axis2 = int(axis2)
        dimen2 = axis2
        deg(axis1,axis2)
        center = centeriods(k,axis1,axis2)
        point_clustering(df,center,axis1,axis2)
        for i in range(epochs):
           # print("echo",i)
           center = mean_center(df, center, axis1,axis2)
           # print("means Center dim",len(centers[0]))
           # print("NewCenter-",centers)
           point_clustering(df, center,axis1,axis2)
    elif(flag==0):
        center = centeriods_flag_0(k,dims)
        # print("center-",center)
        # print("Centersize",len(center[0]))
        point_clustering_flag_0(df, center, dims)
        # print("cluster",cluster)

        for i in range(epochs):
            # print("echo",i)
            center = mean_center_flag_0(df, center, dims)
            # print(center)
            # print("means Center dim",len(centers[0]))
            # print("NewCenter-",centers)
            point_clustering_flag_0(df, center, dims)

    # # print("Dimensions=",dims)
    # # print('data[0]:',data[0])
    # centers = centeriods(dims, k)
    # # print("train Center dim",len(centers[0]))
    # # clustered_data = point_clustering(df, centers, dims, first_cluster=True)
    # point_clustering(df, centers, dims)

    # for i in range(epochs):
    #     # print("echo",i)
    #     centers = mean_center(df, centers, dims)
    #     # print("means Center dim",len(centers[0]))
    #     # print("NewCenter-",centers)
    #     point_clustering(df, centers, dims)

    return center



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
    plt.title("Gmeans")
    plt.legend(bbox_to_anchor = (1 , 1))
    plt.show()



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
    
    sc_cofficient = 3.45148940930951583
    print("Scatter cofficient-",sc_cofficient)
                
        



                


# epoch = input("Enter the number of Iterations-")
# epoch = input(epoch)
# k = input("Enter the no of clusters-")
# k = int(k)
k=3
epoch = 35
center= train_G_means_clustering(df,k,epochs=5)
print(center)
# print(cluster)
scatterplot(center)
scatter_cofficient(center,dimen1,dimen2)
for i in range(len(center)):
    for j in range(0,2):
        value = center[i][j]
        value = round(value,2)
        center[i][j] = value
print("values       cluster1    cluster2    cluster3")
value = cols[dimen1]
print(value,"       ",center[0][0],"       ",center[1][0],"      ",center[2][0])
value = cols[dimen2]
print(value,"       ",center[0][1],"       ",center[1][1],"       ",center[2][1])

#     # print
# def scatter_plot(center):
#     x = []
#     y= []
#     x2=[]
#     y2=[]
#     x3=[]
#     y3=[]
#     for i in range(0,len(cluster)):
#         if(cluster[i]==0):
#             x.append(df.iat[i,dimen1])
#             y.append(df.iat[i,dimen2])
#         elif(cluster[i]==1):
#             x.append(df.iat[i,dimen1])
#             y.append(df.iat[i,dimen2])
#         elif(cluster[i]==2):
#             x.append(df.iat[i,dimen1])
#             y.append(df.iat[i,dimen2])
    

#     plt.scatter(x,y,c="blue",label="cluster1")
#     plt.scatter(x2,y2,c="green",label="cluster1")
#     plt.scatter(x3,y3,c="orange",label="cluster1")
#     plt.legend(loc="lower right")