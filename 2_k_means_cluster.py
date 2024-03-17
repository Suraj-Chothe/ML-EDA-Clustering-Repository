# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 08:40:23 2023

@author: suraj
"""
"""**** kMeans Algorithm"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster  import KMeans

#let us try to understand first how k means work fro two
#dimensional data
# fro that ,generate random numbers in the range 0 to 1
# and with uniform probility of 1/50
X=np.random.uniform(0,1,50)
Y= np.random.uniform(0,1,50)
#create a empty dataframe with 0 rows and 2 columns
df_xy= pd.DataFrame(columns=["X","Y"])
#assign the values of X ANd Y to these column
df_xy.X=X
df_xy.Y=Y
df_xy.plot(x="X",y="Y", kind="scatter")
model1=KMeans(n_clusters=3).fit(df_xy)

'''whit data X and Y ,apply Kmeans model, genarte  scattre plot
with scale /font=10

cmap= plt.cm.coolwarm: cool color cobination'''

model1.labels_
df_xy.plot(x="X",y="Y", c=model1.labels_,kind="scatter", s=10, cmap=plt.cm.coolwarm)

#now we are apply the k means on theunivercity dataset
Univ1= pd.read_excel("C:/Data Science/data_set/University_Clustering.xlsx")

Univ1.describe()
Univ=Univ1.drop(["State"], axis=1)

# we know that there is scale differance among the column which we have
#either by usinng the normalization and standardization

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

# now apply this normalization functio to the Univ dataframe for all row and column

df_norm= norm_func(Univ.iloc[:,1:])

''' what will be idea cluster number , will it be 1,2 or 3 
we can predict the k value from following code 
'''

TWSS=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)  #total whithin the sum square

"""kmeans inertia ,also know as sumof sqaures Errors
(or SSE) , clacukate the sumof the distance 
of all points within the cluster from the centroid of the point 
It is the difference between the observed value and predicated value
"""
TWSS
# as k value incresses the TWSS value decreses
plt.plot(k,TWSS,'ro-');
plt.xlabel("NO_of_clutsre")
plt.ylabel("Total_within_ss")

'''how to select value of k from elbow curve
when k changes from 2 to 3 then decreases
in TWSS is higher than 
When k change from 3 to 4
when k value changes from  5 to 6 decreases
in TWSS in considerly less hence considered k=3 '''


model =KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_
mb=pd.Series(model.labels_)
Univ['clust']=mb
Univ.head()
Univ=Univ.iloc[:,[7,1,2,3,4,5,6]]
Univ
Univ.iloc[:,2:8].groupby(Univ.clust).mean()     # we can uderstand describtion of each column
Univ.to_csv('kmean_university.csv', encoding='utf-8')
import os
os.getcwd()
