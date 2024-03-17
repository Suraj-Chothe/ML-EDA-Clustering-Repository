# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:12:12 2023

@author: suraj
"""

# Ml
#clustering technique for arrangeing the data
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#now import  the file from  data set and create a datframe
univ1=pd.read_excel("C:/Data Science/data_set/University_Clustering.xlsx")
univ1
univ1.describe()        #five number summary
a=univ1.describe()      #check it in the varible exploere
# we have one column 'State which really not useful we will drop it
univ=univ1.drop(["State"],axis=1)
# we know that there is scale difference among the column 
#which we have to remove
#either by using the normalization or standardization
#whenever there is mixed data apply normalization
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#now we can apply this noramlization function to Univ dataframe for all the rows and column from 1 until end
# since 0 th column has Universcity name hence skipped
df_norm= norm_fun(univ.iloc[:,1:])
#here we can use the first column i.e. 1 ie is drop the 1 st column and take the remaing column
# you can check the df_norm dataframe which is scaled
#between the value 0 to 1
#you can apply describe function to new data frame
b=df_norm.describe()
#before you can applying the clustering you need to plot dendrogram first
# now to create the dendrogram , we need to measure distance
#we have import the linkage
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#linkage function gives us hierarchical or aglomerative clustering
#ref the help for linkage
z=linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8));
plt.title("Hierarchical Clustering Dendrogram");
plt.xlabel("Index");
plt.ylabel("Distance")
#ref help of dendogram
#sch.dendrogram
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()
#dendrogram()
#applying  agglomertive clustering choosing 3 a clusters
#from dendrogram
#whatever has been dispalyed in dendrogram is not clustering
#it is just showing number of possible clusters
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(df_norm)
#apply labels to the cluster
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
#assign this series to Univ dataframe as column and name the column as 'cluster'
univ['clust']=cluster_labels
#we want to realocate th ecolumnn 7 to 0 th postion
Univ1=univ.iloc[:,[7,1,2,3,4,5,6]]
#now check the Univ1 Dataframe
Univ1.iloc[:,2:].groupby(Univ1.clust).mean()
#from the output cluster 2 has got highest top10
#lowest accept ratio , best faculty ratio and highest expenses
#highest graduates ratio
Univ1.to_csv("University.csv",encoding="utf-8")
import os
os.getcwd()
#this file is created in  the working direcatory where we want to store 
# file as the working directory are shon on the above top most corner


####################################################
#****************** auto insurance************************** file
#apply the same clustering logic on the auto insurance file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("C:/Data Science/data_set/AutoInsurance.csv") 
df

df.columns
df.shape

df.describe()
b=df.describe()

# we have one column 'State which really not useful we will drop it
df2=df.drop(['Customer', 'State', 'Response', 'Coverage','Effective To Date',
       'Education', 'EmploymentStatus', 'Gender',
        'Location Code', 'Marital Status', 'Policy Type',
       'Policy', 'Renew Offer Type', 'Sales Channel',
       'Vehicle Class', 'Vehicle Size'],axis=1)
# we know that there is scale difference among the column 
#which we have to remove
#either by using the normalization or standardization
#whenever there is mixed data apply normalization
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#now we can apply this noramlization function to df2 dataframe for all the rows and column from 1 until end
# since 0 th column has Universcity name hence skipped
df_norm= norm_fun(df2.iloc[:,1:])
# you can check the df_norm dataframe which is scaled
#between the value 0 to 1
#you can apply describe function to new data frame
c=df_norm.describe()
#before you can applying the clustering you need to plot dendrogram first
# now to create the dendrogram , we need to measure distance
#we have import the linkage
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#linkage function gives us hierarchical or aglomerative clustering
#ref the help for linkage
z=linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8));
plt.title("Hierarchical Clustering Dendrogram");
plt.xlabel("Index");
plt.ylabel("Distance")
#ref help of dendogram
#sch.dendrogram
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()
#dendrogram()
#applying  agglomertive clustering choosing 3 a clusters
#from dendrogram
#whatever has been dispalyed in dendrogram is not clustering
#it is just showing number of possible clusters
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(df_norm)
#apply labels to the cluster
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
#assign this series to Univ dataframe as column and name the column as 'cluster'
df2['clust']=cluster_labels
#we want to realocate th ecolumnn 7 to 0 th postion
df=df2.iloc[:,[8,1,2,3,4,5,6,7]]
#now check the Univ1 Dataframe
df.iloc[:,2:].groupby(df.clust).mean()
#from the output cluster 2 has got highest top10
#lowest accept ratio , best faculty ratio and highest expenses
#highest graduates ratio
df.to_csv("auto_cluster.csv",encoding="utf-8")
import os
os.getcwd()


print(100/20)
print(100//20)




