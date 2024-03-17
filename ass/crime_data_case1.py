# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 00:18:31 2023

@author: suraj
"""
'''2.	Perform clustering for the crime data and identify the number of clusters     
       formed and draw inferences. Refer to crime_data.csv dataset.'''
'''***********model building***************'''
'''5.1 Build the model on the scaled data (try multiple options).
5.2 Perform the hierarchical clustering and visualize the clusters using dendrogram.
5.3 Validate the clusters (try with different number of clusters) – 
label the clusters and derive insights (compare the results from multiple approaches).
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("C:/Data Science/data_set/crime_data.csv")
df
#######################
#1.column
df.columns
#######################
#2.shape
df.shape    #Out[7]: (3999, 12)
##################
#description of the dataframe 
df.describe()
# we can store the above description in the one varible
a=df.describe()

# we are checking which column is not necserray or
# the column which is  numerical data  that can be place in the datframe
df1=df.drop(['Unnamed: 0'],axis=1) 
df1.columns
df1.shape
#there are scale differenece between the column value so we 
#can remove that value from the Dataframe
# by using the normalization or standardization
#so we can use the normalization
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#now we can apply this noramlization function to df2 dataframe for all the rows and column from 1 until end
# since 0 th column has Universcity name hence skipped
df2= norm_fun(df1.iloc[:,:])   #here we can use the data make it as noramlize that is in the 0 and 1 form and we can 
# now we can descrine the df2 after we can make it in the normalize form
b=df2.describe()
#####################
#before you can applying the clustering you need to plot dendrogram first
# now to create the dendrogram , we need to measure distance
#we have import the linkage
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#linkage function gives us hierarchical or aglomerative clustering
#ref the help for linkage
p=linkage(df2,method="complete",metric="euclidean")
plt.figure(figsize=(16,8));
plt.title("Hierarchical Clustering Dendrogram");
plt.xlabel("phone");
plt.ylabel("internet_servise")
#ref help of dendogram
#sch.dendrogram
sch.dendrogram(p,leaf_rotation=0,leaf_font_size=10)
plt.show()

#######################
#dendrogram()
# now we can draw the dendgram
#applying  agglomertive clustering choosing 3 a clusters from dendrogram
#whatever has been dispalyed in dendrogram is not clustering
#it is just showing number of possible clusters
from sklearn.cluster import AgglomerativeClustering
dendo=AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(df2)
#apply labels to the cluster
dendo.labels_
cluster_labels=pd.Series(dendo.labels_)
#assign this series to df2 dataframe as column and name the column as 'cluster'
df2['clust1']=cluster_labels

########################################
df1.shape
#we want to realocate th ecolumnn 7 to 0 th postion
df=df2.iloc[:,[4,1,2,3]]
#now check the Univ1 Dataframe
df.iloc[:,2:].groupby(df.clust1).mean()
#from the output cluster 2 has got highest top10
#lowest accept ratio , best faculty ratio and highest expenses
#highest graduates ratio
##################################################
df.to_csv("crime.csv",encoding="utf-8")
import os
os.getcwd()
#this file is created in  the working direcatory where we want to store 
# file as the working directory are shon on the above top most corner