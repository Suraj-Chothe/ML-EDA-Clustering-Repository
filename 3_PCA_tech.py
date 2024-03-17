# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:22:15 2023

@author: suraj
"""
'''----------PCA---------------'''
#step 1:-#importing the packge
#importing the packge
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig

#step1 :-# defined the simple data
#we are ceating the simple 2-d array and and 
# generation of the data matrix A
#step a:- defined the simple data
Marks=np.array([[3,4],[2,8],[6,9]])
print(Marks)


#step b:- crete on dataframe passed to multidimension array
Marks_df=pd. DataFrame(Marks, columns=["Physics",'maths'])
Marks_df

# step c:- plot the scatter plot
plt.scatter(Marks_df["Physics"], Marks_df["maths"])

# step 2 :- scalinng  of the data
# normalization and standrdization and scalling
#step a:-
# make data mean centric
Meanbycolumn= np.mean(Marks.T, axis=1)  # normalization
print(Meanbycolumn)

Scaled_data=Marks- Meanbycolumn

#step b:-
# taking the transpoce of the column
Marks.T

#Scaled data
Scaled_data
#######################################
# step3:- Find covariance matrix of aboved scaled data
Cov_mat= np.cov(Scaled_data.T)
Cov_mat
###########################################
# step 4 :- finding the eigen value and eigen vector
# find the correscponding egin value and eigen vector of 
#above covarience matrix
Eval,Evec= eig(Cov_mat)
print(Eval)
print(Evec)

######################################
# step 5 project the data on the new axis
# get original data projected to principle componenet on the new axis
Projected_data= Evec.T.dot(Scaled_data.T)
print(Projected_data)


#######################################
#step 6:- alternat method using the sklearn

from sklearn.decomposition import PCA
pca =PCA(n_components=2)
pca.fit_transform(Marks)


########################################################
''' PCA opertion of the Univercity_cluster we are apply'''
# we are applying the PCA on the file

import pandas as pd
import numpy as np

uni1= pd.read_excel("C:/Data Science/data_set/University_Clustering.xlsx")
uni1.describe()

uni1.info()
uni=uni1.drop(["State"], axis=1)
uni

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

# considering only numerical data
uni.data=uni.iloc[:,1:]

# normalization the numerical data
uni_normal= scale(uni.data)
uni_normal

pca=PCA(n_components= 6)
pca_values = pca. fit_transform(uni_normal)

# the amount of variance that each PCA explian is
var = pca.explained_variance_ratio_
var



#PCA weights
#pca.components_
# pca.components_[0]



#cumulative variance

var1= np.cumsum(np.round(var, decimals= 4) *100)
var1

# variance plot for PCA componenets obtanined

plt.plot(var1 , color= "red")

# PCA scores
pca_values


pca_data= pd.DataFrame (pca_values)

pca_data.columns="comp0","comp1","comp2","comp3","comp4","comp5"
final=pd.concat([uni.Univ, pca_data.iloc[:,0:3]], axis=1)
# this is Univ column of uni data frame


#scatter plot

import matplotlib.pyplot as plt

ax= final.plot(x="comp0", y="comp1", kind="scatter", figsize=(12,8))
final[["comp0","comp1", "Univ"]].apply(lambda x:ax.text(*x),axis=1)




