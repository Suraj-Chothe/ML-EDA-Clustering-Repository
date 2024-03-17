# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 15:14:37 2023

@author: suraj
"""
'''SVD TEChnice'''
import numpy as np
from numpy import array
from scipy.linalg import svd

A= array([[1,0,0,0,2],[0,0,3,0,0],[0,0,0,0,0],[0,4,0,0,0]]) 
print(A)


#svd
U,d ,Vt= svd(A)

print(U)
print(d)
print(Vt)

print(np.diag(d))
######################################################

# now we can apply the svd on the dataset
import pandas as pd
import numpy as np

data= pd.read_excel("C:/Data Science/data_set/University_Clustering.xlsx")
data.head()

data=data.iloc[:,2:]    # reomve the non numerical data

data
from sklearn.decomposition import TruncatedSVD
svd=TruncatedSVD(n_components=3)
svd.fit(data)
result= pd.DataFrame(svd.transform(data))
result.head()
result.columns="pc0","pc1","pc2"
result.head()

#scatter diagram

import matplotlib.pyplot as plt
plt.scatter(x=result.pc0, y=result.pc1)
