# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:49:24 2023

@author: suraj
"""

''' problem statement 1.	
Perform clustering for the airlines data to obtain 
optimum number of clusters. 
Draw the inferences from the clusters obtained. 
Refer to EastWestAirlines.xlsx dataset.

# this is column are present in the dataset EastwestAirlines  excel file:
ID#:- here the id is unique
'Balance':-	Number of miles eligible for award travel
'Qual_miles':-	Number of miles counted as qualifying for Topflight status
'cc1_miles':-Has member earned miles with airline freq. flyer credit card in the past 12 months
cc2_miles:-	Has member earned miles with Rewards credit card in the past 12 months (1=Yes/0=No)?
cc3_miles:-	Has member earned miles with Small Business credit card in the past 12 months (1=Yes/0=No)?
Bonus_miles:-	Number of miles earned from non-flight bonus transactions in the past 12 months
Bonus_trans:-	Number of non-flight bonus transactions in the past 12 months
Flight_miles_12mo:-	Number of flight miles in the past 12 months
'Flight_trans_12:-Number of flight transactions in the past 12 months
Days_since_enroll:-Number of days since Enroll_date
Award?:-Dummy variable for Last_award (1=not null, 0=null)

'''
#######################################################################
'''***********************EDA / Expolatry data analysis****************************'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# we import the dataset excel file
df=pd.read_excel("C:/Data Science/data_set/EastWestAirlines.xlsx") 
df
#######################
#1.column
df.columns
'''Out[3]: 
Index(['ID#', 
       'Balance', 
       'Qual_miles', 
       'cc1_miles', 
       'cc2_miles', 
       'cc3_miles',
       'Bonus_miles',
       'Bonus_trans', 
       'Flight_miles_12mo',
       'Flight_trans_12',
       'Days_since_enroll',
       'Award?'],
      dtype='object')

#In this dataset there are 12 column and 
#in there are ID# column has ordinal data and  other all  column has the nominal data'''
#######################
#2.shape
df.shape    #Out[7]: (3999, 12)
# in the EastWestAirline has there are 3999 records and 12 column
#######################
#3.1 Now we count the number of the datapoint in the Balance clomun
df["Balance"].value_counts()
'''1000     10
500       5
2000      5
1500      4
5000      3
         ..
9554      1
30130     1
12665     1
9860      1
3016      1
Name: Balance, Length: 3904, dtype: int64
# in this dataset there are 1000 value are 10 inthis dataset 
and 500,2000 are 5 in this dataset
1500 has  4 value in this dataset 
5000 has 3 entry in this dataset
and other has the value has the 1  time come in this dataset
'''
#######################
#3.data types 
df.dtypes
'''ID#                  int64
Balance              int64
Qual_miles           int64
cc1_miles            int64
cc2_miles            int64
cc3_miles            int64
Bonus_miles          int64
Bonus_trans          int64
Flight_miles_12mo    int64
Flight_trans_12      int64
Days_since_enroll    int64
Award?               int64
dtype: object'''
######################
#4. missing value
a=df.isnull()
a.sum()
##########################
#5.scatter

df.plot(kind='scatter', x='Balance', y='Bonus_miles') ;
plt.show()

###########################
#6 2-D Scatter plot with color-coding for each flower type/class.
# Here 'sns' corresponds to seaborn.
#we are making the code with the Balance and Bonus_miles  with respect to the Awards? 
sns.set_style("whitegrid");
sns.FacetGrid(df, hue="Award?", height=4) \
   .map(plt.scatter, "Balance", "Bonus_miles") \
   .add_legend();
plt.show();
###########################
#6 2-D Scatter plot with color-coding for each flower type/class.
# Here 'sns' corresponds to seaborn.
#we are making the code with the Balance and Bonus_miles  with respect to the Awards? 
sns.set_style("whitegrid");
sns.FacetGrid(df, hue="Award?", height=4) \
   .map(plt.scatter, "Balance", "Days_since_enroll") \
   .add_legend();
plt.show();
############################
# pairwise scatter plot: Pair-Plot
# Dis-advantages: 
##Can be used when number of features are high.
##Cannot visualize higher dimensional patterns in 3-D and 4-D. 
#Only possible to view 2D patterns.
#pair plot 
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df, hue="Award?", height=3);
plt.show()
############################
#histogram
# What about 1-D scatter plot using just one feature?
#1-D scatter plot of petal-length
import numpy as np
df_balance = df.loc[df["Award?"] == "Balance"];
df_Bonus_miles= df.loc[df["Award?"] == "Bonus_trans"];
df_Days_since_enroll= df.loc[df["Award?"] == "Days_since_enroll"];
plt.plot(df_balance["Balance"], np.zeros_like(df_balance["Balance"]), 'o')
plt.plot(df_Bonus_miles["Bonus_miles"], np.zeros_like(df_Bonus_miles['Bonus_miles']), 'o')
plt.plot(df_Days_since_enroll["Days_since_enroll"], np.zeros_like(df_Days_since_enroll['Days_since_enroll']), 'o')

plt.show()
############################

#Plot CDF of Balance
import numpy as np 
counts, bin_edges = np.histogram(df_Bonus_miles['Bonus_miles'], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


counts, bin_edges = np.histogram(df_Bonus_miles['Bonus_miles'], bins=20, density = True)
pdf = counts/(sum(counts))
plt.plot(bin_edges[1:],pdf);

plt.show();
##############################
#first moment business descision

#Mean,  
print("Means:")
print(np.mean(df_balance["Balance"]))
#Mean with an outlier.
print(np.mean(np.append(df_balance["Balance"],50)));
print(np.mean(df_Bonus_miles["Bonus_miles"]))
print(np.mean(df_Bonus_miles["Bonus_miles"]))
#median
print("Median:")
print(np.median(df_balance["Balance"]))
#Mean with an outlier.
print(np.median(np.append(df_balance["Balance"],50)));
print(np.median(df_Bonus_miles["Bonus_miles"]))
print(np.median(df_Bonus_miles["Bonus_miles"]))
#mode
print("Mode:")
print(np.mode(df_balance["Balance"]))
#Mean with an outlier.
print(np.mode(np.append(df_balance["Balance"],50)));
print(np.mode(df_Bonus_miles["Bonus_miles"]))
print(np.mode(df_Bonus_miles["Bonus_miles"]))

#standard deviation
print("\nStd-dev:");
print(np.std(df_balance["Balance"]))
print(np.std(df_Bonus_miles["Bonus_miles"]))

############################################
#Box-plot can be visualized as a PDF on the side-ways.

sns.boxplot(x='Award?',y='Balance', data=df)
plt.show()
##################################
#outlier are occuurs
#box plot
sns.boxplot(df["Balance"])
##########################
#five number summary
df.describe()
##########################################################
'''********************data preparation*******'''
#1 type casting
import pandas as pd
#let us import the dataset
df=pd.read_excel("C:/Data Science/data_set/EastWestAirlines.xlsx") 
df
#check all the phase of EDA
#let check the shape of the dataset
df.shape
#check the column
df.columns

#check datatypes 
df.dtypes
'''ID#                  int64
Balance              int64
Qual_miles           int64
cc1_miles            int64
cc2_miles            int64
cc3_miles            int64
Bonus_miles          int64
Bonus_trans          int64
Flight_miles_12mo    int64
Flight_trans_12      int64
Days_since_enroll    int64
Award?               int64
dtype: object'''
#check is null
a=df.isnull()
a.sum()
'''Out[17]: 
ID#                  0
Balance              0
Qual_miles           0
cc1_miles            0
cc2_miles            0
cc3_miles            0
Bonus_miles          0
Bonus_trans          0
Flight_miles_12mo    0
Flight_trans_12      0
Days_since_enroll    0
Award?               0
dtype: int64'''
##########################################################################
#2 outlier treatement
import pandas as pd
import seaborn as sns

#let us import the dataset
df=pd.read_excel("C:/Data Science/data_set/EastWestAirlines.xlsx") 
df

#now let us find the outlier in the Balance column
sns.boxplot(df.Balance)
#there are outlier
#let us check there are outlier in the Bouns_miles column
sns.boxplot(df.Bonus_miles)
#there are no outlier
#we can calclaute the IQR

IQR=df.Balance.quantile(0.75)-df.Balance.quantile(0.25)

#have observation that the IQR in the variable explore
#no becaue the IQR are in the capaitalluze letter
#treated as constant
IQR
#Out[48]: 28359.945

#but if we will try as I,Iqr or iqr then it is showing
#I=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
#Iqr=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
#iqr=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)

lower_limit=df.Balance.quantile(0.25)-1.5*IQR    #make the lower limit value as 0 not the negative
lower_limit         #Out[51]: -19446.9675
upper_limit=df.Balance.quantile(0.75)-1.5*IQR 
upper_limit         #Out[52]: 8912.9775

#negative vaule are not the lowre limit so make it as 0
# for change go to the varible explore and make it as 0 directly
###############################################
#trimming
import numpy as np
outliers_df=np.where(df.Balance>upper_limit,True,np.where(df.Balance<lower_limit,True,False))
#you can check outlier_df column in the varible explore
#floting the point number,if possible in varible explore
#now trimm that
df_trimmed=df.loc[~outliers_df]
df_trimmed      #it can show the trimmed element
df.shape        #without the trimed shape is
#Out[60]: (310, 13)
df_trimmed.shape        #we trimmed this elemnt
#Out[61]: (34, 13)

################################################################
##repalcemet technique
#masking technique
#drawback of trimming is we can loosing the data
import pandas as pd
import seaborn as sns
import numpy as np
#let us import the dataset
df=pd.read_excel("C:/Data Science/data_set/EastWestAirlines.xlsx") 
df
df.describe()

#recored number 23 has got the outliers
# map all the outlier to the upper limit
df_replace=pd.DataFrame(np.where(df.Balance>upper_limit,upper_limit,np.where(df.Balance<lower_limit,lower_limit,df.Balance)))
df_replace

# if the value is lower than the lower limit ie is it has oulier
#so make it as the lower limit value  to that entry 
#if the value is greater than the upper limit ie is it has outlier
#so make it upper limit value to it 
#other wise make it as the same  for that columns
###################

sns.boxplot(df_replace[0])
#all the outiler are remove
###########################################################
#Winsorizer
#install the feature engine :- pip install feature_engine
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr',
                    tail='both',
                    fold=1.5,
                    variables=['Balance']
                    )
#copy Winsorizer and paste in help tab of
#top right Windows , study the method

df_t=winsor.fit_transform(df[['Balance']])
sns.boxplot(df['Balance'])
sns.boxplot(df_t['Balance'])
##############################
#variance
#zero and naer zero variance features
#if there are no variance in the feature , then ml model
#will not get any intellgence , so it is betttre  to ignore that feature

import pandas as pd
#let us import the dataset
df=pd.read_excel("C:/Data Science/data_set/EastWestAirlines.xlsx") 
df
df.var()

#######################
df.var()==0
###############################
#axis =0 then it is gonig to give the sam e results 
# if there are varienace then it give true 
df.var(axis=0)==0

#############################################
#4.missing value
# missing value
a= df.isnull()
#we can store the null value  in 'a' varible and we take the null value count from the 'a' varible
a.sum()
#there are no any missing value 
#############################################
#discretization
import pandas as pd
import numpy as np
data=pd.read_excel("C:/Data Science/data_set/EastWestAirlines.xlsx") 
data.head()     #show the first five record of the dataset/smaple data
data.head(10)  # 0 to 9 record are display
data.info()         #size of the data
#it gives size , null values,rows ,columns and datatype of the columns

data.describe()     #appllicabe fro numerical value  only
data['Balance_new']=pd.cut(data['Balance'],bins=[min(data.Balance),data.Balance.mean(),max(data.Balance)],labels=['low','High'])
data.Balance_new.value_counts()

#in the above code we can descriteze the data from the variou ppoint and mark them
#with the high value and low value  
# we can divide that data from the mena and label with them with the high value
# and more value from the mean are mark with the high value


data['Balance_new']=pd.cut(data['Balance'], bins=[min(data.Balance),data.Balance.quantile(0.25),data.Balance.mean(), data.Balance.quantile(0.75),max(data.Balance)], labels=['group1','group2','group3','group4'])
data.Balance_new.value_counts()

###############################################
#dummy variable
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("C:/Data Science/data_set/animal_category.csv")
df
#shape
df.shape
#Out[8]: (30, 5)
#drop 
df.drop(['Index'],axis=1, inplace=True)
#check of again
df_new=pd.get_dummies(df)
df_new.shape
#Out[12]: (30, 14)
#here we are getting 30 rows and 14 column
# we are getting two columns for homely and gender , one column is sufficient for each catogeriacl
#delete the second column of gender and second column of homely
df_new.drop(['Gender_Male', 'Homly_Yes'],axis=1,inplace=True)
df_new.shape
#Out[14]: (30, 12)
#now we are getting the 30,12
df_new.rename(columns={'Gender_female':'Gender','Homly_No':'house_no'})

###############################################################
#dummy varaible on the another dataset 
#do the same process for the ethnic diversity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_excel("C:/Data Science/data_set/EastWestAirlines.xlsx") 
df
df.dtypes
#shape
df.shape


df.drop(['ID#'],axis=1,inplace=True)
df.shape
#Out[24]: (310, 9)
#drop the column the dummy variable
df_new2=pd.get_dummies(df)
df_new2.shape
############################################
#one hot encoding


#################################
#label encoder


#################################
#standardization 
#normalization(x)=(x-xmin)/(xmax-xmin)

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
d=pd.read_excel("C:/Data Science/data_set/EastWestAirlines.xlsx")
d.describe()
a=d.describe()

#intialize the scalar
scalar=StandardScaler()
df=scalar.fit_transform(d)
dataset=pd.DataFrame(df)
res=dataset.describe()
#here if you will check res, in varaible explorer or in the vraible environment then
##########################################
#standardization 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
d=pd.read_excel("C:/Data Science/data_set/EastWestAirlines.xlsx")
d.describe()
a=d.describe()

#intialize the scalar
scalar=StandardScaler()
df=scalar.fit_transform(d)
dataset=pd.DataFrame(df)
res=dataset.describe()
#fromthe bove we can get the standardize data
###################################
#Normalization
import pandas as pd
ethnic=pd.read_excel("C:/Data Science/data_set/EastWestAirlines.xlsx")
ethnic
#now read the columns
ethnic.columns
#there are some column which are not useful , we need to drop
ethnic.drop(['ID#'],axis=1, inplace=True)
#now read minimum value and maximum value of salarie s and age
a1=ethnic.describe()
#check a1 data frame in the varible explorer
#you find minimum saralies is 0 and max is 108304
#same way you check for age, there is huge difference
#in min and max .value . Hence we are going for normalization
#first we will have to convert non-numerical data to label encoding'
ethnic =pd.get_dummies(ethnic, drop_first=True)
#Normalization function written where ethnic argument is passed
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_fun(ethnic)
b=df_norm.describe()
#error in the function are braces
#if you will observe the b frame
#it has dimension 8,81
#earlier in a they were 8,11 it is because all non_numeric
#data has been converted to numeric using label encoding

#######################################################################
'''***********model building***************'''
'''5.1 Build the model on the scaled data (try multiple options).
5.2 Perform the hierarchical clustering and visualize the clusters using dendrogram.
5.3 Validate the clusters (try with different number of clusters) – label the clusters and derive insights (compare the results from multiple approaches).
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#we are importing the  xcel file file 
df=pd.read_excel("C:/Data Science/data_set/EastWestAirlines.xlsx") 
df

#shape ot the dataframe
df.shape
#Out[7]: (3999, 12)

#column in the dataframe
df.columns
'''Index(['ID#', 'Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles',
       'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12',
       'Days_since_enroll', 'Award?'],
      dtype='object')'''

#description of the dataframe 
df.describe()
# we can store the above description in the one varible
a=df.describe()

# we are checking which column is not necserray or
# the column which is  numerical data  that can be place in the datframe
df1=df.drop(["ID#"],axis=1) 
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
plt.xlabel("miles");
plt.ylabel("flights")
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
df2['clust']=cluster_labels

########################################
df1.shape
#we want to realocate th ecolumnn 7 to 0 th postion
df=df2.iloc[:,[11,1,2,3,4,5,6,7,8,9,10]]
#now check the Univ1 Dataframe
df.iloc[:,2:].groupby(df.clust).mean()
#from the output cluster 2 has got highest top10
#lowest accept ratio , best faculty ratio and highest expenses
#highest graduates ratio
#######################
df.to_csv("Airline.csv",encoding="utf-8")
import os
os.getcwd()
#this file is created in  the working direcatory where we want to store 
# file as the working directory are shon on the above top most corner

