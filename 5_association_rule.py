# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:27:15 2023

@author: suraj
"""
'''Asscoiation Rule day 7/11/23'''
#pip install mlxtend

from mlxtend.frequent_patterns import apriori ,association_rules
# here we are going to use transcational data where in size of each row is not consistent
# we can not use pandas to load this unstructured data
# here function called open( ) is used
# create an empty list

groceries=[]

with open("C:/Data Science/data_set/groceries.csv") as f:groceries=f.read()
#splilting the data into separet transaction using separeator , it is comma
#we can use new line charater "\n"

groceries=groceries.split("\n")
#ealier groceries datastructure was in string format , now it will change to
#9836 , each item is comma separeted
# we will have to separate out each item from each transaction
groceries_list=[]
for i in groceries:
    groceries_list.append(i.split(","))

#split function will separted each item from each list, wherever it will find
# in order to generate asscocitaion rules, you can directly use groceries_list
# now let us separate out each item from the groceries list
all_groceries_list=[i for item in groceries_list for i in item]
#you will get all the items occured in all tranasaction
#we will get 43368 items in various transaction
 
# now let us count the frequency of each item
# we will import collection package will has counter function which will create
from collections import Counter
item_frequencies= Counter(all_groceries_list)

#item_frequencies os basillcaly dictionary having x[0] as key and x[1]=values
# we want to access values and sort based on the count that occured in it.
# it will show the count of each item purched in eavery transcation
# now let us sort these frequencies in ascendiing order
item_frequencies=sorted(item_frequencies.items(),key=lambda x:x[1])
#when we excute this item frequencies will be in sorted form in the form
#item name with count
# let us separete out items and their count

items=list(reversed([i[0] for i in item_frequencies]))
#this is list comprehension for each item in item frequencies access the key 
#here we will get item list
frequencies=list(reversed([i[1] for i in item_frequencies]))
# here you will get count of purchases of each item

# now let us plot bar graph of item frequencies
import matplotlib.pyplot as plt
# here we are taking frequencies from zero to 11 you can try 0-15 ar any otherwise
plt.bar(height=frequencies[0:11], x=list(range(0,11)))
plt.xticks(list(range(0,11)),items[0:11])

#plt.xticks you can specify a rotaation for the tick
# labels in degrees or with keywords

plt.xlabel("items")
plt.ylabel("count")
plt.show()
import pandas as pd
#now let us try to establish asscocitaion rule mining
# we have groceries list in the list format we need to convert it in dataframe
groceries_series= pd.DataFrame(pd.Series(groceries_list))
# now we will get the dataframe  of size 9831x1 size , columns compries of multiple items
#we had extra row created , check the groceries _series , last row is empty , let us first delete it
groceries_series= groceries_series.iloc[:9835,:]
# we hava taken rows from 0 to 9834 and columns 0 to all
# groceries series has column having name 0, let us rename as transcation
groceries_series.columns=["Transcation"]
# now we will have to apply 1- hot encoding , before  that in
# on columns  there are various items separated by ',
#',' let us separete it with"*"
x=groceries_series['Transcation'].str.join(sep='*')
# check the x in varibal explore which has * separetor rather the','
x=x.str.get_dummies(sep='*')
# now will get one hot encoding  dataframe of size 9835x169
# this is our input data to apply to apriori algorithum , it wi generate 
# is 0.0075(it must be between o to 1) , you can give any number  but must be
# between  0 and 1
frequent_itemsets= apriori(x, min_support=0.0075, max_len=4, use_colnames= True)
# you will get support value s for 1,2,3and 4 max items
# let us sort these supporetd values
frequent_itemsets.sort_values('support',ascending=False, inplace=True)
# supported values will be sorted  descending order
# Even EDA was also have the same trend in , EDA there was count
# and  here it is supported value
# we will generate asscocition rules, this association
# rule e=will calculate all the matrix
# of each and every combination
rules=association_rules(frequent_itemsets, metric='lift', min_threshold=1)
# this generate asscocition rules of size 1198x9 columns
#comprizes of antescends , consequences
rules.head(20)
rules.sort_values('lift',ascending=False).head(10)


