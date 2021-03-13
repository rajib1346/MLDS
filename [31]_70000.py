#!/usr/bin/env python
# coding: utf-8

# In[11]:


from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor 


# In[5]:


from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

#Data Set spilting
DataFrame=pd.read_csv("C:/Users/acer/Downloads/25/cardio_train.CSV")

x = DataFrame[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y = DataFrame[['target']]

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=1/8)

x_train['target']=y_train
x_test['target']=y_test

x_train.to_csv('C:/Users/acer/Downloads/25/Initial_Train4.csv')
x_test.to_csv('C:/Users/acer/Downloads/25/Initial_Test4.csv')


# In[12]:


from sklearn import tree
from sklearn.model_selection import cross_val_score


# In[26]:


DataFrame1=pd.read_csv("C:/Users/acer/Downloads/25/Initial_Train1.CSV")
DataFrame2=pd.read_csv("C:/Users/acer/Downloads/25/Initial_Test1.CSV")

x1 = DataFrame1[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y1 = DataFrame1[['target']]
x2 = DataFrame2[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y2 = DataFrame2[['target']]

DataFrame2=pd.read_csv("C:/Users/acer/Downloads/25/Initial_Train2.CSV")
DataFrame3=pd.read_csv("C:/Users/acer/Downloads/25/Initial_Test2.CSV")

x3 = DataFrame2[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y3 = DataFrame2[['target']]
x4 = DataFrame3[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y4 = DataFrame3[['target']]

DataFrame4=pd.read_csv("C:/Users/acer/Downloads/25/Initial_Train3.CSV")
DataFrame5=pd.read_csv("C:/Users/acer/Downloads/25/Initial_Test3.CSV")

x5 = DataFrame4[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y5 = DataFrame4[['target']]
x6 = DataFrame5[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y6 = DataFrame5[['target']]

DataFrame7=pd.read_csv("C:/Users/acer/Downloads/25/Initial_Train4.CSV")
DataFrame8=pd.read_csv("C:/Users/acer/Downloads/25/Initial_Test4.CSV")

x7 = DataFrame7[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y7 = DataFrame7[['target']]
x8 = DataFrame8[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y8 = DataFrame8[['target']]

dt.fit(x1,y1)
pre_1=dt.predict(x2)
pre_1=pre_1>0.5
acc_dt_1=metrics.accuracy_score(y2,pre_1)*100
print(acc_dt_1)

dt.fit(x3,y3)
pre_1=dt.predict(x4)
pre_1=pre_1>0.5
acc_dt_2=metrics.accuracy_score(y4,pre_1)*100
print(acc_dt_2)

dt.fit(x5,y5)
pre_1=dt.predict(x6)
pre_1=pre_1>0.5
acc_dt_3=metrics.accuracy_score(y6,pre_1)*100
print(acc_dt_3)

dt.fit(x7,y7)
pre_1=dt.predict(x8)
pre_1=pre_1>0.5
acc_dt_4=metrics.accuracy_score(y8,pre_1)*100
print(acc_dt_4)


# In[ ]:




