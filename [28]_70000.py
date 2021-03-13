#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn import metrics


# In[27]:


from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


# In[21]:


DataFrame1=pd.read_csv("C:/Users/acer/Downloads/17/mean_17_(70000)/Initial_Train.CSV")
DataFrame2=pd.read_csv("C:/Users/acer/Downloads/17/mean_17_(70000)/Initial_Test.CSV")

x_train = DataFrame1[['age','weight','ap_hi','height','ap_lo','cholesterol']]
y_train = DataFrame1[['target']]
x_test = DataFrame2[['age','weight','ap_hi','height','ap_lo','cholesterol']]
y_test = DataFrame2[['target']]


# In[9]:


nv = GaussianNB()
nv.fit(x_train,y_train)
pre_nv=nv.predict(x_test)
acc_nv=metrics.accuracy_score(y_test,pre_nv)*100
print('Accuracu_nv= \n',acc_nv)


# In[30]:


rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model =KNeighborsClassifier(n_neighbors = K)
    model.fit(x_train, y_train) 
    pred=model.predict(x_test) 
    print('Accuracy for k= ' , K , 'is:',metrics.accuracy_score(y_test,pred)*100)


# In[31]:


svc_rbf = svm.SVC(kernel ='rbf')
svc_rbf.fit(x_train,y_train)
pre_rbf=svc_rbf.predict(x_test)
acc_sv_rbf=metrics.accuracy_score(y_test,pre_rbf)*100
print('Accuracu_svm_rbf= \n',acc_sv_rbf)


# In[32]:


svc_li = svm.SVC(kernel ='linear')
svc_li.fit(x_train,y_train)
pre_li=svc_li.predict(x_test)
acc_sv_li=metrics.accuracy_score(y_test,pre_li)*100
print('Accuracu_svm_li= \n',acc_sv_li)


# In[ ]:




