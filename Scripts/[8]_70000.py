#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier


# In[2]:


data_frame=pd.read_csv("E:/85_15/Cardio_Data.CSV")
data_frame


# In[ ]:





# In[5]:


#mRmR
x1 = data_frame[['weight','smoke','ap_hi','ap_lo','height','gluc']]
y1 = data_frame[['target']]

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# In[15]:


nv_1=GaussianNB()
print('mRmR_nv_1=',cross_val_score(nv_1, x1, y1, cv=10, scoring ='accuracy').mean())


# In[12]:


lr_1=LogisticRegression(C=1)
print('mRmR_lr_1=',cross_val_score(lr_1, x1, y1, cv=10, scoring ='accuracy').mean())
lr_2=LogisticRegression(C=10)
print('mRmR_lr_2=',cross_val_score(lr_2, x1, y1, cv=10, scoring ='accuracy').mean())
lr_3=LogisticRegression(C=100)
print('mRmR_lr_3=',cross_val_score(lr_3, x1, y1, cv=10, scoring ='accuracy').mean())


# In[70]:


svm_1 = svm.SVC(kernel ='rbf',C=100)
print('mRmR_svm_1',cross_val_score(svm_1, x1, y1, cv=10, scoring ='accuracy').mean())
svm_2 = svm.SVC(kernel ='rbf',C=10)
print('mRmR_svm_2',cross_val_score(svm_2, x1, y1, cv=10, scoring ='accuracy').mean())
svm_3 = svm.SVC(kernel ='linear',C=10)
print('mRmR_lr_3',cross_val_score(svm_3, x1, y1, cv=10, scoring ='accuracy').mean())
svm_4 = svm.SVC(kernel ='linear',C=100)
print('mRmR_svm_4',cross_val_score(svm_4, x1, y1, cv=10, scoring ='accuracy').mean())


# In[49]:


dt_1=tree.DecisionTreeClassifier(100)
print('mRmR_dt_1',cross_val_score(dt_1, x1, y1, cv=10, scoring ='accuracy').mean())
dt_2=tree.DecisionTreeClassifier(50)
print('mRmR_dt_2',cross_val_score(dt_2, x1, y1, cv=10, scoring ='accuracy').mean())


# In[13]:


rf_1=RandomForestClassifier(50)
print('mRmR_rf_1',cross_val_score(rf_1, x1, y1, cv=10, scoring ='accuracy').mean())
rf_2=RandomForestClassifier(100)
print('mRmR_rf_2',cross_val_score(rf_2, x1, y1, cv=10, scoring ='accuracy').mean())


# In[14]:


kn_1 =KNeighborsClassifier(n_neighbors = 1)
print('mRmR_knn_1',cross_val_score(kn_1, x1, y1, cv=10, scoring ='accuracy').mean())
kn_2 =KNeighborsClassifier(n_neighbors = 3)
print('mRmR_knn_2',cross_val_score(kn_1, x1, y1, cv=10, scoring ='accuracy').mean())
kn_3 =KNeighborsClassifier(n_neighbors = 7)
print('mRmR_knn_3',cross_val_score(kn_1, x1, y1, cv=10, scoring ='accuracy').mean())


# In[6]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

def built_classfier():
    classifier = Sequential()
    classifier.add(Dense(output_dim=4,init='uniform',activation='relu',input_dim=6))
    classifier.add(Dense(output_dim=4,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,activation='sigmodoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classfier
classifier=KerasClassifier(build_fn=built_classfier,batch_size=100,epochs=100)
print('mRmR_ann_1',cross_val_score(estimator=classifier,X=x1,y=y1,cv=10))


# In[7]:


x2 =data_frame[['age','weight','ap_hi','height','ap_lo','cholesterol']]
y2 = data_frame[['target']]


# In[22]:


lr_1=LogisticRegression(C=1)
print('re_lr_1=',cross_val_score(lr_1, x2, y2, cv=10, scoring ='accuracy').mean())
lr_2=LogisticRegression(C=10)
print('re_lr_2=',cross_val_score(lr_2, x2, y2, cv=10, scoring ='accuracy').mean())
lr_3=LogisticRegression(C=100)
print('re_lr_3=',cross_val_score(lr_3, x2, y2, cv=10, scoring ='accuracy').mean())
lr_4=LogisticRegression(C=0.001)
print('re_lr_3=',cross_val_score(lr_4, x2, y2, cv=10, scoring ='accuracy').mean())


# In[77]:


kn_1 =KNeighborsClassifier(n_neighbors = 1)
print('re_knn_1',cross_val_score(kn_1, x2, y2, cv=10, scoring ='accuracy').mean())
kn_2 =KNeighborsClassifier(n_neighbors = 3)
print('re_knn_2',cross_val_score(kn_2, x2, y2, cv=10, scoring ='accuracy').mean())
kn_3 =KNeighborsClassifier(n_neighbors = 7)
print('re_knn_3',cross_val_score(kn_3, x2, y2, cv=10, scoring ='accuracy').mean())
kn_4 =KNeighborsClassifier(n_neighbors = 9)
print('re_knn_3',cross_val_score(kn_4, x2, y2, cv=10, scoring ='accuracy').mean())
kn_5 =KNeighborsClassifier(n_neighbors = 13)
print('re_knn_3',cross_val_score(kn_5, x2, y2, cv=10, scoring ='accuracy').mean())


# In[ ]:


svm_1 = svm.SVC(kernel ='rbf',C=100)
print('re_svm_1',cross_val_score(svm_1, x2, y2, cv=10, scoring ='accuracy').mean())
svm_2 = svm.SVC(kernel ='rbf',C=10)
print('re_svm_2',cross_val_score(svm_2, x2, y2, cv=10, scoring ='accuracy').mean())
svm_3 = svm.SVC(kernel ='linear',C=10)
print('re_lr_3',cross_val_score(svm_3, x2, y2, cv=10, scoring ='accuracy').mean())
svm_4 = svm.SVC(kernel ='linear',C=100)
print('re_svm_4',cross_val_score(svm_4, x2, y2, cv=10, scoring ='accuracy').mean())
svm_5 = svm.SVC(kernel ='rbf',C=1)
print('re_svm_2',cross_val_score(svm_5, x2, y2, cv=10, scoring ='accuracy').mean())


# In[23]:


nv_1=GaussianNB()
print('re_nv_1=',cross_val_score(nv_1, x2, y2, cv=10, scoring ='accuracy').mean())


# In[24]:


dt_1=tree.DecisionTreeClassifier(100)
print('re_dt_1',cross_val_score(dt_1, x2, y2, cv=10, scoring ='accuracy').mean())
dt_2=tree.DecisionTreeClassifier(500)
print('re_dt_2',cross_val_score(dt_2, x2, y2, cv=10, scoring ='accuracy').mean())


# In[ ]:


rf_1=RandomForestClassifier(100)
print('re_rf_1',cross_val_score(rf_1, x2, y2, cv=10, scoring ='accuracy').mean())
rf_2=RandomForestClassifier(50)
print('re_rf_2',cross_val_score(rf_2, x2, y2, cv=10, scoring ='accuracy').mean())
rf_3=RandomForestClassifier(20)
print('re_rf_3',cross_val_score(rf_3, x2, y2, cv=10, scoring ='accuracy').mean())


# In[8]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

def built_classfier():
    classifier = Sequential()
    classifier.add(Dense(output_dim=4,init='uniform',activation='relu',input_dim=6))
    classifier.add(Dense(output_dim=4,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,activation='sigmodoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classfier
classifier=KerasClassifier(build_fn=built_classfier,batch_size=100,epochs=100)
print('mRmR_ann_1',cross_val_score(estimator=classifier,X=x2,y=y2,cv=10))


# In[10]:


#Lasso
data_frame1=pd.read_csv("E:/85_15/Cardio_Data.CSV")
x3 =data_frame1[['age','gluc','ap_hi','weight','ap_lo','cholesterol']]
y3 = data_frame1[['target']]


# In[14]:


lr_1=LogisticRegression(C=1)
print('re_lr_1=',cross_val_score(lr_1, x3, y3, cv=10, scoring ='accuracy').mean())
lr_2=LogisticRegression(C=10)
print('re_lr_2=',cross_val_score(lr_2, x3, y3, cv=10, scoring ='accuracy').mean())
lr_3=LogisticRegression(C=100)
print('re_lr_3=',cross_val_score(lr_3, x3, y3, cv=10, scoring ='accuracy').mean())
lr_4=LogisticRegression(C=0.001)
print('re_lr_3=',cross_val_score(lr_4, x3, y3, cv=10, scoring ='accuracy').mean())


# In[15]:


kn_1 =KNeighborsClassifier(n_neighbors = 1)
print('re_knn_1',cross_val_score(kn_1, x3, y3, cv=10, scoring ='accuracy').mean())
kn_2 =KNeighborsClassifier(n_neighbors = 3)
print('re_knn_2',cross_val_score(kn_2, x3, y3, cv=10, scoring ='accuracy').mean())
kn_3 =KNeighborsClassifier(n_neighbors = 7)
print('re_knn_3',cross_val_score(kn_3, x3, y3, cv=10, scoring ='accuracy').mean())
kn_4 =KNeighborsClassifier(n_neighbors = 9)
print('re_knn_3',cross_val_score(kn_4, x3, y3, cv=10, scoring ='accuracy').mean())
kn_5 =KNeighborsClassifier(n_neighbors = 13)
print('re_knn_3',cross_val_score(kn_5, x3, y3, cv=10, scoring ='accuracy').mean())


# In[ ]:


svm_1 = svm.SVC(kernel ='rbf',C=100)
print('re_svm_1',cross_val_score(svm_1, x3, y3, cv=10, scoring ='accuracy').mean())
svm_2 = svm.SVC(kernel ='rbf',C=10)
print('re_svm_2',cross_val_score(svm_2, x3, y3, cv=10, scoring ='accuracy').mean())
svm_3 = svm.SVC(kernel ='linear',C=10)
print('re_lr_3',cross_val_score(svm_3, x3, y3, cv=10, scoring ='accuracy').mean())
svm_4 = svm.SVC(kernel ='linear',C=100)
print('re_svm_4',cross_val_score(svm_4, x3, y3, cv=10, scoring ='accuracy').mean())
svm_5 = svm.SVC(kernel ='rbf',C=1)
print('re_svm_2',cross_val_score(svm_5, x3, y3, cv=10, scoring ='accuracy').mean())


# In[20]:


nv_1=GaussianNB()
print('re_nv_1=',cross_val_score(nv_1, x3, y3, cv=10, scoring ='accuracy').mean())


# In[19]:


dt_1=tree.DecisionTreeClassifier(100)
print('re_dt_1',cross_val_score(dt_1, x3, y3, cv=10, scoring ='accuracy').mean())
dt_2=tree.DecisionTreeClassifier(500)
print('re_dt_2',cross_val_score(dt_2, x3, y3, cv=10, scoring ='accuracy').mean())


# In[18]:


rf_1=RandomForestClassifier(100)
print('re_rf_1',cross_val_score(rf_1, x3, y3, cv=10, scoring ='accuracy').mean())
rf_2=RandomForestClassifier(50)
print('re_rf_2',cross_val_score(rf_2, x3, y3, cv=10, scoring ='accuracy').mean())
rf_3=RandomForestClassifier(20)
print('re_rf_3',cross_val_score(rf_3, x3, y3, cv=10, scoring ='accuracy').mean())


# In[11]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

def built_classfier():
    classifier = Sequential()
    classifier.add(Dense(output_dim=4,init='uniform',activation='relu',input_dim=6))
    classifier.add(Dense(output_dim=4,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,activation='sigmodoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classfier
classifier=KerasClassifier(build_fn=built_classfier,batch_size=100,epochs=100)
print('mRmR_ann_1',cross_val_score(estimator=classifier,X=x3,y=y3,cv=10))


# In[ ]:




