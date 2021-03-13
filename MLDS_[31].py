#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn import metrics


# In[2]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier


# In[3]:


data_frame=pd.read_csv("C:/Users/acer/Downloads/25/claveland.CSV")
data_frame


# In[4]:


data_frame.columns[data_frame.isnull().any()].tolist()


# In[5]:


import pandas as pd
import numpy as np

DataFrame=pd.read_csv("C:/Users/acer/Downloads/25/claveland.csv")
x = DataFrame[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
y = DataFrame[['target']]

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(6).plot(kind='barh')
plt.show()
feat_importances


# In[9]:


from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

#Data Set spilting
DataFrame=pd.read_csv("C:/Users/acer/Downloads/25/claveland.csv")

x = DataFrame[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
y = DataFrame[['target']]

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3,random_state=5)

x_train['target']=y_train
x_test['target']=y_test

x_train.to_csv('C:/Users/acer/Downloads/25/Initial_Train.csv')
x_test.to_csv('C:/Users/acer/Downloads/25/Initial_Test.csv')


# # 1st_iteraition

# In[12]:


train1=pd.read_csv("C:/Users/acer/Downloads/25/Train.CSV")
test1=pd.read_csv("C:/Users/acer/Downloads/25/1st_iteration/Test_1.csv")

x_train1 =train1[['cp','thalach','exang','oldpeak','ca','thal']]
y_train1 = train1[['target']]

x_test1 = test1[['cp','thalach','exang','oldpeak','ca','thal']]
y_test1 = test1[['target']]

random_forest=RandomForestClassifier(random_state=50)
random_forest.fit(x_train1,y_train1)
rf_1=random_forest.predict(x_test1)
acc_rf_1=metrics.accuracy_score(y_test1,rf_1)*100

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train1,y_train1)
Naive_1=Naive_bayes.predict(x_test1)
acc_naive_1=metrics.accuracy_score(y_test1,Naive_1)*100

gb = GradientBoostingClassifier(random_state=50)
gb.fit(x_train1, y_train1)
gb_1=gb.predict(x_test1)
acc_gb_1=metrics.accuracy_score(y_test1,gb_1)*100

print("Train_size:",len(train1),"| Test_size:",len(test1))
print("RF:",acc_rf_1)
print("Naivebayes:",acc_naive_1)
print("Gradient Boost:",acc_gb_1)
if (acc_rf_1 > acc_naive_1 and acc_rf_1 > acc_gb_1) or (acc_rf_1==acc_naive_1 and acc_rf_1>acc_gb_1) or (acc_rf_1==acc_gb_1 and acc_rf_1>acc_naive_1) or (acc_naive_1==acc_gb_1 and acc_rf_1>acc_naive_1):
    test1['rf_pre']=rf_1
    test1['match'] = np.where(test1['target'] == test1['rf_pre'], 'True', 'False')
    FP_FN_RF_1=test1[test1['match']=='False']
    FP_FN_RF_1=FP_FN_RF_1.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_1.to_csv('C:/Users/acer/Downloads/25/2nd_iteration/Test_2.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_1))
    
    TP_TN_RF_1=test1[test1['match']=='True']
    TP_TN_RF_1.to_csv('C:/Users/acer/Downloads/25/1st_iteration/1.CSV')
    TP_TN_RF_1=TP_TN_RF_1.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_1.to_csv('C:/Users/acer/Downloads/25/1st_iteration/TN_TP_1.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_1))

elif (acc_naive_1>acc_rf_1 and acc_naive_1>acc_gb_1) or (acc_rf_1==acc_naive_1 and acc_naive_1>acc_gb_1) or (acc_naive_1==acc_gb_1 and acc_naive_1>acc_rf_1) or (acc_rf_1==acc_gb_1 and acc_naive_1>acc_rf_1):  
    test1['nb_pre']=Naive_1      
    test1['match'] = np.where(test1['target'] == test1['nb_pre'], 'True', 'False')
    FP_FN_Na_1=test1[test1['match']=='False']
    FP_FN_Na_1=FP_FN_Na_1.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_1.to_csv('C:/Users/acer/Downloads/25/2nd_iteration/Test_2.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_1))
    
    TP_TN_Na_1=test1[test1['match']=='True']
    TP_TN_Na_1.to_csv('C:/Users/acer/Downloads/25/1st_iteration/1.CSV')
    TP_TN_Na_1=TP_TN_Na_1.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_1.to_csv('C:/Users/acer/Downloads/25/1st_iteration/TN_TP_1.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_1))
      
    
elif (acc_gb_1>acc_rf_1 and acc_gb_1>acc_naive_1) or (acc_rf_1==acc_gb_1 and acc_gb_1>acc_naive_1) or (acc_naive_1==acc_gb_1 and acc_gb_1>acc_rf_1) or (acc_rf_1==acc_naive_1 and acc_gb_1>acc_rf_1) :
    test1['gb_pre']=gb_1
    test1['match'] = np.where(test1['target'] == test1['gb_pre'], 'True', 'False')
    FP_FN_gb_1=test1[test1['match']=='False']
    FP_FN_gb_1=FP_FN_gb_1.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_1.to_csv('C:/Users/acer/Downloads/25/2nd_iteration/Test_2.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_1))
    
    TP_TN_gb_1=test1[test1['match']=='True']
    TP_TN_gb_1.to_csv('C:/Users/acer/Downloads/25/1st_iteration/1.CSV')
    TP_TN_gb_1=TP_TN_gb_1.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_1.to_csv('C:/Users/acer/Downloads/25/1st_iteration/TN_TP_1.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_1))

elif (acc_rf_1 != 0 and acc_naive_1!=0 and acc_gb_1!=0) and (acc_rf_1 == acc_naive_1 and acc_naive_1==acc_gb_1):
    test1['rf_pre']=rf_1
    test1['match'] = np.where(test1['target'] == test1['rf_pre'], 'True', 'False')
    FP_FN_RF_1=test1[test1['match']=='False']
    FP_FN_RF_1=FP_FN_RF_1.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_1.to_csv('C:/Users/acer/Downloads/25/2nd_iteration/Test_2.CSV')
    print("All Classifier Provides Same Accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_1))
    
    TP_TN_RF_1=test1[test1['match']=='True']
    TP_TN_RF_1.to_csv('C:/Users/acer/Downloads/25/1st_iteration/1.CSV')
    TP_TN_RF_1=TP_TN_RF_1.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_1.to_csv('C:/Users/acer/Downloads/25/1st_iteration/TN_TP_1.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_1))

else:
    print ("STOP")


# # 2nd_iteraition

# In[13]:


train2=pd.read_csv("C:/Users/acer/Downloads/25/Train.CSV")
test2=pd.read_csv("C:/Users/acer/Downloads/25/2nd_iteration/Test_2.csv")

x_train2 =train2[['cp','thalach','exang','oldpeak','ca','thal']]
y_train2 = train2[['target']]

x_test2 = test2[['cp','thalach','exang','oldpeak','ca','thal']]
y_test2 = test2[['target']]

random_forest=RandomForestClassifier(random_state=50)
random_forest.fit(x_train2,y_train2)
rf_2=random_forest.predict(x_test2)
acc_rf_2=metrics.accuracy_score(y_test2,rf_2)*100

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train2,y_train2)
Naive_2=Naive_bayes.predict(x_test2)
acc_naive_2=metrics.accuracy_score(y_test2,Naive_2)*100

gb = GradientBoostingClassifier(random_state=50)
gb.fit(x_train2, y_train2)
gb_2=gb.predict(x_test2)
acc_gb_2=metrics.accuracy_score(y_test2,gb_2)*100

print("Train_size:",len(train2),"| Test_size:",len(test2))
print("RF:",acc_rf_2)
print("Naivebayes:",acc_naive_2)
print("Gradient Boost:",acc_gb_2)
if (acc_rf_2 > acc_naive_2 and acc_rf_2 > acc_gb_2) or (acc_rf_2==acc_naive_2 and acc_rf_2>acc_gb_2) or (acc_rf_2==acc_gb_2 and acc_rf_2>acc_naive_2) or (acc_naive_2==acc_gb_2 and acc_rf_2>acc_naive_2):
    test2['rf_pre']=rf_2
    test2['match'] = np.where(test2['target'] == test2['rf_pre'], 'True', 'False')
    FP_FN_RF_2=test2[test2['match']=='False']
    FP_FN_RF_2=FP_FN_RF_2.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_2.to_csv('C:/Users/acer/Downloads/25/3rd_iteration/Test_3.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_2))
    
    TP_TN_RF_2=test2[test2['match']=='True']
    TP_TN_RF_2.to_csv('C:/Users/acer/Downloads/25/2st_iteration/2.CSV')
    TP_TN_RF_2=TP_TN_RF_2.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_2.to_csv('C:/Users/acer/Downloads/25/2nd_iteration/TN_TP_2.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_2))

elif (acc_naive_2>acc_rf_2 and acc_naive_2>acc_gb_2) or (acc_rf_2==acc_naive_2 and acc_naive_2>acc_gb_2) or (acc_naive_2==acc_gb_2 and acc_naive_2>acc_rf_2) or (acc_rf_2==acc_gb_2 and acc_naive_2>acc_rf_2):  
    test2['nb_pre']=Naive_2      
    test2['match'] = np.where(test2['target'] == test2['nb_pre'], 'True', 'False')
    FP_FN_Na_2=test2[test2['match']=='False']
    FP_FN_Na_2=FP_FN_Na_2.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_2.to_csv('C:/Users/acer/Downloads/25/3rd_iteration/Test_3.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_2))
    
    TP_TN_Na_2=test2[test2['match']=='True']
    TP_TN_Na_2.to_csv('C:/Users/acer/Downloads/25/2nd_iteration/2.CSV')
    TP_TN_Na_2=TP_TN_Na_2.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_2.to_csv('C:/Users/acer/Downloads/25/2nd_iteration/TN_TP_2.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_2))
      
    
elif (acc_gb_2>acc_rf_2 and acc_gb_2>acc_naive_2) or (acc_rf_2==acc_gb_2 and acc_gb_2>acc_naive_2) or (acc_naive_2==acc_gb_2 and acc_gb_2>acc_rf_2) or (acc_rf_2==acc_naive_2 and acc_gb_2>acc_rf_2) :
    test2['gb_pre']=gb_2
    test2['match'] = np.where(test2['target'] == test2['gb_pre'], 'True', 'False')
    FP_FN_gb_2=test2[test2['match']=='False']
    FP_FN_gb_2=FP_FN_gb_2.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_2.to_csv('C:/Users/acer/Downloads/25/3rd_iteration/Test_3.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_2))
    
    TP_TN_gb_2=test2[test2['match']=='True']
    TP_TN_gb_2.to_csv('C:/Users/acer/Downloads/25/2nd_iteration/2.CSV')
    TP_TN_gb_2=TP_TN_gb_2.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_2.to_csv('C:/Users/acer/Downloads/25/2nd_iteration/TN_TP_2.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_2))

elif (acc_rf_2 != 0 and acc_naive_2!=0 and acc_gb_2!=0) and (acc_rf_2 == acc_naive_2 and acc_naive_2==acc_gb_2):
    test2['rf_pre']=rf_2
    test2['match'] = np.where(test2['target'] == test2['rf_pre'], 'True', 'False')
    FP_FN_RF_2=test2[test2['match']=='False']
    FP_FN_RF_2=FP_FN_RF_2.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_2.to_csv('C:/Users/acer/Downloads/25/3rd_iteration/Test_3.CSV')
    print("All Classifier Provides Same Accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_2))
    
    TP_TN_RF_2=test2[test2['match']=='True']
    TP_TN_RF_2.to_csv('C:/Users/acer/Downloads/25/2st_iteration/2.CSV')
    TP_TN_RF_2=TP_TN_RF_2.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_2.to_csv('C:/Users/acer/Downloads/25/2nd_iteration/TN_TP_2.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_2))

else:
    print ("STOP")


# # 3rd_iteraition

# In[17]:


train3=pd.read_csv("C:/Users/acer/Downloads/25/Train.CSV")
test3=pd.read_csv("C:/Users/acer/Downloads/25/3rd_iteration/Test_3.csv")

x_train3 =train3[['cp','thalach','exang','oldpeak','ca','thal']]
y_train3 = train3[['target']]

x_test3 = test3[['cp','thalach','exang','oldpeak','ca','thal']]
y_test3 = test3[['target']]

random_forest=RandomForestClassifier(random_state=50)
random_forest.fit(x_train3,y_train3)
rf_3=random_forest.predict(x_test3)
acc_rf_3=metrics.accuracy_score(y_test3,rf_3)*100

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train3,y_train3)
Naive_3=Naive_bayes.predict(x_test3)
acc_naive_3=metrics.accuracy_score(y_test3,Naive_3)*100

gb = GradientBoostingClassifier(random_state=50)
gb.fit(x_train3, y_train3)
gb_3=gb.predict(x_test3)
acc_gb_3=metrics.accuracy_score(y_test3,gb_3)*100

print("Train_size:",len(train3),"| Test_size:",len(test3))
print("RF:",acc_rf_3)
print("Naivebayes:",acc_naive_3)
print("Gradient Boost:",acc_gb_3)
if (acc_rf_3 > acc_naive_3 and acc_rf_3 > acc_gb_3) or (acc_rf_3==acc_naive_3 and acc_rf_3>acc_gb_3) or (acc_rf_3==acc_gb_3 and acc_rf_3>acc_naive_3) or (acc_naive_3==acc_gb_3 and acc_rf_3>acc_naive_3):
    test3['rf_pre']=rf_3
    test3['match'] = np.where(test3['target'] == test3['rf_pre'], 'True', 'False')
    FP_FN_RF_3=test3[test3['match']=='False']
    FP_FN_RF_3=FP_FN_RF_3.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_3.to_csv('C:/Users/acer/Downloads/25/4th_iteration/Test_4.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_3))
    
    TP_TN_RF_3=test3[test3['match']=='True']
    TP_TN_RF_3.to_csv('C:/Users/acer/Downloads/25/3rd_iteration/3.CSV')
    TP_TN_RF_3=TP_TN_RF_3.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_3.to_csv('C:/Users/acer/Downloads/25/3rd_iteration/TN_TP_3.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_3))

elif (acc_naive_3>acc_rf_3 and acc_naive_3>acc_gb_3) or (acc_rf_3==acc_naive_3 and acc_naive_3>acc_gb_3) or (acc_naive_3==acc_gb_3 and acc_naive_3>acc_rf_3) or (acc_rf_3==acc_gb_3 and acc_naive_3>acc_rf_3):  
    test3['nb_pre']=Naive_3      
    test3['match'] = np.where(test3['target'] == test3['nb_pre'], 'True', 'False')
    FP_FN_Na_3=test3[test3['match']=='False']
    FP_FN_Na_3=FP_FN_Na_3.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_3.to_csv('C:/Users/acer/Downloads/25/4th_iteration/Test_4.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_3))
    
    TP_TN_Na_3=test3[test3['match']=='True']
    TP_TN_Na_3.to_csv('C:/Users/acer/Downloads/25/3rd_iteration/3.CSV')
    TP_TN_Na_3=TP_TN_Na_3.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_3.to_csv('C:/Users/acer/Downloads/25/3rd_iteration/TN_TP_3.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_3))
      
    
elif (acc_gb_3>acc_rf_3 and acc_gb_3>acc_naive_3) or (acc_rf_3==acc_gb_3 and acc_gb_3>acc_naive_3) or (acc_naive_3==acc_gb_3 and acc_gb_3>acc_rf_3) or (acc_rf_3==acc_naive_3 and acc_gb_3>acc_rf_3) :
    test3['gb_pre']=gb_3
    test3['match'] = np.where(test3['target'] == test3['gb_pre'], 'True', 'False')
    FP_FN_gb_3=test3[test3['match']=='False']
    FP_FN_gb_3=FP_FN_gb_3.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_3.to_csv('C:/Users/acer/Downloads/25/4th_iteration/Test_4.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_3))
    
    TP_TN_gb_3=test3[test3['match']=='True']
    TP_TN_gb_3.to_csv('C:/Users/acer/Downloads/25/3rd_iteration/3.CSV')
    TP_TN_gb_3=TP_TN_gb_3.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_3.to_csv('C:/Users/acer/Downloads/25/3rd_iteration/TN_TP_3.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_3))

elif (acc_rf_3 != 0 and acc_naive_3!=0 and acc_gb_3!=0) and (acc_rf_3 == acc_naive_3 and acc_naive_3==acc_gb_3):
    test3['rf_pre']=rf_3
    test3['match'] = np.where(test3['target'] == test3['rf_pre'], 'True', 'False')
    FP_FN_RF_3=test3[test3['match']=='False']
    FP_FN_RF_3=FP_FN_RF_3.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_3.to_csv('C:/Users/acer/Downloads/25/4th_iteration/Test_4.CSV')
    print("All Classifier Provides Same Accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_3))
    
    TP_TN_RF_3=test3[test3['match']=='True']
    TP_TN_RF_3.to_csv('C:/Users/acer/Downloads/25/3rd_iteration/3.CSV')
    TP_TN_RF_3=TP_TN_RF_3.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_3.to_csv('C:/Users/acer/Downloads/25/3rd_iteration/TN_TP_3.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_3))

else:
    print ("STOP")


# # Continuous

# In[ ]:




