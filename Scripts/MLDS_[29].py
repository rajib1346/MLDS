#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn import metrics


# In[7]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier


# In[28]:


data_frame=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/Cleveland.CSV")
data_frame


# In[30]:


data_frame.columns[data_frame.isnull().any()].tolist()


# In[29]:


data_frame.isnull().sum()


# In[37]:


process=data_frame.dropna()
process.to_csv("C:/Users/acer/Downloads/22/MLPS(303)/processed_data.csv")


# In[60]:


import pandas as pd
import numpy as np

DataFrame=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/processed_data.csv")
x = DataFrame[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
y = DataFrame[['target']]

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(12).plot(kind='barh')
plt.show()
feat_importances


# In[59]:


from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

#Data Set spilting
DataFrame=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/processed_data.csv")

x = DataFrame[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
y = DataFrame[['target']]

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3,random_state=5)

x_train['target']=y_train
x_test['target']=y_test

x_train.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/Initial_Train.csv')
x_test.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/Initial_Test.csv')


# # 1st_iteraition

# In[87]:


train1=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/Train.CSV")
test1=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/1st_iteration/Test_1.csv")

x_train1 =train1[['age','cp','trestbps','chol','thalach','exang','oldpeak','slope','ca','thal']]
y_train1 = train1[['target']]

x_test1 = test1[['age','cp','trestbps','chol','thalach','exang','oldpeak','slope','ca','thal']]
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
    FP_FN_RF_1.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/2nd_iteration/Test_2.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_1))
    
    TP_TN_RF_1=test1[test1['match']=='True']
    TP_TN_RF_1.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/1st_iteration/1.CSV')
    TP_TN_RF_1=TP_TN_RF_1.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_1.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/1st_iteration/TN_TP_1.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_1))

elif (acc_naive_1>acc_rf_1 and acc_naive_1>acc_gb_1) or (acc_rf_1==acc_naive_1 and acc_naive_1>acc_gb_1) or (acc_naive_1==acc_gb_1 and acc_naive_1>acc_rf_1) or (acc_rf_1==acc_gb_1 and acc_naive_1>acc_rf_1):  
    test1['nb_pre']=Naive_1      
    test1['match'] = np.where(test1['target'] == test1['nb_pre'], 'True', 'False')
    FP_FN_Na_1=test1[test1['match']=='False']
    FP_FN_Na_1=FP_FN_Na_1.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_1.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/2nd_iteration/Test_2.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_1))
    
    TP_TN_Na_1=test1[test1['match']=='True']
    TP_TN_Na_1.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/1st_iteration/1.CSV')
    TP_TN_Na_1=TP_TN_Na_1.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_1.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/1st_iteration/TN_TP_1.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_1))
      
    
elif (acc_gb_1>acc_rf_1 and acc_gb_1>acc_naive_1) or (acc_rf_1==acc_gb_1 and acc_gb_1>acc_naive_1) or (acc_naive_1==acc_gb_1 and acc_gb_1>acc_rf_1) or (acc_rf_1==acc_naive_1 and acc_gb_1>acc_rf_1) :
    test1['gb_pre']=gb_1
    test1['match'] = np.where(test1['target'] == test1['gb_pre'], 'True', 'False')
    FP_FN_gb_1=test1[test1['match']=='False']
    FP_FN_gb_1=FP_FN_gb_1.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_1.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/2nd_iteration/Test_2.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_1))
    
    TP_TN_gb_1=test1[test1['match']=='True']
    TP_TN_gb_1.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/1st_iteration/1.CSV')
    TP_TN_gb_1=TP_TN_gb_1.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_1.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/1st_iteration/TN_TP_1.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_1))

elif (acc_rf_1 != 0 and acc_naive_1!=0 and acc_gb_1!=0) and (acc_rf_1 == acc_naive_1 and acc_naive_1==acc_gb_1):
    test1['rf_pre']=rf_1
    test1['match'] = np.where(test1['target'] == test1['rf_pre'], 'True', 'False')
    FP_FN_RF_1=test1[test1['match']=='False']
    FP_FN_RF_1=FP_FN_RF_1.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_1.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/2nd_iteration/Test_2.CSV')
    print("All Classifier Provides Same Accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_1))
    
    TP_TN_RF_1=test1[test1['match']=='True']
    TP_TN_RF_1.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/1st_iteration/1.CSV')
    TP_TN_RF_1=TP_TN_RF_1.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_1.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/1st_iteration/TN_TP_1.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_1))

else:
    print ("STOP")


# # 2nd_iteraition

# In[88]:


train2=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/Train.CSV")
test2=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/2nd_iteration/Test_2.csv")

x_train2 =train2[['age','cp','trestbps','chol','thalach','exang','oldpeak','slope','ca','thal']]
y_train2 = train2[['target']]

x_test2 = test2[['age','cp','trestbps','chol','thalach','exang','oldpeak','slope','ca','thal']]
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
    FP_FN_RF_2.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/3rd_iteration/Test_3.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_2))
    
    TP_TN_RF_2=test2[test2['match']=='True']
    TP_TN_RF_2.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/2nd_iteration/2.CSV')
    TP_TN_RF_2=TP_TN_RF_2.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_2.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/2nd_iteration/TN_TP_2.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_2))

elif (acc_naive_2>acc_rf_2 and acc_naive_2>acc_gb_2) or (acc_rf_2==acc_naive_2 and acc_naive_2>acc_gb_2) or (acc_naive_2==acc_gb_2 and acc_naive_2>acc_rf_2) or (acc_rf_2==acc_gb_2 and acc_naive_2>acc_rf_2):  
    test2['nb_pre']=Naive_2      
    test2['match'] = np.where(test2['target'] == test2['nb_pre'], 'True', 'False')
    FP_FN_Na_2=test2[test2['match']=='False']
    FP_FN_Na_2=FP_FN_Na_2.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_2.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/3rd_iteration/Test_3.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_2))
    
    TP_TN_Na_2=test2[test2['match']=='True']
    TP_TN_Na_2.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/2nd_iteration/2.CSV')
    TP_TN_Na_2=TP_TN_Na_2.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_2.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/2nd_iteration/TN_TP_2.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_2))
      
    
elif (acc_gb_2>acc_rf_2 and acc_gb_2>acc_naive_2) or (acc_rf_2==acc_gb_2 and acc_gb_2>acc_naive_2) or (acc_naive_2==acc_gb_2 and acc_gb_2>acc_rf_2) or (acc_rf_2==acc_naive_2 and acc_gb_2>acc_rf_2) :
    test2['gb_pre']=gb_2
    test2['match'] = np.where(test2['target'] == test2['gb_pre'], 'True', 'False')
    FP_FN_gb_2=test2[test2['match']=='False']
    FP_FN_gb_2=FP_FN_gb_2.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_2.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/3rd_iteration/Test_3.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_2))
    
    TP_TN_gb_2=test2[test2['match']=='True']
    TP_TN_gb_2.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/2nd_iteration/2.CSV')
    TP_TN_gb_2=TP_TN_gb_2.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_2.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/2nd_iteration/TN_TP_2.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_2))

elif (acc_rf_2 != 0 and acc_naive_2!=0 and acc_gb_2!=0) and (acc_rf_2 == acc_naive_2 and acc_naive_2==acc_gb_2):
    test2['rf_pre']=rf_2
    test2['match'] = np.where(test2['target'] == test2['rf_pre'], 'True', 'False')
    FP_FN_RF_2=test2[test2['match']=='False']
    FP_FN_RF_2=FP_FN_RF_2.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_2.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/3rd_iteration/Test_3.CSV')
    print("All Classifier Provides Same Accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_2))
    
    TP_TN_RF_2=test2[test2['match']=='True']
    TP_TN_RF_2.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/2nd_iteration/2.CSV')
    TP_TN_RF_2=TP_TN_RF_2.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_2.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/2nd_iteration/TN_TP_2.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_2))

else:
    print ("STOP")


# # 3rd_iteraition

# In[95]:


train3=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/Train.CSV")
test3=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/3rd_iteration/Test_3.csv")

x_train3 =train3[['age','cp','trestbps','chol','thalach','exang','oldpeak','slope','ca','thal']]
y_train3 = train3[['target']]

x_test3 = test3[['age','cp','trestbps','chol','thalach','exang','oldpeak','slope','ca','thal']]
y_test3 = test3[['target']]

random_forest=RandomForestClassifier(random_state=20)
random_forest.fit(x_train3,y_train3)
rf_3=random_forest.predict(x_test3)
acc_rf_3=metrics.accuracy_score(y_test3,rf_3)*100

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train3,y_train3)
Naive_3=Naive_bayes.predict(x_test3)
acc_naive_3=metrics.accuracy_score(y_test3,Naive_3)*100

gb = GradientBoostingClassifier(random_state=20)
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
    FP_FN_RF_3.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/4th_iteration/Test_4.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_3))
    
    TP_TN_RF_3=test3[test3['match']=='True']
    TP_TN_RF_3.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/3rd_iteration/3.CSV')
    TP_TN_RF_3=TP_TN_RF_3.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_3.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/3rd_iteration/TN_TP_3.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_3))

elif (acc_naive_3>acc_rf_3 and acc_naive_3>acc_gb_3) or (acc_rf_3==acc_naive_3 and acc_naive_3>acc_gb_3) or (acc_naive_3==acc_gb_3 and acc_naive_3>acc_rf_3) or (acc_rf_3==acc_gb_3 and acc_naive_3>acc_rf_3):  
    test3['nb_pre']=Naive_3      
    test3['match'] = np.where(test3['target'] == test3['nb_pre'], 'True', 'False')
    FP_FN_Na_3=test3[test3['match']=='False']
    FP_FN_Na_3=FP_FN_Na_3.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_3.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/4th_iteration/Test_4.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_3))
    
    TP_TN_Na_3=test3[test3['match']=='True']
    TP_TN_Na_3.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/3rd_iteration/3.CSV')
    TP_TN_Na_3=TP_TN_Na_3.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_3.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/3rd_iteration/TN_TP_3.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_3))
      
    
elif (acc_gb_3>acc_rf_3 and acc_gb_3>acc_naive_3) or (acc_rf_3==acc_gb_3 and acc_gb_3>acc_naive_3) or (acc_naive_3==acc_gb_3 and acc_gb_3>acc_rf_3) or (acc_rf_3==acc_naive_3 and acc_gb_3>acc_rf_3) :
    test3['gb_pre']=gb_3
    test3['match'] = np.where(test3['target'] == test3['gb_pre'], 'True', 'False')
    FP_FN_gb_3=test3[test3['match']=='False']
    FP_FN_gb_3=FP_FN_gb_3.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_3.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/4th_iteration/Test_4.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_3))
    
    TP_TN_gb_3=test3[test3['match']=='True']
    TP_TN_gb_3.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/3rd_iteration/3.CSV')
    TP_TN_gb_3=TP_TN_gb_3.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_3.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/3rd_iteration/TN_TP_3.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_3))

elif (acc_rf_3 != 0 and acc_naive_3!=0 and acc_gb_3!=0) and (acc_rf_3 == acc_naive_3 and acc_naive_3==acc_gb_3):
    test3['rf_pre']=rf_3
    test3['match'] = np.where(test3['target'] == test3['rf_pre'], 'True', 'False')
    FP_FN_RF_3=test3[test3['match']=='False']
    FP_FN_RF_3=FP_FN_RF_3.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_3.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/4th_iteration/Test_4.CSV')
    print("All Classifier Provides Same Accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_3))
    
    TP_TN_RF_3=test3[test3['match']=='True']
    TP_TN_RF_3.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/3rd_iteration/3.CSV')
    TP_TN_RF_3=TP_TN_RF_3.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_3.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/3rd_iteration/TN_TP_3.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_3))

else:
    print ("STOP")


# # 4th_iteraition

# In[100]:


train4=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/Train.CSV")
test4=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/4th_iteration/Test_4.csv")

x_train4 =train4[['age','cp','trestbps','chol','thalach','exang','oldpeak','slope','ca','thal']]
y_train4 = train4[['target']]

x_test4 = test4[['age','cp','trestbps','chol','thalach','exang','oldpeak','slope','ca','thal']]
y_test4 = test4[['target']]

random_forest=RandomForestClassifier(random_state=40)
random_forest.fit(x_train4,y_train4)
rf_4=random_forest.predict(x_test4)
acc_rf_4=metrics.accuracy_score(y_test4,rf_4)*100

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train4,y_train4)
Naive_4=Naive_bayes.predict(x_test4)
acc_naive_4=metrics.accuracy_score(y_test4,Naive_4)*100

gb = GradientBoostingClassifier(random_state=40)
gb.fit(x_train4, y_train4)
gb_4=gb.predict(x_test4)
acc_gb_4=metrics.accuracy_score(y_test4,gb_4)*100

print("Train_size:",len(train4),"| Test_size:",len(test4))
print("RF:",acc_rf_4)
print("Naivebayes:",acc_naive_4)
print("Gradient Boost:",acc_gb_4)
if (acc_rf_4 > acc_naive_4 and acc_rf_4 > acc_gb_4) or (acc_rf_4==acc_naive_4 and acc_rf_4>acc_gb_4) or (acc_rf_4==acc_gb_4 and acc_rf_4>acc_naive_4) or (acc_naive_4==acc_gb_4 and acc_rf_4>acc_naive_4):
    test4['rf_pre']=rf_4
    test4['match'] = np.where(test4['target'] == test4['rf_pre'], 'True', 'False')
    FP_FN_RF_4=test4[test4['match']=='False']
    FP_FN_RF_4=FP_FN_RF_4.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_4.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/5th_iteration/Test_5.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_4))
    
    TP_TN_RF_4=test4[test4['match']=='True']
    TP_TN_RF_4.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/4th_iteration/4.CSV')
    TP_TN_RF_4=TP_TN_RF_4.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_4.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/4th_iteration/TN_TP_4.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_4))

elif (acc_naive_4>acc_rf_4 and acc_naive_4>acc_gb_4) or (acc_rf_4==acc_naive_4 and acc_naive_4>acc_gb_4) or (acc_naive_4==acc_gb_4 and acc_naive_4>acc_rf_4) or (acc_rf_4==acc_gb_4 and acc_naive_4>acc_rf_4):  
    test4['nb_pre']=Naive_4      
    test4['match'] = np.where(test4['target'] == test4['nb_pre'], 'True', 'False')
    FP_FN_Na_4=test4[test4['match']=='False']
    FP_FN_Na_4=FP_FN_Na_4.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_4.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/5th_iteration/Test_5.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_4))
    
    TP_TN_Na_4=test4[test4['match']=='True']
    TP_TN_Na_4.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/4th_iteration/4.CSV')
    TP_TN_Na_4=TP_TN_Na_4.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_4.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/4th_iteration/TN_TP_4.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_4))
      
    
elif (acc_gb_4>acc_rf_4 and acc_gb_4>acc_naive_4) or (acc_rf_4==acc_gb_4 and acc_gb_4>acc_naive_4) or (acc_naive_4==acc_gb_4 and acc_gb_4>acc_rf_4) or (acc_rf_4==acc_naive_4 and acc_gb_4>acc_rf_4) :
    test4['gb_pre']=gb_4
    test4['match'] = np.where(test4['target'] == test4['gb_pre'], 'True', 'False')
    FP_FN_gb_4=test4[test4['match']=='False']
    FP_FN_gb_4=FP_FN_gb_4.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_4.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/5th_iteration/Test_5.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_4))
    
    TP_TN_gb_4=test4[test4['match']=='True']
    TP_TN_gb_4.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/4th_iteration/4.CSV')
    TP_TN_gb_4=TP_TN_gb_4.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_4.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/4th_iteration/TN_TP_4.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_4))

elif (acc_rf_4 != 0 and acc_naive_4!=0 and acc_gb_4!=0) and (acc_rf_4 == acc_naive_4 and acc_naive_4==acc_gb_4):
    test4['rf_pre']=rf_4
    test4['match'] = np.where(test4['target'] == test4['rf_pre'], 'True', 'False')
    FP_FN_RF_4=test4[test4['match']=='False']
    FP_FN_RF_4=FP_FN_RF_4.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_4.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/5th_iteration/Test_5.CSV')
    print("All Classifier Provides Same Accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_4))
    
    TP_TN_RF_4=test4[test4['match']=='True']
    TP_TN_RF_4.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/4th_iteration/4.CSV')
    TP_TN_RF_4=TP_TN_RF_4.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_4.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/4th_iteration/TN_TP_4.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_4))

else:
    print ("STOP")


# # 5th_iteraition

# In[112]:


train5=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/Train.CSV")
test5=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/5th_iteration/Test_5.csv")

x_train5 =train5[['age','cp','trestbps','chol','thalach','exang','oldpeak','slope','ca','thal']]
y_train5 = train5[['target']]

x_test5 = test5[['age','cp','trestbps','chol','thalach','exang','oldpeak','slope','ca','thal']]
y_test5 = test5[['target']]

random_forest=RandomForestClassifier(random_state=10)
random_forest.fit(x_train5,y_train5)
rf_5=random_forest.predict(x_test5)
acc_rf_5=metrics.accuracy_score(y_test5,rf_5)*100

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train5,y_train5)
Naive_5=Naive_bayes.predict(x_test5)
acc_naive_5=metrics.accuracy_score(y_test5,Naive_5)*100

gb = GradientBoostingClassifier(random_state=10)
gb.fit(x_train5, y_train5)
gb_5=gb.predict(x_test5)
acc_gb_5=metrics.accuracy_score(y_test5,gb_5)*100

print("Train_size:",len(train5),"| Test_size:",len(test5))
print("RF:",acc_rf_5)
print("Naivebayes:",acc_naive_5)
print("Gradient Boost:",acc_gb_5)
if (acc_rf_5 > acc_naive_5 and acc_rf_5 > acc_gb_5) or (acc_rf_5==acc_naive_5 and acc_rf_5>acc_gb_5) or (acc_rf_5==acc_gb_5 and acc_rf_5>acc_naive_5) or (acc_naive_5==acc_gb_5 and acc_rf_5>acc_naive_5):
    test5['rf_pre']=rf_5
    test5['match'] = np.where(test5['target'] == test5['rf_pre'], 'True', 'False')
    FP_FN_RF_5=test5[test5['match']=='False']
    FP_FN_RF_5=FP_FN_RF_5.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_5.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/6th_iteration/Test_6.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_5))
    
    TP_TN_RF_5=test5[test5['match']=='True']
    TP_TN_RF_5.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/5th_iteration/5.CSV')
    TP_TN_RF_5=TP_TN_RF_5.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_5.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/5th_iteration/TN_TP_5.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_5))

elif (acc_naive_5>acc_rf_5 and acc_naive_5>acc_gb_5) or (acc_rf_5==acc_naive_5 and acc_naive_5>acc_gb_5) or (acc_naive_5==acc_gb_5 and acc_naive_5>acc_rf_5) or (acc_rf_5==acc_gb_5 and acc_naive_5>acc_rf_5):  
    test5['nb_pre']=Naive_5      
    test5['match'] = np.where(test5['target'] == test5['nb_pre'], 'True', 'False')
    FP_FN_Na_5=test5[test5['match']=='False']
    FP_FN_Na_5=FP_FN_Na_5.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_5.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/6th_iteration/Test_6.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_5))
    
    TP_TN_Na_5=test5[test5['match']=='True']
    TP_TN_Na_5.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/5th_iteration/5.CSV')
    TP_TN_Na_5=TP_TN_Na_5.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_5.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/5th_iteration/TN_TP_5.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_5))
      
    
elif (acc_gb_5>acc_rf_5 and acc_gb_5>acc_naive_5) or (acc_rf_5==acc_gb_5 and acc_gb_5>acc_naive_5) or (acc_naive_5==acc_gb_5 and acc_gb_5>acc_rf_5) or (acc_rf_5==acc_naive_5 and acc_gb_5>acc_rf_5) :
    test5['gb_pre']=gb_5
    test5['match'] = np.where(test5['target'] == test5['gb_pre'], 'True', 'False')
    FP_FN_gb_5=test5[test5['match']=='False']
    FP_FN_gb_5=FP_FN_gb_5.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_5.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/6th_iteration/Test_6.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_5))
    
    TP_TN_gb_5=test5[test5['match']=='True']
    TP_TN_gb_5.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/5th_iteration/5.CSV')
    TP_TN_gb_5=TP_TN_gb_5.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_5.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/5th_iteration/TN_TP_5.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_5))

elif (acc_rf_5 != 0 and acc_naive_5!=0 and acc_gb_5!=0) and (acc_rf_5 == acc_naive_5 and acc_naive_5==acc_gb_5):
    test5['rf_pre']=rf_5
    test5['match'] = np.where(test5['target'] == test5['rf_pre'], 'True', 'False')
    FP_FN_RF_5=test5[test5['match']=='False']
    FP_FN_RF_5=FP_FN_RF_5.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_5.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/6th_iteration/Test_6.CSV')
    print("All Classifier Provides Same Accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_5))
    
    TP_TN_RF_5=test5[test5['match']=='True']
    TP_TN_RF_5.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/5th_iteration/5.CSV')
    TP_TN_RF_5=TP_TN_RF_5.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_5.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/5th_iteration/TN_TP_5.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_5))

else:
    print ("STOP")


# # 6th_iteraition

# In[123]:


train6=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/Train.CSV")
test6=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/6th_iteration/Test_6.csv")

x_train6 =train6[['age','cp','trestbps','chol','thalach','exang','oldpeak','slope','ca','thal']]
y_train6 = train6[['target']]

x_test6 = test6[['age','cp','trestbps','chol','thalach','exang','oldpeak','slope','ca','thal']]
y_test6 = test6[['target']]

random_forest=RandomForestClassifier(n_estimators=1,max_depth=2)
random_forest.fit(x_train6,y_train6)
rf_6=random_forest.predict(x_test6)
acc_rf_6=metrics.accuracy_score(y_test6,rf_6)*100

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train6,y_train6)
Naive_6=Naive_bayes.predict(x_test6)
acc_naive_6=metrics.accuracy_score(y_test6,Naive_6)*100

gb = GradientBoostingClassifier(random_state=10)
gb.fit(x_train6, y_train6)
gb_6=gb.predict(x_test6)
acc_gb_6=metrics.accuracy_score(y_test6,gb_6)*100

print("Train_size:",len(train6),"| Test_size:",len(test6))
print("RF:",acc_rf_6)
print("Naivebayes:",acc_naive_6)
print("Gradient Boost:",acc_gb_6)
if (acc_rf_6 > acc_naive_6 and acc_rf_6 > acc_gb_6) or (acc_rf_6==acc_naive_6 and acc_rf_6>acc_gb_6) or (acc_rf_6==acc_gb_6 and acc_rf_6>acc_naive_6) or (acc_naive_6==acc_gb_6 and acc_rf_6>acc_naive_6):
    test6['rf_pre']=rf_6
    test6['match'] = np.where(test6['target'] == test6['rf_pre'], 'True', 'False')
    FP_FN_RF_6=test6[test6['match']=='False']
    FP_FN_RF_6=FP_FN_RF_6.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_6.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/7th_iteration/Test_7.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_6))
    
    TP_TN_RF_6=test6[test6['match']=='True']
    TP_TN_RF_6.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/6th_iteration/6.CSV')
    TP_TN_RF_6=TP_TN_RF_6.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_6.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/6th_iteration/TN_TP_6.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_6))

elif (acc_naive_6>acc_rf_6 and acc_naive_6>acc_gb_6) or (acc_rf_6==acc_naive_6 and acc_naive_6>acc_gb_6) or (acc_naive_6==acc_gb_6 and acc_naive_6>acc_rf_6) or (acc_rf_6==acc_gb_6 and acc_naive_6>acc_rf_6):  
    test6['nb_pre']=Naive_6      
    test6['match'] = np.where(test6['target'] == test6['nb_pre'], 'True', 'False')
    FP_FN_Na_6=test6[test6['match']=='False']
    FP_FN_Na_6=FP_FN_Na_6.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_6.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/7th_iteration/Test_7.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_6))
    
    TP_TN_Na_6=test6[test6['match']=='True']
    TP_TN_Na_6.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/6th_iteration/6.CSV')
    TP_TN_Na_6=TP_TN_Na_6.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_6.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/6th_iteration/TN_TP_6.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_6))
      
    
elif (acc_gb_6>acc_rf_6 and acc_gb_6>acc_naive_6) or (acc_rf_6==acc_gb_6 and acc_gb_6>acc_naive_6) or (acc_naive_6==acc_gb_6 and acc_gb_6>acc_rf_6) or (acc_rf_6==acc_naive_6 and acc_gb_6>acc_rf_6) :
    test6['gb_pre']=gb_6
    test6['match'] = np.where(test6['target'] == test6['gb_pre'], 'True', 'False')
    FP_FN_gb_6=test6[test6['match']=='False']
    FP_FN_gb_6=FP_FN_gb_6.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_6.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/7th_iteration/Test_7.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_6))
    
    TP_TN_gb_6=test6[test6['match']=='True']
    TP_TN_gb_6.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/6th_iteration/6.CSV')
    TP_TN_gb_6=TP_TN_gb_6.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_6.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/6th_iteration/TN_TP_6.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_6))

elif (acc_rf_6 != 0 and acc_naive_6!=0 and acc_gb_6!=0) and (acc_rf_6 == acc_naive_6 and acc_naive_6==acc_gb_6):
    test6['rf_pre']=rf_6
    test6['match'] = np.where(test6['target'] == test6['rf_pre'], 'True', 'False')
    FP_FN_RF_6=test6[test6['match']=='False']
    FP_FN_RF_6=FP_FN_RF_6.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_6.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/7th_iteration/Test_7.CSV')
    print("All Classifier Provides Same Accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_6))
    
    TP_TN_RF_6=test6[test6['match']=='True']
    TP_TN_RF_6.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/6th_iteration/6.CSV')
    TP_TN_RF_6=TP_TN_RF_6.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_6.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/6th_iteration/TN_TP_6.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_6))

else:
    print ("STOP")


# # 7th_iteraition

# In[132]:


train7=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/Train.CSV")
test7=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/7th_iteration/Test_7.csv")

x_train7 =train7[['age','cp','trestbps','chol','thalach','exang','oldpeak','slope','ca','thal']]
y_train7 = train7[['target']]

x_test7 = test7[['age','cp','trestbps','chol','thalach','exang','oldpeak','slope','ca','thal']]
y_test7 = test7[['target']]

random_forest=RandomForestClassifier(n_estimators=2,max_depth=2)
random_forest.fit(x_train7,y_train7)
rf_7=random_forest.predict(x_test7)
acc_rf_7=metrics.accuracy_score(y_test7,rf_7)*100

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train7,y_train7)
Naive_7=Naive_bayes.predict(x_test7)
acc_naive_7=metrics.accuracy_score(y_test7,Naive_7)*100

gb = GradientBoostingClassifier(random_state=70)
gb.fit(x_train7, y_train7)
gb_7=gb.predict(x_test7)
acc_gb_7=metrics.accuracy_score(y_test7,gb_7)*100

print("Train_size:",len(train7),"| Test_size:",len(test7))
print("RF:",acc_rf_7)
print("Naivebayes:",acc_naive_7)
print("Gradient Boost:",acc_gb_7)
if (acc_rf_7 > acc_naive_7 and acc_rf_7 > acc_gb_7) or (acc_rf_7==acc_naive_7 and acc_rf_7>acc_gb_7) or (acc_rf_7==acc_gb_7 and acc_rf_7>acc_naive_7) or (acc_naive_7==acc_gb_7 and acc_rf_7>acc_naive_7):
    test7['rf_pre']=rf_7
    test7['match'] = np.where(test7['target'] == test7['rf_pre'], 'True', 'False')
    FP_FN_RF_7=test7[test7['match']=='False']
    FP_FN_RF_7=FP_FN_RF_7.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_7.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/8th_iteration/Test_8.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_7))
    
    TP_TN_RF_7=test7[test7['match']=='True']
    TP_TN_RF_7.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/7th_iteration/7.CSV')
    TP_TN_RF_7=TP_TN_RF_7.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_7.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/7th_iteration/TN_TP_7.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_7))

elif (acc_naive_7>acc_rf_7 and acc_naive_7>acc_gb_7) or (acc_rf_7==acc_naive_7 and acc_naive_7>acc_gb_7) or (acc_naive_7==acc_gb_7 and acc_naive_7>acc_rf_7) or (acc_rf_7==acc_gb_7 and acc_naive_7>acc_rf_7):  
    test7['nb_pre']=Naive_7      
    test7['match'] = np.where(test7['target'] == test7['nb_pre'], 'True', 'False')
    FP_FN_Na_7=test7[test7['match']=='False']
    FP_FN_Na_7=FP_FN_Na_7.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_7.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/8th_iteration/Test_8.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_7))
    
    TP_TN_Na_7=test7[test7['match']=='True']
    TP_TN_Na_7.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/7th_iteration/7.CSV')
    TP_TN_Na_7=TP_TN_Na_7.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_7.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/7th_iteration/TN_TP_7.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_7))
      
    
elif (acc_gb_7>acc_rf_7 and acc_gb_7>acc_naive_7) or (acc_rf_7==acc_gb_7 and acc_gb_7>acc_naive_7) or (acc_naive_7==acc_gb_7 and acc_gb_7>acc_rf_7) or (acc_rf_7==acc_naive_7 and acc_gb_7>acc_rf_7) :
    test7['gb_pre']=gb_7
    test7['match'] = np.where(test7['target'] == test7['gb_pre'], 'True', 'False')
    FP_FN_gb_7=test7[test7['match']=='False']
    FP_FN_gb_7=FP_FN_gb_7.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_7.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/8th_iteration/Test_8.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_7))
    
    TP_TN_gb_7=test7[test7['match']=='True']
    TP_TN_gb_7.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/7th_iteration/7.CSV')
    TP_TN_gb_7=TP_TN_gb_7.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_7.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/7th_iteration/TN_TP_7.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_7))

elif (acc_rf_7 != 0 and acc_naive_7!=0 and acc_gb_7!=0) and (acc_rf_7 == acc_naive_7 and acc_naive_7==acc_gb_7):
    test7['rf_pre']=rf_7
    test7['match'] = np.where(test7['target'] == test7['rf_pre'], 'True', 'False')
    FP_FN_RF_7=test7[test7['match']=='False']
    FP_FN_RF_7=FP_FN_RF_7.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_7.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/8th_iteration/Test_8.CSV')
    print("All Classifier Provides Same Accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_7))
    
    TP_TN_RF_7=test7[test7['match']=='True']
    TP_TN_RF_7.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/7th_iteration/7.CSV')
    TP_TN_RF_7=TP_TN_RF_7.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_7.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/7th_iteration/TN_TP_7.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_7))

else:
    print ("STOP")


# # 8th_iteraition

# In[3]:


train8=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/Train.CSV")
test8=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/8th_iteration/Test_8.csv")

x_train8 =train8[['age','cp','trestbps','chol','thalach','exang','oldpeak','slope','ca','thal']]
y_train8 = train8[['target']]

x_test8 = test8[['age','cp','trestbps','chol','thalach','exang','oldpeak','slope','ca','thal']]
y_test8 = test8[['target']]

random_forest=RandomForestClassifier(n_estimators=1,max_depth=2)
random_forest.fit(x_train8,y_train8)
rf_8=random_forest.predict(x_test8)
acc_rf_8=metrics.accuracy_score(y_test8,rf_8)*100

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train8,y_train8)
Naive_8=Naive_bayes.predict(x_test8)
acc_naive_8=metrics.accuracy_score(y_test8,Naive_8)*100

gb = GradientBoostingClassifier(random_state=80)
gb.fit(x_train8, y_train8)
gb_8=gb.predict(x_test8)
acc_gb_8=metrics.accuracy_score(y_test8,gb_8)*100

print("Train_size:",len(train8),"| Test_size:",len(test8))
print("RF:",acc_rf_8)
print("Naivebayes:",acc_naive_8)
print("Gradient Boost:",acc_gb_8)
if (acc_rf_8 > acc_naive_8 and acc_rf_8 > acc_gb_8) or (acc_rf_8==acc_naive_8 and acc_rf_8>acc_gb_8) or (acc_rf_8==acc_gb_8 and acc_rf_8>acc_naive_8) or (acc_naive_8==acc_gb_8 and acc_rf_8>acc_naive_8):
    test8['rf_pre']=rf_8
    test8['match'] = np.where(test8['target'] == test8['rf_pre'], 'True', 'False')
    FP_FN_RF_8=test8[test8['match']=='False']
    FP_FN_RF_8=FP_FN_RF_8.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_8.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/9th_iteration/Test_9.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_8))
    
    TP_TN_RF_8=test8[test8['match']=='True']
    TP_TN_RF_8.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/8th_iteration/8.CSV')
    TP_TN_RF_8=TP_TN_RF_8.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_8.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/8th_iteration/TN_TP_8.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_8))

elif (acc_naive_8>acc_rf_8 and acc_naive_8>acc_gb_8) or (acc_rf_8==acc_naive_8 and acc_naive_8>acc_gb_8) or (acc_naive_8==acc_gb_8 and acc_naive_8>acc_rf_8) or (acc_rf_8==acc_gb_8 and acc_naive_8>acc_rf_8):  
    test8['nb_pre']=Naive_8      
    test8['match'] = np.where(test8['target'] == test8['nb_pre'], 'True', 'False')
    FP_FN_Na_8=test8[test8['match']=='False']
    FP_FN_Na_8=FP_FN_Na_8.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_8.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/9th_iteration/Test_9.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_8))
    
    TP_TN_Na_8=test8[test8['match']=='True']
    TP_TN_Na_8.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/8th_iteration/8.CSV')
    TP_TN_Na_8=TP_TN_Na_8.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_8.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/8th_iteration/TN_TP_8.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_8))
      
    
elif (acc_gb_8>acc_rf_8 and acc_gb_8>acc_naive_8) or (acc_rf_8==acc_gb_8 and acc_gb_8>acc_naive_8) or (acc_naive_8==acc_gb_8 and acc_gb_8>acc_rf_8) or (acc_rf_8==acc_naive_8 and acc_gb_8>acc_rf_8) :
    test8['gb_pre']=gb_8
    test8['match'] = np.where(test8['target'] == test8['gb_pre'], 'True', 'False')
    FP_FN_gb_8=test8[test8['match']=='False']
    FP_FN_gb_8=FP_FN_gb_8.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_8.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/9th_iteration/Test_9.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_8))
    
    TP_TN_gb_8=test8[test8['match']=='True']
    TP_TN_gb_8.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/8th_iteration/8.CSV')
    TP_TN_gb_8=TP_TN_gb_8.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_8.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/8th_iteration/TN_TP_8.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_8))

elif (acc_rf_8 != 0 and acc_naive_8!=0 and acc_gb_8!=0) and (acc_rf_8 == acc_naive_8 and acc_naive_8==acc_gb_8):
    test8['rf_pre']=rf_8
    test8['match'] = np.where(test8['target'] == test8['rf_pre'], 'True', 'False')
    FP_FN_RF_8=test8[test8['match']=='False']
    FP_FN_RF_8=FP_FN_RF_8.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_8.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/9th_iteration/Test_9.CSV')
    print("All Classifier Provides Same Accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_8))
    
    TP_TN_RF_8=test8[test8['match']=='True']
    TP_TN_RF_8.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/8th_iteration/8.CSV')
    TP_TN_RF_8=TP_TN_RF_8.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_8.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/8th_iteration/TN_TP_8.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_8))

else:
    print ("STOP")


# # 9th_iteraition

# In[16]:


train9=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/Train.CSV")
test9=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/9th_iteration/Test_9.csv")

x_train9 =train9[['age','cp','trestbps','chol','thalach','exang','oldpeak','slope','ca','thal']]
y_train9 = train9[['target']]

x_test9 = test9[['age','cp','trestbps','chol','thalach','exang','oldpeak','slope','ca','thal']]
y_test9 = test9[['target']]

random_forest=RandomForestClassifier(n_estimators=1,max_depth=2)
random_forest.fit(x_train9,y_train9)
rf_9=random_forest.predict(x_test9)
acc_rf_9=metrics.accuracy_score(y_test9,rf_9)*100

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train9,y_train9)
Naive_9=Naive_bayes.predict(x_test9)
acc_naive_9=metrics.accuracy_score(y_test9,Naive_9)*100

gb = GradientBoostingClassifier(random_state=90)
gb.fit(x_train9, y_train9)
gb_9=gb.predict(x_test9)
acc_gb_9=metrics.accuracy_score(y_test9,gb_9)*100

print("Train_size:",len(train9),"| Test_size:",len(test9))
print("RF:",acc_rf_9)
print("Naivebayes:",acc_naive_9)
print("Gradient Boost:",acc_gb_9)
if (acc_rf_9 > acc_naive_9 and acc_rf_9 > acc_gb_9) or (acc_rf_9==acc_naive_9 and acc_rf_9>acc_gb_9) or (acc_rf_9==acc_gb_9 and acc_rf_9>acc_naive_9) or (acc_naive_9==acc_gb_9 and acc_rf_9>acc_naive_9):
    test9['rf_pre']=rf_9
    test9['match'] = np.where(test9['target'] == test9['rf_pre'], 'True', 'False')
    FP_FN_RF_9=test9[test9['match']=='False']
    FP_FN_RF_9=FP_FN_RF_9.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_9.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/10th_iteration/Test_10.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_9))
    
    TP_TN_RF_9=test9[test9['match']=='True']
    TP_TN_RF_9.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/9th_iteration/9.CSV')
    TP_TN_RF_9=TP_TN_RF_9.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_9.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/9th_iteration/TN_TP_9.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_9))

elif (acc_naive_9>acc_rf_9 and acc_naive_9>acc_gb_9) or (acc_rf_9==acc_naive_9 and acc_naive_9>acc_gb_9) or (acc_naive_9==acc_gb_9 and acc_naive_9>acc_rf_9) or (acc_rf_9==acc_gb_9 and acc_naive_9>acc_rf_9):  
    test9['nb_pre']=Naive_9      
    test9['match'] = np.where(test9['target'] == test9['nb_pre'], 'True', 'False')
    FP_FN_Na_9=test9[test9['match']=='False']
    FP_FN_Na_9=FP_FN_Na_9.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_9.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/10th_iteration/Test_10.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_9))
    
    TP_TN_Na_9=test9[test9['match']=='True']
    TP_TN_Na_9.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/9th_iteration/9.CSV')
    TP_TN_Na_9=TP_TN_Na_9.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_9.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/9th_iteration/TN_TP_9.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_9))
      
    
elif (acc_gb_9>acc_rf_9 and acc_gb_9>acc_naive_9) or (acc_rf_9==acc_gb_9 and acc_gb_9>acc_naive_9) or (acc_naive_9==acc_gb_9 and acc_gb_9>acc_rf_9) or (acc_rf_9==acc_naive_9 and acc_gb_9>acc_rf_9) :
    test9['gb_pre']=gb_9
    test9['match'] = np.where(test9['target'] == test9['gb_pre'], 'True', 'False')
    FP_FN_gb_9=test9[test9['match']=='False']
    FP_FN_gb_9=FP_FN_gb_9.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_9.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/10th_iteration/Test_10.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_9))
    
    TP_TN_gb_9=test9[test9['match']=='True']
    TP_TN_gb_9.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/9th_iteration/9.CSV')
    TP_TN_gb_9=TP_TN_gb_9.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_9.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/9th_iteration/TN_TP_9.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_9))

elif (acc_rf_9 != 0 and acc_naive_9!=0 and acc_gb_9!=0) and (acc_rf_9 == acc_naive_9 and acc_naive_9==acc_gb_9):
    test9['rf_pre']=rf_9
    test9['match'] = np.where(test9['target'] == test9['rf_pre'], 'True', 'False')
    FP_FN_RF_9=test9[test9['match']=='False']
    FP_FN_RF_9=FP_FN_RF_9.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_9.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/10th_iteration/Test_10.CSV')
    print("All Classifier Provides Same Accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_9))
    
    TP_TN_RF_9=test9[test9['match']=='True']
    TP_TN_RF_9.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/9th_iteration/9.CSV')
    TP_TN_RF_9=TP_TN_RF_9.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_9.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/9th_iteration/TN_TP_9.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_9))

else:
    print ("STOP")

train9=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/Train.CSV")
test9=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/9th_iteration/Test_9.csv")

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors = 1)
knn.fit(x_train9,y_train9)
knn_9=knn.predict(x_test9)
acc_knn_9=metrics.accuracy_score(y_test9,knn_9)*100

test9['knn_pre']=knn_9
test9['match'] = np.where(test9['target'] == test9['knn_pre'], 'True', 'False')
FP_FN_knn_9=test9[test9['match']=='False']
FP_FN_knn_9=FP_FN_knn_9.drop(['knn_pre','match'], axis=1)
FP_FN_knn_9.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/10th_iteration/Test_10.CSV')
print("knn_1 Accuracy ",acc_knn_9)
print("knn_FP_FN: ",len(FP_FN_knn_9))
    
TP_TN_knn_9=test9[test9['match']=='True']
TP_TN_knn_9.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/9th_iteration/9.CSV')
TP_TN_knn_9=TP_TN_knn_9.drop(['knn_pre','match'], axis=1)
TP_TN_knn_9.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/9th_iteration/TN_TP_9.CSV')
print("knn_TN_TP: ",len(TP_TN_knn_9))


# # 10th_iteraition

# In[19]:


train10=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/Train.CSV")
test10=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/10th_iteration/Test_10.csv")

x_train10 =train10[['age','cp','trestbps','chol','thalach','exang','oldpeak','slope','ca','thal']]
y_train10 = train10[['target']]

x_test10 = test10[['age','cp','trestbps','chol','thalach','exang','oldpeak','slope','ca','thal']]
y_test10 = test10[['target']]

random_forest=RandomForestClassifier(n_estimators=1,max_depth=2)
random_forest.fit(x_train10,y_train10)
rf_10=random_forest.predict(x_test10)
acc_rf_10=metrics.accuracy_score(y_test10,rf_10)*110

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train10,y_train10)
Naive_10=Naive_bayes.predict(x_test10)
acc_naive_10=metrics.accuracy_score(y_test10,Naive_10)*110

gb = GradientBoostingClassifier(random_state=100)
gb.fit(x_train10, y_train10)
gb_10=gb.predict(x_test10)
acc_gb_10=metrics.accuracy_score(y_test10,gb_10)*110

print("Train_size:",len(train10),"| Test_size:",len(test10))
print("RF:",acc_rf_10)
print("Naivebayes:",acc_naive_10)
print("Gradient Boost:",acc_gb_10)
if (acc_rf_10 > acc_naive_10 and acc_rf_10 > acc_gb_10) or (acc_rf_10==acc_naive_10 and acc_rf_10>acc_gb_10) or (acc_rf_10==acc_gb_10 and acc_rf_10>acc_naive_10) or (acc_naive_10==acc_gb_10 and acc_rf_10>acc_naive_10):
    test10['rf_pre']=rf_10
    test10['match'] = np.where(test10['target'] == test10['rf_pre'], 'True', 'False')
    FP_FN_RF_10=test10[test10['match']=='False']
    FP_FN_RF_10=FP_FN_RF_10.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_10.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/11th_iteration/Test_11.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_10))
    
    TP_TN_RF_10=test10[test10['match']=='True']
    TP_TN_RF_10.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/10th_iteration/10.CSV')
    TP_TN_RF_10=TP_TN_RF_10.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_10.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/10th_iteration/TN_TP_10.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_10))

elif (acc_naive_10>acc_rf_10 and acc_naive_10>acc_gb_10) or (acc_rf_10==acc_naive_10 and acc_naive_10>acc_gb_10) or (acc_naive_10==acc_gb_10 and acc_naive_10>acc_rf_10) or (acc_rf_10==acc_gb_10 and acc_naive_10>acc_rf_10):  
    test10['nb_pre']=Naive_10      
    test10['match'] = np.where(test10['target'] == test10['nb_pre'], 'True', 'False')
    FP_FN_Na_10=test10[test10['match']=='False']
    FP_FN_Na_10=FP_FN_Na_10.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_10.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/11th_iteration/Test_11.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_10))
    
    TP_TN_Na_10=test10[test10['match']=='True']
    TP_TN_Na_10.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/10th_iteration/10.CSV')
    TP_TN_Na_10=TP_TN_Na_10.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_10.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/10th_iteration/TN_TP_10.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_10))
      
    
elif (acc_gb_10>acc_rf_10 and acc_gb_10>acc_naive_10) or (acc_rf_10==acc_gb_10 and acc_gb_10>acc_naive_10) or (acc_naive_10==acc_gb_10 and acc_gb_10>acc_rf_10) or (acc_rf_10==acc_naive_10 and acc_gb_10>acc_rf_10) :
    test10['gb_pre']=gb_10
    test10['match'] = np.where(test10['target'] == test10['gb_pre'], 'True', 'False')
    FP_FN_gb_10=test10[test10['match']=='False']
    FP_FN_gb_10=FP_FN_gb_10.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_10.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/11th_iteration/Test_11.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_10))
    
    TP_TN_gb_10=test10[test10['match']=='True']
    TP_TN_gb_10.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/10th_iteration/10.CSV')
    TP_TN_gb_10=TP_TN_gb_10.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_10.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/10th_iteration/TN_TP_10.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_10))

elif (acc_rf_10 != 0 and acc_naive_10!=0 and acc_gb_10!=0) and (acc_rf_10 == acc_naive_10 and acc_naive_10==acc_gb_10):
    test10['rf_pre']=rf_10
    test10['match'] = np.where(test10['target'] == test10['rf_pre'], 'True', 'False')
    FP_FN_RF_10=test10[test10['match']=='False']
    FP_FN_RF_10=FP_FN_RF_10.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_10.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/11th_iteration/Test_11.CSV')
    print("All Classifier Provides Same Accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_10))
    
    TP_TN_RF_10=test10[test10['match']=='True']
    TP_TN_RF_10.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/10th_iteration/10.CSV')
    TP_TN_RF_10=TP_TN_RF_10.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_10.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/10th_iteration/TN_TP_10.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_10))

else:
    print ("STOP")

train10=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/Train.CSV")
test10=pd.read_csv("C:/Users/acer/Downloads/22/MLPS(303)/10th_iteration/Test_10.csv")

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors = 1)
knn.fit(x_train10,y_train10)
knn_10=knn.predict(x_test10)
acc_knn_10=metrics.accuracy_score(y_test10,knn_10)*110

test10['knn_pre']=knn_10
test10['match'] = np.where(test10['target'] == test10['knn_pre'], 'True', 'False')
FP_FN_knn_10=test10[test10['match']=='False']
FP_FN_knn_10=FP_FN_knn_10.drop(['knn_pre','match'], axis=1)
FP_FN_knn_10.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/11th_iteration/Test_11.CSV')
print("knn_1 Accuracy ",acc_knn_10)
print("knn_FP_FN: ",len(FP_FN_knn_10))
    
TP_TN_knn_10=test10[test10['match']=='True']
TP_TN_knn_10.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/10th_iteration/10.CSV')
TP_TN_knn_10=TP_TN_knn_10.drop(['knn_pre','match'], axis=1)
TP_TN_knn_10.to_csv('C:/Users/acer/Downloads/22/MLPS(303)/10th_iteration/TN_TP_10.CSV')
print("knn_TN_TP: ",len(TP_TN_knn_10))

