#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###### from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier


# # 1st_iteraition

# In[3]:


train1=pd.read_csv("E:/1st_iteration/Train.CSV")
test1=pd.read_csv("E:/1st_iteration/Test_1.csv")

x_train1 =train1[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train1 = train1[['target']]

x_test1 = test1[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test1 = test1[['target']]

random_forest=RandomForestClassifier(random_state=0)
random_forest.fit(x_train1,y_train1)
rf_1=random_forest.predict(x_test1)
acc_rf_1=metrics.accuracy_score(y_test1,rf_1)*100

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train1,y_train1)
Naive_1=Naive_bayes.predict(x_test1)
acc_naive_1=metrics.accuracy_score(y_test1,Naive_1)*100

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train1, y_train1)
gb_1=gb.predict(x_test1)
acc_gb_1=metrics.accuracy_score(y_test1,gb_1)*100

print("Train_size:",len(train1),"| Test_size:",len(test1))
print("RF:",acc_rf_1)
print("Naivebayes:",acc_naive_1)
print("Gradient Boost:",acc_gb_1)

if acc_rf_1 > acc_naive_1 and acc_rf_1 > acc_gb_1 :
    test1['rf_pre']=rf_1
    test1['match'] = np.where(test1['target'] == test1['rf_pre'], 'True', 'False')
    FP_FN_RF_1=test1[test1['match']=='False']
    FP_FN_RF_1=FP_FN_RF_1.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_1.to_csv('E:/2nd_iteration/Test_2.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_1))
    
    TP_TN_RF_1=test1[test1['match']=='True']
    TP_TN_RF_1.to_csv('E:/1st_iteration/1.CSV')
    TP_TN_RF_1=TP_TN_RF_1.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_1.to_csv('E:/1st_iteration/TN_TP_1.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_1))

elif acc_naive_1>acc_rf_1 and acc_naive_1>acc_gb_1 :
    test1['nb_pre']=Naive_1      
    test1['match'] = np.where(test1['target'] == test1['nb_pre'], 'True', 'False')
    FP_FN_Na_1=test1[test1['match']=='False']
    FP_FN_Na_1=FP_FN_Na_1.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_1.to_csv('E:/2nd_iteration/Test_2.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_1))
    
    TP_TN_Na_1=test1[test1['match']=='True']
    TP_TN_Na_1.to_csv('E:/1st_iteration/1.CSV')
    TP_TN_Na_1=TP_TN_Na_1.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_1.to_csv('E:/1st_iteration/TN_TP_1.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_1))
      
    
else:
    test1['gb_pre']=gb_1
    test1['match'] = np.where(test1['target'] == test1['gb_pre'], 'True', 'False')
    FP_FN_gb_1=test1[test1['match']=='False']
    FP_FN_gb_1=FP_FN_gb_1.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_1.to_csv('E:/2nd_iteration/Test_2.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_1))
    
    TP_TN_gb_1=test1[test1['match']=='True']
    TP_TN_gb_1.to_csv('E:/1st_iteration/1.CSV')
    TP_TN_gb_1=TP_TN_gb_1.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_1.to_csv('E:/1st_iteration/TN_TP_1.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_1))


# # 2nd_iteraition

# In[4]:


train2=pd.read_csv("E:/1st_iteration/Train.CSV")
test2=pd.read_csv("E:/2nd_iteration/Test_2.csv")

x_train2 =train2[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train2 = train2[['target']]

x_test2 = test2[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test2 = test2[['target']]

random_forest=RandomForestClassifier(random_state=0)
random_forest.fit(x_train2,y_train2)
rf_2=random_forest.predict(x_test2)
acc_rf_2=metrics.accuracy_score(y_test2,rf_2)*100

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train2,y_train2)
Naive_2=Naive_bayes.predict(x_test2)
acc_naive_2=metrics.accuracy_score(y_test2,Naive_2)*100

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train2, y_train2)
gb_2=gb.predict(x_test2)
acc_gb_2=metrics.accuracy_score(y_test2,gb_2)*100

print("Train_size:",len(train2),"| Test_size:",len(test2))
print("RF:",acc_rf_2)
print("Naivebayes:",acc_naive_2)
print("Gradient Boost:",acc_gb_2)

if acc_rf_2 > acc_naive_2 and acc_rf_2 > acc_gb_2 :
    test2['rf_pre']=rf_2
    test2['match'] = np.where(test2['target'] == test2['rf_pre'], 'True', 'False')
    FP_FN_RF_2=test2[test2['match']=='False']
    FP_FN_RF_2=FP_FN_RF_2.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_2.to_csv('E:/3rd_iteration/Test_3.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_2))
    
    TP_TN_RF_2=test2[test2['match']=='True']
    TP_TN_RF_2.to_csv('E:/2nd_iteration/2.CSV')
    TP_TN_RF_2=TP_TN_RF_2.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_2.to_csv('E:/2nd_iteration/TN_TP_2.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_2))

elif acc_naive_2>acc_rf_2 and acc_naive_2>acc_gb_2 :
    test2['nb_pre']=Naive_2      
    test2['match'] = np.where(test2['target'] == test2['nb_pre'], 'True', 'False')
    FP_FN_Na_2=test2[test2['match']=='False']
    FP_FN_Na_2=FP_FN_Na_2.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_2.to_csv('E:/3rd_iteration/Test_3.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_2))
    
    TP_TN_Na_2=test2[test2['match']=='True']
    TP_TN_Na_2.to_csv('E:/2nd_iteration/2.CSV')
    TP_TN_Na_2=TP_TN_Na_2.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_2.to_csv('E:/2nd_iteration/TN_TP_2.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_2))
      
    
else:
    test2['gb_pre']=gb_2
    test2['match'] = np.where(test2['target'] == test2['gb_pre'], 'True', 'False')
    FP_FN_gb_2=test2[test2['match']=='False']
    FP_FN_gb_2=FP_FN_gb_2.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_2.to_csv('E:/3rd_iteration/Test_3.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_2))
    
    TP_TN_gb_2=test2[test2['match']=='True']
    TP_TN_gb_2.to_csv('E:/2nd_iteration/2.CSV')
    TP_TN_gb_2=TP_TN_gb_2.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_2.to_csv('E:/2nd_iteration/TN_TP_2.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_2))


# # 3rd_iteraition

# In[5]:


train3=pd.read_csv("E:/1st_iteration/Train.CSV")
test3=pd.read_csv("E:/3rd_iteration/Test_3.csv")

x_train3 =train3[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train3 = train3[['target']]

x_test3 = test3[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test3 = test3[['target']]

random_forest=RandomForestClassifier(random_state=0)
random_forest.fit(x_train3,y_train3)
rf_3=random_forest.predict(x_test3)
acc_rf_3=metrics.accuracy_score(y_test3,rf_3)*100

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train3,y_train3)
Naive_3=Naive_bayes.predict(x_test3)
acc_naive_3=metrics.accuracy_score(y_test3,Naive_3)*100

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train3, y_train3)
gb_3=gb.predict(x_test3)
acc_gb_3=metrics.accuracy_score(y_test3,gb_3)*100

print("Train_size:",len(train3),"| Test_size:",len(test3))
print("RF:",acc_rf_3)
print("Naivebayes:",acc_naive_3)
print("Gradient Boost:",acc_gb_3)

if acc_rf_3 > acc_naive_3 and acc_rf_3 > acc_gb_3 :
    test3['rf_pre']=rf_3
    test3['match'] = np.where(test3['target'] == test3['rf_pre'], 'True', 'False')
    FP_FN_RF_3=test3[test3['match']=='False']
    FP_FN_RF_3=FP_FN_RF_3.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_3.to_csv('E:/4th_iteration/Test_4.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_3))
    
    TP_TN_RF_3=test3[test3['match']=='True']
    TP_TN_RF_3.to_csv('E:/3rd_iteration/3.CSV')
    TP_TN_RF_3=TP_TN_RF_3.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_3.to_csv('E:/3rd_iteration/TN_TP_3.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_3))

elif acc_naive_3>acc_rf_3 and acc_naive_3>acc_gb_3 :
    test3['nb_pre']=Naive_3      
    test3['match'] = np.where(test3['target'] == test3['nb_pre'], 'True', 'False')
    FP_FN_Na_3=test3[test3['match']=='False']
    FP_FN_Na_3=FP_FN_Na_3.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_3.to_csv('E:/4th_iteration/Test_4.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_3))
    
    TP_TN_Na_3=test3[test3['match']=='True']
    TP_TN_Na_3.to_csv('E:/3rd_iteration/3.CSV')
    TP_TN_Na_3=TP_TN_Na_3.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_3.to_csv('E:/3rd_iteration/TN_TP_3.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_3))
      
    
else:
    test3['gb_pre']=gb_3
    test3['match'] = np.where(test3['target'] == test3['gb_pre'], 'True', 'False')
    FP_FN_gb_3=test3[test3['match']=='False']
    FP_FN_gb_3=FP_FN_gb_3.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_3.to_csv('E:/4th_iteration/Test_4.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_3))
    
    TP_TN_gb_3=test3[test3['match']=='True']
    TP_TN_gb_3.to_csv('E:/3rd_iteration/3.CSV')
    TP_TN_gb_3=TP_TN_gb_3.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_3.to_csv('E:/3rd_iteration/TN_TP_3.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_3))


# # 4th_iteraition

# In[6]:


train4=pd.read_csv("E:/1st_iteration/Train.CSV")
test4=pd.read_csv("E:/4th_iteration/Test_4.csv")

x_train4 =train4[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train4 = train4[['target']]

x_test4 = test4[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test4 = test4[['target']]

random_forest=RandomForestClassifier(random_state=20)
random_forest.fit(x_train4,y_train4)
rf_4=random_forest.predict(x_test4)
acc_rf_4=metrics.accuracy_score(y_test4,rf_4)*100

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train4,y_train4)
Naive_4=Naive_bayes.predict(x_test4)
acc_naive_4=metrics.accuracy_score(y_test4,Naive_4)*100

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train4, y_train4)
gb_4=gb.predict(x_test4)
acc_gb_4=metrics.accuracy_score(y_test4,gb_4)*100

print("Train_size:",len(train4),"| Test_size:",len(test4))
print("RF:",acc_rf_4)
print("Naivebayes:",acc_naive_4)
print("Gradient Boost:",acc_gb_4)

if acc_rf_4 > acc_naive_4 and acc_rf_4 > acc_gb_4 :
    test4['rf_pre']=rf_4
    test4['match'] = np.where(test4['target'] == test4['rf_pre'], 'True', 'False')
    FP_FN_RF_4=test4[test4['match']=='False']
    FP_FN_RF_4=FP_FN_RF_4.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_4.to_csv('E:/5th_iteration/Test_5.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_4))
    
    TP_TN_RF_4=test4[test4['match']=='True']
    TP_TN_RF_4.to_csv('E:/4th_iteration/4.CSV')
    TP_TN_RF_4=TP_TN_RF_4.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_4.to_csv('E:/4th_iteration/TN_TP_4.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_4))

elif acc_naive_4>acc_rf_4 and acc_naive_4>acc_gb_4 :
    test4['nb_pre']=Naive_4      
    test4['match'] = np.where(test4['target'] == test4['nb_pre'], 'True', 'False')
    FP_FN_Na_4=test4[test4['match']=='False']
    FP_FN_Na_4=FP_FN_Na_4.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_4.to_csv('E:/5th_iteration/Test_5.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_4))
    
    TP_TN_Na_4=test4[test4['match']=='True']
    TP_TN_Na_4.to_csv('E:/4th_iteration/4.CSV')
    TP_TN_Na_4=TP_TN_Na_4.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_4.to_csv('E:/4th_iteration/TN_TP_4.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_4))
         
else:
    test4['gb_pre']=gb_4
    test4['match'] = np.where(test4['target'] == test4['gb_pre'], 'True', 'False')
    FP_FN_gb_4=test4[test4['match']=='False']
    FP_FN_gb_4=FP_FN_gb_4.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_4.to_csv('E:/5th_iteration/Test_5.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_4))
    
    TP_TN_gb_4=test4[test4['match']=='True']
    TP_TN_gb_4.to_csv('E:/4th_iteration/4.CSV')
    TP_TN_gb_4=TP_TN_gb_4.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_4.to_csv('E:/4th_iteration/TN_TP_4.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_4))


# # 5th_iteraition

# In[7]:


train5=pd.read_csv("E:/1st_iteration/Train.CSV")
test5=pd.read_csv("E:/5th_iteration/Test_5.csv")

x_train5 =train5[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train5 = train5[['target']]

x_test5 = test5[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test5 = test5[['target']]

random_forest=RandomForestClassifier(random_state=0)
random_forest.fit(x_train5,y_train5)
rf_5=random_forest.predict(x_test5)
acc_rf_5=metrics.accuracy_score(y_test5,rf_5)*100

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train5,y_train5)
Naive_5=Naive_bayes.predict(x_test5)
acc_naive_5=metrics.accuracy_score(y_test5,Naive_5)*100

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train5, y_train5)
gb_5=gb.predict(x_test5)
acc_gb_5=metrics.accuracy_score(y_test5,gb_5)*100

print("Train_size:",len(train5),"| Test_size:",len(test5))
print("RF:",acc_rf_5)
print("Naivebayes:",acc_naive_5)
print("Gradient Boost:",acc_gb_5)

if acc_rf_5 > acc_naive_5 and acc_rf_5 > acc_gb_5 :
    test5['rf_pre']=rf_5
    test5['match'] = np.where(test5['target'] == test5['rf_pre'], 'True', 'False')
    FP_FN_RF_5=test5[test5['match']=='False']
    FP_FN_RF_5=FP_FN_RF_5.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_5.to_csv('E:/6th_iteration/Test_6.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_5))
    
    TP_TN_RF_5=test5[test5['match']=='True']
    TP_TN_RF_5.to_csv('E:/5th_iteration/5.CSV')
    TP_TN_RF_5=TP_TN_RF_5.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_5.to_csv('E:/5th_iteration/TN_TP_5.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_5))

elif acc_naive_5>acc_rf_5 and acc_naive_5>acc_gb_5 :
    test5['nb_pre']=Naive_5      
    test5['match'] = np.where(test5['target'] == test5['nb_pre'], 'True', 'False')
    FP_FN_Na_5=test5[test5['match']=='False']
    FP_FN_Na_5=FP_FN_Na_5.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_5.to_csv('E:/6th_iteration/Test_6.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_5))
    
    TP_TN_Na_5=test5[test5['match']=='True']
    TP_TN_Na_5.to_csv('E:/5th_iteration/5.CSV')
    TP_TN_Na_5=TP_TN_Na_5.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_5.to_csv('E:/5th_iteration/TN_TP_5.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_5))
         
else:
    test5['gb_pre']=gb_5
    test5['match'] = np.where(test5['target'] == test5['gb_pre'], 'True', 'False')
    FP_FN_gb_5=test5[test5['match']=='False']
    FP_FN_gb_5=FP_FN_gb_5.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_5.to_csv('E:/6th_iteration/Test_6.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_5))
    
    TP_TN_gb_5=test5[test5['match']=='True']
    TP_TN_gb_5.to_csv('E:/5th_iteration/5.CSV')
    TP_TN_gb_5=TP_TN_gb_5.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_5.to_csv('E:/5th_iteration/TN_TP_5.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_5))


# # 6th_iteraition

# In[2]:


train6=pd.read_csv("E:/1st_iteration/Train.CSV")
test6=pd.read_csv("E:/6th_iteration/Test_6.csv")

x_train6 =train6[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train6 = train6[['target']]

x_test6 = test6[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test6 = test6[['target']]

random_forest=RandomForestClassifier(random_state=100)
random_forest.fit(x_train6,y_train6)
rf_6=random_forest.predict(x_test6)
acc_rf_6=metrics.accuracy_score(y_test6,rf_6)*100

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train6,y_train6)
Naive_6=Naive_bayes.predict(x_test6)
acc_naive_6=metrics.accuracy_score(y_test6,Naive_6)*100

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train6, y_train6)
gb_6=gb.predict(x_test6)
acc_gb_6=metrics.accuracy_score(y_test6,gb_6)*100

print("Train_size:",len(train6),"| Test_size:",len(test6))
print("RF:",acc_rf_6)
print("Naivebayes:",acc_naive_6)
print("Gradient Boost:",acc_gb_6)

if acc_rf_6 > acc_naive_6 and acc_rf_6 > acc_gb_6 :
    test6['rf_pre']=rf_6
    test6['match'] = np.where(test6['target'] == test6['rf_pre'], 'True', 'False')
    FP_FN_RF_6=test6[test6['match']=='False']
    FP_FN_RF_6=FP_FN_RF_6.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_6.to_csv('E:/7th_iteration/Test_7.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_6))
    
    TP_TN_RF_6=test6[test6['match']=='True']
    TP_TN_RF_6.to_csv('E:/6th_iteration/6.CSV')
    TP_TN_RF_6=TP_TN_RF_6.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_6.to_csv('E:/6th_iteration/TN_TP_6.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_6))

elif acc_naive_6>acc_rf_6 and acc_naive_6>acc_gb_6 :
    test6['nb_pre']=Naive_6      
    test6['match'] = np.where(test6['target'] == test6['nb_pre'], 'True', 'False')
    FP_FN_Na_6=test5[test6['match']=='False']
    FP_FN_Na_6=FP_FN_Na_6.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_6.to_csv('E:/7th_iteration/Test_7.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_6))
    
    TP_TN_Na_6=test6[test6['match']=='True']
    TP_TN_Na_6.to_csv('E:/6th_iteration/6.CSV')
    TP_TN_Na_6=TP_TN_Na_6.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_6.to_csv('E:/6th_iteration/TN_TP_6.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_6))
         
else:
    test6['gb_pre']=gb_6
    test6['match'] = np.where(test6['target'] == test6['gb_pre'], 'True', 'False')
    FP_FN_gb_6=test6[test6['match']=='False']
    FP_FN_gb_6=FP_FN_gb_6.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_6.to_csv('E:/7th_iteration/Test_7.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_6))
    
    TP_TN_gb_6=test6[test6['match']=='True']
    TP_TN_gb_6.to_csv('E:/6th_iteration/6.CSV')
    TP_TN_gb_6=TP_TN_gb_6.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_6.to_csv('E:/6th_iteration/TN_TP_6.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_6))


# # 7th_iteraition

# In[3]:


train7=pd.read_csv("E:/1st_iteration/Train.CSV")
test7=pd.read_csv("E:/7th_iteration/Test_7.csv")

x_train7 =train7[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train7 = train7[['target']]

x_test7 = test7[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test7 = test7[['target']]

random_forest=RandomForestClassifier(random_state=50)
random_forest.fit(x_train7,y_train7)
rf_7=random_forest.predict(x_test7)
acc_rf_7=metrics.accuracy_score(y_test7,rf_7)*100

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train7,y_train7)
Naive_7=Naive_bayes.predict(x_test7)
acc_naive_7=metrics.accuracy_score(y_test7,Naive_7)*100

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train7, y_train7)
gb_7=gb.predict(x_test7)
acc_gb_7=metrics.accuracy_score(y_test7,gb_7)*100

print("Train_size:",len(train7),"| Test_size:",len(test7))
print("RF:",acc_rf_7)
print("Naivebayes:",acc_naive_7)
print("Gradient Boost:",acc_gb_7)

if acc_rf_7 > acc_naive_7 and acc_rf_7 > acc_gb_7 :
    test7['rf_pre']=rf_7
    test7['match'] = np.where(test7['target'] == test7['rf_pre'], 'True', 'False')
    FP_FN_RF_7=test7[test7['match']=='False']
    FP_FN_RF_7=FP_FN_RF_7.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_7.to_csv('E:/8th_iteration/Test_8.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_7))
    
    TP_TN_RF_7=test7[test7['match']=='True']
    TP_TN_RF_7.to_csv('E:/7th_iteration/7.CSV')
    TP_TN_RF_7=TP_TN_RF_7.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_7.to_csv('E:/7th_iteration/TN_TP_7.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_7))

elif acc_naive_7>acc_rf_7 and acc_naive_7>acc_gb_7 :
    test7['nb_pre']=Naive_7      
    test7['match'] = np.where(test7['target'] == test7['nb_pre'], 'True', 'False')
    FP_FN_Na_7=test7[test7['match']=='False']
    FP_FN_Na_7=FP_FN_Na_7.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_7.to_csv('E:/8th_iteration/Test_8.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_7))
    
    TP_TN_Na_7=test7[test7['match']=='True']
    TP_TN_Na_7.to_csv('E:/7th_iteration/7.CSV')
    TP_TN_Na_7=TP_TN_Na_7.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_7.to_csv('E:/7th_iteration/TN_TP_7.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_7))
         
else:
    test7['gb_pre']=gb_7
    test7['match'] = np.where(test7['target'] == test7['gb_pre'], 'True', 'False')
    FP_FN_gb_7=test7[test7['match']=='False']
    FP_FN_gb_7=FP_FN_gb_7.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_7.to_csv('E:/8th_iteration/Test_8.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_7))
    
    TP_TN_gb_7=test7[test7['match']=='True']
    TP_TN_gb_7.to_csv('E:/7th_iteration/7.CSV')
    TP_TN_gb_7=TP_TN_gb_7.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_7.to_csv('E:/7th_iteration/TN_TP_7.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_7))


# # 8th_iteration

# In[4]:


train8=pd.read_csv("E:/1st_iteration/Train.CSV")
test8=pd.read_csv("E:/8th_iteration/Test_8.csv")

x_train8 =train8[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train8 = train8[['target']]

x_test8 = test8[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test8 = test8[['target']]

random_forest=RandomForestClassifier(random_state=10)
random_forest.fit(x_train8,y_train8)
rf_8=random_forest.predict(x_test8)
acc_rf_8=metrics.accuracy_score(y_test8,rf_8)*100

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train8,y_train8)
Naive_8=Naive_bayes.predict(x_test8)
acc_naive_8=metrics.accuracy_score(y_test8,Naive_8)*100

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train8, y_train8)
gb_8=gb.predict(x_test8)
acc_gb_8=metrics.accuracy_score(y_test8,gb_8)*100

print("Train_size:",len(train8),"| Test_size:",len(test8))
print("RF:",acc_rf_8)
print("Naivebayes:",acc_naive_8)
print("Gradient Boost:",acc_gb_8)

if acc_rf_8 > acc_naive_8 and acc_rf_8 > acc_gb_8 :
    test8['rf_pre']=rf_8
    test8['match'] = np.where(test8['target'] == test8['rf_pre'], 'True', 'False')
    FP_FN_RF_8=test8[test8['match']=='False']
    FP_FN_RF_8=FP_FN_RF_8.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_8.to_csv('E:/9th_iteration/Test_9.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_8))
    
    TP_TN_RF_8=test8[test8['match']=='True']
    TP_TN_RF_8.to_csv('E:/8th_iteration/8.CSV')
    TP_TN_RF_8=TP_TN_RF_8.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_8.to_csv('E:/8th_iteration/TN_TP_8.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_8))

elif acc_naive_8>acc_rf_8 and acc_naive_8>acc_gb_8 :
    test8['nb_pre']=Naive_8      
    test8['match'] = np.where(test8['target'] == test8['nb_pre'], 'True', 'False')
    FP_FN_Na_8=test8[test8['match']=='False']
    FP_FN_Na_8=FP_FN_Na_8.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_8.to_csv('E:/9th_iteration/Test_9.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_8))
    
    TP_TN_Na_8=test8[test8['match']=='True']
    TP_TN_Na_8.to_csv('E:/8th_iteration/8.CSV')
    TP_TN_Na_8=TP_TN_Na_8.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_8.to_csv('E:/8th_iteration/TN_TP_8.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_8))
         
else:
    test8['gb_pre']=gb_8
    test8['match'] = np.where(test8['target'] == test8['gb_pre'], 'True', 'False')
    FP_FN_gb_8=test8[test8['match']=='False']
    FP_FN_gb_8=FP_FN_gb_8.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_8.to_csv('E:/9th_iteration/Test_9.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_8))
    
    TP_TN_gb_8=test8[test8['match']=='True']
    TP_TN_gb_8.to_csv('E:/8th_iteration/8.CSV')
    TP_TN_gb_8=TP_TN_gb_8.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_8.to_csv('E:/8th_iteration/TN_TP_8.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_8))


# # 9th_iteration

# In[5]:


train9=pd.read_csv("E:/1st_iteration/Train.CSV")
test9=pd.read_csv("E:/9th_iteration/Test_9.csv")

x_train9 =train9[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train9 = train9[['target']]

x_test9 = test9[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test9 = test9[['target']]

random_forest=RandomForestClassifier(random_state=50)
random_forest.fit(x_train9,y_train9)
rf_9=random_forest.predict(x_test9)
acc_rf_9=metrics.accuracy_score(y_test9,rf_9)*100

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train9,y_train9)
Naive_9=Naive_bayes.predict(x_test9)
acc_naive_9=metrics.accuracy_score(y_test9,Naive_9)*100

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train9, y_train9)
gb_9=gb.predict(x_test9)
acc_gb_9=metrics.accuracy_score(y_test9,gb_9)*100

print("Train_size:",len(train9),"| Test_size:",len(test9))
print("RF:",acc_rf_9)
print("Naivebayes:",acc_naive_9)
print("Gradient Boost:",acc_gb_9)

if acc_rf_9 > acc_naive_9 and acc_rf_9 > acc_gb_9 :
    test9['rf_pre']=rf_9
    test9['match'] = np.where(test9['target'] == test9['rf_pre'], 'True', 'False')
    FP_FN_RF_9=test9[test9['match']=='False']
    FP_FN_RF_9=FP_FN_RF_9.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_9.to_csv('E:/10th_iteration/Test_10.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_9))
    
    TP_TN_RF_9=test9[test9['match']=='True']
    TP_TN_RF_9.to_csv('E:/9th_iteration/9.CSV')
    TP_TN_RF_9=TP_TN_RF_9.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_9.to_csv('E:/9th_iteration/TN_TP_9.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_9))

elif acc_naive_9>acc_rf_9 and acc_naive_9>acc_gb_9 :
    test9['nb_pre']=Naive_9      
    test9['match'] = np.where(test9['target'] == test9['nb_pre'], 'True', 'False')
    FP_FN_Na_9=test9[test9['match']=='False']
    FP_FN_Na_9=FP_FN_Na_9.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_9.to_csv('E:/10th_iteration/Test_10.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_9))
    
    TP_TN_Na_9=test9[test9['match']=='True']
    TP_TN_Na_9.to_csv('E:/9th_iteration/9.CSV')
    TP_TN_Na_9=TP_TN_Na_9.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_9.to_csv('E:/9th_iteration/TN_TP_9.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_9))
         
else:
    test9['gb_pre']=gb_9
    test9['match'] = np.where(test9['target'] == test9['gb_pre'], 'True', 'False')
    FP_FN_gb_9=test9[test9['match']=='False']
    FP_FN_gb_9=FP_FN_gb_9.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_9.to_csv('E:/10th_iteration/Test_10.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_9))
    
    TP_TN_gb_9=test9[test9['match']=='True']
    TP_TN_gb_9.to_csv('E:/9th_iteration/9.CSV')
    TP_TN_gb_9=TP_TN_gb_9.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_9.to_csv('E:/9th_iteration/TN_TP_9.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_9))


# # 10th_iteration

# In[6]:


train10=pd.read_csv("E:/1st_iteration/Train.CSV")
test10=pd.read_csv("E:/10th_iteration/Test_10.csv")

x_train10 =train10[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train10 = train10[['target']]

x_test10 = test10[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test10 = test10[['target']]

random_forest=RandomForestClassifier(random_state=50)
random_forest.fit(x_train10,y_train10)
rf_10=random_forest.predict(x_test10)
acc_rf_10=metrics.accuracy_score(y_test10,rf_10)*110

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train10,y_train10)
Naive_10=Naive_bayes.predict(x_test10)
acc_naive_10=metrics.accuracy_score(y_test10,Naive_10)*110

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train10, y_train10)
gb_10=gb.predict(x_test10)
acc_gb_10=metrics.accuracy_score(y_test10,gb_10)*110

print("Train_size:",len(train10),"| Test_size:",len(test10))
print("RF:",acc_rf_10)
print("Naivebayes:",acc_naive_10)
print("Gradient Boost:",acc_gb_10)

if acc_rf_10 > acc_naive_10 and acc_rf_10 > acc_gb_10 :
    test10['rf_pre']=rf_10
    test10['match'] = np.where(test10['target'] == test10['rf_pre'], 'True', 'False')
    FP_FN_RF_10=test10[test10['match']=='False']
    FP_FN_RF_10=FP_FN_RF_10.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_10.to_csv('E:/11th_iteration/Test_11.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_10))
    
    TP_TN_RF_10=test10[test10['match']=='True']
    TP_TN_RF_10.to_csv('E:/10th_iteration/10.CSV')
    TP_TN_RF_10=TP_TN_RF_10.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_10.to_csv('E:/10th_iteration/TN_TP_10.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_10))

elif acc_naive_10>acc_rf_10 and acc_naive_10>acc_gb_10 :
    test10['nb_pre']=Naive_10      
    test10['match'] = np.where(test10['target'] == test10['nb_pre'], 'True', 'False')
    FP_FN_Na_10=test10[test10['match']=='False']
    FP_FN_Na_10=FP_FN_Na_10.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_10.to_csv('E:/11th_iteration/Test_11.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_10))
    
    TP_TN_Na_10=test10[test10['match']=='True']
    TP_TN_Na_10.to_csv('E:/10th_iteration/10.CSV')
    TP_TN_Na_10=TP_TN_Na_10.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_10.to_csv('E:/10th_iteration/TN_TP_10.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_10))
         
else:
    test10['gb_pre']=gb_10
    test10['match'] = np.where(test10['target'] == test10['gb_pre'], 'True', 'False')
    FP_FN_gb_10=test10[test10['match']=='False']
    FP_FN_gb_10=FP_FN_gb_10.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_10.to_csv('E:/11th_iteration/Test_11.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_10))
    
    TP_TN_gb_10=test10[test10['match']=='True']
    TP_TN_gb_10.to_csv('E:/10th_iteration/10.CSV')
    TP_TN_gb_10=TP_TN_gb_10.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_10.to_csv('E:/10th_iteration/TN_TP_10.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_10))


# # 11th_iteration

# In[7]:


train11=pd.read_csv("E:/1st_iteration/Train.CSV")
test11=pd.read_csv("E:/11th_iteration/Test_11.csv")

x_train11 =train11[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train11 = train11[['target']]

x_test11 = test11[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test11 = test11[['target']]

random_forest=RandomForestClassifier(random_state=10)
random_forest.fit(x_train11,y_train11)
rf_11=random_forest.predict(x_test11)
acc_rf_11=metrics.accuracy_score(y_test11,rf_11)*120

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train11,y_train11)
Naive_11=Naive_bayes.predict(x_test11)
acc_naive_11=metrics.accuracy_score(y_test11,Naive_11)*120

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train11, y_train11)
gb_11=gb.predict(x_test11)
acc_gb_11=metrics.accuracy_score(y_test11,gb_11)*120

print("Train_size:",len(train11),"| Test_size:",len(test11))
print("RF:",acc_rf_11)
print("Naivebayes:",acc_naive_11)
print("Gradient Boost:",acc_gb_11)

if acc_rf_11 > acc_naive_11 and acc_rf_11 > acc_gb_11 :
    test11['rf_pre']=rf_11
    test11['match'] = np.where(test11['target'] == test11['rf_pre'], 'True', 'False')
    FP_FN_RF_11=test11[test11['match']=='False']
    FP_FN_RF_11=FP_FN_RF_11.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_11.to_csv('E:/12th_iteration/Test_12.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_11))
    
    TP_TN_RF_11=test11[test11['match']=='True']
    TP_TN_RF_11.to_csv('E:/11th_iteration/11.CSV')
    TP_TN_RF_11=TP_TN_RF_11.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_11.to_csv('E:/11th_iteration/TN_TP_11.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_11))

elif acc_naive_11>acc_rf_11 and acc_naive_11>acc_gb_11 :
    test11['nb_pre']=Naive_11      
    test11['match'] = np.where(test11['target'] == test11['nb_pre'], 'True', 'False')
    FP_FN_Na_11=test11[test11['match']=='False']
    FP_FN_Na_11=FP_FN_Na_11.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_11.to_csv('E:/12th_iteration/Test_12.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_11))
    
    TP_TN_Na_11=test11[test11['match']=='True']
    TP_TN_Na_11.to_csv('E:/11th_iteration/11.CSV')
    TP_TN_Na_11=TP_TN_Na_11.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_11.to_csv('E:/11th_iteration/TN_TP_11.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_11))
         
else:
    test11['gb_pre']=gb_11
    test11['match'] = np.where(test11['target'] == test11['gb_pre'], 'True', 'False')
    FP_FN_gb_11=test11[test11['match']=='False']
    FP_FN_gb_11=FP_FN_gb_11.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_11.to_csv('E:/12th_iteration/Test_12.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_11))
    
    TP_TN_gb_11=test11[test11['match']=='True']
    TP_TN_gb_11.to_csv('E:/11th_iteration/11.CSV')
    TP_TN_gb_11=TP_TN_gb_11.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_11.to_csv('E:/11th_iteration/TN_TP_11.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_11))


# # 12th_iteration

# In[8]:


train12=pd.read_csv("E:/1st_iteration/Train.CSV")
test12=pd.read_csv("E:/12th_iteration/Test_12.csv")

x_train12 =train12[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train12 = train12[['target']]

x_test12 = test12[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test12 = test12[['target']]

random_forest=RandomForestClassifier(random_state=50)
random_forest.fit(x_train12,y_train12)
rf_12=random_forest.predict(x_test12)
acc_rf_12=metrics.accuracy_score(y_test12,rf_12)*130

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train12,y_train12)
Naive_12=Naive_bayes.predict(x_test12)
acc_naive_12=metrics.accuracy_score(y_test12,Naive_12)*130

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train12, y_train12)
gb_12=gb.predict(x_test12)
acc_gb_12=metrics.accuracy_score(y_test12,gb_12)*130

print("Train_size:",len(train12),"| Test_size:",len(test12))
print("RF:",acc_rf_12)
print("Naivebayes:",acc_naive_12)
print("Gradient Boost:",acc_gb_12)

if acc_rf_12 > acc_naive_12 and acc_rf_12 > acc_gb_12 :
    test12['rf_pre']=rf_12
    test12['match'] = np.where(test12['target'] == test12['rf_pre'], 'True', 'False')
    FP_FN_RF_12=test12[test12['match']=='False']
    FP_FN_RF_12=FP_FN_RF_12.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_12.to_csv('E:/13th_iteration/Test_13.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_12))
    
    TP_TN_RF_12=test12[test12['match']=='True']
    TP_TN_RF_12.to_csv('E:/12th_iteration/12.CSV')
    TP_TN_RF_12=TP_TN_RF_12.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_12.to_csv('E:/12th_iteration/TN_TP_12.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_12))

elif acc_naive_12>acc_rf_12 and acc_naive_12>acc_gb_12 :
    test12['nb_pre']=Naive_12      
    test12['match'] = np.where(test12['target'] == test12['nb_pre'], 'True', 'False')
    FP_FN_Na_12=test12[test12['match']=='False']
    FP_FN_Na_12=FP_FN_Na_12.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_12.to_csv('E:/13th_iteration/Test_13.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_12))
    
    TP_TN_Na_12=test12[test12['match']=='True']
    TP_TN_Na_12.to_csv('E:/12th_iteration/12.CSV')
    TP_TN_Na_12=TP_TN_Na_12.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_12.to_csv('E:/12th_iteration/TN_TP_12.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_12))
         
else:
    test12['gb_pre']=gb_12
    test12['match'] = np.where(test12['target'] == test12['gb_pre'], 'True', 'False')
    FP_FN_gb_12=test12[test12['match']=='False']
    FP_FN_gb_12=FP_FN_gb_12.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_12.to_csv('E:/13th_iteration/Test_13.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_12))
    
    TP_TN_gb_12=test12[test12['match']=='True']
    TP_TN_gb_12.to_csv('E:/12th_iteration/12.CSV')
    TP_TN_gb_12=TP_TN_gb_12.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_12.to_csv('E:/12th_iteration/TN_TP_12.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_12))


# # 13th_iteration

# In[9]:


train13=pd.read_csv("E:/1st_iteration/Train.CSV")
test13=pd.read_csv("E:/13th_iteration/Test_13.csv")

x_train13 =train13[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train13 = train13[['target']]

x_test13 = test13[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test13 = test13[['target']]

random_forest=RandomForestClassifier(random_state=10)
random_forest.fit(x_train13,y_train13)
rf_13=random_forest.predict(x_test13)
acc_rf_13=metrics.accuracy_score(y_test13,rf_13)*140

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train13,y_train13)
Naive_13=Naive_bayes.predict(x_test13)
acc_naive_13=metrics.accuracy_score(y_test13,Naive_13)*140

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train13, y_train13)
gb_13=gb.predict(x_test13)
acc_gb_13=metrics.accuracy_score(y_test13,gb_13)*140

print("Train_size:",len(train13),"| Test_size:",len(test13))
print("RF:",acc_rf_13)
print("Naivebayes:",acc_naive_13)
print("Gradient Boost:",acc_gb_13)

if acc_rf_13 > acc_naive_13 and acc_rf_13 > acc_gb_13 :
    test13['rf_pre']=rf_13
    test13['match'] = np.where(test13['target'] == test13['rf_pre'], 'True', 'False')
    FP_FN_RF_13=test13[test13['match']=='False']
    FP_FN_RF_13=FP_FN_RF_13.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_13.to_csv('E:/14th_iteration/Test_14.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_13))
    
    TP_TN_RF_13=test13[test13['match']=='True']
    TP_TN_RF_13.to_csv('E:/13th_iteration/13.CSV')
    TP_TN_RF_13=TP_TN_RF_13.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_13.to_csv('E:/13th_iteration/TN_TP_13.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_13))

elif acc_naive_13>acc_rf_13 and acc_naive_13>acc_gb_13 :
    test13['nb_pre']=Naive_13      
    test13['match'] = np.where(test13['target'] == test13['nb_pre'], 'True', 'False')
    FP_FN_Na_13=test13[test13['match']=='False']
    FP_FN_Na_13=FP_FN_Na_13.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_13.to_csv('E:/14th_iteration/Test_14.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_13))
    
    TP_TN_Na_13=test13[test13['match']=='True']
    TP_TN_Na_13.to_csv('E:/13th_iteration/13.CSV')
    TP_TN_Na_13=TP_TN_Na_13.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_13.to_csv('E:/13th_iteration/TN_TP_13.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_13))
         
else:
    test13['gb_pre']=gb_13
    test13['match'] = np.where(test13['target'] == test13['gb_pre'], 'True', 'False')
    FP_FN_gb_13=test13[test13['match']=='False']
    FP_FN_gb_13=FP_FN_gb_13.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_13.to_csv('E:/14th_iteration/Test_14.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_13))
    
    TP_TN_gb_13=test13[test13['match']=='True']
    TP_TN_gb_13.to_csv('E:/13th_iteration/13.CSV')
    TP_TN_gb_13=TP_TN_gb_13.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_13.to_csv('E:/13th_iteration/TN_TP_13.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_13))


# # 14th_iteration

# In[10]:


train14=pd.read_csv("E:/1st_iteration/Train.CSV")
test14=pd.read_csv("E:/14th_iteration/Test_14.csv")

x_train14 =train14[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train14 = train14[['target']]

x_test14 = test14[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test14 = test14[['target']]

random_forest=RandomForestClassifier(random_state=15)
random_forest.fit(x_train14,y_train14)
rf_14=random_forest.predict(x_test14)
acc_rf_14=metrics.accuracy_score(y_test14,rf_14)*150

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train14,y_train14)
Naive_14=Naive_bayes.predict(x_test14)
acc_naive_14=metrics.accuracy_score(y_test14,Naive_14)*150

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train14, y_train14)
gb_14=gb.predict(x_test14)
acc_gb_14=metrics.accuracy_score(y_test14,gb_14)*150

print("Train_size:",len(train14),"| Test_size:",len(test14))
print("RF:",acc_rf_14)
print("Naivebayes:",acc_naive_14)
print("Gradient Boost:",acc_gb_14)

if acc_rf_14 > acc_naive_14 and acc_rf_14 > acc_gb_14 :
    test14['rf_pre']=rf_14
    test14['match'] = np.where(test14['target'] == test14['rf_pre'], 'True', 'False')
    FP_FN_RF_14=test14[test14['match']=='False']
    FP_FN_RF_14=FP_FN_RF_14.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_14.to_csv('E:/15th_iteration/Test_15.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_14))
    
    TP_TN_RF_14=test14[test14['match']=='True']
    TP_TN_RF_14.to_csv('E:/14th_iteration/14.CSV')
    TP_TN_RF_14=TP_TN_RF_14.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_14.to_csv('E:/14th_iteration/TN_TP_14.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_14))

elif acc_naive_14>acc_rf_14 and acc_naive_14>acc_gb_14 :
    test14['nb_pre']=Naive_14      
    test14['match'] = np.where(test14['target'] == test14['nb_pre'], 'True', 'False')
    FP_FN_Na_14=test14[test14['match']=='False']
    FP_FN_Na_14=FP_FN_Na_14.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_14.to_csv('E:/15th_iteration/Test_15.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_14))
    
    TP_TN_Na_14=test14[test14['match']=='True']
    TP_TN_Na_14.to_csv('E:/14th_iteration/14.CSV')
    TP_TN_Na_14=TP_TN_Na_14.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_14.to_csv('E:/14th_iteration/TN_TP_14.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_14))
         
else:
    test14['gb_pre']=gb_14
    test14['match'] = np.where(test14['target'] == test14['gb_pre'], 'True', 'False')
    FP_FN_gb_14=test14[test14['match']=='False']
    FP_FN_gb_14=FP_FN_gb_14.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_14.to_csv('E:/15th_iteration/Test_15.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_14))
    
    TP_TN_gb_14=test14[test14['match']=='True']
    TP_TN_gb_14.to_csv('E:/14th_iteration/14.CSV')
    TP_TN_gb_14=TP_TN_gb_14.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_14.to_csv('E:/14th_iteration/TN_TP_14.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_14))


# # 15th_iteration

# In[11]:


train15=pd.read_csv("E:/1st_iteration/Train.CSV")
test15=pd.read_csv("E:/15th_iteration/Test_15.csv")

x_train15 =train15[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train15 = train15[['target']]

x_test15 = test15[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test15 = test15[['target']]

random_forest=RandomForestClassifier(random_state=20)
random_forest.fit(x_train15,y_train15)
rf_15=random_forest.predict(x_test15)
acc_rf_15=metrics.accuracy_score(y_test15,rf_15)*160

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train15,y_train15)
Naive_15=Naive_bayes.predict(x_test15)
acc_naive_15=metrics.accuracy_score(y_test15,Naive_15)*160

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train15, y_train15)
gb_15=gb.predict(x_test15)
acc_gb_15=metrics.accuracy_score(y_test15,gb_15)*160

print("Train_size:",len(train15),"| Test_size:",len(test15))
print("RF:",acc_rf_15)
print("Naivebayes:",acc_naive_15)
print("Gradient Boost:",acc_gb_15)

if acc_rf_15 > acc_naive_15 and acc_rf_15 > acc_gb_15 :
    test15['rf_pre']=rf_15
    test15['match'] = np.where(test15['target'] == test15['rf_pre'], 'True', 'False')
    FP_FN_RF_15=test15[test15['match']=='False']
    FP_FN_RF_15=FP_FN_RF_15.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_15.to_csv('E:/16th_iteration/Test_16.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_15))
    
    TP_TN_RF_15=test15[test15['match']=='True']
    TP_TN_RF_15.to_csv('E:/15th_iteration/15.CSV')
    TP_TN_RF_15=TP_TN_RF_15.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_15.to_csv('E:/15th_iteration/TN_TP_15.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_15))

elif acc_naive_15>acc_rf_15 and acc_naive_15>acc_gb_15 :
    test15['nb_pre']=Naive_15      
    test15['match'] = np.where(test15['target'] == test15['nb_pre'], 'True', 'False')
    FP_FN_Na_15=test15[test15['match']=='False']
    FP_FN_Na_15=FP_FN_Na_15.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_15.to_csv('E:/16th_iteration/Test_16.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_15))
    
    TP_TN_Na_15=test15[test15['match']=='True']
    TP_TN_Na_15.to_csv('E:/15th_iteration/15.CSV')
    TP_TN_Na_15=TP_TN_Na_15.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_15.to_csv('E:/15th_iteration/TN_TP_15.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_15))
         
else:
    test15['gb_pre']=gb_15
    test15['match'] = np.where(test15['target'] == test15['gb_pre'], 'True', 'False')
    FP_FN_gb_15=test15[test15['match']=='False']
    FP_FN_gb_15=FP_FN_gb_15.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_15.to_csv('E:/16th_iteration/Test_16.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_15))
    
    TP_TN_gb_15=test15[test15['match']=='True']
    TP_TN_gb_15.to_csv('E:/15th_iteration/15.CSV')
    TP_TN_gb_15=TP_TN_gb_15.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_15.to_csv('E:/15th_iteration/TN_TP_15.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_15))


# # 16th_iteration

# In[12]:


train16=pd.read_csv("E:/1st_iteration/Train.CSV")
test16=pd.read_csv("E:/16th_iteration/Test_16.csv")

x_train16 =train16[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train16 = train16[['target']]

x_test16 = test16[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test16 = test16[['target']]

random_forest=RandomForestClassifier(random_state=30)
random_forest.fit(x_train16,y_train16)
rf_16=random_forest.predict(x_test16)
acc_rf_16=metrics.accuracy_score(y_test16,rf_16)*170

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train16,y_train16)
Naive_16=Naive_bayes.predict(x_test16)
acc_naive_16=metrics.accuracy_score(y_test16,Naive_16)*170

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train16, y_train16)
gb_16=gb.predict(x_test16)
acc_gb_16=metrics.accuracy_score(y_test16,gb_16)*170

print("Train_size:",len(train16),"| Test_size:",len(test16))
print("RF:",acc_rf_16)
print("Naivebayes:",acc_naive_16)
print("Gradient Boost:",acc_gb_16)

if acc_rf_16 > acc_naive_16 and acc_rf_16 > acc_gb_16 :
    test16['rf_pre']=rf_16
    test16['match'] = np.where(test16['target'] == test16['rf_pre'], 'True', 'False')
    FP_FN_RF_16=test16[test16['match']=='False']
    FP_FN_RF_16=FP_FN_RF_16.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_16.to_csv('E:/17th_iteration/Test_17.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_16))
    
    TP_TN_RF_16=test16[test16['match']=='True']
    TP_TN_RF_16.to_csv('E:/16th_iteration/16.CSV')
    TP_TN_RF_16=TP_TN_RF_16.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_16.to_csv('E:/16th_iteration/TN_TP_16.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_16))

elif acc_naive_16>acc_rf_16 and acc_naive_16>acc_gb_16 :
    test16['nb_pre']=Naive_16      
    test16['match'] = np.where(test16['target'] == test16['nb_pre'], 'True', 'False')
    FP_FN_Na_16=test16[test16['match']=='False']
    FP_FN_Na_16=FP_FN_Na_16.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_16.to_csv('E:/17th_iteration/Test_17.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_16))
    
    TP_TN_Na_16=test16[test16['match']=='True']
    TP_TN_Na_16.to_csv('E:/16th_iteration/16.CSV')
    TP_TN_Na_16=TP_TN_Na_16.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_16.to_csv('E:/16th_iteration/TN_TP_16.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_16))
         
else:
    test16['gb_pre']=gb_16
    test16['match'] = np.where(test16['target'] == test16['gb_pre'], 'True', 'False')
    FP_FN_gb_16=test16[test16['match']=='False']
    FP_FN_gb_16=FP_FN_gb_16.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_16.to_csv('E:/17th_iteration/Test_17.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_16))
    
    TP_TN_gb_16=test16[test16['match']=='True']
    TP_TN_gb_16.to_csv('E:/16th_iteration/16.CSV')
    TP_TN_gb_16=TP_TN_gb_16.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_16.to_csv('E:/16th_iteration/TN_TP_16.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_16))


# # 17th_iteration

# In[13]:


train17=pd.read_csv("E:/1st_iteration/Train.CSV")
test17=pd.read_csv("E:/17th_iteration/Test_17.csv")

x_train17 =train17[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train17 = train17[['target']]

x_test17 = test17[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test17 = test17[['target']]

random_forest=RandomForestClassifier(random_state=30)
random_forest.fit(x_train17,y_train17)
rf_17=random_forest.predict(x_test17)
acc_rf_17=metrics.accuracy_score(y_test17,rf_17)*180

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train17,y_train17)
Naive_17=Naive_bayes.predict(x_test17)
acc_naive_17=metrics.accuracy_score(y_test17,Naive_17)*180

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train17, y_train17)
gb_17=gb.predict(x_test17)
acc_gb_17=metrics.accuracy_score(y_test17,gb_17)*180

print("Train_size:",len(train17),"| Test_size:",len(test17))
print("RF:",acc_rf_17)
print("Naivebayes:",acc_naive_17)
print("Gradient Boost:",acc_gb_17)

if acc_rf_17 > acc_naive_17 and acc_rf_17 > acc_gb_17 :
    test17['rf_pre']=rf_17
    test17['match'] = np.where(test17['target'] == test17['rf_pre'], 'True', 'False')
    FP_FN_RF_17=test17[test17['match']=='False']
    FP_FN_RF_17=FP_FN_RF_17.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_17.to_csv('E:/18th_iteration/Test_18.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_17))
    
    TP_TN_RF_17=test17[test17['match']=='True']
    TP_TN_RF_17.to_csv('E:/17th_iteration/17.CSV')
    TP_TN_RF_17=TP_TN_RF_17.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_17.to_csv('E:/17th_iteration/TN_TP_17.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_17))

elif acc_naive_17>acc_rf_17 and acc_naive_17>acc_gb_17 :
    test17['nb_pre']=Naive_17      
    test17['match'] = np.where(test17['target'] == test17['nb_pre'], 'True', 'False')
    FP_FN_Na_17=test17[test17['match']=='False']
    FP_FN_Na_17=FP_FN_Na_17.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_17.to_csv('E:/18th_iteration/Test_18.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_17))
    
    TP_TN_Na_17=test17[test17['match']=='True']
    TP_TN_Na_17.to_csv('E:/17th_iteration/17.CSV')
    TP_TN_Na_17=TP_TN_Na_17.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_17.to_csv('E:/17th_iteration/TN_TP_17.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_17))
         
else:
    test17['gb_pre']=gb_17
    test17['match'] = np.where(test17['target'] == test17['gb_pre'], 'True', 'False')
    FP_FN_gb_17=test17[test17['match']=='False']
    FP_FN_gb_17=FP_FN_gb_17.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_17.to_csv('E:/18th_iteration/Test_18.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_17))
    
    TP_TN_gb_17=test17[test17['match']=='True']
    TP_TN_gb_17.to_csv('E:/17th_iteration/17.CSV')
    TP_TN_gb_17=TP_TN_gb_17.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_17.to_csv('E:/17th_iteration/TN_TP_17.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_17))


# # 18th_iteration

# In[14]:


train18=pd.read_csv("E:/1st_iteration/Train.CSV")
test18=pd.read_csv("E:/18th_iteration/Test_18.csv")

x_train18 =train18[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train18 = train18[['target']]

x_test18 = test18[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test18 = test18[['target']]

random_forest=RandomForestClassifier(random_state=10)
random_forest.fit(x_train18,y_train18)
rf_18=random_forest.predict(x_test18)
acc_rf_18=metrics.accuracy_score(y_test18,rf_18)*190

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train18,y_train18)
Naive_18=Naive_bayes.predict(x_test18)
acc_naive_18=metrics.accuracy_score(y_test18,Naive_18)*190

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train18, y_train18)
gb_18=gb.predict(x_test18)
acc_gb_18=metrics.accuracy_score(y_test18,gb_18)*190

print("Train_size:",len(train18),"| Test_size:",len(test18))
print("RF:",acc_rf_18)
print("Naivebayes:",acc_naive_18)
print("Gradient Boost:",acc_gb_18)

if acc_rf_18 > acc_naive_18 and acc_rf_18 > acc_gb_18 :
    test18['rf_pre']=rf_18
    test18['match'] = np.where(test18['target'] == test18['rf_pre'], 'True', 'False')
    FP_FN_RF_18=test18[test18['match']=='False']
    FP_FN_RF_18=FP_FN_RF_18.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_18.to_csv('E:/19th_iteration/Test_19.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_18))
    
    TP_TN_RF_18=test18[test18['match']=='True']
    TP_TN_RF_18.to_csv('E:/18th_iteration/18.CSV')
    TP_TN_RF_18=TP_TN_RF_18.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_18.to_csv('E:/18th_iteration/TN_TP_18.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_18))

elif acc_naive_18>acc_rf_18 and acc_naive_18>acc_gb_18 :
    test18['nb_pre']=Naive_18      
    test18['match'] = np.where(test18['target'] == test18['nb_pre'], 'True', 'False')
    FP_FN_Na_18=test18[test18['match']=='False']
    FP_FN_Na_18=FP_FN_Na_18.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_18.to_csv('E:/19th_iteration/Test_19.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_18))
    
    TP_TN_Na_18=test18[test18['match']=='True']
    TP_TN_Na_18.to_csv('E:/18th_iteration/18.CSV')
    TP_TN_Na_18=TP_TN_Na_18.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_18.to_csv('E:/18th_iteration/TN_TP_18.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_18))
         
else:
    test18['gb_pre']=gb_18
    test18['match'] = np.where(test18['target'] == test18['gb_pre'], 'True', 'False')
    FP_FN_gb_18=test18[test18['match']=='False']
    FP_FN_gb_18=FP_FN_gb_18.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_18.to_csv('E:/19th_iteration/Test_19.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_18))
    
    TP_TN_gb_18=test18[test18['match']=='True']
    TP_TN_gb_18.to_csv('E:/18th_iteration/18.CSV')
    TP_TN_gb_18=TP_TN_gb_18.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_18.to_csv('E:/18th_iteration/TN_TP_18.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_18))


# # 19th_iteration

# In[15]:


train19=pd.read_csv("E:/1st_iteration/Train.CSV")
test19=pd.read_csv("E:/19th_iteration/Test_19.csv")

x_train19 =train19[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train19 = train19[['target']]

x_test19 = test19[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test19 = test19[['target']]

random_forest=RandomForestClassifier(random_state=0)
random_forest.fit(x_train19,y_train19)
rf_19=random_forest.predict(x_test19)
acc_rf_19=metrics.accuracy_score(y_test19,rf_19)*200

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train19,y_train19)
Naive_19=Naive_bayes.predict(x_test19)
acc_naive_19=metrics.accuracy_score(y_test19,Naive_19)*200

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train19, y_train19)
gb_19=gb.predict(x_test19)
acc_gb_19=metrics.accuracy_score(y_test19,gb_19)*200

print("Train_size:",len(train19),"| Test_size:",len(test19))
print("RF:",acc_rf_19)
print("Naivebayes:",acc_naive_19)
print("Gradient Boost:",acc_gb_19)

if acc_rf_19 > acc_naive_19 and acc_rf_19 > acc_gb_19 :
    test19['rf_pre']=rf_19
    test19['match'] = np.where(test19['target'] == test19['rf_pre'], 'True', 'False')
    FP_FN_RF_19=test19[test19['match']=='False']
    FP_FN_RF_19=FP_FN_RF_19.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_19.to_csv('E:/20th_iteration/Test_20.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_19))
    
    TP_TN_RF_19=test19[test19['match']=='True']
    TP_TN_RF_19.to_csv('E:/19th_iteration/19.CSV')
    TP_TN_RF_19=TP_TN_RF_19.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_19.to_csv('E:/19th_iteration/TN_TP_19.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_19))

elif acc_naive_19>acc_rf_19 and acc_naive_19>acc_gb_19 :
    test19['nb_pre']=Naive_19      
    test19['match'] = np.where(test19['target'] == test19['nb_pre'], 'True', 'False')
    FP_FN_Na_19=test19[test19['match']=='False']
    FP_FN_Na_19=FP_FN_Na_19.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_19.to_csv('E:/20th_iteration/Test_20.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_19))
    
    TP_TN_Na_19=test19[test19['match']=='True']
    TP_TN_Na_19.to_csv('E:/19th_iteration/19.CSV')
    TP_TN_Na_19=TP_TN_Na_19.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_19.to_csv('E:/19th_iteration/TN_TP_19.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_19))
         
else:
    test19['gb_pre']=gb_19
    test19['match'] = np.where(test19['target'] == test19['gb_pre'], 'True', 'False')
    FP_FN_gb_19=test19[test19['match']=='False']
    FP_FN_gb_19=FP_FN_gb_19.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_19.to_csv('E:/20th_iteration/Test_20.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_19))
    
    TP_TN_gb_19=test19[test19['match']=='True']
    TP_TN_gb_19.to_csv('E:/19th_iteration/19.CSV')
    TP_TN_gb_19=TP_TN_gb_19.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_19.to_csv('E:/19th_iteration/TN_TP_19.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_19))


# # 20th_iteration

# In[16]:


train20=pd.read_csv("E:/1st_iteration/Train.CSV")
test20=pd.read_csv("E:/20th_iteration/Test_20.csv")

x_train20 =train20[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train20 = train20[['target']]

x_test20 = test20[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test20 = test20[['target']]

random_forest=RandomForestClassifier(random_state=10)
random_forest.fit(x_train20,y_train20)
rf_20=random_forest.predict(x_test20)
acc_rf_20=metrics.accuracy_score(y_test20,rf_20)*210

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train20,y_train20)
Naive_20=Naive_bayes.predict(x_test20)
acc_naive_20=metrics.accuracy_score(y_test20,Naive_20)*210

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train20, y_train20)
gb_20=gb.predict(x_test20)
acc_gb_20=metrics.accuracy_score(y_test20,gb_20)*210

print("Train_size:",len(train20),"| Test_size:",len(test20))
print("RF:",acc_rf_20)
print("Naivebayes:",acc_naive_20)
print("Gradient Boost:",acc_gb_20)

if acc_rf_20 > acc_naive_20 and acc_rf_20 > acc_gb_20 :
    test20['rf_pre']=rf_20
    test20['match'] = np.where(test20['target'] == test20['rf_pre'], 'True', 'False')
    FP_FN_RF_20=test20[test20['match']=='False']
    FP_FN_RF_20=FP_FN_RF_20.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_20.to_csv('E:/21th_iteration/Test_21.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_20))
    
    TP_TN_RF_20=test20[test20['match']=='True']
    TP_TN_RF_20.to_csv('E:/20th_iteration/20.CSV')
    TP_TN_RF_20=TP_TN_RF_20.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_20.to_csv('E:/20th_iteration/TN_TP_20.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_20))

elif acc_naive_20>acc_rf_20 and acc_naive_20>acc_gb_20 :
    test20['nb_pre']=Naive_20      
    test20['match'] = np.where(test20['target'] == test20['nb_pre'], 'True', 'False')
    FP_FN_Na_20=test20[test20['match']=='False']
    FP_FN_Na_20=FP_FN_Na_20.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_20.to_csv('E:/21th_iteration/Test_21.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_20))
    
    TP_TN_Na_20=test20[test20['match']=='True']
    TP_TN_Na_20.to_csv('E:/20th_iteration/20.CSV')
    TP_TN_Na_20=TP_TN_Na_20.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_20.to_csv('E:/20th_iteration/TN_TP_20.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_20))
         
else:
    test20['gb_pre']=gb_20
    test20['match'] = np.where(test20['target'] == test20['gb_pre'], 'True', 'False')
    FP_FN_gb_20=test20[test20['match']=='False']
    FP_FN_gb_20=FP_FN_gb_20.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_20.to_csv('E:/21th_iteration/Test_21.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_20))
    
    TP_TN_gb_20=test20[test20['match']=='True']
    TP_TN_gb_20.to_csv('E:/20th_iteration/20.CSV')
    TP_TN_gb_20=TP_TN_gb_20.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_20.to_csv('E:/20th_iteration/TN_TP_20.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_20))


# # 21th_iteration

# In[17]:


train21=pd.read_csv("E:/1st_iteration/Train.CSV")
test21=pd.read_csv("E:/21th_iteration/Test_21.csv")

x_train21 =train21[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train21 = train21[['target']]

x_test21 = test21[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test21 = test21[['target']]

random_forest=RandomForestClassifier(random_state=10)
random_forest.fit(x_train21,y_train21)
rf_21=random_forest.predict(x_test21)
acc_rf_21=metrics.accuracy_score(y_test21,rf_21)*221

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train21,y_train21)
Naive_21=Naive_bayes.predict(x_test21)
acc_naive_21=metrics.accuracy_score(y_test21,Naive_21)*221

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train21, y_train21)
gb_21=gb.predict(x_test21)
acc_gb_21=metrics.accuracy_score(y_test21,gb_21)*221

print("Train_size:",len(train21),"| Test_size:",len(test21))
print("RF:",acc_rf_21)
print("Naivebayes:",acc_naive_21)
print("Gradient Boost:",acc_gb_21)

if acc_rf_21 > acc_naive_21 and acc_rf_21 > acc_gb_21 :
    test21['rf_pre']=rf_21
    test21['match'] = np.where(test21['target'] == test21['rf_pre'], 'True', 'False')
    FP_FN_RF_21=test21[test21['match']=='False']
    FP_FN_RF_21=FP_FN_RF_21.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_21.to_csv('E:/22th_iteration/Test_22.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_21))
    
    TP_TN_RF_21=test21[test21['match']=='True']
    TP_TN_RF_21.to_csv('E:/21th_iteration/21.CSV')
    TP_TN_RF_21=TP_TN_RF_21.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_21.to_csv('E:/21th_iteration/TN_TP_21.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_21))

elif acc_naive_21>acc_rf_21 and acc_naive_21>acc_gb_21 :
    test21['nb_pre']=Naive_21      
    test21['match'] = np.where(test21['target'] == test21['nb_pre'], 'True', 'False')
    FP_FN_Na_21=test21[test21['match']=='False']
    FP_FN_Na_21=FP_FN_Na_21.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_21.to_csv('E:/22th_iteration/Test_22.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_21))
    
    TP_TN_Na_21=test21[test21['match']=='True']
    TP_TN_Na_21.to_csv('E:/21th_iteration/21.CSV')
    TP_TN_Na_21=TP_TN_Na_21.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_21.to_csv('E:/21th_iteration/TN_TP_21.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_21))
         
else:
    test21['gb_pre']=gb_21
    test21['match'] = np.where(test21['target'] == test21['gb_pre'], 'True', 'False')
    FP_FN_gb_21=test21[test21['match']=='False']
    FP_FN_gb_21=FP_FN_gb_21.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_21.to_csv('E:/22th_iteration/Test_22.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_21))
    
    TP_TN_gb_21=test21[test21['match']=='True']
    TP_TN_gb_21.to_csv('E:/21th_iteration/21.CSV')
    TP_TN_gb_21=TP_TN_gb_21.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_21.to_csv('E:/21th_iteration/TN_TP_21.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_21))


# # 22th_iteration

# In[ ]:


###### train22=pd.read_csv("E:/1st_iteration/Train.CSV")
test22=pd.read_csv("E:/22th_iteration/Test_22.csv")

x_train22 =train22[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train22 = train22[['target']]

x_test22 = test22[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test22 = test22[['target']]

random_forest=RandomForestClassifier(random_state=20)
random_forest.fit(x_train22,y_train22)
rf_22=random_forest.predict(x_test22)
acc_rf_22=metrics.accuracy_score(y_test22,rf_22)*231

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train22,y_train22)
Naive_22=Naive_bayes.predict(x_test22)
acc_naive_22=metrics.accuracy_score(y_test22,Naive_22)*231

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train22, y_train22)
gb_22=gb.predict(x_test22)
acc_gb_22=metrics.accuracy_score(y_test22,gb_22)*231

print("Train_size:",len(train22),"| Test_size:",len(test22))
print("RF:",acc_rf_22)
print("Naivebayes:",acc_naive_22)
print("Gradient Boost:",acc_gb_22)

if acc_rf_22 > acc_naive_22 and acc_rf_22 > acc_gb_22 :
    test22['rf_pre']=rf_22
    test22['match'] = np.where(test22['target'] == test22['rf_pre'], 'True', 'False')
    FP_FN_RF_22=test22[test22['match']=='False']
    FP_FN_RF_22=FP_FN_RF_22.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_22.to_csv('E:/23th_iteration/Test_23.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_22))
    
    TP_TN_RF_22=test22[test22['match']=='True']
    TP_TN_RF_22.to_csv('E:/22th_iteration/22.CSV')
    TP_TN_RF_22=TP_TN_RF_22.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_22.to_csv('E:/22th_iteration/TN_TP_22.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_22))

elif acc_naive_22>acc_rf_22 and acc_naive_22>acc_gb_22 :
    test22['nb_pre']=Naive_22      
    test22['match'] = np.where(test22['target'] == test22['nb_pre'], 'True', 'False')
    FP_FN_Na_22=test22[test22['match']=='False']
    FP_FN_Na_22=FP_FN_Na_22.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_22.to_csv('E:/23th_iteration/Test_23.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_22))
    
    TP_TN_Na_22=test22[test22['match']=='True']
    TP_TN_Na_22.to_csv('E:/22th_iteration/22.CSV')
    TP_TN_Na_22=TP_TN_Na_22.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_22.to_csv('E:/22th_iteration/TN_TP_22.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_22))
         
else:
    test22['gb_pre']=gb_22
    test22['match'] = np.where(test22['target'] == test22['gb_pre'], 'True', 'False')
    FP_FN_gb_22=test22[test22['match']=='False']
    FP_FN_gb_22=FP_FN_gb_22.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_22.to_csv('E:/23th_iteration/Test_23.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_22))
    
    TP_TN_gb_22=test22[test22['match']=='True']
    TP_TN_gb_22.to_csv('E:/22th_iteration/22.CSV')
    TP_TN_gb_22=TP_TN_gb_22.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_22.to_csv('E:/22th_iteration/TN_TP_22.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_22))


# # 23th_iteration

# In[19]:


train23=pd.read_csv("E:/1st_iteration/Train.CSV")
test23=pd.read_csv("E:/23th_iteration/Test_23.csv")

x_train23 =train23[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train23 = train23[['target']]

x_test23 = test23[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test23 = test23[['target']]

random_forest=RandomForestClassifier(random_state=20)
random_forest.fit(x_train23,y_train23)
rf_23=random_forest.predict(x_test23)
acc_rf_23=metrics.accuracy_score(y_test23,rf_23)*241

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train23,y_train23)
Naive_23=Naive_bayes.predict(x_test23)
acc_naive_23=metrics.accuracy_score(y_test23,Naive_23)*241

gb = GradientBoostingClassifier(random_state=10)
gb.fit(x_train23, y_train23)
gb_23=gb.predict(x_test23)
acc_gb_23=metrics.accuracy_score(y_test23,gb_23)*241

print("Train_size:",len(train23),"| Test_size:",len(test23))
print("RF:",acc_rf_23)
print("Naivebayes:",acc_naive_23)
print("Gradient Boost:",acc_gb_23)

if acc_rf_23 > acc_naive_23 and acc_rf_23 > acc_gb_23 :
    test23['rf_pre']=rf_23
    test23['match'] = np.where(test23['target'] == test23['rf_pre'], 'True', 'False')
    FP_FN_RF_23=test23[test23['match']=='False']
    FP_FN_RF_23=FP_FN_RF_23.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_23.to_csv('E:/24th_iteration/Test_24.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_23))
    
    TP_TN_RF_23=test23[test23['match']=='True']
    TP_TN_RF_23.to_csv('E:/23th_iteration/23.CSV')
    TP_TN_RF_23=TP_TN_RF_23.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_23.to_csv('E:/23th_iteration/TN_TP_23.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_23))

elif acc_naive_23>acc_rf_23 and acc_naive_23>acc_gb_23 :
    test23['nb_pre']=Naive_23      
    test23['match'] = np.where(test23['target'] == test23['nb_pre'], 'True', 'False')
    FP_FN_Na_23=test23[test23['match']=='False']
    FP_FN_Na_23=FP_FN_Na_23.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_23.to_csv('E:/24th_iteration/Test_24.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_23))
    
    TP_TN_Na_23=test23[test23['match']=='True']
    TP_TN_Na_23.to_csv('E:/23th_iteration/23.CSV')
    TP_TN_Na_23=TP_TN_Na_23.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_23.to_csv('E:/23th_iteration/TN_TP_23.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_23))
         
else:
    test23['gb_pre']=gb_23
    test23['match'] = np.where(test23['target'] == test23['gb_pre'], 'True', 'False')
    FP_FN_gb_23=test23[test23['match']=='False']
    FP_FN_gb_23=FP_FN_gb_23.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_23.to_csv('E:/24th_iteration/Test_24.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_23))
    
    TP_TN_gb_23=test23[test23['match']=='True']
    TP_TN_gb_23.to_csv('E:/23th_iteration/23.CSV')
    TP_TN_gb_23=TP_TN_gb_23.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_23.to_csv('E:/23th_iteration/TN_TP_23.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_23))


# # 24th_iteration

# In[20]:


train24=pd.read_csv("E:/1st_iteration/Train.CSV")
test24=pd.read_csv("E:/24th_iteration/Test_24.csv")

x_train24 =train24[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train24 = train24[['target']]

x_test24 = test24[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test24 = test24[['target']]

random_forest=RandomForestClassifier(random_state=10)
random_forest.fit(x_train24,y_train24)
rf_24=random_forest.predict(x_test24)
acc_rf_24=metrics.accuracy_score(y_test24,rf_24)*251

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train24,y_train24)
Naive_24=Naive_bayes.predict(x_test24)
acc_naive_24=metrics.accuracy_score(y_test24,Naive_24)*251

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train24, y_train24)
gb_24=gb.predict(x_test24)
acc_gb_24=metrics.accuracy_score(y_test24,gb_24)*251

print("Train_size:",len(train24),"| Test_size:",len(test24))
print("RF:",acc_rf_24)
print("Naivebayes:",acc_naive_24)
print("Gradient Boost:",acc_gb_24)

if acc_rf_24 > acc_naive_24 and acc_rf_24 > acc_gb_24 :
    test24['rf_pre']=rf_24
    test24['match'] = np.where(test24['target'] == test24['rf_pre'], 'True', 'False')
    FP_FN_RF_24=test24[test24['match']=='False']
    FP_FN_RF_24=FP_FN_RF_24.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_24.to_csv('E:/25th_iteration/Test_25.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_24))
    
    TP_TN_RF_24=test24[test24['match']=='True']
    TP_TN_RF_24.to_csv('E:/24th_iteration/24.CSV')
    TP_TN_RF_24=TP_TN_RF_24.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_24.to_csv('E:/24th_iteration/TN_TP_24.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_24))

elif acc_naive_24>acc_rf_24 and acc_naive_24>acc_gb_24 :
    test24['nb_pre']=Naive_24      
    test24['match'] = np.where(test24['target'] == test24['nb_pre'], 'True', 'False')
    FP_FN_Na_24=test24[test24['match']=='False']
    FP_FN_Na_24=FP_FN_Na_24.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_24.to_csv('E:/25th_iteration/Test_25.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_24))
    
    TP_TN_Na_24=test24[test24['match']=='True']
    TP_TN_Na_24.to_csv('E:/24th_iteration/24.CSV')
    TP_TN_Na_24=TP_TN_Na_24.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_24.to_csv('E:/24th_iteration/TN_TP_24.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_24))
         
else:
    test24['gb_pre']=gb_24
    test24['match'] = np.where(test24['target'] == test24['gb_pre'], 'True', 'False')
    FP_FN_gb_24=test24[test24['match']=='False']
    FP_FN_gb_24=FP_FN_gb_24.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_24.to_csv('E:/25th_iteration/Test_25.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_24))
    
    TP_TN_gb_24=test24[test24['match']=='True']
    TP_TN_gb_24.to_csv('E:/24th_iteration/24.CSV')
    TP_TN_gb_24=TP_TN_gb_24.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_24.to_csv('E:/24th_iteration/TN_TP_24.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_24))


# # 25th_iteration

# In[21]:


train25=pd.read_csv("E:/1st_iteration/Train.CSV")
test25=pd.read_csv("E:/25th_iteration/Test_25.csv")

x_train25 =train25[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train25 = train25[['target']]

x_test25 = test25[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test25 = test25[['target']]

random_forest=RandomForestClassifier(random_state=10)
random_forest.fit(x_train25,y_train25)
rf_25=random_forest.predict(x_test25)
acc_rf_25=metrics.accuracy_score(y_test25,rf_25)*261

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train25,y_train25)
Naive_25=Naive_bayes.predict(x_test25)
acc_naive_25=metrics.accuracy_score(y_test25,Naive_25)*261

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train25, y_train25)
gb_25=gb.predict(x_test25)
acc_gb_25=metrics.accuracy_score(y_test25,gb_25)*261

print("Train_size:",len(train25),"| Test_size:",len(test25))
print("RF:",acc_rf_25)
print("Naivebayes:",acc_naive_25)
print("Gradient Boost:",acc_gb_25)

if acc_rf_25 > acc_naive_25 and acc_rf_25 > acc_gb_25 :
    test25['rf_pre']=rf_25
    test25['match'] = np.where(test25['target'] == test25['rf_pre'], 'True', 'False')
    FP_FN_RF_25=test25[test25['match']=='False']
    FP_FN_RF_25=FP_FN_RF_25.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_25.to_csv('E:/26th_iteration/Test_26.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_25))
    
    TP_TN_RF_25=test25[test25['match']=='True']
    TP_TN_RF_25.to_csv('E:/25th_iteration/25.CSV')
    TP_TN_RF_25=TP_TN_RF_25.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_25.to_csv('E:/25th_iteration/TN_TP_25.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_25))

elif acc_naive_25>acc_rf_25 and acc_naive_25>acc_gb_25 :
    test25['nb_pre']=Naive_25      
    test25['match'] = np.where(test25['target'] == test25['nb_pre'], 'True', 'False')
    FP_FN_Na_25=test25[test25['match']=='False']
    FP_FN_Na_25=FP_FN_Na_25.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_25.to_csv('E:/26th_iteration/Test_26.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_25))
    
    TP_TN_Na_25=test25[test25['match']=='True']
    TP_TN_Na_25.to_csv('E:/25th_iteration/25.CSV')
    TP_TN_Na_25=TP_TN_Na_25.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_25.to_csv('E:/25th_iteration/TN_TP_25.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_25))
         
else:
    test25['gb_pre']=gb_25
    test25['match'] = np.where(test25['target'] == test25['gb_pre'], 'True', 'False')
    FP_FN_gb_25=test25[test25['match']=='False']
    FP_FN_gb_25=FP_FN_gb_25.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_25.to_csv('E:/26th_iteration/Test_26.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_25))
    
    TP_TN_gb_25=test25[test25['match']=='True']
    TP_TN_gb_25.to_csv('E:/25th_iteration/25.CSV')
    TP_TN_gb_25=TP_TN_gb_25.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_25.to_csv('E:/25th_iteration/TN_TP_25.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_25))


# # 26th_iteration

# In[22]:


train26=pd.read_csv("E:/1st_iteration/Train.CSV")
test26=pd.read_csv("E:/26th_iteration/Test_26.csv")

x_train26 =train26[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train26 = train26[['target']]

x_test26 = test26[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test26 = test26[['target']]

random_forest=RandomForestClassifier(random_state=30)
random_forest.fit(x_train26,y_train26)
rf_26=random_forest.predict(x_test26)
acc_rf_26=metrics.accuracy_score(y_test26,rf_26)*271

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train26,y_train26)
Naive_26=Naive_bayes.predict(x_test26)
acc_naive_26=metrics.accuracy_score(y_test26,Naive_26)*271

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train26, y_train26)
gb_26=gb.predict(x_test26)
acc_gb_26=metrics.accuracy_score(y_test26,gb_26)*271

print("Train_size:",len(train26),"| Test_size:",len(test26))
print("RF:",acc_rf_26)
print("Naivebayes:",acc_naive_26)
print("Gradient Boost:",acc_gb_26)

if acc_rf_26 > acc_naive_26 and acc_rf_26 > acc_gb_26 :
    test26['rf_pre']=rf_26
    test26['match'] = np.where(test26['target'] == test26['rf_pre'], 'True', 'False')
    FP_FN_RF_26=test26[test26['match']=='False']
    FP_FN_RF_26=FP_FN_RF_26.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_26.to_csv('E:/27th_iteration/Test_27.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_26))
    
    TP_TN_RF_26=test26[test26['match']=='True']
    TP_TN_RF_26.to_csv('E:/26th_iteration/26.CSV')
    TP_TN_RF_26=TP_TN_RF_26.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_26.to_csv('E:/26th_iteration/TN_TP_26.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_26))

elif acc_naive_26>acc_rf_26 and acc_naive_26>acc_gb_26 :
    test26['nb_pre']=Naive_26      
    test26['match'] = np.where(test26['target'] == test26['nb_pre'], 'True', 'False')
    FP_FN_Na_26=test26[test26['match']=='False']
    FP_FN_Na_26=FP_FN_Na_26.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_26.to_csv('E:/27th_iteration/Test_27.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_26))
    
    TP_TN_Na_26=test26[test26['match']=='True']
    TP_TN_Na_26.to_csv('E:/26th_iteration/26.CSV')
    TP_TN_Na_26=TP_TN_Na_26.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_26.to_csv('E:/26th_iteration/TN_TP_26.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_26))
         
else:
    test26['gb_pre']=gb_26
    test26['match'] = np.where(test26['target'] == test26['gb_pre'], 'True', 'False')
    FP_FN_gb_26=test26[test26['match']=='False']
    FP_FN_gb_26=FP_FN_gb_26.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_26.to_csv('E:/27th_iteration/Test_27.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_26))
    
    TP_TN_gb_26=test26[test26['match']=='True']
    TP_TN_gb_26.to_csv('E:/26th_iteration/26.CSV')
    TP_TN_gb_26=TP_TN_gb_26.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_26.to_csv('E:/26th_iteration/TN_TP_26.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_26))


# # 27th_iteration

# In[23]:


train27=pd.read_csv("E:/1st_iteration/Train.CSV")
test27=pd.read_csv("E:/27th_iteration/Test_27.csv")

x_train27 =train27[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train27 = train27[['target']]

x_test27 = test27[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test27 = test27[['target']]

random_forest=RandomForestClassifier(random_state=30)
random_forest.fit(x_train27,y_train27)
rf_27=random_forest.predict(x_test27)
acc_rf_27=metrics.accuracy_score(y_test27,rf_27)*281

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train27,y_train27)
Naive_27=Naive_bayes.predict(x_test27)
acc_naive_27=metrics.accuracy_score(y_test27,Naive_27)*281

gb = GradientBoostingClassifier(random_state=10)
gb.fit(x_train27, y_train27)
gb_27=gb.predict(x_test27)
acc_gb_27=metrics.accuracy_score(y_test27,gb_27)*281

print("Train_size:",len(train27),"| Test_size:",len(test27))
print("RF:",acc_rf_27)
print("Naivebayes:",acc_naive_27)
print("Gradient Boost:",acc_gb_27)

if acc_rf_27 > acc_naive_27 and acc_rf_27 > acc_gb_27 :
    test27['rf_pre']=rf_27
    test27['match'] = np.where(test27['target'] == test27['rf_pre'], 'True', 'False')
    FP_FN_RF_27=test27[test27['match']=='False']
    FP_FN_RF_27=FP_FN_RF_27.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_27.to_csv('E:/28th_iteration/Test_28.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_27))
    
    TP_TN_RF_27=test27[test27['match']=='True']
    TP_TN_RF_27.to_csv('E:/27th_iteration/27.CSV')
    TP_TN_RF_27=TP_TN_RF_27.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_27.to_csv('E:/27th_iteration/TN_TP_27.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_27))

elif acc_naive_27>acc_rf_27 and acc_naive_27>acc_gb_27 :
    test27['nb_pre']=Naive_27      
    test27['match'] = np.where(test27['target'] == test27['nb_pre'], 'True', 'False')
    FP_FN_Na_27=test27[test27['match']=='False']
    FP_FN_Na_27=FP_FN_Na_27.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_27.to_csv('E:/28th_iteration/Test_28.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_27))
    
    TP_TN_Na_27=test27[test27['match']=='True']
    TP_TN_Na_27.to_csv('E:/27th_iteration/27.CSV')
    TP_TN_Na_27=TP_TN_Na_27.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_27.to_csv('E:/27th_iteration/TN_TP_27.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_27))
         
else:
    test27['gb_pre']=gb_27
    test27['match'] = np.where(test27['target'] == test27['gb_pre'], 'True', 'False')
    FP_FN_gb_27=test27[test27['match']=='False']
    FP_FN_gb_27=FP_FN_gb_27.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_27.to_csv('E:/28th_iteration/Test_28.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_27))
    
    TP_TN_gb_27=test27[test27['match']=='True']
    TP_TN_gb_27.to_csv('E:/27th_iteration/27.CSV')
    TP_TN_gb_27=TP_TN_gb_27.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_27.to_csv('E:/27th_iteration/TN_TP_27.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_27))


# # 28th_iteration

# In[24]:


train28=pd.read_csv("E:/1st_iteration/Train.CSV")
test28=pd.read_csv("E:/28th_iteration/Test_28.csv")

x_train28 =train28[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train28 = train28[['target']]

x_test28 = test28[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test28 = test28[['target']]

random_forest=RandomForestClassifier(random_state=0)
random_forest.fit(x_train28,y_train28)
rf_28=random_forest.predict(x_test28)
acc_rf_28=metrics.accuracy_score(y_test28,rf_28)*291

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train28,y_train28)
Naive_28=Naive_bayes.predict(x_test28)
acc_naive_28=metrics.accuracy_score(y_test28,Naive_28)*291

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train28, y_train28)
gb_28=gb.predict(x_test28)
acc_gb_28=metrics.accuracy_score(y_test28,gb_28)*291

print("Train_size:",len(train28),"| Test_size:",len(test28))
print("RF:",acc_rf_28)
print("Naivebayes:",acc_naive_28)
print("Gradient Boost:",acc_gb_28)

if acc_rf_28 > acc_naive_28 and acc_rf_28 > acc_gb_28 :
    test28['rf_pre']=rf_28
    test28['match'] = np.where(test28['target'] == test28['rf_pre'], 'True', 'False')
    FP_FN_RF_28=test28[test28['match']=='False']
    FP_FN_RF_28=FP_FN_RF_28.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_28.to_csv('E:/29th_iteration/Test_29.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_28))
    
    TP_TN_RF_28=test28[test28['match']=='True']
    TP_TN_RF_28.to_csv('E:/28th_iteration/28.CSV')
    TP_TN_RF_28=TP_TN_RF_28.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_28.to_csv('E:/28th_iteration/TN_TP_28.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_28))

elif acc_naive_28>acc_rf_28 and acc_naive_28>acc_gb_28 :
    test28['nb_pre']=Naive_28      
    test28['match'] = np.where(test28['target'] == test28['nb_pre'], 'True', 'False')
    FP_FN_Na_28=test28[test28['match']=='False']
    FP_FN_Na_28=FP_FN_Na_28.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_28.to_csv('E:/29th_iteration/Test_29.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_28))
    
    TP_TN_Na_28=test28[test28['match']=='True']
    TP_TN_Na_28.to_csv('E:/28th_iteration/28.CSV')
    TP_TN_Na_28=TP_TN_Na_28.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_28.to_csv('E:/28th_iteration/TN_TP_28.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_28))
         
else:
    test28['gb_pre']=gb_28
    test28['match'] = np.where(test28['target'] == test28['gb_pre'], 'True', 'False')
    FP_FN_gb_28=test28[test28['match']=='False']
    FP_FN_gb_28=FP_FN_gb_28.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_28.to_csv('E:/29th_iteration/Test_29.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_28))
    
    TP_TN_gb_28=test28[test28['match']=='True']
    TP_TN_gb_28.to_csv('E:/28th_iteration/28.CSV')
    TP_TN_gb_28=TP_TN_gb_28.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_28.to_csv('E:/28th_iteration/TN_TP_28.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_28))


# # 29th_iteration

# In[25]:


train29=pd.read_csv("E:/1st_iteration/Train.CSV")
test29=pd.read_csv("E:/29th_iteration/Test_29.csv")

x_train29 =train29[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train29 = train29[['target']]

x_test29 = test29[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test29 = test29[['target']]

random_forest=RandomForestClassifier(random_state=0)
random_forest.fit(x_train29,y_train29)
rf_29=random_forest.predict(x_test29)
acc_rf_29=metrics.accuracy_score(y_test29,rf_29)*301

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train29,y_train29)
Naive_29=Naive_bayes.predict(x_test29)
acc_naive_29=metrics.accuracy_score(y_test29,Naive_29)*301

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train29, y_train29)
gb_29=gb.predict(x_test29)
acc_gb_29=metrics.accuracy_score(y_test29,gb_29)*301

print("Train_size:",len(train29),"| Test_size:",len(test29))
print("RF:",acc_rf_29)
print("Naivebayes:",acc_naive_29)
print("Gradient Boost:",acc_gb_29)

if acc_rf_29 > acc_naive_29 and acc_rf_29 > acc_gb_29 :
    test29['rf_pre']=rf_29
    test29['match'] = np.where(test29['target'] == test29['rf_pre'], 'True', 'False')
    FP_FN_RF_29=test29[test29['match']=='False']
    FP_FN_RF_29=FP_FN_RF_29.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_29.to_csv('E:/30th_iteration/Test_30.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_29))
    
    TP_TN_RF_29=test29[test29['match']=='True']
    TP_TN_RF_29.to_csv('E:/29th_iteration/29.CSV')
    TP_TN_RF_29=TP_TN_RF_29.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_29.to_csv('E:/29th_iteration/TN_TP_29.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_29))

elif acc_naive_29>acc_rf_29 and acc_naive_29>acc_gb_29 :
    test29['nb_pre']=Naive_29      
    test29['match'] = np.where(test29['target'] == test29['nb_pre'], 'True', 'False')
    FP_FN_Na_29=test29[test29['match']=='False']
    FP_FN_Na_29=FP_FN_Na_29.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_29.to_csv('E:/30th_iteration/Test_30.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_29))
    
    TP_TN_Na_29=test29[test29['match']=='True']
    TP_TN_Na_29.to_csv('E:/29th_iteration/29.CSV')
    TP_TN_Na_29=TP_TN_Na_29.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_29.to_csv('E:/29th_iteration/TN_TP_29.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_29))
         
else:
    test29['gb_pre']=gb_29
    test29['match'] = np.where(test29['target'] == test29['gb_pre'], 'True', 'False')
    FP_FN_gb_29=test29[test29['match']=='False']
    FP_FN_gb_29=FP_FN_gb_29.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_29.to_csv('E:/30th_iteration/Test_30.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_29))
    
    TP_TN_gb_29=test29[test29['match']=='True']
    TP_TN_gb_29.to_csv('E:/29th_iteration/29.CSV')
    TP_TN_gb_29=TP_TN_gb_29.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_29.to_csv('E:/29th_iteration/TN_TP_29.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_29))


# # 30th_iteration

# In[26]:


train30=pd.read_csv("E:/1st_iteration/Train.CSV")
test30=pd.read_csv("E:/30th_iteration/Test_30.csv")

x_train30 =train30[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train30 = train30[['target']]

x_test30 = test30[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test30 = test30[['target']]

random_forest=RandomForestClassifier(random_state=0)
random_forest.fit(x_train30,y_train30)
rf_30=random_forest.predict(x_test30)
acc_rf_30=metrics.accuracy_score(y_test30,rf_30)*311

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train30,y_train30)
Naive_30=Naive_bayes.predict(x_test30)
acc_naive_30=metrics.accuracy_score(y_test30,Naive_30)*311

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train30, y_train30)
gb_30=gb.predict(x_test30)
acc_gb_30=metrics.accuracy_score(y_test30,gb_30)*311

print("Train_size:",len(train30),"| Test_size:",len(test30))
print("RF:",acc_rf_30)
print("Naivebayes:",acc_naive_30)
print("Gradient Boost:",acc_gb_30)

if acc_rf_30 > acc_naive_30 and acc_rf_30 > acc_gb_30 :
    test30['rf_pre']=rf_30
    test30['match'] = np.where(test30['target'] == test30['rf_pre'], 'True', 'False')
    FP_FN_RF_30=test30[test30['match']=='False']
    FP_FN_RF_30=FP_FN_RF_30.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_30.to_csv('E:/31th_iteration/Test_31.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_30))
    
    TP_TN_RF_30=test30[test30['match']=='True']
    TP_TN_RF_30.to_csv('E:/30th_iteration/30.CSV')
    TP_TN_RF_30=TP_TN_RF_30.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_30.to_csv('E:/30th_iteration/TN_TP_30.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_30))

elif acc_naive_30>acc_rf_30 and acc_naive_30>acc_gb_30 :
    test30['nb_pre']=Naive_30      
    test30['match'] = np.where(test30['target'] == test30['nb_pre'], 'True', 'False')
    FP_FN_Na_30=test30[test30['match']=='False']
    FP_FN_Na_30=FP_FN_Na_30.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_30.to_csv('E:/31th_iteration/Test_31.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_30))
    
    TP_TN_Na_30=test30[test30['match']=='True']
    TP_TN_Na_30.to_csv('E:/30th_iteration/30.CSV')
    TP_TN_Na_30=TP_TN_Na_30.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_30.to_csv('E:/30th_iteration/TN_TP_30.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_30))
         
else:
    test30['gb_pre']=gb_30
    test30['match'] = np.where(test30['target'] == test30['gb_pre'], 'True', 'False')
    FP_FN_gb_30=test30[test30['match']=='False']
    FP_FN_gb_30=FP_FN_gb_30.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_30.to_csv('E:/31th_iteration/Test_31.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_30))
    
    TP_TN_gb_30=test30[test30['match']=='True']
    TP_TN_gb_30.to_csv('E:/30th_iteration/30.CSV')
    TP_TN_gb_30=TP_TN_gb_30.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_30.to_csv('E:/30th_iteration/TN_TP_30.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_30))


# # 31th_iteration

# In[ ]:


# train31=pd.read_csv("E:/1st_iteration/Train.CSV")
test31=pd.read_csv("E:/31th_iteration/Test_31.csv")

x_train31 =train31[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train31 = train31[['target']]

x_test31 = test31[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test31 = test31[['target']]

random_forest=RandomForestClassifier(random_state=0)
random_forest.fit(x_train31,y_train31)
rf_31=random_forest.predict(x_test31)
acc_rf_31=metrics.accuracy_score(y_test31,rf_31)*321

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train31,y_train31)
Naive_31=Naive_bayes.predict(x_test31)
acc_naive_31=metrics.accuracy_score(y_test31,Naive_31)*321

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train31, y_train31)
gb_31=gb.predict(x_test31)
acc_gb_31=metrics.accuracy_score(y_test31,gb_31)*321

print("Train_size:",len(train31),"| Test_size:",len(test31))
print("RF:",acc_rf_31)
print("Naivebayes:",acc_naive_31)
print("Gradient Boost:",acc_gb_31)

if acc_rf_31 > acc_naive_31 and acc_rf_31 > acc_gb_31 :
    test31['rf_pre']=rf_31
    test31['match'] = np.where(test31['target'] == test31['rf_pre'], 'True', 'False')
    FP_FN_RF_31=test31[test31['match']=='False']
    FP_FN_RF_31=FP_FN_RF_31.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_31.to_csv('E:/32th_iteration/Test_32.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_31))
    
    TP_TN_RF_31=test31[test31['match']=='True']
    TP_TN_RF_31.to_csv('E:/31th_iteration/31.CSV')
    TP_TN_RF_31=TP_TN_RF_31.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_31.to_csv('E:/31th_iteration/TN_TP_31.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_31))

elif acc_naive_31>acc_rf_31 and acc_naive_31>acc_gb_31 :
    test31['nb_pre']=Naive_31      
    test31['match'] = np.where(test31['target'] == test31['nb_pre'], 'True', 'False')
    FP_FN_Na_31=test31[test31['match']=='False']
    FP_FN_Na_31=FP_FN_Na_31.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_31.to_csv('E:/32th_iteration/Test_32.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_31))
    
    TP_TN_Na_31=test31[test31['match']=='True']
    TP_TN_Na_31.to_csv('E:/31th_iteration/31.CSV')
    TP_TN_Na_31=TP_TN_Na_31.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_31.to_csv('E:/31th_iteration/TN_TP_31.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_31))
         
else:
    test31['gb_pre']=gb_31
    test31['match'] = np.where(test31['target'] == test31['gb_pre'], 'True', 'False')
    FP_FN_gb_31=test31[test31['match']=='False']
    FP_FN_gb_31=FP_FN_gb_31.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_31.to_csv('E:/32th_iteration/Test_32.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_31))
    
    TP_TN_gb_31=test31[test31['match']=='True']
    TP_TN_gb_31.to_csv('E:/31th_iteration/31.CSV')
    TP_TN_gb_31=TP_TN_gb_31.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_31.to_csv('E:/31th_iteration/TN_TP_31.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_31))


# # 32th_iteration

# In[28]:


train32=pd.read_csv("E:/1st_iteration/Train.CSV")
test32=pd.read_csv("E:/32th_iteration/Test_32.csv")

x_train32 =train32[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train32 = train32[['target']]

x_test32 = test32[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test32 = test32[['target']]

random_forest=RandomForestClassifier(random_state=20)
random_forest.fit(x_train32,y_train32)
rf_32=random_forest.predict(x_test32)
acc_rf_32=metrics.accuracy_score(y_test32,rf_32)*332

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train32,y_train32)
Naive_32=Naive_bayes.predict(x_test32)
acc_naive_32=metrics.accuracy_score(y_test32,Naive_32)*332

gb = GradientBoostingClassifier(random_state=20)
gb.fit(x_train32, y_train32)
gb_32=gb.predict(x_test32)
acc_gb_32=metrics.accuracy_score(y_test32,gb_32)*332

print("Train_size:",len(train32),"| Test_size:",len(test32))
print("RF:",acc_rf_32)
print("Naivebayes:",acc_naive_32)
print("Gradient Boost:",acc_gb_32)

if (acc_rf_32 > acc_naive_32 and acc_rf_32 > acc_gb_32) or (acc_rf_32==acc_naive_32 and acc_rf_32>acc_gb_32) or (acc_rf_32==acc_gb_32 and acc_rf_32>acc_naive_32) or (acc_naive_32==acc_gb_32 and acc_rf_32>acc_naive_32):
    test32['rf_pre']=rf_32
    test32['match'] = np.where(test32['target'] == test32['rf_pre'], 'True', 'False')
    FP_FN_RF_32=test32[test32['match']=='False']
    FP_FN_RF_32=FP_FN_RF_32.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_32.to_csv('E:/33th_iteration/Test_33.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_32))
    
    TP_TN_RF_32=test32[test32['match']=='True']
    TP_TN_RF_32.to_csv('E:/32th_iteration/32.CSV')
    TP_TN_RF_32=TP_TN_RF_32.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_32.to_csv('E:/32th_iteration/TN_TP_32.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_32))

elif (acc_naive_32>acc_rf_32 and acc_naive_32>acc_gb_32) or (acc_rf_32==acc_naive_32 and acc_naive_32>acc_gb_32) or (acc_naive_32==acc_gb_32 and acc_naive_32>acc_rf_32) or (acc_rf_32==acc_gb_32 and acc_naive_32>acc_rf_32):
    test32['nb_pre']=Naive_32      
    test32['match'] = np.where(test32['target'] == test32['nb_pre'], 'True', 'False')
    FP_FN_Na_32=test32[test32['match']=='False']
    FP_FN_Na_32=FP_FN_Na_32.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_32.to_csv('E:/33th_iteration/Test_33.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_32))
    
    TP_TN_Na_32=test32[test32['match']=='True']
    TP_TN_Na_32.to_csv('E:/32th_iteration/32.CSV')
    TP_TN_Na_32=TP_TN_Na_32.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_32.to_csv('E:/32th_iteration/TN_TP_32.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_32))
         
elif (acc_gb_32>acc_rf_32 and acc_gb_32>acc_naive_32) or (acc_rf_32==acc_gb_32 and acc_gb_32>acc_naive_32) or (acc_naive_32==acc_gb_32 and acc_gb_32>acc_rf_32) or (acc_rf_32==acc_naive_32 and acc_gb_32>acc_rf_32) :
    test32['gb_pre']=gb_32
    test32['match'] = np.where(test32['target'] == test32['gb_pre'], 'True', 'False')
    FP_FN_gb_32=test32[test32['match']=='False']
    FP_FN_gb_32=FP_FN_gb_32.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_32.to_csv('E:/33th_iteration/Test_33.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_32))
    
    TP_TN_gb_32=test32[test32['match']=='True']
    TP_TN_gb_32.to_csv('E:/32th_iteration/32.CSV')
    TP_TN_gb_32=TP_TN_gb_32.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_32.to_csv('E:/32th_iteration/TN_TP_32.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_32))
    
else:
    test32['rf_pre']=rf_32
    test32['match'] = np.where(test32['target'] == test32['rf_pre'], 'True', 'False')
    FP_FN_RF_32=test32[test32['match']=='False']
    FP_FN_RF_32=FP_FN_RF_32.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_32.to_csv('E:/33th_iteration/Test_33.CSV')
    print("All Classifier Provides Same Accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_32))
    
    TP_TN_RF_32=test32[test32['match']=='True']
    TP_TN_RF_32.to_csv('E:/32th_iteration/32.CSV')
    TP_TN_RF_32=TP_TN_RF_32.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_32.to_csv('E:/32th_iteration/TN_TP_32.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_32))


# # 33th_iteration

# In[29]:


train33=pd.read_csv("E:/1st_iteration/Train.CSV")
test33=pd.read_csv("E:/33th_iteration/Test_33.csv")

x_train33 =train33[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train33 = train33[['target']]

x_test33 = test33[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test33 = test33[['target']]

random_forest=RandomForestClassifier(random_state=0)
random_forest.fit(x_train33,y_train33)
rf_33=random_forest.predict(x_test33)
acc_rf_33=metrics.accuracy_score(y_test33,rf_33)*342

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train33,y_train33)
Naive_33=Naive_bayes.predict(x_test33)
acc_naive_33=metrics.accuracy_score(y_test33,Naive_33)*342

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train33, y_train33)
gb_33=gb.predict(x_test33)
acc_gb_33=metrics.accuracy_score(y_test33,gb_33)*342

print("Train_size:",len(train33),"| Test_size:",len(test33))
print("RF:",acc_rf_33)
print("Naivebayes:",acc_naive_33)
print("Gradient Boost:",acc_gb_33)

if (acc_rf_33 > acc_naive_33 and acc_rf_33 > acc_gb_33) or (acc_rf_33==acc_naive_33 and acc_rf_33>acc_gb_33) or (acc_rf_33==acc_gb_33 and acc_rf_33>acc_naive_33) or (acc_naive_33==acc_gb_33 and acc_rf_33>acc_naive_33):
    test33['rf_pre']=rf_33
    test33['match'] = np.where(test33['target'] == test33['rf_pre'], 'True', 'False')
    FP_FN_RF_33=test33[test33['match']=='False']
    FP_FN_RF_33=FP_FN_RF_33.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_33.to_csv('E:/34th_iteration/Test_34.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_33))
    
    TP_TN_RF_33=test33[test33['match']=='True']
    TP_TN_RF_33.to_csv('E:/33th_iteration/33.CSV')
    TP_TN_RF_33=TP_TN_RF_33.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_33.to_csv('E:/33th_iteration/TN_TP_33.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_33))

elif (acc_naive_33>acc_rf_33 and acc_naive_33>acc_gb_33) or (acc_rf_33==acc_naive_33 and acc_naive_33>acc_gb_33) or (acc_naive_33==acc_gb_33 and acc_naive_33>acc_rf_33) or (acc_rf_33==acc_gb_33 and acc_naive_33>acc_rf_33):
    test33['nb_pre']=Naive_33      
    test33['match'] = np.where(test33['target'] == test33['nb_pre'], 'True', 'False')
    FP_FN_Na_33=test33[test33['match']=='False']
    FP_FN_Na_33=FP_FN_Na_33.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_33.to_csv('E:/34th_iteration/Test_34.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_33))
    
    TP_TN_Na_33=test33[test33['match']=='True']
    TP_TN_Na_33.to_csv('E:/33th_iteration/33.CSV')
    TP_TN_Na_33=TP_TN_Na_33.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_33.to_csv('E:/33th_iteration/TN_TP_33.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_33))
         
elif (acc_gb_33>acc_rf_33 and acc_gb_33>acc_naive_33) or (acc_rf_33==acc_gb_33 and acc_gb_33>acc_naive_33) or (acc_naive_33==acc_gb_33 and acc_gb_33>acc_rf_33) or (acc_rf_33==acc_naive_33 and acc_gb_33>acc_rf_33) :
    test33['gb_pre']=gb_33
    test33['match'] = np.where(test33['target'] == test33['gb_pre'], 'True', 'False')
    FP_FN_gb_33=test33[test33['match']=='False']
    FP_FN_gb_33=FP_FN_gb_33.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_33.to_csv('E:/34th_iteration/Test_34.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_33))
    
    TP_TN_gb_33=test33[test33['match']=='True']
    TP_TN_gb_33.to_csv('E:/33th_iteration/33.CSV')
    TP_TN_gb_33=TP_TN_gb_33.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_33.to_csv('E:/33th_iteration/TN_TP_33.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_33))
    
else:
    test33['rf_pre']=rf_33
    test33['match'] = np.where(test33['target'] == test33['rf_pre'], 'True', 'False')
    FP_FN_RF_33=test33[test33['match']=='False']
    FP_FN_RF_33=FP_FN_RF_33.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_33.to_csv('E:/34th_iteration/Test_34.CSV')
    print("All Classifier Provides Same Accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_33))
    
    TP_TN_RF_33=test33[test33['match']=='True']
    TP_TN_RF_33.to_csv('E:/33th_iteration/33.CSV')
    TP_TN_RF_33=TP_TN_RF_33.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_33.to_csv('E:/33th_iteration/TN_TP_33.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_33))


# # 34th_iteration

# In[30]:


train34=pd.read_csv("E:/1st_iteration/Train.CSV")
test34=pd.read_csv("E:/34th_iteration/Test_34.csv")

x_train34 =train34[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train34 = train34[['target']]

x_test34 = test34[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test34 = test34[['target']]

random_forest=RandomForestClassifier(random_state=20)
random_forest.fit(x_train34,y_train34)
rf_34=random_forest.predict(x_test34)
acc_rf_34=metrics.accuracy_score(y_test34,rf_34)*352

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train34,y_train34)
Naive_34=Naive_bayes.predict(x_test34)
acc_naive_34=metrics.accuracy_score(y_test34,Naive_34)*352

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train34, y_train34)
gb_34=gb.predict(x_test34)
acc_gb_34=metrics.accuracy_score(y_test34,gb_34)*352

print("Train_size:",len(train34),"| Test_size:",len(test34))
print("RF:",acc_rf_34)
print("Naivebayes:",acc_naive_34)
print("Gradient Boost:",acc_gb_34)

if (acc_rf_34 > acc_naive_34 and acc_rf_34 > acc_gb_34) or (acc_rf_34==acc_naive_34 and acc_rf_34>acc_gb_34) or (acc_rf_34==acc_gb_34 and acc_rf_34>acc_naive_34) or (acc_naive_34==acc_gb_34 and acc_rf_34>acc_naive_34):
    test34['rf_pre']=rf_34
    test34['match'] = np.where(test34['target'] == test34['rf_pre'], 'True', 'False')
    FP_FN_RF_34=test34[test34['match']=='False']
    FP_FN_RF_34=FP_FN_RF_34.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_34.to_csv('E:/35th_iteration/Test_35.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_34))
    
    TP_TN_RF_34=test34[test34['match']=='True']
    TP_TN_RF_34.to_csv('E:/34th_iteration/34.CSV')
    TP_TN_RF_34=TP_TN_RF_34.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_34.to_csv('E:/34th_iteration/TN_TP_34.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_34))

elif (acc_naive_34>acc_rf_34 and acc_naive_34>acc_gb_34) or (acc_rf_34==acc_naive_34 and acc_naive_34>acc_gb_34) or (acc_naive_34==acc_gb_34 and acc_naive_34>acc_rf_34) or (acc_rf_34==acc_gb_34 and acc_naive_34>acc_rf_34):
    test34['nb_pre']=Naive_34      
    test34['match'] = np.where(test34['target'] == test34['nb_pre'], 'True', 'False')
    FP_FN_Na_34=test34[test34['match']=='False']
    FP_FN_Na_34=FP_FN_Na_34.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_34.to_csv('E:/35th_iteration/Test_35.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_34))
    
    TP_TN_Na_34=test34[test34['match']=='True']
    TP_TN_Na_34.to_csv('E:/34th_iteration/34.CSV')
    TP_TN_Na_34=TP_TN_Na_34.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_34.to_csv('E:/34th_iteration/TN_TP_34.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_34))
         
elif (acc_gb_34>acc_rf_34 and acc_gb_34>acc_naive_34) or (acc_rf_34==acc_gb_34 and acc_gb_34>acc_naive_34) or (acc_naive_34==acc_gb_34 and acc_gb_34>acc_rf_34) or (acc_rf_34==acc_naive_34 and acc_gb_34>acc_rf_34) :
    test34['gb_pre']=gb_34
    test34['match'] = np.where(test34['target'] == test34['gb_pre'], 'True', 'False')
    FP_FN_gb_34=test34[test34['match']=='False']
    FP_FN_gb_34=FP_FN_gb_34.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_34.to_csv('E:/35th_iteration/Test_35.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_34))
    
    TP_TN_gb_34=test34[test34['match']=='True']
    TP_TN_gb_34.to_csv('E:/34th_iteration/34.CSV')
    TP_TN_gb_34=TP_TN_gb_34.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_34.to_csv('E:/34th_iteration/TN_TP_34.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_34))
    
else:
    test34['rf_pre']=rf_34
    test34['match'] = np.where(test34['target'] == test34['rf_pre'], 'True', 'False')
    FP_FN_RF_34=test34[test34['match']=='False']
    FP_FN_RF_34=FP_FN_RF_34.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_34.to_csv('E:/35th_iteration/Test_35.CSV')
    print("All Classifier Provides Same Accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_34))
    
    TP_TN_RF_34=test34[test34['match']=='True']
    TP_TN_RF_34.to_csv('E:/34th_iteration/34.CSV')
    TP_TN_RF_34=TP_TN_RF_34.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_34.to_csv('E:/34th_iteration/TN_TP_34.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_34))


# # 35th_iteration

# In[31]:


train35=pd.read_csv("E:/1st_iteration/Train.CSV")
test35=pd.read_csv("E:/35th_iteration/Test_35.csv")

x_train35 =train35[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train35 = train35[['target']]

x_test35 = test35[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test35 = test35[['target']]

random_forest=RandomForestClassifier(random_state=20)
random_forest.fit(x_train35,y_train35)
rf_35=random_forest.predict(x_test35)
acc_rf_35=metrics.accuracy_score(y_test35,rf_35)*362

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train35,y_train35)
Naive_35=Naive_bayes.predict(x_test35)
acc_naive_35=metrics.accuracy_score(y_test35,Naive_35)*362

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train35, y_train35)
gb_35=gb.predict(x_test35)
acc_gb_35=metrics.accuracy_score(y_test35,gb_35)*362

print("Train_size:",len(train35),"| Test_size:",len(test35))
print("RF:",acc_rf_35)
print("Naivebayes:",acc_naive_35)
print("Gradient Boost:",acc_gb_35)

if (acc_rf_35 > acc_naive_35 and acc_rf_35 > acc_gb_35) or (acc_rf_35==acc_naive_35 and acc_rf_35>acc_gb_35) or (acc_rf_35==acc_gb_35 and acc_rf_35>acc_naive_35) or (acc_naive_35==acc_gb_35 and acc_rf_35>acc_naive_35):
    test35['rf_pre']=rf_35
    test35['match'] = np.where(test35['target'] == test35['rf_pre'], 'True', 'False')
    FP_FN_RF_35=test35[test35['match']=='False']
    FP_FN_RF_35=FP_FN_RF_35.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_35.to_csv('E:/36th_iteration/Test_36.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_35))
    
    TP_TN_RF_35=test35[test35['match']=='True']
    TP_TN_RF_35.to_csv('E:/35th_iteration/35.CSV')
    TP_TN_RF_35=TP_TN_RF_35.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_35.to_csv('E:/35th_iteration/TN_TP_35.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_35))

elif (acc_naive_35>acc_rf_35 and acc_naive_35>acc_gb_35) or (acc_rf_35==acc_naive_35 and acc_naive_35>acc_gb_35) or (acc_naive_35==acc_gb_35 and acc_naive_35>acc_rf_35) or (acc_rf_35==acc_gb_35 and acc_naive_35>acc_rf_35):
    test35['nb_pre']=Naive_35      
    test35['match'] = np.where(test35['target'] == test35['nb_pre'], 'True', 'False')
    FP_FN_Na_35=test35[test35['match']=='False']
    FP_FN_Na_35=FP_FN_Na_35.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_35.to_csv('E:/36th_iteration/Test_36.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_35))
    
    TP_TN_Na_35=test35[test35['match']=='True']
    TP_TN_Na_35.to_csv('E:/35th_iteration/35.CSV')
    TP_TN_Na_35=TP_TN_Na_35.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_35.to_csv('E:/35th_iteration/TN_TP_35.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_35))
         
elif (acc_gb_35>acc_rf_35 and acc_gb_35>acc_naive_35) or (acc_rf_35==acc_gb_35 and acc_gb_35>acc_naive_35) or (acc_naive_35==acc_gb_35 and acc_gb_35>acc_rf_35) or (acc_rf_35==acc_naive_35 and acc_gb_35>acc_rf_35) :
    test35['gb_pre']=gb_35
    test35['match'] = np.where(test35['target'] == test35['gb_pre'], 'True', 'False')
    FP_FN_gb_35=test35[test35['match']=='False']
    FP_FN_gb_35=FP_FN_gb_35.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_35.to_csv('E:/36th_iteration/Test_36.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_35))
    
    TP_TN_gb_35=test35[test35['match']=='True']
    TP_TN_gb_35.to_csv('E:/35th_iteration/35.CSV')
    TP_TN_gb_35=TP_TN_gb_35.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_35.to_csv('E:/35th_iteration/TN_TP_35.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_35))
    
else:
    test35['rf_pre']=rf_35
    test35['match'] = np.where(test35['target'] == test35['rf_pre'], 'True', 'False')
    FP_FN_RF_35=test35[test35['match']=='False']
    FP_FN_RF_35=FP_FN_RF_35.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_35.to_csv('E:/36th_iteration/Test_36.CSV')
    print("All Classifier Provides Same Accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_35))
    
    TP_TN_RF_35=test35[test35['match']=='True']
    TP_TN_RF_35.to_csv('E:/35th_iteration/35.CSV')
    TP_TN_RF_35=TP_TN_RF_35.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_35.to_csv('E:/35th_iteration/TN_TP_35.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_35))


# # 36th_iteration

# In[32]:


train36=pd.read_csv("E:/1st_iteration/Train.CSV")
test36=pd.read_csv("E:/36th_iteration/Test_36.csv")

x_train36 =train36[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train36 = train36[['target']]

x_test36 = test36[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test36 = test36[['target']]

random_forest=RandomForestClassifier(random_state=20)
random_forest.fit(x_train36,y_train36)
rf_36=random_forest.predict(x_test36)
acc_rf_36=metrics.accuracy_score(y_test36,rf_36)*372

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train36,y_train36)
Naive_36=Naive_bayes.predict(x_test36)
acc_naive_36=metrics.accuracy_score(y_test36,Naive_36)*372

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train36, y_train36)
gb_36=gb.predict(x_test36)
acc_gb_36=metrics.accuracy_score(y_test36,gb_36)*372

print("Train_size:",len(train36),"| Test_size:",len(test36))
print("RF:",acc_rf_36)
print("Naivebayes:",acc_naive_36)
print("Gradient Boost:",acc_gb_36)

if (acc_rf_36 > acc_naive_36 and acc_rf_36 > acc_gb_36) or (acc_rf_36==acc_naive_36 and acc_rf_36>acc_gb_36) or (acc_rf_36==acc_gb_36 and acc_rf_36>acc_naive_36) or (acc_naive_36==acc_gb_36 and acc_rf_36>acc_naive_36):
    test36['rf_pre']=rf_36
    test36['match'] = np.where(test36['target'] == test36['rf_pre'], 'True', 'False')
    FP_FN_RF_36=test36[test36['match']=='False']
    FP_FN_RF_36=FP_FN_RF_36.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_36.to_csv('E:/37th_iteration/Test_37.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_36))
    
    TP_TN_RF_36=test36[test36['match']=='True']
    TP_TN_RF_36.to_csv('E:/36th_iteration/36.CSV')
    TP_TN_RF_36=TP_TN_RF_36.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_36.to_csv('E:/36th_iteration/TN_TP_36.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_36))

elif (acc_naive_36>acc_rf_36 and acc_naive_36>acc_gb_36) or (acc_rf_36==acc_naive_36 and acc_naive_36>acc_gb_36) or (acc_naive_36==acc_gb_36 and acc_naive_36>acc_rf_36) or (acc_rf_36==acc_gb_36 and acc_naive_36>acc_rf_36):
    test36['nb_pre']=Naive_36      
    test36['match'] = np.where(test36['target'] == test36['nb_pre'], 'True', 'False')
    FP_FN_Na_36=test36[test36['match']=='False']
    FP_FN_Na_36=FP_FN_Na_36.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_36.to_csv('E:/37th_iteration/Test_37.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_36))
    
    TP_TN_Na_36=test36[test36['match']=='True']
    TP_TN_Na_36.to_csv('E:/36th_iteration/36.CSV')
    TP_TN_Na_36=TP_TN_Na_36.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_36.to_csv('E:/36th_iteration/TN_TP_36.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_36))
         
elif (acc_gb_36>acc_rf_36 and acc_gb_36>acc_naive_36) or (acc_rf_36==acc_gb_36 and acc_gb_36>acc_naive_36) or (acc_naive_36==acc_gb_36 and acc_gb_36>acc_rf_36) or (acc_rf_36==acc_naive_36 and acc_gb_36>acc_rf_36) :
    test36['gb_pre']=gb_36
    test36['match'] = np.where(test36['target'] == test36['gb_pre'], 'True', 'False')
    FP_FN_gb_36=test36[test36['match']=='False']
    FP_FN_gb_36=FP_FN_gb_36.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_36.to_csv('E:/37th_iteration/Test_37.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_36))
    
    TP_TN_gb_36=test36[test36['match']=='True']
    TP_TN_gb_36.to_csv('E:/36th_iteration/36.CSV')
    TP_TN_gb_36=TP_TN_gb_36.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_36.to_csv('E:/36th_iteration/TN_TP_36.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_36))
    
else:
    test36['rf_pre']=rf_36
    test36['match'] = np.where(test36['target'] == test36['rf_pre'], 'True', 'False')
    FP_FN_RF_36=test36[test36['match']=='False']
    FP_FN_RF_36=FP_FN_RF_36.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_36.to_csv('E:/37th_iteration/Test_37.CSV')
    print("All Classifier Provides Same Accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_36))
    
    TP_TN_RF_36=test36[test36['match']=='True']
    TP_TN_RF_36.to_csv('E:/36th_iteration/36.CSV')
    TP_TN_RF_36=TP_TN_RF_36.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_36.to_csv('E:/36th_iteration/TN_TP_36.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_36))


# # 37th_iteration

# In[33]:


train37=pd.read_csv("E:/1st_iteration/Train.CSV")
test37=pd.read_csv("E:/37th_iteration/Test_37.csv")

x_train37 =train37[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train37 = train37[['target']]

x_test37 = test37[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test37 = test37[['target']]

random_forest=RandomForestClassifier(random_state=20)
random_forest.fit(x_train37,y_train37)
rf_37=random_forest.predict(x_test37)
acc_rf_37=metrics.accuracy_score(y_test37,rf_37)*382

Naive_bayes = GaussianNB()
Naive_bayes.fit(x_train37,y_train37)
Naive_37=Naive_bayes.predict(x_test37)
acc_naive_37=metrics.accuracy_score(y_test37,Naive_37)*382

gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train37, y_train37)
gb_37=gb.predict(x_test37)
acc_gb_37=metrics.accuracy_score(y_test37,gb_37)*382

print("Train_size:",len(train37),"| Test_size:",len(test37))
print("RF:",acc_rf_37)
print("Naivebayes:",acc_naive_37)
print("Gradient Boost:",acc_gb_37)

if (acc_rf_37 > acc_naive_37 and acc_rf_37 > acc_gb_37) or (acc_rf_37==acc_naive_37 and acc_rf_37>acc_gb_37) or (acc_rf_37==acc_gb_37 and acc_rf_37>acc_naive_37) or (acc_naive_37==acc_gb_37 and acc_rf_37>acc_naive_37):
    test37['rf_pre']=rf_37
    test37['match'] = np.where(test37['target'] == test37['rf_pre'], 'True', 'False')
    FP_FN_RF_37=test37[test37['match']=='False']
    FP_FN_RF_37=FP_FN_RF_37.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_37.to_csv('E:/38th_iteration/Test_38.CSV')
    print("RF Provides best accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_37))
    
    TP_TN_RF_37=test37[test37['match']=='True']
    TP_TN_RF_37.to_csv('E:/37th_iteration/37.CSV')
    TP_TN_RF_37=TP_TN_RF_37.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_37.to_csv('E:/37th_iteration/TN_TP_37.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_37))

elif (acc_naive_37>acc_rf_37 and acc_naive_37>acc_gb_37) or (acc_rf_37==acc_naive_37 and acc_naive_37>acc_gb_37) or (acc_naive_37==acc_gb_37 and acc_naive_37>acc_rf_37) or (acc_rf_37==acc_gb_37 and acc_naive_37>acc_rf_37):
    test37['nb_pre']=Naive_37      
    test37['match'] = np.where(test37['target'] == test37['nb_pre'], 'True', 'False')
    FP_FN_Na_37=test37[test37['match']=='False']
    FP_FN_Na_37=FP_FN_Na_37.drop(['nb_pre','match'], axis=1)
    FP_FN_Na_37.to_csv('E:/38th_iteration/Test_38.CSV')
    print("Naive Bayes Provides best accuracy")
    print("na_FP_FN: ",len(FP_FN_Na_37))
    
    TP_TN_Na_37=test37[test37['match']=='True']
    TP_TN_Na_37.to_csv('E:/37th_iteration/37.CSV')
    TP_TN_Na_37=TP_TN_Na_37.drop(['nb_pre','match'], axis=1)
    TP_TN_Na_37.to_csv('E:/37th_iteration/TN_TP_37.CSV')
    print("na_TN_TP: ",len(TP_TN_Na_37))
         
elif (acc_gb_37>acc_rf_37 and acc_gb_37>acc_naive_37) or (acc_rf_37==acc_gb_37 and acc_gb_37>acc_naive_37) or (acc_naive_37==acc_gb_37 and acc_gb_37>acc_rf_37) or (acc_rf_37==acc_naive_37 and acc_gb_37>acc_rf_37) :
    test37['gb_pre']=gb_37
    test37['match'] = np.where(test37['target'] == test37['gb_pre'], 'True', 'False')
    FP_FN_gb_37=test37[test37['match']=='False']
    FP_FN_gb_37=FP_FN_gb_37.drop(['gb_pre','match'], axis=1)
    FP_FN_gb_37.to_csv('E:/38th_iteration/Test_38.CSV')
    print("GB Provides best accuracy: ")
    print("gb_FP_FN: ",len(FP_FN_gb_37))
    
    TP_TN_gb_37=test37[test37['match']=='True']
    TP_TN_gb_37.to_csv('E:/37th_iteration/37.CSV')
    TP_TN_gb_37=TP_TN_gb_37.drop(['gb_pre','match'], axis=1)
    TP_TN_gb_37.to_csv('E:/37th_iteration/TN_TP_37.CSV')
    print("gb_TN_TP: ",len(TP_TN_gb_37))
    
else:
    test37['rf_pre']=rf_37
    test37['match'] = np.where(test37['target'] == test37['rf_pre'], 'True', 'False')
    FP_FN_RF_37=test37[test37['match']=='False']
    FP_FN_RF_37=FP_FN_RF_37.drop(['rf_pre','match'], axis=1)
    FP_FN_RF_37.to_csv('E:/38th_iteration/Test_38.CSV')
    print("All Classifier Provides Same Accuracy")
    print("rf_FP_FN: ",len(FP_FN_RF_37))
    
    TP_TN_RF_37=test37[test37['match']=='True']
    TP_TN_RF_37.to_csv('E:/37th_iteration/37.CSV')
    TP_TN_RF_37=TP_TN_RF_37.drop(['rf_pre','match'], axis=1)
    TP_TN_RF_37.to_csv('E:/37th_iteration/TN_TP_37.CSV')
    print("rf_TN_TP: ",len(TP_TN_RF_37))


# # 38h_iteration

# In[36]:


from sklearn.neighbors import KNeighborsClassifier

train37=pd.read_csv("E:/1st_iteration/Train.CSV")
test37=pd.read_csv("E:/37th_iteration/Test_37.csv")

x_train37 =train37[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train37 = train37[['target']]

x_test37 = test37[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test37 = test37[['target']]


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train37,y_train37)
knn_37=knn.predict(x_test37)
acc_knn_37=metrics.accuracy_score(y_test37,knn_37)*100

print("Train_size:",len(train37),"| Test_size:",len(test37))
print("Knn:",acc_knn_37)


test37['knn_pre']=knn_37
test37['match'] = np.where(test37['target'] == test37['knn_pre'], 'True', 'False')
FP_FN_knn_37=test37[test37['match']=='False']
FP_FN_knn_37=FP_FN_knn_37.drop(['knn_pre','match'], axis=1)
FP_FN_knn_37.to_csv('E:/38th_iteration/Test_38.CSV')
print("knn_FP_FN: ",len(FP_FN_knn_37))
    
TP_TN_knn_37=test37[test37['match']=='True']
TP_TN_knn_37.to_csv('E:/37th_iteration/37.CSV')
TP_TN_knn_37=TP_TN_knn_37.drop(['knn_pre','match'], axis=1)
TP_TN_knn_37.to_csv('E:/37th_iteration/TN_TP_37.CSV')
print("knn_TN_TP: ",len(TP_TN_knn_37))


# # 39th_iteration

# In[38]:


from sklearn.neighbors import KNeighborsClassifier

train38=pd.read_csv("E:/1st_iteration/Train.CSV")
test38=pd.read_csv("E:/38th_iteration/Test_38.csv")

x_train38 =train38[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train38 = train38[['target']]

x_test38 = test38[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test38 = test38[['target']]


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train38,y_train38)
knn_38=knn.predict(x_test38)
acc_knn_38=metrics.accuracy_score(y_test38,knn_38)*100

print("Train_size:",len(train38),"| Test_size:",len(test38))
print("Knn:",acc_knn_38)


test38['knn_pre']=knn_38
test38['match'] = np.where(test38['target'] == test38['knn_pre'], 'True', 'False')
FP_FN_knn_38=test38[test38['match']=='False']
FP_FN_knn_38=FP_FN_knn_38.drop(['knn_pre','match'], axis=1)
FP_FN_knn_38.to_csv('E:/39th_iteration/Test_39.CSV')
print("knn_FP_FN: ",len(FP_FN_knn_38))
    
TP_TN_knn_38=test38[test38['match']=='True']
TP_TN_knn_38.to_csv('E:/38th_iteration/38.CSV')
TP_TN_knn_38=TP_TN_knn_38.drop(['knn_pre','match'], axis=1)
TP_TN_knn_38.to_csv('E:/38th_iteration/TN_TP_38.CSV')
print("knn_TN_TP: ",len(TP_TN_knn_38))


# # 40th iteration

# In[40]:


from sklearn.neighbors import KNeighborsClassifier

train39=pd.read_csv("E:/1st_iteration/Train.CSV")
test39=pd.read_csv("E:/39th_iteration/Test_39.csv")

x_train39 =train39[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train39 = train39[['target']]

x_test39 = test39[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test39 = test39[['target']]
error = []

from sklearn.neighbors import KNeighborsClassifier

train39=pd.read_csv("E:/1st_iteration/Train.CSV")
test39=pd.read_csv("E:/39th_iteration/Test_39.csv")

x_train39 =train39[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train39 = train39[['target']]

x_test39 = test39[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test39 = test39[['target']]


knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train39,y_train39)
knn_39=knn.predict(x_test39)
acc_knn_39=metrics.accuracy_score(y_test39,knn_39)*100

print("Train_size:",len(train39),"| Test_size:",len(test39))
print("Knn:",acc_knn_39)


test39['knn_pre']=knn_39
test39['match'] = np.where(test39['target'] == test39['knn_pre'], 'True', 'False')
FP_FN_knn_39=test39[test39['match']=='False']
FP_FN_knn_39=FP_FN_knn_39.drop(['knn_pre','match'], axis=1)
FP_FN_knn_39.to_csv('E:/40th_iteration/Test_40.CSV')
print("knn_FP_FN: ",len(FP_FN_knn_39))
    
TP_TN_knn_39=test39[test39['match']=='True']
TP_TN_knn_39.to_csv('E:/39th_iteration/39.CSV')
TP_TN_knn_39=TP_TN_knn_39.drop(['knn_pre','match'], axis=1)
TP_TN_knn_39.to_csv('E:/39th_iteration/TN_TP_39.CSV')
print("knn_TN_TP: ",len(TP_TN_knn_39))


# # 41th_iteration

# In[42]:


from sklearn.neighbors import KNeighborsClassifier

train40=pd.read_csv("E:/1st_iteration/Train.CSV")
test40=pd.read_csv("E:/40th_iteration/Test_40.csv")

x_train40 =train40[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train40 = train40[['target']]

x_test40 = test40[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test40 = test40[['target']]


knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train40,y_train40)
knn_40=knn.predict(x_test40)
acc_knn_40=metrics.accuracy_score(y_test40,knn_40)*100

print("Train_size:",len(train40),"| Test_size:",len(test40))
print("Knn:",acc_knn_40)


test40['knn_pre']=knn_40
test40['match'] = np.where(test40['target'] == test40['knn_pre'], 'True', 'False')
FP_FN_knn_40=test40[test40['match']=='False']
FP_FN_knn_40=FP_FN_knn_40.drop(['knn_pre','match'], axis=1)
FP_FN_knn_40.to_csv('E:/41th_iteration/Test_41.CSV')
print("knn_FP_FN: ",len(FP_FN_knn_40))
    
TP_TN_knn_40=test40[test40['match']=='True']
TP_TN_knn_40.to_csv('E:/40th_iteration/40.CSV')
TP_TN_knn_40=TP_TN_knn_40.drop(['knn_pre','match'], axis=1)
TP_TN_knn_40.to_csv('E:/40th_iteration/TN_TP_40.CSV')
print("knn_TN_TP: ",len(TP_TN_knn_40))


# # 42th iteration

# In[3]:


from sklearn.neighbors import KNeighborsClassifier

train41=pd.read_csv("E:/1st_iteration/Train.CSV")
test41=pd.read_csv("E:/41th_iteration/Test_41.csv")

x_train41 =train41[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train41 = train41[['target']]

x_test41 = test41[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test41 = test41[['target']]


knn = KNeighborsClassifier(n_neighbors=39)
knn.fit(x_train41,y_train41)
knn_41=knn.predict(x_test41)
acc_knn_41=metrics.accuracy_score(y_test41,knn_41)*100

print("Train_size:",len(train41),"| Test_size:",len(test41))
print("Knn:",acc_knn_41)


test41['knn_pre']=knn_41
test41['match'] = np.where(test41['target'] == test41['knn_pre'], 'True', 'False')
FP_FN_knn_41=test41[test41['match']=='False']
FP_FN_knn_41=FP_FN_knn_41.drop(['knn_pre','match'], axis=1)
FP_FN_knn_41.to_csv('E:/42th_iteration/Test_42.CSV')
print("knn_FP_FN: ",len(FP_FN_knn_41))
    
TP_TN_knn_41=test41[test41['match']=='True']
TP_TN_knn_41.to_csv('E:/41th_iteration/41.CSV')
TP_TN_knn_41=TP_TN_knn_41.drop(['knn_pre','match'], axis=1)
TP_TN_knn_41.to_csv('E:/41th_iteration/TN_TP_41.CSV')
print("knn_TN_TP: ",len(TP_TN_knn_41))


# # 43th_iteration

# In[5]:


from sklearn.neighbors import KNeighborsClassifier

train42=pd.read_csv("E:/1st_iteration/Train.CSV")
test42=pd.read_csv("E:/42th_iteration/Test_42.csv")

x_train42 =train42[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train42 = train42[['target']]

x_test42 = test42[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test42 = test42[['target']]


knn = KNeighborsClassifier(n_neighbors=19)
knn.fit(x_train42,y_train42)
knn_42=knn.predict(x_test42)
acc_knn_42=metrics.accuracy_score(y_test42,knn_42)*100

print("Train_size:",len(train42),"| Test_size:",len(test42))
print("Knn:",acc_knn_42)


test42['knn_pre']=knn_42
test42['match'] = np.where(test42['target'] == test42['knn_pre'], 'True', 'False')
FP_FN_knn_42=test42[test42['match']=='False']
FP_FN_knn_42=FP_FN_knn_42.drop(['knn_pre','match'], axis=1)
FP_FN_knn_42.to_csv('E:/43th_iteration/Test_43.CSV')
print("knn_FP_FN: ",len(FP_FN_knn_42))
    
TP_TN_knn_42=test42[test42['match']=='True']
TP_TN_knn_42.to_csv('E:/42th_iteration/42.CSV')
TP_TN_knn_42=TP_TN_knn_42.drop(['knn_pre','match'], axis=1)
TP_TN_knn_42.to_csv('E:/42th_iteration/TN_TP_42.CSV')
print("knn_TN_TP: ",len(TP_TN_knn_42))


# # 44th_iteration

# In[17]:


from sklearn.neighbors import KNeighborsClassifier


train43=pd.read_csv("E:/1st_iteration/Train.CSV")
test43=pd.read_csv("E:/43th_iteration/Test_43.csv")

train43['age']=train43['age'].apply(np.floor) 
x_train43 =train43[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train43 = train43[['target']]

test43['age']=test43['age'].apply(np.floor) 
x_test43 = test43[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test43 = test43[['target']]

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train43,y_train43)
knn_43=knn.predict(x_test43)
acc_knn_43=metrics.accuracy_score(y_test43,knn_43)*100

print("Train_size:",len(train43),"| Test_size:",len(test43))
print("Knn:",acc_knn_43)


test43['knn_pre']=knn_43
test43['match'] = np.where(test43['target'] == test43['knn_pre'], 'True', 'False')
FP_FN_knn_43=test43[test43['match']=='False']
FP_FN_knn_43=FP_FN_knn_43.drop(['knn_pre','match'], axis=1)
FP_FN_knn_43.to_csv('E:/44th_iteration/Test_44.CSV')
print("knn_FP_FN: ",len(FP_FN_knn_43))
    
TP_TN_knn_43=test43[test43['match']=='True']
TP_TN_knn_43.to_csv('E:/43th_iteration/43.CSV')
TP_TN_knn_43=TP_TN_knn_43.drop(['knn_pre','match'], axis=1)
TP_TN_knn_43.to_csv('E:/43th_iteration/TN_TP_43.CSV')
print("knn_TN_TP: ",len(TP_TN_knn_43))


# In[ ]:





# In[ ]:




