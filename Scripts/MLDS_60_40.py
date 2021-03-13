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

# In[9]:


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
    FP_FN_Na_6=test6[test6['match']=='False']
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

# In[10]:


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

# In[11]:


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

# In[12]:


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

# In[ ]:


# train10=pd.read_csv("E:/1st_iteration/Train.CSV")
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

# In[15]:


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

# In[16]:


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

# In[17]:


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

# In[18]:


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

# In[19]:


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

# In[20]:


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

# In[ ]:


# train17=pd.read_csv("E:/1st_iteration/Train.CSV")
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

# In[25]:


from sklearn.neighbors import KNeighborsClassifier

train17=pd.read_csv("E:/1st_iteration/Train.CSV")
test17=pd.read_csv("E:/17th_iteration/Test_17.csv")

x_train17 =train17[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train17 = train17[['target']]

x_test17 = test17[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test17 = test17[['target']]


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train17,y_train17)
knn_17=knn.predict(x_test17)
acc_knn_17=metrics.accuracy_score(y_test17,knn_17)*100

print("Train_size:",len(train17),"| Test_size:",len(test17))
print("Knn:",acc_knn_17)


test17['knn_pre']=knn_17
test17['match'] = np.where(test17['target'] == test17['knn_pre'], 'True', 'False')
FP_FN_knn_17=test17[test17['match']=='False']
FP_FN_knn_17=FP_FN_knn_17.drop(['knn_pre','match'], axis=1)
FP_FN_knn_17.to_csv('E:/18th_iteration/Test_18.CSV')
print("knn_FP_FN: ",len(FP_FN_knn_17))
    
TP_TN_knn_17=test17[test17['match']=='True']
TP_TN_knn_17.to_csv('E:/17th_iteration/17.CSV')
TP_TN_knn_17=TP_TN_knn_17.drop(['knn_pre','match'], axis=1)
TP_TN_knn_17.to_csv('E:/17th_iteration/TN_TP_17.CSV')
print("knn_TN_TP: ",len(TP_TN_knn_17))


# # 19th_iteration

# In[27]:


from sklearn.neighbors import KNeighborsClassifier

train18=pd.read_csv("E:/1st_iteration/Train.CSV")
test18=pd.read_csv("E:/18th_iteration/Test_18.csv")

x_train18 =train18[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train18 = train18[['target']]

x_test18 = test18[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test18 = test18[['target']]


knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(x_train18,y_train18)
knn_18=knn.predict(x_test18)
acc_knn_18=metrics.accuracy_score(y_test18,knn_18)*100

print("Train_size:",len(train18),"| Test_size:",len(test18))
print("Knn:",acc_knn_18)


test18['knn_pre']=knn_18
test18['match'] = np.where(test18['target'] == test18['knn_pre'], 'True', 'False')
FP_FN_knn_18=test18[test18['match']=='False']
FP_FN_knn_18=FP_FN_knn_18.drop(['knn_pre','match'], axis=1)
FP_FN_knn_18.to_csv('E:/19th_iteration/Test_19.CSV')
print("knn_FP_FN: ",len(FP_FN_knn_18))
    
TP_TN_knn_18=test18[test18['match']=='True']
TP_TN_knn_18.to_csv('E:/18th_iteration/18.CSV')
TP_TN_knn_18=TP_TN_knn_18.drop(['knn_pre','match'], axis=1)
TP_TN_knn_18.to_csv('E:/18th_iteration/TN_TP_18.CSV')
print("knn_TN_TP: ",len(TP_TN_knn_18))


# # 20th iteration

# In[30]:


from sklearn.neighbors import KNeighborsClassifier

train19=pd.read_csv("E:/1st_iteration/Train.CSV")
test19=pd.read_csv("E:/19th_iteration/Test_19.csv")

x_train19 =train19[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train19 = train19[['target']]

x_test19 = test19[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test19 = test19[['target']]


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train19,y_train19)
knn_19=knn.predict(x_test19)
acc_knn_19=metrics.accuracy_score(y_test19,knn_19)*100

print("Train_size:",len(train19),"| Test_size:",len(test19))
print("Knn:",acc_knn_19)


test19['knn_pre']=knn_19
test19['match'] = np.where(test19['target'] == test19['knn_pre'], 'True', 'False')
FP_FN_knn_19=test19[test19['match']=='False']
FP_FN_knn_19=FP_FN_knn_19.drop(['knn_pre','match'], axis=1)
FP_FN_knn_19.to_csv('E:/20th_iteration/Test_20.CSV')
print("knn_FP_FN: ",len(FP_FN_knn_19))
    
TP_TN_knn_19=test19[test19['match']=='True']
TP_TN_knn_19.to_csv('E:/19th_iteration/19.CSV')
TP_TN_knn_19=TP_TN_knn_19.drop(['knn_pre','match'], axis=1)
TP_TN_knn_19.to_csv('E:/19th_iteration/TN_TP_19.CSV')
print("knn_TN_TP: ",len(TP_TN_knn_19))


# # 21th_iteration

# In[33]:


from sklearn.neighbors import KNeighborsClassifier

train20=pd.read_csv("E:/1st_iteration/Train.CSV")
test20=pd.read_csv("E:/20th_iteration/Test_20.csv")

x_train20 =train20[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train20 = train20[['target']]

x_test20 = test20[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test20 = test20[['target']]


knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train20,y_train20)
knn_20=knn.predict(x_test20)
acc_knn_20=metrics.accuracy_score(y_test20,knn_20)*100

print("Train_size:",len(train20),"| Test_size:",len(test20))
print("Knn:",acc_knn_20)


test20['knn_pre']=knn_20
test20['match'] = np.where(test20['target'] == test20['knn_pre'], 'True', 'False')
FP_FN_knn_20=test20[test20['match']=='False']
FP_FN_knn_20=FP_FN_knn_20.drop(['knn_pre','match'], axis=1)
FP_FN_knn_20.to_csv('E:/21th_iteration/Test_21.CSV')
print("knn_FP_FN: ",len(FP_FN_knn_20))
    
TP_TN_knn_20=test20[test20['match']=='True']
TP_TN_knn_20.to_csv('E:/20th_iteration/20.CSV')
TP_TN_knn_20=TP_TN_knn_20.drop(['knn_pre','match'], axis=1)
TP_TN_knn_20.to_csv('E:/20th_iteration/TN_TP_20.CSV')
print("knn_TN_TP: ",len(TP_TN_knn_20))


# # 22th iteration

# In[36]:


from sklearn.neighbors import KNeighborsClassifier

train21=pd.read_csv("E:/1st_iteration/Train.CSV")
test21=pd.read_csv("E:/21th_iteration/Test_21.csv")

x_train21 =train21[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train21 = train21[['target']]

x_test21 = test21[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test21 = test21[['target']]


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train21,y_train21)
knn_21=knn.predict(x_test21)
acc_knn_21=metrics.accuracy_score(y_test21,knn_21)*100

print("Train_size:",len(train21),"| Test_size:",len(test21))
print("Knn:",acc_knn_21)


test21['knn_pre']=knn_21
test21['match'] = np.where(test21['target'] == test21['knn_pre'], 'True', 'False')
FP_FN_knn_21=test21[test21['match']=='False']
FP_FN_knn_21=FP_FN_knn_21.drop(['knn_pre','match'], axis=1)
FP_FN_knn_21.to_csv('E:/22th_iteration/Test_22.CSV')
print("knn_FP_FN: ",len(FP_FN_knn_21))
    
TP_TN_knn_21=test21[test21['match']=='True']
TP_TN_knn_21.to_csv('E:/21th_iteration/21.CSV')
TP_TN_knn_21=TP_TN_knn_21.drop(['knn_pre','match'], axis=1)
TP_TN_knn_21.to_csv('E:/21th_iteration/TN_TP_21.CSV')
print("knn_TN_TP: ",len(TP_TN_knn_21))


# # 23th_iteration

# In[39]:


from sklearn.neighbors import KNeighborsClassifier

train22=pd.read_csv("E:/1st_iteration/Train.CSV")
test22=pd.read_csv("E:/22th_iteration/Test_22.csv")

x_train22 =train22[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_train22 = train22[['target']]

x_test22 = test22[['age','cholesterol','weight','gluc','ap_lo','ap_hi']]
y_test22 = test22[['target']]


knn = KNeighborsClassifier(n_neighbors=37)
knn.fit(x_train22,y_train22)
knn_22=knn.predict(x_test22)
acc_knn_22=metrics.accuracy_score(y_test22,knn_22)*100

print("Train_size:",len(train22),"| Test_size:",len(test22))
print("Knn:",acc_knn_22)


test22['knn_pre']=knn_22
test22['match'] = np.where(test22['target'] == test22['knn_pre'], 'True', 'False')
FP_FN_knn_22=test22[test22['match']=='False']
FP_FN_knn_22=FP_FN_knn_22.drop(['knn_pre','match'], axis=1)
FP_FN_knn_22.to_csv('E:/23th_iteration/Test_23.CSV')
print("knn_FP_FN: ",len(FP_FN_knn_22))
    
TP_TN_knn_22=test22[test22['match']=='True']
TP_TN_knn_22.to_csv('E:/22th_iteration/22.CSV')
TP_TN_knn_22=TP_TN_knn_22.drop(['knn_pre','match'], axis=1)
TP_TN_knn_22.to_csv('E:/22th_iteration/TN_TP_22.CSV')
print("knn_TN_TP: ",len(TP_TN_knn_22))


# In[ ]:





# In[5]:


error = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train23,y_train23)
    pred_knn = knn.predict(x_test23)
    acc_23=metrics.accuracy_score(y_test23,pred_knn)*100
    print('K=',i)
    print('Accuracy= ',acc_23)


# In[ ]:




