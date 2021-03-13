#!/usr/bin/env python
# coding: utf-8

# # 50:50

# In[7]:


import xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix

DataFrame1=pd.read_csv("E:/50_50/Initial_Train.CSV")
DataFrame2=pd.read_csv("E:/50_50/Initial_Test.CSV")

x_train = DataFrame1[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y_train = DataFrame1[['target']]
x_test = DataFrame2[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y_test = DataFrame2[['target']]
clssifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.5, gamma=0.4, learning_rate=0.1,
       max_delta_step=0, max_depth=6, min_child_weight=7, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)
clssifier.fit(x_train,y_train)
XB_pridict=clssifier.predict(x_test)
accuracy_XB=metrics.accuracy_score(y_test,XB_pridict)*100
Precession=metrics.precision_score(y_test,XB_pridict)
ConMat_XB=metrics.confusion_matrix(y_test,XB_pridict)
error_rate=1-(accuracy_XB/100)
TP=ConMat_XB[1,1]
TN=ConMat_XB[0,0]
FP=ConMat_XB[0,1]
FN=ConMat_XB[1,0]
print('XG Boost')
print(accuracy_XB)
print(TP,TN,FP,FN)
print("Precison=",Precession)
print('Error Rate=',error_rate)

TPR=TP/float(TP+FN)
FPR=FP/float(TN+FP)
TNR=TN/float(TN+FP)
TPR=metrics.recall_score(y_test,XB_pridict)
FNR=1-TPR
print("TP=",TP)
print("FP=",FP)
print("TN=",TN)
print("FN=",FN)

print("TPR=",TPR)
print("FPR=",FPR)
print("TNR=",TNR)
print("FNR=",FNR)


# In[4]:


pip install xgboost


# # 60:40

# In[9]:


import xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix

DataFrame1=pd.read_csv("E:/60_40/Initial_Train.CSV")
DataFrame2=pd.read_csv("E:/60_40/Initial_Test.CSV")

x_train = DataFrame1[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y_train = DataFrame1[['target']]
x_test = DataFrame2[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y_test = DataFrame2[['target']]
clssifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.5, gamma=0.4, learning_rate=0.1,
       max_delta_step=0, max_depth=6, min_child_weight=7, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)
clssifier.fit(x_train,y_train)
XB_pridict=clssifier.predict(x_test)
accuracy_XB=metrics.accuracy_score(y_test,XB_pridict)*100
Precession=metrics.precision_score(y_test,XB_pridict)
ConMat_XB=metrics.confusion_matrix(y_test,XB_pridict)
error_rate=1-(accuracy_XB/100)
TP=ConMat_XB[1,1]
TN=ConMat_XB[0,0]
FP=ConMat_XB[0,1]
FN=ConMat_XB[1,0]
print('XG Boost')
print(accuracy_XB)
print(TP,TN,FP,FN)
print("Precison=",Precession)
print('Error Rate=',error_rate)

TPR=TP/float(TP+FN)
FPR=FP/float(TN+FP)
TNR=TN/float(TN+FP)
TPR=metrics.recall_score(y_test,XB_pridict)
FNR=1-TPR
print("TP=",TP)
print("FP=",FP)
print("TN=",TN)
print("FN=",FN)

print("TPR=",TPR)
print("FPR=",FPR)
print("TNR=",TNR)
print("FNR=",FNR)


# # 70:30

# In[1]:


import xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix

DataFrame1=pd.read_csv("E:/70_30/Initial_Train.CSV")
DataFrame2=pd.read_csv("E:/70_30/Initial_Test.CSV")

x_train = DataFrame1[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y_train = DataFrame1[['target']]
x_test = DataFrame2[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y_test = DataFrame2[['target']]
clssifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.5, gamma=0.4, learning_rate=0.1,
       max_delta_step=0, max_depth=6, min_child_weight=7, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)
clssifier.fit(x_train,y_train)
XB_pridict=clssifier.predict(x_test)
accuracy_XB=metrics.accuracy_score(y_test,XB_pridict)*100
Precession=metrics.precision_score(y_test,XB_pridict)
ConMat_XB=metrics.confusion_matrix(y_test,XB_pridict)
error_rate=1-(accuracy_XB/100)
TP=ConMat_XB[1,1]
TN=ConMat_XB[0,0]
FP=ConMat_XB[0,1]
FN=ConMat_XB[1,0]
print('XG Boost')
print(accuracy_XB)
print(TP,TN,FP,FN)
print("Precison=",Precession)
print('Error Rate=',error_rate)

TPR=TP/float(TP+FN)
FPR=FP/float(TN+FP)
TNR=TN/float(TN+FP)
TPR=metrics.recall_score(y_test,XB_pridict)
FNR=1-TPR
print("TP=",TP)
print("FP=",FP)
print("TN=",TN)
print("FN=",FN)

print("TPR=",TPR)
print("FPR=",FPR)
print("TNR=",TNR)
print("FNR=",FNR)


# # 80:20

# In[2]:


import xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix

DataFrame1=pd.read_csv("E:/80_20/Initial_Train.CSV")
DataFrame2=pd.read_csv("E:/80_20/Initial_Test.CSV")

x_train = DataFrame1[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y_train = DataFrame1[['target']]
x_test = DataFrame2[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y_test = DataFrame2[['target']]
clssifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.5, gamma=0.4, learning_rate=0.1,
       max_delta_step=0, max_depth=6, min_child_weight=7, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)
clssifier.fit(x_train,y_train)
XB_pridict=clssifier.predict(x_test)
accuracy_XB=metrics.accuracy_score(y_test,XB_pridict)*100
Precession=metrics.precision_score(y_test,XB_pridict)
ConMat_XB=metrics.confusion_matrix(y_test,XB_pridict)
error_rate=1-(accuracy_XB/100)
TP=ConMat_XB[1,1]
TN=ConMat_XB[0,0]
FP=ConMat_XB[0,1]
FN=ConMat_XB[1,0]
print('XG Boost')
print(accuracy_XB)
print(TP,TN,FP,FN)
print("Precison=",Precession)
print('Error Rate=',error_rate)

TPR=TP/float(TP+FN)
FPR=FP/float(TN+FP)
TNR=TN/float(TN+FP)
TPR=metrics.recall_score(y_test,XB_pridict)
FNR=1-TPR
print("TP=",TP)
print("FP=",FP)
print("TN=",TN)
print("FN=",FN)

print("TPR=",TPR)
print("FPR=",FPR)
print("TNR=",TNR)
print("FNR=",FNR)


# # 85:15

# In[4]:


import xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix

DataFrame1=pd.read_csv("E:/85_15/Initial_Train.CSV")
DataFrame2=pd.read_csv("E:/85_15/Initial_Test.CSV")

x_train = DataFrame1[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y_train = DataFrame1[['target']]
x_test = DataFrame2[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y_test = DataFrame2[['target']]
clssifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.5, gamma=0.4, learning_rate=0.1,
       max_delta_step=0, max_depth=6, min_child_weight=7, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)
clssifier.fit(x_train,y_train)
XB_pridict=clssifier.predict(x_test)
accuracy_XB=metrics.accuracy_score(y_test,XB_pridict)*100
Precession=metrics.precision_score(y_test,XB_pridict)
ConMat_XB=metrics.confusion_matrix(y_test,XB_pridict)
error_rate=1-(accuracy_XB/100)
TP=ConMat_XB[1,1]
TN=ConMat_XB[0,0]
FP=ConMat_XB[0,1]
FN=ConMat_XB[1,0]
print('XG Boost')
print(accuracy_XB)
print(TP,TN,FP,FN)
print("Precison=",Precession)
print('Error Rate=',error_rate)

TPR=TP/float(TP+FN)
FPR=FP/float(TN+FP)
TNR=TN/float(TN+FP)
TPR=metrics.recall_score(y_test,XB_pridict)
FNR=1-TPR
print("TP=",TP)
print("FP=",FP)
print("TN=",TN)
print("FN=",FN)

print("TPR=",TPR)
print("FPR=",FPR)
print("TNR=",TNR)
print("FNR=",FNR)


# In[ ]:




