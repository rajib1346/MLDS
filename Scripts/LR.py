#!/usr/bin/env python
# coding: utf-8

# ###### 50:50

# In[2]:


from sklearn.linear_model import LogisticRegression
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
LR=LogisticRegression()
LR.fit(x_train,y_train)
LR_pre=LR.predict(x_test)
accuracy_lr=metrics.accuracy_score(y_test,LR_pre)*100
Precession=metrics.precision_score(y_test,LR_pre)
ConMat_LR=metrics.confusion_matrix(y_test,LR_pre)
error_rate=1-(accuracy_lr/100)
TP=ConMat_LR[1,1]
TN=ConMat_LR[0,0]
FP=ConMat_LR[0,1]
FN=ConMat_LR[1,0]
print('Logistic Regression')
print(accuracy_lr)
print(TP,TN,FP,FN)
print("Precison=",Precession)
print('Error Rate=',error_rate)
print("TP=",TP)
print("FP=",FP)
print("TN=",TN)
print("FN=",FN)
TPR=TP/float(TP+FN)
FPR=FP/float(TN+FP)
TNR=TN/float(TN+FP)
TPR=metrics.recall_score(y_test,LR_pre)
FNR=1-TPR
print("TPR=",TPR)
print("FPR=",FPR)
print("TNR=",TNR)
print("FNR=",FNR)


# # 60:40

# In[3]:


from sklearn.linear_model import LogisticRegression
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
LR=LogisticRegression()
LR.fit(x_train,y_train)
LR_pre=LR.predict(x_test)
accuracy_lr=metrics.accuracy_score(y_test,LR_pre)*100
Precession=metrics.precision_score(y_test,LR_pre)
ConMat_LR=metrics.confusion_matrix(y_test,LR_pre)
error_rate=1-(accuracy_lr/100)
TP=ConMat_LR[1,1]
TN=ConMat_LR[0,0]
FP=ConMat_LR[0,1]
FN=ConMat_LR[1,0]
print('Logistic Regression')
print(accuracy_lr)
print(TP,TN,FP,FN)
print("Precison=",Precession)
print('Error Rate=',error_rate)
print("TP=",TP)
print("FP=",FP)
print("TN=",TN)
print("FN=",FN)
TPR=TP/float(TP+FN)
FPR=FP/float(TN+FP)
TNR=TN/float(TN+FP)
TPR=metrics.recall_score(y_test,LR_pre)
FNR=1-TPR
print("TPR=",TPR)
print("FPR=",FPR)
print("TNR=",TNR)
print("FNR=",FNR)


# In[1]:


from sklearn.linear_model import LogisticRegression
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
LR=LogisticRegression()
LR.fit(x_train,y_train)
LR_pre=LR.predict(x_test)
accuracy_lr=metrics.accuracy_score(y_test,LR_pre)*100
Precession=metrics.precision_score(y_test,LR_pre)
ConMat_LR=metrics.confusion_matrix(y_test,LR_pre)
error_rate=1-(accuracy_lr/100)
TP=ConMat_LR[1,1]
TN=ConMat_LR[0,0]
FP=ConMat_LR[0,1]
FN=ConMat_LR[1,0]
print('Logistic Regression')
print(accuracy_lr)
print(TP,TN,FP,FN)
print("Precison=",Precession)
print('Error Rate=',error_rate)
print("TP=",TP)
print("FP=",FP)
print("TN=",TN)
print("FN=",FN)
TPR=TP/float(TP+FN)
FPR=FP/float(TN+FP)
TNR=TN/float(TN+FP)
TPR=metrics.recall_score(y_test,LR_pre)
FNR=1-TPR
print("TPR=",TPR)
print("FPR=",FPR)
print("TNR=",TNR)
print("FNR=",FNR)


# # 80:20

# In[3]:


from sklearn.linear_model import LogisticRegression
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
LR=LogisticRegression()
LR.fit(x_train,y_train)
LR_pre=LR.predict(x_test)
accuracy_lr=metrics.accuracy_score(y_test,LR_pre)*100
Precession=metrics.precision_score(y_test,LR_pre)
ConMat_LR=metrics.confusion_matrix(y_test,LR_pre)
error_rate=1-(accuracy_lr/100)
TP=ConMat_LR[1,1]
TN=ConMat_LR[0,0]
FP=ConMat_LR[0,1]
FN=ConMat_LR[1,0]
print('Logistic Regression')
print(accuracy_lr)
print(TP,TN,FP,FN)
print("Precison=",Precession)
print('Error Rate=',error_rate)
print("TP=",TP)
print("FP=",FP)
print("TN=",TN)
print("FN=",FN)
TPR=TP/float(TP+FN)
FPR=FP/float(TN+FP)
TNR=TN/float(TN+FP)
TPR=metrics.recall_score(y_test,LR_pre)
FNR=1-TPR
print("TPR=",TPR)
print("FPR=",FPR)
print("TNR=",TNR)
print("FNR=",FNR)


# # 85:15

# In[5]:


from sklearn.linear_model import LogisticRegression
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
LR=LogisticRegression()
LR.fit(x_train,y_train)
LR_pre=LR.predict(x_test)
accuracy_lr=metrics.accuracy_score(y_test,LR_pre)*100
Precession=metrics.precision_score(y_test,LR_pre)
ConMat_LR=metrics.confusion_matrix(y_test,LR_pre)
error_rate=1-(accuracy_lr/100)
TP=ConMat_LR[1,1]
TN=ConMat_LR[0,0]
FP=ConMat_LR[0,1]
FN=ConMat_LR[1,0]
print('Logistic Regression')
print(accuracy_lr)
print(TP,TN,FP,FN)
print("Precison=",Precession)
print('Error Rate=',error_rate)
print("TP=",TP)
print("FP=",FP)
print("TN=",TN)
print("FN=",FN)
TPR=TP/float(TP+FN)
FPR=FP/float(TN+FP)
TNR=TN/float(TN+FP)
TPR=metrics.recall_score(y_test,LR_pre)
FNR=1-TPR
print("TPR=",TPR)
print("FPR=",FPR)
print("TNR=",TNR)
print("FNR=",FNR)


# In[1]:


from sklearn.linear_model import LogisticRegression
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
LR=LogisticRegression()


# In[2]:





# In[ ]:




