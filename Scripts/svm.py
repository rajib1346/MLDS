#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
total=pd.read_csv("E:/TP_TN_FP_FN.CSV")
y_test =total[['target']]
y_pre = total[['predict']]
len(total)


# # 50:50

# In[3]:



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

from sklearn import svm
svm_model= svm.SVC(gamma='scale')
svm_model.fit(x_train,y_train)
svm_pre=svm_model.predict(x_test)
accuracy_svm=metrics.accuracy_score(y_test,svm_pre)*100
Precession=metrics.precision_score(y_test,svm_pre)
ConMat_Svm=metrics.confusion_matrix(y_test,svm_pre)
error_rate=1-(accuracy_svm/100)
TP=ConMat_Svm[1,1]
TN=ConMat_Svm[0,0]
FP=ConMat_Svm[0,1]
FN=ConMat_Svm[1,0]
print('SVM')
print(accuracy_svm)
print(TP,TN,FP,FN)
print("Precison=",Precession)
print("Error Rate= ",error_rate)
print("TP=",TP)
print("FP=",FP)
print("TN=",TN)
print("FN=",FN)
TPR=TP/float(TP+FN)
FPR=FP/float(TN+FP)
TNR=TN/float(TN+FP)
TPR=metrics.recall_score(y_test,svm_pre)
FNR=1-TPR
print("TPR=",TPR)
print("FPR=",FPR)
print("TNR=",TNR)
print("FNR=",FNR)


# # 60:40

# In[4]:



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

from sklearn import svm
svm_model= svm.SVC(gamma='scale')
svm_model.fit(x_train,y_train)
svm_pre=svm_model.predict(x_test)
accuracy_svm=metrics.accuracy_score(y_test,svm_pre)*100
Precession=metrics.precision_score(y_test,svm_pre)
ConMat_Svm=metrics.confusion_matrix(y_test,svm_pre)
error_rate=1-(accuracy_svm/100)
TP=ConMat_Svm[1,1]
TN=ConMat_Svm[0,0]
FP=ConMat_Svm[0,1]
FN=ConMat_Svm[1,0]
print('SVM')
print(accuracy_svm)
print(TP,TN,FP,FN)
print("Precison=",Precession)
print("Error Rate= ",error_rate)
print("TP=",TP)
print("FP=",FP)
print("TN=",TN)
print("FN=",FN)
TPR=TP/float(TP+FN)
FPR=FP/float(TN+FP)
TNR=TN/float(TN+FP)
TPR=metrics.recall_score(y_test,svm_pre)
FNR=1-TPR
print("TPR=",TPR)
print("FPR=",FPR)
print("TNR=",TNR)
print("FNR=",FNR)


# In[1]:



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

from sklearn import svm
svm_model= svm.SVC(gamma='scale')
svm_model.fit(x_train,y_train)
svm_pre=svm_model.predict(x_test)
accuracy_svm=metrics.accuracy_score(y_test,svm_pre)*100
Precession=metrics.precision_score(y_test,svm_pre)
ConMat_Svm=metrics.confusion_matrix(y_test,svm_pre)
error_rate=1-(accuracy_svm/100)
TP=ConMat_Svm[1,1]
TN=ConMat_Svm[0,0]
FP=ConMat_Svm[0,1]
FN=ConMat_Svm[1,0]
print('SVM')
print(accuracy_svm)
print(TP,TN,FP,FN)
print("Precison=",Precession)
print("Error Rate= ",error_rate)
print("TP=",TP)
print("FP=",FP)
print("TN=",TN)
print("FN=",FN)
TPR=TP/float(TP+FN)
FPR=FP/float(TN+FP)
TNR=TN/float(TN+FP)
TPR=metrics.recall_score(y_test,svm_pre)
FNR=1-TPR
print("TPR=",TPR)
print("FPR=",FPR)
print("TNR=",TNR)
print("FNR=",FNR)


# # 80:20

# In[3]:



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

from sklearn import svm
svm_model= svm.SVC(gamma='scale')
svm_model.fit(x_train,y_train)
svm_pre=svm_model.predict(x_test)
accuracy_svm=metrics.accuracy_score(y_test,svm_pre)*100
Precession=metrics.precision_score(y_test,svm_pre)
ConMat_Svm=metrics.confusion_matrix(y_test,svm_pre)
error_rate=1-(accuracy_svm/100)
TP=ConMat_Svm[1,1]
TN=ConMat_Svm[0,0]
FP=ConMat_Svm[0,1]
FN=ConMat_Svm[1,0]
print('SVM')
print(accuracy_svm)
print(TP,TN,FP,FN)
print("Precison=",Precession)
print("Error Rate= ",error_rate)
print("TP=",TP)
print("FP=",FP)
print("TN=",TN)
print("FN=",FN)
TPR=TP/float(TP+FN)
FPR=FP/float(TN+FP)
TNR=TN/float(TN+FP)
TPR=metrics.recall_score(y_test,svm_pre)
FNR=1-TPR
print("TPR=",TPR)
print("FPR=",FPR)
print("TNR=",TNR)
print("FNR=",FNR)


# # 85:15

# In[5]:



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

from sklearn import svm
svm_model= svm.SVC(gamma='scale')
svm_model.fit(x_train,y_train)
svm_pre=svm_model.predict(x_test)
accuracy_svm=metrics.accuracy_score(y_test,svm_pre)*100
Precession=metrics.precision_score(y_test,svm_pre)
ConMat_Svm=metrics.confusion_matrix(y_test,svm_pre)
error_rate=1-(accuracy_svm/100)
TP=ConMat_Svm[1,1]
TN=ConMat_Svm[0,0]
FP=ConMat_Svm[0,1]
FN=ConMat_Svm[1,0]
print('SVM')
print(accuracy_svm)
print(TP,TN,FP,FN)
print("Precison=",Precession)
print("Error Rate= ",error_rate)
print("TP=",TP)
print("FP=",FP)
print("TN=",TN)
print("FN=",FN)
TPR=TP/float(TP+FN)
FPR=FP/float(TN+FP)
TNR=TN/float(TN+FP)
TPR=metrics.recall_score(y_test,svm_pre)
FNR=1-TPR
print("TPR=",TPR)
print("FPR=",FPR)
print("TNR=",TNR)
print("FNR=",FNR)


# In[ ]:





# In[ ]:





# In[ ]:




