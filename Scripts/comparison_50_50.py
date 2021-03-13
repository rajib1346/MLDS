#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
remain=pd.read_csv("E:/50_50/30th_iteration/Test_30.csv")
remain

remain['predict'] = np.where(remain['target'] ==1, 0, remain['predict'])
remain['predict'] = np.where(remain['target'] ==0, 1, remain['predict'])
remain.to_csv("E:/50_50/FP_FN.CSV")


# In[5]:


total=pd.read_csv("E:/50_50/TP_TN_FP_FN.CSV")
y_test =total[['target']]
y_pre = total[['predict']]
len(total)


# In[6]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix
ConMat=metrics.confusion_matrix(y_test,y_pre)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pre)*100
Precession=metrics.precision_score(y_test,y_pre)
ConMat=metrics.confusion_matrix(y_test,y_pre)
ConMat
error_rate=1-(accuracy/100)
TP=ConMat[1,1]
TN=ConMat[0,0]
FP=ConMat[0,1]
FN=ConMat[1,0]
print('PM')
print(accuracy)
print(TP,TN,FP,FN)
print("Precison=",Precession)
print("Error Rate= ",error_rate)

TPR=TP/float(TP+FN)
FPR=FP/float(TN+FP)
TNR=TN/float(TN+FP)
TPR=metrics.recall_score(y_test,y_pre)
FNR=1-TPR
print("TP=",TP)
print("FP=",FP)
print("TN=",TN)
print("FN=",FN)

print("TPR=",TPR)
print("FPR=",FPR)
print("TNR=",TNR)
print("FNR=",FNR)

print(ConMat)


# In[1]:


from sklearn.metrics import roc_curve,auc
import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt

frame=pd.read_csv("E:/50_50/TP_TN_FP_FN.CSV")
x = frame[['target']]
y = frame[['predict']]

fpr,tpr,treshold= roc_curve(x,y)
auc_sc=auc(fpr,tpr)

plt.figure(figsize=(5,5),dpi=100)
plt.plot([0,1],[0,1], color='orange',linestyle='--',label='model_auc (auc=%0.2f)' %auc_sc)
plt.plot(fpr,tpr,linestyle='-')


plt.xlabel('False Positive Rate-->')
plt.ylabel('True Positive Rate-->')

plt.legend()
plt.show()


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox ,TextArea,HPacker,VPacker

ax=plt.subplot(111)
n_groups=3


PM=[88.84571428571428,0.8158450905352145*100,0.9545545679342378*100]
XGB=[73.66857142857143,0.6897241103558577*100,0.7613973138281103*100]
LR=[70.56857142857143,0.6700748272119723*100,0.7216412401574803*100]
SVM=[71.97714285714287,0.6220368995259039*100,0.7733825722604929*100]
KNN=[68.92285714285714,0.6694465071114412*100,0.6972040452111838*100]
DT=[62.825714285714284, 0.6264922602387616*100,0.6288990825688073*100]
fig,ax=plt.subplots()
index=np.arange(n_groups)
bar_width=0.15
opacity=1
rects1=plt.bar(index,PM,bar_width,alpha=opacity,color='orange',label='MLDS')
rects2=plt.bar(index+bar_width,XGB,bar_width,alpha=opacity,color='green',label='XGB')
rects3=plt.bar(index+bar_width+bar_width,LR,bar_width,alpha=opacity,color='magenta',label='LR')
rects4=plt.bar(index+bar_width+bar_width+bar_width,SVM,bar_width,alpha=opacity,color='red',label='SVM')
rects5=plt.bar(index+bar_width+bar_width+bar_width+bar_width,KNN,bar_width,alpha=opacity,color='blue',label='KNN')
rects6=plt.bar(index+bar_width+bar_width+bar_width+bar_width+bar_width,DT,bar_width,alpha=opacity,color='cyan',label='DT')

autolabel=(rects1,"center")
autolabel=(rects2,"right")
autolabel=(rects3,"center")
autolabel=(rects4,"right")
autolabel=(rects5,"center")
autolabel=(rects6,"right")

ybox1=TextArea("MLDS",textprops=dict(color="orange",size=8,ha='left',va='bottom'))
ybox2=TextArea("XGB",textprops=dict(color="green",size=8,ha='left',va='bottom'))
ybox3=TextArea("LR",textprops=dict(color="magenta",size=8,ha='left',va='bottom'))
ybox4=TextArea("SVM",textprops=dict(color="red",size=8,ha='left',va='bottom'))
ybox5=TextArea("KNN",textprops=dict(color="blue",size=8,ha='left',va='bottom'))
ybox6=TextArea("DT",textprops=dict(color="cyan",size=8,ha='left',va='bottom'))

ybox=VPacker(children=[ybox1,ybox2,ybox3,ybox4,ybox5,ybox6],align="buttom",pad=0,sep=5)
anchored_ybox=AnchoredOffsetbox(loc=8,child=ybox,pad=0,frameon=False,bbox_to_anchor=(.99,-0.1),bbox_transform=ax.transAxes,borderpad=0.)
plt.xticks(index+bar_width,('Accuracy','TPR','Precisison'))
legend_x = 1
legend_y = 0.5
plt.legend(loc='center left',bbox_to_anchor=(legend_x, legend_y),fontsize=8)
plt.tight_layout()
plt.show()


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox ,TextArea,HPacker,VPacker

ax=plt.subplot(111)
n_groups=3


PM=[ 0.038872691933916424*100,0.9611273080660836*100,0.18415490946478552*100]
XGB=[0.21631509746755845*100,0.7836849025324415*100,0.3102758896441423*100]
LR=[0.2586748985308409*100,0.7413251014691591*100,0.3299251727880277*100]
SVM=[0.18241582347224605*100,0.817584176527754*100,0.3779631004740961*100]
KNN=[0.29097353227005085*100,0.7090264677299492*100,0.33055349288855884*100]
DT=[0.3699765620533928*100, 0.6300234379466072*100,0.3735077397612384*100]
fig,ax=plt.subplots()
index=np.arange(n_groups)
bar_width=0.15
opacity=1
rects1=plt.bar(index,PM,bar_width,alpha=opacity,color='orange',label='MLDS')
rects2=plt.bar(index+bar_width,XGB,bar_width,alpha=opacity,color='green',label='XGB')
rects3=plt.bar(index+bar_width+bar_width,LR,bar_width,alpha=opacity,color='magenta',label='LR')
rects4=plt.bar(index+bar_width+bar_width+bar_width,SVM,bar_width,alpha=opacity,color='red',label='SVM')
rects5=plt.bar(index+bar_width+bar_width+bar_width+bar_width,KNN,bar_width,alpha=opacity,color='blue',label='KNN')
rects6=plt.bar(index+bar_width+bar_width+bar_width+bar_width+bar_width,DT,bar_width,alpha=opacity,color='cyan',label='DT')

autolabel=(rects1,"center")
autolabel=(rects2,"right")
autolabel=(rects3,"center")
autolabel=(rects4,"right")
autolabel=(rects5,"center")
autolabel=(rects6,"right")

ybox1=TextArea("MLDS",textprops=dict(color="orange",size=8,ha='left',va='bottom'))
ybox2=TextArea("XGB",textprops=dict(color="green",size=8,ha='left',va='bottom'))
ybox3=TextArea("LR",textprops=dict(color="magenta",size=8,ha='left',va='bottom'))
ybox4=TextArea("SVM",textprops=dict(color="red",size=8,ha='left',va='bottom'))
ybox5=TextArea("KNN",textprops=dict(color="blue",size=8,ha='left',va='bottom'))
ybox6=TextArea("DT",textprops=dict(color="cyan",size=8,ha='left',va='bottom'))

ybox=VPacker(children=[ybox1,ybox2,ybox3,ybox4,ybox5,ybox6],align="buttom",pad=0,sep=5)
anchored_ybox=AnchoredOffsetbox(loc=8,child=ybox,pad=0,frameon=False,bbox_to_anchor=(.99,-0.1),bbox_transform=ax.transAxes,borderpad=0.)
plt.xticks(index+bar_width,('FPR','TNR','FNR'))
legend_x = 1
legend_y = 0.5
plt.legend(loc='center left',bbox_to_anchor=(legend_x, legend_y),fontsize=8)
plt.tight_layout()
plt.show()


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox ,TextArea,HPacker,VPacker

ax=plt.subplot(111)
n_groups=6


Normal=[17493,17493,17493,17493,17493,17493]
TN=[16813,13709 ,12968,14302,12403,10996]
FP=[680,3784,4525,3191,5090,6497]

fig,ax=plt.subplots()
index=np.arange(n_groups)
bar_width=0.15
opacity=1
rects1=plt.bar(index,Normal,bar_width,alpha=opacity,color='orange',label='MLPM')
rects2=plt.bar(index+bar_width,TN,bar_width,alpha=opacity,color='green',label='XGB')
rects3=plt.bar(index+bar_width+bar_width,FP,bar_width,alpha=opacity,color='magenta',label='LR')

autolabel=(rects1,"center")
autolabel=(rects2,"right")
autolabel=(rects3,"center")

ybox1=TextArea("Normal",textprops=dict(color="orange",size=8,ha='left',va='bottom'))
ybox2=TextArea("TN",textprops=dict(color="green",size=8,ha='left',va='bottom'))
ybox3=TextArea("FP",textprops=dict(color="magenta",size=8,ha='left',va='bottom'))

ybox=VPacker(children=[ybox1,ybox2,ybox3],align="buttom",pad=0,sep=5)
anchored_ybox=AnchoredOffsetbox(loc=8,child=ybox,pad=0,frameon=False,bbox_to_anchor=(.99,-0.1),bbox_transform=ax.transAxes,borderpad=0.)
#plt.title('Confusion Matrix Result Comparison(Normal Patients,TN,FP)')
plt.xticks(index+bar_width,('MLDS','XGB','LR','SVM','KNN','DT'))
plt.tight_layout()
plt.show()


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox ,TextArea,HPacker,VPacker

ax=plt.subplot(111)
n_groups=6

Affected=[17507,17507,17507,17507,17507,17507]
TP=[14283,12075,11731,10890,11720,11000]
FN=[3224,5432,5776,6617,5787,6507]

fig,ax=plt.subplots()
index=np.arange(n_groups)
bar_width=0.15
opacity=1
rects1=plt.bar(index,Affected,bar_width,alpha=opacity,color='orange',label='Proposed Method')
rects2=plt.bar(index+bar_width,TP,bar_width,alpha=opacity,color='green',label='XGB')
rects3=plt.bar(index+bar_width+bar_width,FN,bar_width,alpha=opacity,color='magenta',label='LR')

autolabel=(rects1,"center")
autolabel=(rects2,"right")
autolabel=(rects3,"center")

ybox1=TextArea("Affected",textprops=dict(color="orange",size=8,ha='left',va='bottom'))
ybox2=TextArea("TP",textprops=dict(color="green",size=8,ha='left',va='bottom'))
ybox3=TextArea("FN",textprops=dict(color="magenta",size=8,ha='left',va='bottom'))

ybox=VPacker(children=[ybox1,ybox2,ybox3],align="buttom",pad=0,sep=5)
anchored_ybox=AnchoredOffsetbox(loc=8,child=ybox,pad=0,frameon=False,bbox_to_anchor=(.99,-0.1),bbox_transform=ax.transAxes,borderpad=0.)
#plt.title('Confusion Matrix Result Comparison(Afected,TP,FN)')
plt.xticks(index+bar_width,('MLDS','XGB','LR','SVM','KNN','DT'))
plt.tight_layout()
plt.show()


# In[ ]:




