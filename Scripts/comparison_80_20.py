#!/usr/bin/env python
# coding: utf-8

# In[6]:


from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
remain=pd.read_csv("E:/80_20/41th_iteration/Test_41.csv")
remain

remain['predict'] = np.where(remain['target'] ==1, 0, remain['predict'])
remain['predict'] = np.where(remain['target'] ==0, 1, remain['predict'])
remain.to_csv("E:/80_20/FP_FN.CSV")


# In[7]:


total=pd.read_csv("E:/80_20/TP_TN_FP_FN.CSV")
y_test =total[['target']]
y_pre = total[['predict']]
len(total)


# In[8]:


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

frame=pd.read_csv("E:/80_20/TP_TN_FP_FN.CSV")
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


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox ,TextArea,HPacker,VPacker

ax=plt.subplot(111)
n_groups=3


PM=[92.72857142857143,0.8922223802075928*100,0.9602142310635042*100]
XGB=[73.52142857142857,0.694013934309683*100,0.7583903045369795*100]
LR=[70.7,0.6751030854542869*100, 0.7232292460015233*100]
SVM=[72.46428571428571,0.6371392009099958*100,0.7747233748271093*100]
KNN=[69.22857142857143,0.6751030854542869*100,0.7012258159799143*100]
DT=[63.142857142857146, 0.633442343239016*100,0.6330822793804178*100]
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


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox ,TextArea,HPacker,VPacker

ax=plt.subplot(111)
n_groups=3


PM=[ 0.03731878857470934*100,0.9626812114252906*100,0.10777761979240719*100]
XGB=[0.22319506243720397*100,0.776804937562796*100,0.30598606569031706*100]
LR=[0.26080091861633414*100,0.7391990813836659*100,0.3248969145457131*100]
SVM=[0.18702454428017798*100,0.812975455719822*100,0.36286079909000424*100]
KNN=[0.2903688818716808*100,0.7096311181283192*100,0.3248969145457131*100]
DT=[ 0.3706042773073059*100,  0.6293957226926942*100,0.36655765676098395*100]
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


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox ,TextArea,HPacker,VPacker

ax=plt.subplot(111)
n_groups=6


Normal=[6967,6967,6967,6967,6967,6967]
TN=[6707,5412 ,5150,5664,4944,4385]
FP=[260,1555,1817,1303,2023,2582]

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


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox ,TextArea,HPacker,VPacker

ax=plt.subplot(111)
n_groups=6

Affected=[7033,7033,7033,7033,7033,7033]
TP=[6275,4881,4748,4481,4748,4455]
FN=[758,2152,2285,2552,2285,2578]

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

