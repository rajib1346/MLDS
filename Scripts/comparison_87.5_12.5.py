#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
remain=pd.read_csv("E:/87.5_12.5/47th_iteration/Test_47.csv")
remain

remain['predict'] = np.where(remain['target'] ==1, 0, remain['predict'])
remain['predict'] = np.where(remain['target'] ==0, 1, remain['predict'])
remain.to_csv("E:/87.5_12.5/FP_FN.CSV")


# In[3]:


total=pd.read_csv("E:/87.5_12.5/TP_TN_FP_FN.CSV")
y_test =total[['target']]
y_pre = total[['predict']]
len(total)


# In[4]:


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


# In[5]:


from sklearn.metrics import roc_curve,auc
import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt

frame=pd.read_csv("E:/85_15/TP_TN_FP_FN.CSV")
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


PM=[94.16,0.9111111111111111*100,0.97*100]
XGB=[73.87428571428572,0.7010309278350515*100, 0.7572383073496659*100]
LR=[ 71.10857142857144,0.6829324169530355*100,0.7226666666666667*100]
SVM=[72.88,0.6421534936998855*100,0.7755949086884338*100]
KNN=[69.23428571428572,0.6799541809851088*100,0.6962233169129721*100]
DT=[63.37142857142857, 0.6398625429553264*100,0.631043831902395*100]
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


PM=[  0.028050171037628278*100,0.9719498289623717*100,0.0888888888888889*100]
XGB=[0.22371721778791334*100,0.7762827822120867*100,0.2989690721649485*100]
LR=[0.2608893956670468*100,0.7391106043329533*100,0.3170675830469645*100]
SVM=[0.18494868871151654*100,0.8150513112884835*100,0.3578465063001145*100]
KNN=[0.2953249714937286*100,0.7046750285062714*100,0.32004581901489115*100]
DT=[0.372405929304447*100, 0.6275940706955531*100,0.3601374570446736*100]
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


Normal=[4385,4385,4385,4385,4385,4385]
TN=[4262,3404 ,3241,3574,3090,2752]
FP=[125,981,1144,811,1295,1633]

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

Affected=[4365,4365,4365,4365,4365,4365]
TP=[3977,3060,2981,2803,2968,2793]
FN=[388,1305,1384,1562,1397,1572]

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




