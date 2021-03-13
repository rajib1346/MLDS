#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
remain=pd.read_csv("E:/60_40/23th_iteration/Test_23.csv")
remain

remain['predict'] = np.where(remain['target'] ==1, 0, remain['predict'])
remain['predict'] = np.where(remain['target'] ==0, 1, remain['predict'])
remain.to_csv("E:/60_40/FP_FN.CSV")


# In[4]:


total=pd.read_csv("E:/60_40/TP_TN_FP_FN.CSV")
y_test =total[['target']]
y_pre = total[['predict']]
len(total)


# In[5]:


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

frame=pd.read_csv("E:/60_40/TP_TN_FP_FN.CSV")
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


# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox ,TextArea,HPacker,VPacker

ax=plt.subplot(111)
n_groups=3


PM=[89.44285714285715,0.8260436668800227*100, 0.9579381443298969*100]
XGB=[73.625, 0.6916293293506863*100,0.7613120400814154*100]
LR=[70.70357142857144,0.6688002275798307*100,0.7261776061776062*100]
SVM=[72.01071428571429,0.6232131427352251*100,0.7753494956644842*100]
KNN=[68.87857142857143,0.6688002275798307*100,0.6986108015749202*100]
DT=[63.535714285714285,  0.6367968138823696*100,0.6369780180692893*100]
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


# In[8]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox ,TextArea,HPacker,VPacker

ax=plt.subplot(111)
n_groups=3


PM=[ 0.036587990530167154*100,0.9634120094698329*100,0.17395633311997727*100]
XGB=[0.21873879044407776*100,0.7812612095559223*100,0.30837067064931367*100]
LR=[0.25439414592151516*100,0.7456058540784848*100,0.3311997724201693*100]
SVM=[0.18215079991391062*100,0.8178492000860894*100,0.3767868572647749*100]
KNN=[0.2910538776095846*100,0.7089461223904154*100,0.3311997724201693*100]
DT=[0.3660951287753784*100, 0.6339048712246216*100,0.36320318611763036*100]
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


# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox ,TextArea,HPacker,VPacker

ax=plt.subplot(111)
n_groups=6


Normal=[13939,13939,13939,13939,13939,13939]
TN=[13429,10890 ,10393,11400,9882,8836]
FP=[510,3049,3546,2539,4057,5103]

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


# In[25]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox ,TextArea,HPacker,VPacker

ax=plt.subplot(111)
n_groups=6

Affected=[14061,14061,14061,14061,14061,14061]
TP=[11615,9725,9404,8763,9404,8954]
FN=[2446,4336,4657,5298,4657,5107]

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




