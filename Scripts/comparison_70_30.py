#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
remain=pd.read_csv("E:/70_30/44th_iteration/Test_44.csv")
remain

remain['predict'] = np.where(remain['target'] ==1, 0, remain['predict'])
remain['predict'] = np.where(remain['target'] ==0, 1, remain['predict'])
remain.to_csv("E:/70_30/FP_FN.CSV")


# In[9]:


total=pd.read_csv("E:/70_30/TP_TN_FP_FN.CSV")
y_test =total[['target']]
y_pre = total[['predict']]
len(total)


# In[10]:


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


# In[11]:


from sklearn.metrics import roc_curve,auc
import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt

frame=pd.read_csv("E:/70_30/TP_TN_FP_FN.CSV")
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


PM=[91.56666666666666,0.866793349168646*100,0.9611251580278128*100]
XGB=[73.70476190476191,0.6914964370546318*100,0.7618549146864859*100]
LR=[70.54285714285714,0.667458432304038*100, 0.7234064462980125*100]
SVM=[72.0904761904762,0.6258432304038005*100,0.7740305522914218*100]
KNN=[69.15714285714286,0.6725890736342043*100,0.7001978239366964*100]
DT=[63.14761904761905, 0.6264133016627078*100,0.6339423076923076*100]
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


PM=[ 0.03522673031026253*100,0.9647732696897374*100,0.13320665083135397*100]
XGB=[0.2171837708830549*100,0.7828162291169452*100,0.3085035629453682*100]
LR=[ 0.2564200477326969*100,0.7435799522673031*100,0.332541567695962*100]
SVM=[ 0.18357995226730311*100,0.8164200477326969*100,0.3741567695961995*100]
KNN=[0.2893556085918854*100,0.7106443914081145*100,0.3274109263657957*100]
DT=[0.363436754176611*100, 0.636563245823389*100,0.3735866983372922*100]
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


Normal=[10475,10475,10475,10475,10475,10475]
TN=[10106,8200 ,7789,8552,7444,6668]
FP=[369,2275,2686,1923,3031,3807]

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


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox ,TextArea,HPacker,VPacker

ax=plt.subplot(111)
n_groups=6

Affected=[10525,10525,10525,10525,10525,10525]
TP=[9123,7278,7025,6587,7079,6593]
FN=[1402,3247,3500,3938,3446,3932]

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




